//! MoE estimator.
//!
//! Applies to: llama4, qwen3moe, qwen3vlmoe, deepseek2, mixtral, gpt-oss.
//!
//! Identifies expert tensors by the `_exps` suffix on
//! `blk.N.ffn_{gate,up,down}_exps.weight`. When `n_cpu_moe > 0`, the
//! top-N expert-bearing layers move their expert bytes to CPU.

use std::{cmp::Reverse, collections::BTreeMap};

use smol_str::SmolStr;

use super::{
    kv,
    llama::{collect_non_layer, layer_index},
    types::Estimate,
};
use crate::{config::ServiceConfig, gguf::GgufSummary};

pub const MOE_FAMILY: &[&str] = &[
    "llama4",
    "qwen3moe",
    "qwen3vlmoe",
    "deepseek2",
    "mixtral",
    "gpt-oss",
];

pub fn is_moe(arch: &str) -> bool {
    MOE_FAMILY.contains(&arch)
}

pub fn estimate(summary: &GgufSummary, svc: &ServiceConfig) -> Estimate {
    let arch = summary.architecture.as_str();
    let context = svc.raw.context.unwrap_or(4096);

    let n_layers = summary.block_count.unwrap_or(0);

    // Per-layer split into {non-expert, expert} bytes.
    let mut per_layer_nonexp = vec![0u64; n_layers as usize];
    let mut per_layer_exp = vec![0u64; n_layers as usize];

    for (name, t) in &summary.tensors {
        let Some(idx) = layer_index(name) else {
            continue;
        };
        if (idx as usize) >= per_layer_nonexp.len() {
            continue;
        }
        if is_expert_tensor(name) {
            per_layer_exp[idx as usize] += t.byte_size;
        } else {
            per_layer_nonexp[idx as usize] += t.byte_size;
        }
    }

    let non_layer = collect_non_layer(summary);

    let n_cpu_moe = svc.raw.n_cpu_moe.unwrap_or(0) as usize;

    // Pick the top-N layers by expert byte count for offload.
    let mut layer_scores: Vec<(u32, u64)> = per_layer_exp
        .iter()
        .enumerate()
        .map(|(i, b)| (i as u32, *b))
        .collect();
    layer_scores.sort_by_key(|&(_, b)| Reverse(b));
    let offload_layers: Vec<u32> = layer_scores
        .iter()
        .take(n_cpu_moe)
        .map(|(i, _)| *i)
        .collect();

    // Build per-layer total (non-expert + kept experts).
    let mut per_layer_total: Vec<u64> = per_layer_nonexp
        .iter()
        .zip(per_layer_exp.iter())
        .map(|(a, b)| *a + *b)
        .collect();
    let mut expert_layer_cpu_bytes = BTreeMap::new();
    for layer in &offload_layers {
        let exp_bytes = per_layer_exp[*layer as usize];
        per_layer_total[*layer as usize] =
            per_layer_total[*layer as usize].saturating_sub(exp_bytes);
        expert_layer_cpu_bytes.insert(*layer, exp_bytes);
    }

    let weights_bytes = per_layer_total.iter().sum::<u64>()
        + non_layer.output_head_bytes
        + non_layer.token_embd_bytes
        + non_layer.other_bytes;

    // KV: same formula as llama-family.
    let n_kv_heads = summary
        .metadata
        .get(&*format!("{arch}.attention.head_count_kv"))
        .and_then(|v| v.as_u32())
        .unwrap_or(0) as u64;
    let key_length = summary
        .metadata
        .get(&*format!("{arch}.attention.key_length"))
        .and_then(|v| v.as_u32())
        .unwrap_or(128) as u64;
    let value_length = summary
        .metadata
        .get(&*format!("{arch}.attention.value_length"))
        .and_then(|v| v.as_u32())
        .unwrap_or(128) as u64;

    let cache_k = svc.raw.cache_type_k.as_deref().unwrap_or("f16");
    let cache_v = svc.raw.cache_type_v.as_deref().unwrap_or("f16");

    let kv_per_token = if n_layers > 0 && n_kv_heads > 0 {
        let per_layer_bytes_kv = n_kv_heads
            * ((key_length as f64 * kv::kv_bytes_per_element(cache_k))
                + (value_length as f64 * kv::kv_bytes_per_element(cache_v))) as u64;
        n_layers as u64 * per_layer_bytes_kv
    } else {
        0
    };

    let expert_layers: Vec<u32> = per_layer_exp
        .iter()
        .enumerate()
        .filter_map(|(i, b)| if *b > 0 { Some(i as u32) } else { None })
        .collect();

    Estimate {
        weights_bytes,
        kv_per_token,
        compute_buffer_mb: svc
            .raw
            .estimation
            .as_ref()
            .and_then(|e| e.compute_buffer_mb)
            .unwrap_or(400),
        per_layer_bytes: Some(per_layer_total),
        attention_layers: None,
        non_layer,
        override_tensor_bytes: BTreeMap::new(),
        expert_layers,
        expert_layer_cpu_bytes,
        context,
        architecture: SmolStr::new(arch),
    }
}

/// Does this tensor name denote an expert weight?
/// Pattern: `blk.N.ffn_{gate,up,down}_exps.weight` (and `_shexp` counterparts
/// are *not* considered experts for offload purposes).
pub(crate) fn is_expert_tensor(name: &str) -> bool {
    if let Some(rest) = name.strip_prefix("blk.")
        && let Some((_, kind)) = rest.split_once('.')
    {
        return (kind.starts_with("ffn_gate_exps")
            || kind.starts_with("ffn_up_exps")
            || kind.starts_with("ffn_down_exps"))
            && !kind.contains("shexp");
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_pattern_matches() {
        assert!(is_expert_tensor("blk.0.ffn_gate_exps.weight"));
        assert!(is_expert_tensor("blk.1.ffn_up_exps.weight"));
        assert!(is_expert_tensor("blk.5.ffn_down_exps.weight"));
        assert!(!is_expert_tensor("blk.0.ffn_gate.weight"));
        assert!(!is_expert_tensor("blk.0.ffn_gate_shexp.weight"));
        assert!(!is_expert_tensor("output.weight"));
    }

    #[test]
    fn n_cpu_moe_offloads_top_layers() {
        use smol_str::SmolStr;

        use crate::{
            config::validate::{
                DeviceSlot, PlacementPolicy, ServiceConfig, test_fixtures::minimal_service,
            },
            gguf::types::{GgufSummary, GgufTensor, GgufType, GgufValue},
        };

        let mut tensors = std::collections::BTreeMap::new();
        for layer in 0..3u32 {
            // Base layer tensors: 1 MiB.
            let attn = format!("blk.{layer}.attn_q.weight");
            tensors.insert(
                SmolStr::new(&attn),
                GgufTensor {
                    name: SmolStr::new(&attn),
                    dtype: GgufType::F16,
                    shape: vec![512 * 1024],
                    byte_size: 1024 * 1024,
                    shard_idx: 0,
                    offset: 0,
                },
            );
            // Expert tensors: different size by layer.
            let size = match layer {
                0 => 4,
                1 => 10,
                2 => 2,
                _ => unreachable!(),
            };
            let exp = format!("blk.{layer}.ffn_gate_exps.weight");
            tensors.insert(
                SmolStr::new(&exp),
                GgufTensor {
                    name: SmolStr::new(&exp),
                    dtype: GgufType::F16,
                    shape: vec![size * 512 * 1024],
                    byte_size: size * 1024 * 1024,
                    shard_idx: 0,
                    offset: 0,
                },
            );
        }
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new("general.architecture"),
            GgufValue::String("qwen3moe".into()),
        );
        metadata.insert(SmolStr::new("qwen3moe.block_count"), GgufValue::U32(3));

        let summary = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors,
            metadata,
            block_count: Some(3),
            architecture: SmolStr::new("qwen3moe"),
            shards: vec!["/fake".into()],
        };

        let mut svc: ServiceConfig = minimal_service("demo");
        svc.placement_override.clear();
        svc.placement_override.insert(DeviceSlot::Gpu(0), 1000);
        svc.placement_policy = PlacementPolicy::Hybrid;
        svc.raw.model = Some("/fake".into());
        svc.raw.context = Some(4096);
        svc.raw.n_cpu_moe = Some(1);
        svc.raw.flash_attn = Some(true);

        let e = estimate(&summary, &svc);
        // The layer with the largest expert bytes is layer 1 (10 MiB).
        assert_eq!(e.expert_layer_cpu_bytes.len(), 1);
        assert!(e.expert_layer_cpu_bytes.contains_key(&1));
        assert_eq!(e.expert_layer_cpu_bytes[&1], 10 * 1024 * 1024);
    }
}
