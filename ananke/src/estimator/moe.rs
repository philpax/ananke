//! MoE estimator.
//!
//! Applies to: llama4, qwen3moe, qwen3vlmoe, deepseek2, mixtral, gpt-oss.
//!
//! Identifies expert tensors by the `_exps` suffix on
//! `blk.N.ffn_{gate,up,down}_exps.weight` and itemises them into
//! `Estimate::expert_tensors`. The estimate keeps every expert's bytes in the
//! full `per_layer_bytes` total; the packer decides which experts to offload
//! (from live VRAM) and synthesises the matching `-ot` rules.

use std::collections::BTreeMap;

use smol_str::SmolStr;

use super::{
    llama::{collect_non_layer, layer_index},
    types::{Estimate, EstimatorInputs, ExpertKind, ExpertTensor},
};
use crate::gguf::GgufSummary;

pub const MOE_FAMILY: &[&str] = &[
    "llama4",
    "qwen3moe",
    "qwen3vlmoe",
    "deepseek2",
    "mixtral",
    "gpt-oss",
    // GLM-4.5 series (including glm-4-5-air) uses the standard MoE tensor
    // layout: `blk.N.ffn_{gate,up,down}_exps.weight` + shared expert tensors
    // (`_shexp`). Without this entry the dispatcher falls through to the
    // generic fallback, which has no per-layer breakdown — and the
    // operator's CPU-offload `override_tensor` regex then zeroes the
    // weight estimate entirely, leading to 400 MiB predicted vs 27 GiB
    // observed (a 67× under-reservation).
    "glm4moe",
    // Qwen 3.5+ MoE is a hybrid: every `full_attention_interval`-th layer
    // runs full attention (with KV cache); the others run a linear-
    // attention / gated-delta-net SSM that carries constant per-layer
    // state instead of context-dependent KV. Every layer still has the
    // standard `blk.N.ffn_{gate,up,down}_exps.weight` tensors so the
    // MoE weight accounting applies as-is; the attention interval drops
    // KV by ~`1 / interval`. The SSM state bytes are small (<100 MiB
    // total across all recurrent layers for typical sizes) and are
    // absorbed by the compute-buffer headroom rather than modelled
    // explicitly.
    "qwen35moe",
];

pub fn is_moe(arch: &str) -> bool {
    MOE_FAMILY.contains(&arch)
}

pub fn estimate(summary: &GgufSummary, inputs: &EstimatorInputs<'_>) -> Estimate {
    let arch = summary.architecture.as_str();
    let n_layers = summary.block_count.unwrap_or(0);

    // Per-layer split into {non-expert, expert} bytes, and itemise every
    // offloadable fused expert tensor for the packer.
    let mut per_layer_nonexp = vec![0u64; n_layers as usize];
    let mut per_layer_exp = vec![0u64; n_layers as usize];
    let mut expert_tensors: Vec<ExpertTensor> = Vec::new();

    for (name, t) in &summary.tensors {
        let Some(idx) = layer_index(name) else {
            continue;
        };
        if (idx as usize) >= per_layer_nonexp.len() {
            continue;
        }
        if let Some(kind) = expert_kind(name) {
            per_layer_exp[idx as usize] += t.byte_size;
            expert_tensors.push(ExpertTensor {
                layer: idx,
                kind,
                bytes: t.byte_size,
            });
        } else {
            per_layer_nonexp[idx as usize] += t.byte_size;
        }
    }

    let non_layer = collect_non_layer(summary);

    // Full per-layer cost (non-expert + experts). Experts are itemised in
    // `expert_tensors` but stay counted here; the packer subtracts them only
    // for the experts it actually moves off the GPU.
    let per_layer_total: Vec<u64> = per_layer_nonexp
        .iter()
        .zip(per_layer_exp.iter())
        .map(|(a, b)| *a + *b)
        .collect();

    let weights_bytes = per_layer_total.iter().sum::<u64>()
        + non_layer.output_head_bytes
        + non_layer.token_embd_bytes
        + non_layer.other_bytes;

    // Hybrid families (qwen35moe) expose `full_attention_interval`:
    // only every N-th layer runs full attention; the rest are SSM.
    // Reuse the shared hybrid KV logic.
    let kv_per_token = super::hybrid::kv_for_hybrid(summary, arch, n_layers, inputs);

    let expert_layers: Vec<u32> = per_layer_exp
        .iter()
        .enumerate()
        .filter_map(|(i, b)| if *b > 0 { Some(i as u32) } else { None })
        .collect();

    // Stable order so the packer's offload selection and the synthesised `-ot`
    // rules are deterministic across runs.
    expert_tensors.sort_by_key(|e| (e.layer, e.kind));

    Estimate {
        weights_bytes,
        kv_per_token,
        compute_buffer_mb: inputs
            .compute_buffer_mb
            .unwrap_or_else(|| super::compute_buffer::default_for(summary, inputs.context)),
        mtp_bytes: 0,
        per_layer_bytes: Some(per_layer_total),
        attention_layers: None,
        non_layer,
        override_tensor_bytes: BTreeMap::new(),
        expert_layers,
        expert_tensors: Some(expert_tensors),
        context: inputs.context,
        architecture: SmolStr::new(arch),
    }
}

/// Classify an expert weight tensor by its projection.
/// Pattern: `blk.N.ffn_{gate,up,down}_exps.weight`. The `_shexp` (shared
/// expert) counterparts are *not* offloadable experts and return `None`.
pub(crate) fn expert_kind(name: &str) -> Option<ExpertKind> {
    let rest = name.strip_prefix("blk.")?;
    let (_, kind) = rest.split_once('.')?;
    if kind.contains("shexp") {
        return None;
    }
    if kind.starts_with("ffn_gate_exps") {
        Some(ExpertKind::Gate)
    } else if kind.starts_with("ffn_up_exps") {
        Some(ExpertKind::Up)
    } else if kind.starts_with("ffn_down_exps") {
        Some(ExpertKind::Down)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen35moe_kv_scales_with_full_attention_interval() {
        use std::path::Path;

        use crate::{
            estimator::types::EstimatorInputs,
            gguf::types::{GgufSummary, GgufTensor, GgufType, GgufValue},
        };

        let mut tensors = std::collections::BTreeMap::new();
        for layer in 0..8u32 {
            let name = format!("blk.{layer}.attn_q.weight");
            tensors.insert(
                SmolStr::new(&name),
                GgufTensor {
                    name: SmolStr::new(&name),
                    dtype: GgufType::F16,
                    shape: vec![512 * 1024],
                    byte_size: 1024 * 1024,
                    shard_idx: 0,
                    offset: 0,
                },
            );
        }
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new("general.architecture"),
            GgufValue::String("qwen35moe".into()),
        );
        metadata.insert(SmolStr::new("qwen35moe.block_count"), GgufValue::U32(8));
        metadata.insert(
            SmolStr::new("qwen35moe.attention.head_count_kv"),
            GgufValue::U32(4),
        );
        metadata.insert(
            SmolStr::new("qwen35moe.attention.key_length"),
            GgufValue::U32(128),
        );
        metadata.insert(
            SmolStr::new("qwen35moe.attention.value_length"),
            GgufValue::U32(128),
        );
        // Every 4th layer is full-attention → 2 layers × KV; the other 6
        // are recurrent and contribute no KV.
        metadata.insert(
            SmolStr::new("qwen35moe.full_attention_interval"),
            GgufValue::U32(4),
        );

        let summary = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors,
            metadata,
            block_count: Some(8),
            architecture: SmolStr::new("qwen35moe"),
            shards: vec!["/fake".into()],
        };

        let empty: Vec<String> = Vec::new();
        let inputs = EstimatorInputs {
            name: "demo",
            model: Path::new("/fake"),
            mmproj: None,
            context: 4096,
            cache_type_k: None,
            cache_type_v: None,
            override_tensor: &empty,
            compute_buffer_mb: None,
            allow_fallback: false,
            mtp: false,
        };

        let e = estimate(&summary, &inputs);
        // Per-layer KV: 4 heads × (128 + 128) × 2 bytes (f16) = 2048 bytes/token.
        // With interval=4 we only count 2 layers → kv_per_token = 4096.
        // Without the interval handling we'd naively multiply by 8 → 16384.
        assert_eq!(e.kv_per_token, 4096);
    }

    #[test]
    fn expert_pattern_matches() {
        assert_eq!(
            expert_kind("blk.0.ffn_gate_exps.weight"),
            Some(ExpertKind::Gate)
        );
        assert_eq!(
            expert_kind("blk.1.ffn_up_exps.weight"),
            Some(ExpertKind::Up)
        );
        assert_eq!(
            expert_kind("blk.5.ffn_down_exps.weight"),
            Some(ExpertKind::Down)
        );
        assert_eq!(expert_kind("blk.0.ffn_gate.weight"), None);
        assert_eq!(expert_kind("blk.0.ffn_gate_shexp.weight"), None);
        assert_eq!(expert_kind("output.weight"), None);
    }

    #[test]
    fn itemises_expert_tensors_with_full_per_layer() {
        use std::path::Path;

        use smol_str::SmolStr;

        use crate::{
            estimator::types::EstimatorInputs,
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

        let empty_override: Vec<String> = Vec::new();
        let inputs = EstimatorInputs {
            name: "demo",
            model: Path::new("/fake"),
            mmproj: None,
            context: 4096,
            cache_type_k: None,
            cache_type_v: None,
            override_tensor: &empty_override,
            compute_buffer_mb: None,
            allow_fallback: false,
            mtp: false,
        };

        let e = estimate(&summary, &inputs);
        // Every layer's experts are itemised; nothing is pre-offloaded.
        let experts = e.expert_tensors.expect("MoE arch must itemise experts");
        assert_eq!(experts.len(), 3, "one fused gate tensor per layer");
        // Sorted by (layer, kind).
        assert_eq!(experts[0].layer, 0);
        assert_eq!(experts[0].bytes, 4 * 1024 * 1024);
        assert_eq!(experts[1].layer, 1);
        assert_eq!(experts[1].bytes, 10 * 1024 * 1024);
        assert_eq!(experts[2].layer, 2);
        assert_eq!(experts[2].bytes, 2 * 1024 * 1024);
        // per_layer_bytes keeps the full cost (1 MiB attn + experts).
        let per_layer = e.per_layer_bytes.expect("per-layer breakdown");
        assert_eq!(per_layer[1], (1 + 10) * 1024 * 1024);
    }
}
