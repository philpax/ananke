//! Hybrid-architecture estimator (SSM + attention, no MoE).
//!
//! Applies to: jamba, qwen35.
//!
//! These models mix standard attention layers with SSM (State Space Model)
//! layers. Only every `full_attention_interval`-th layer runs full attention
//! with a KV cache; the rest run a recurrent SSM that carries constant
//! per-layer state instead of context-dependent KV.
//!
//! Weight accounting uses the same `blk.N.*` layout as llama-family.
//! KV cache is scaled down by the attention interval.

use std::collections::BTreeMap;

use smol_str::SmolStr;

use super::{
    kv,
    llama::{collect_non_layer, collect_per_layer},
    types::{Estimate, EstimatorInputs},
};
use crate::gguf::GgufSummary;

/// Architectures that mix attention with recurrent SSM layers (no MoE).
pub const HYBRID_FAMILY: &[&str] = &["jamba", "qwen35"];

pub fn is_hybrid(arch: &str) -> bool {
    HYBRID_FAMILY.contains(&arch)
}

/// Compute the KV cost per token for a hybrid model, scaling by
/// `full_attention_interval`. Only every N-th layer has a KV cache;
/// the rest use SSM with constant per-layer state.
///
/// Returns `kv_per_token` so the caller can multiply by context to
/// recover total KV bytes.
pub fn kv_for_hybrid(
    summary: &GgufSummary,
    arch: &str,
    n_layers: u32,
    inputs: &EstimatorInputs<'_>,
) -> u64 {
    let cache_k = inputs.cache_type_k.unwrap_or("f16");
    let cache_v = inputs.cache_type_v.unwrap_or("f16");
    let bytes_k = kv::kv_bytes_per_element(cache_k);
    let bytes_v = kv::kv_bytes_per_element(cache_v);

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

    // `full_attention_interval = N`: only every N-th layer runs full
    // attention; the rest are SSM with no KV cache. Absent / 1 = every
    // layer has KV (the jamba case).
    let full_attention_interval = summary
        .metadata
        .get(&*format!("{arch}.full_attention_interval"))
        .and_then(|v| v.as_u32())
        .unwrap_or(1)
        .max(1);
    let kv_layer_count = (n_layers / full_attention_interval) as u64;

    if kv_layer_count > 0 && n_kv_heads > 0 {
        let per_layer_bytes_kv =
            n_kv_heads * ((key_length as f64 * bytes_k) + (value_length as f64 * bytes_v)) as u64;
        kv_layer_count * per_layer_bytes_kv
    } else {
        0
    }
}

pub fn estimate(summary: &GgufSummary, inputs: &EstimatorInputs<'_>) -> Estimate {
    let arch = summary.architecture.as_str();
    let n_layers = summary.block_count.unwrap_or(0);

    let per_layer = collect_per_layer(summary, n_layers);
    let non_layer = collect_non_layer(summary);
    let weights_bytes = per_layer.iter().sum::<u64>()
        + non_layer.output_head_bytes
        + non_layer.token_embd_bytes
        + non_layer.other_bytes;

    let kv_per_token = kv_for_hybrid(summary, arch, n_layers, inputs);

    Estimate {
        weights_bytes,
        kv_per_token,
        compute_buffer_mb: inputs
            .compute_buffer_mb
            .unwrap_or_else(|| super::compute_buffer::default_for(summary, inputs.context)),
        per_layer_bytes: Some(per_layer),
        attention_layers: None,
        non_layer,
        override_tensor_bytes: BTreeMap::new(),
        expert_layers: Vec::new(),
        expert_layer_cpu_bytes: BTreeMap::new(),
        context: inputs.context,
        architecture: SmolStr::new(arch),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::gguf::types::{GgufSummary, GgufTensor, GgufType, GgufValue};

    fn fake_hybrid_summary(arch: &str, n_layers: u32, interval: Option<u32>) -> GgufSummary {
        let mut tensors = std::collections::BTreeMap::new();
        for layer in 0..n_layers {
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
        tensors.insert(
            SmolStr::new("output.weight"),
            GgufTensor {
                name: SmolStr::new("output.weight"),
                dtype: GgufType::F16,
                shape: vec![1024 * 1024],
                byte_size: 2 * 1024 * 1024,
                shard_idx: 0,
                offset: 0,
            },
        );
        tensors.insert(
            SmolStr::new("token_embd.weight"),
            GgufTensor {
                name: SmolStr::new("token_embd.weight"),
                dtype: GgufType::F16,
                shape: vec![2 * 1024 * 1024],
                byte_size: 4 * 1024 * 1024,
                shard_idx: 0,
                offset: 0,
            },
        );

        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new("general.architecture"),
            GgufValue::String(arch.into()),
        );
        metadata.insert(
            SmolStr::new(format!("{arch}.block_count")),
            GgufValue::U32(n_layers),
        );
        metadata.insert(
            SmolStr::new(format!("{arch}.attention.head_count_kv")),
            GgufValue::U32(4),
        );
        metadata.insert(
            SmolStr::new(format!("{arch}.attention.key_length")),
            GgufValue::U32(128),
        );
        metadata.insert(
            SmolStr::new(format!("{arch}.attention.value_length")),
            GgufValue::U32(128),
        );
        if let Some(interval) = interval {
            metadata.insert(
                SmolStr::new(format!("{arch}.full_attention_interval")),
                GgufValue::U32(interval),
            );
        }

        GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors,
            metadata,
            block_count: Some(n_layers),
            architecture: SmolStr::new(arch),
            shards: vec!["/fake".into()],
        }
    }

    fn inputs<'a>(context: u32, empty: &'a [String]) -> EstimatorInputs<'a> {
        EstimatorInputs {
            name: "demo",
            model: Path::new("/fake"),
            mmproj: None,
            context,
            cache_type_k: Some("f16"),
            cache_type_v: Some("f16"),
            override_tensor: empty,
            n_cpu_moe: None,
            compute_buffer_mb: None,
            allow_fallback: false,
        }
    }

    #[test]
    fn qwen35_kv_scales_with_full_attention_interval() {
        // 64 layers, interval=4 → 16 attention layers.
        // Per-layer KV: 4 heads × (128+128) × 2 bytes (f16) = 2048 bytes.
        // kv_per_token = 16 × 2048 = 32768.
        let s = fake_hybrid_summary("qwen35", 64, Some(4));
        let empty: Vec<String> = Vec::new();
        let e = estimate(&s, &inputs(4096, &empty));
        assert_eq!(e.kv_per_token, 32768);
    }

    #[test]
    fn jamba_kv_no_interval_scales_all_layers() {
        // No full_attention_interval key → defaults to 1 (all layers).
        let s = fake_hybrid_summary("jamba", 80, None);
        let empty: Vec<String> = Vec::new();
        let e = estimate(&s, &inputs(4096, &empty));
        // 80 layers × 2048 bytes = 163840.
        assert_eq!(e.kv_per_token, 163840);
    }

    #[test]
    fn hybrid_is_recognised() {
        assert!(is_hybrid("jamba"));
        assert!(is_hybrid("qwen35"));
    }
}
