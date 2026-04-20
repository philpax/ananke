//! Llama-family estimator.
//!
//! Applies to: llama, qwen2, qwen3, mistral, gemma(1/2/3), phi3, glm4.
//!
//! Per spec §8.3: weights = Σ per-layer tensor bytes + non-layer bytes;
//! kv_per_token = n_layers × n_kv_heads ×
//!                (key_length × bytes(cache_k) + value_length × bytes(cache_v)).

use std::collections::BTreeMap;

use smol_str::SmolStr;

use super::{
    kv,
    types::{Estimate, EstimatorInputs, NonLayer},
};
use crate::gguf::GgufSummary;

/// Llama-family architectures that use sliding-window attention, and the
/// length of the SWA pattern (`N-1` SWA layers per `1` global-attention
/// layer in each group of `N`). Returns `None` for architectures that
/// use full attention everywhere.
///
/// Gemma 2 / Gemma 3 GGUFs advertise `{arch}.attention.sliding_window`
/// but *not* the pattern length; llama.cpp hardcodes the group size
/// because the HF config isn't round-tripped. The 1-global-per-6 ratio
/// comes from Gemma's reference: "1 global : 5 local per group".
pub fn sliding_window_pattern_for(arch: &str) -> Option<u32> {
    match arch {
        "gemma2" | "gemma3" => Some(6),
        _ => None,
    }
}

pub const LLAMA_FAMILY: &[&str] = &[
    "llama", "qwen2", "qwen3", "mistral", "gemma", "gemma2", "gemma3", "phi3", "glm4",
    // NVIDIA Nemotron ("deci") is a Llama derivative with a compressed attention
    // stack. Same `blk.N.*` tensor naming and same `{arch}.attention.*` metadata
    // keys so the llama-family estimator works unchanged; added here so
    // `dispatch` routes it away from the weights-only fallback that leaves
    // `per_layer_bytes = None` and breaks multi-GPU layer splits.
    "deci",
];

pub fn is_llama_family(arch: &str) -> bool {
    LLAMA_FAMILY.contains(&arch)
}

pub fn estimate(summary: &GgufSummary, inputs: &EstimatorInputs<'_>) -> Estimate {
    let arch = summary.architecture.as_str();
    let n_layers = summary.block_count.unwrap_or(0);

    let per_layer_bytes = collect_per_layer(summary, n_layers);
    let non_layer = collect_non_layer(summary);

    let weights_bytes = per_layer_bytes.iter().sum::<u64>()
        + non_layer.output_head_bytes
        + non_layer.token_embd_bytes
        + non_layer.other_bytes;

    // `{arch}.attention.head_count_kv` may be a scalar (constant across
    // layers) or an array of length `n_layers` (per-layer, e.g. Nvidia's
    // `deci` / Nemotron which aggressively varies attention capacity
    // across blocks). `as_u32_array` coerces both shapes into a vector so
    // the sum below is the total KV-head count across the whole model —
    // which is what multiplies the per-head bytes for `kv_per_token`.
    let kv_heads_per_layer: Vec<u32> = summary
        .metadata
        .get(&*format!("{arch}.attention.head_count_kv"))
        .and_then(|v| v.as_u32_array())
        .unwrap_or_default();
    let total_kv_heads: u64 = kv_heads_per_layer.iter().map(|&h| h as u64).sum();
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

    let cache_k = inputs.cache_type_k.unwrap_or("f16");
    let cache_v = inputs.cache_type_v.unwrap_or("f16");

    let bytes_k = kv::kv_bytes_per_element(cache_k);
    let bytes_v = kv::kv_bytes_per_element(cache_v);

    // Per-head KV bytes are constant; the total for the model is
    // `sum(heads_per_layer) × per_head`. When the metadata is a scalar,
    // `as_u32_array` returns a one-element vector so we multiply by
    // n_layers ourselves.
    let per_head_bytes =
        ((key_length as f64 * bytes_k) + (value_length as f64 * bytes_v)) as u64;
    let full_layers_kv = if kv_heads_per_layer.len() == 1 {
        // Scalar → one entry in the vector; broadcast across layers.
        n_layers as u64 * kv_heads_per_layer[0] as u64
    } else if !kv_heads_per_layer.is_empty() {
        total_kv_heads
    } else {
        0
    };

    // Sliding-window attention: gemma2/gemma3 use a pattern of N-1 local
    // (SWA, bounded to `sliding_window` tokens) + 1 global per group.
    // Llama.cpp hardcodes the pattern length because gemma GGUFs don't
    // expose it; see `sliding_window_pattern_for`. Other llama-family
    // members either don't use SWA (pattern returns None) or expose
    // enough metadata to do so in the future.
    let sliding_window = summary
        .metadata
        .get(&*format!("{arch}.attention.sliding_window"))
        .and_then(|v| v.as_u32());
    let swa_pattern = sliding_window_pattern_for(arch);
    let kv_per_token = match (sliding_window, swa_pattern) {
        (Some(window), Some(pattern)) if pattern > 0 && n_layers > 0 => {
            // In a model with pattern `p`, layers with index `i % p ==
            // p - 1` run global attention (full context); the rest use
            // SWA and cache at most `window` tokens per layer.
            let global_layers = (n_layers / pattern) as u64;
            let swa_layers = n_layers as u64 - global_layers;
            // `kv_per_token × context` needs to equal the real total
            // bytes, so fold the SWA cap into an effective per-token
            // cost. When `context ≤ window` the SWA layers still scale
            // linearly with context so the cap is a no-op.
            let swa_fraction = if inputs.context == 0 || inputs.context as u64 <= window as u64 {
                1.0
            } else {
                window as f64 / inputs.context as f64
            };
            let scalar = if kv_heads_per_layer.len() == 1 {
                kv_heads_per_layer[0] as u64
            } else if !kv_heads_per_layer.is_empty() {
                // Variable-per-layer KV + SWA is a combination we haven't
                // seen in the wild yet. Use the layer-count average and
                // let rolling correction catch any residual.
                full_layers_kv.saturating_div(n_layers as u64)
            } else {
                0
            };
            let global_kv = global_layers * scalar * per_head_bytes;
            let swa_kv =
                (swa_layers * scalar * per_head_bytes) as f64 * swa_fraction;
            global_kv + swa_kv as u64
        }
        _ => full_layers_kv * per_head_bytes,
    };

    Estimate {
        weights_bytes,
        kv_per_token,
        compute_buffer_mb: inputs
            .compute_buffer_mb
            .unwrap_or_else(|| super::compute_buffer::default_for(inputs.context)),
        per_layer_bytes: Some(per_layer_bytes),
        attention_layers: None,
        non_layer,
        override_tensor_bytes: BTreeMap::new(),
        expert_layers: Vec::new(),
        expert_layer_cpu_bytes: BTreeMap::new(),
        context: inputs.context,
        architecture: SmolStr::new(arch),
    }
}

pub(crate) fn collect_per_layer(summary: &GgufSummary, n_layers: u32) -> Vec<u64> {
    let mut out = vec![0u64; n_layers as usize];
    for tensor in summary.tensors.values() {
        if let Some(idx) = layer_index(&tensor.name)
            && (idx as usize) < out.len()
        {
            out[idx as usize] += tensor.byte_size;
        }
    }
    out
}

pub(crate) fn collect_non_layer(summary: &GgufSummary) -> NonLayer {
    let mut nl = NonLayer::default();
    for (name, tensor) in &summary.tensors {
        if layer_index(name).is_some() {
            continue;
        }
        match name.as_str() {
            "output.weight" => nl.output_head_bytes += tensor.byte_size,
            "token_embd.weight" => nl.token_embd_bytes += tensor.byte_size,
            _ => nl.other_bytes += tensor.byte_size,
        }
    }
    nl
}

/// Extract the N in a tensor name like `blk.N.attn_q.weight`.
pub(crate) fn layer_index(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("blk.")?;
    let (idx, _) = rest.split_once('.')?;
    idx.parse().ok()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use smol_str::SmolStr;

    use super::*;
    use crate::gguf::types::{GgufSummary, GgufTensor, GgufType, GgufValue};

    fn tensor(name: &str, bytes: u64) -> GgufTensor {
        GgufTensor {
            name: SmolStr::new(name),
            dtype: GgufType::F16,
            shape: vec![bytes / 2],
            byte_size: bytes,
            shard_idx: 0,
            offset: 0,
        }
    }

    fn fake_summary() -> GgufSummary {
        let mut tensors = std::collections::BTreeMap::new();
        // 2 layers × 3 tensors per layer.
        for layer in 0..2u32 {
            for kind in ["attn_q", "attn_k", "ffn_down"] {
                let name = format!("blk.{layer}.{kind}.weight");
                tensors.insert(SmolStr::new(&name), tensor(&name, 1024 * 1024));
            }
        }
        tensors.insert(
            SmolStr::new("output.weight"),
            tensor("output.weight", 2 * 1024 * 1024),
        );
        tensors.insert(
            SmolStr::new("token_embd.weight"),
            tensor("token_embd.weight", 4 * 1024 * 1024),
        );

        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new("general.architecture"),
            GgufValue::String("qwen3".into()),
        );
        metadata.insert(SmolStr::new("qwen3.block_count"), GgufValue::U32(2));
        metadata.insert(
            SmolStr::new("qwen3.attention.head_count_kv"),
            GgufValue::U32(4),
        );
        metadata.insert(
            SmolStr::new("qwen3.attention.key_length"),
            GgufValue::U32(128),
        );
        metadata.insert(
            SmolStr::new("qwen3.attention.value_length"),
            GgufValue::U32(128),
        );

        GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 6 * 1024 * 1024 + 6 * 1024 * 1024,
            tensors,
            metadata,
            block_count: Some(2),
            architecture: SmolStr::new("qwen3"),
            shards: vec!["/fake".into()],
        }
    }

    fn inputs<'a>(
        cache_k: &'a str,
        cache_v: &'a str,
        context: u32,
        empty: &'a [String],
    ) -> EstimatorInputs<'a> {
        EstimatorInputs {
            name: "demo",
            model: Path::new("/fake"),
            mmproj: None,
            context,
            cache_type_k: Some(cache_k),
            cache_type_v: Some(cache_v),
            override_tensor: empty,
            n_cpu_moe: None,
            compute_buffer_mb: None,
            allow_fallback: false,
        }
    }

    #[test]
    fn sums_per_layer_and_non_layer() {
        let s = fake_summary();
        let empty: Vec<String> = Vec::new();
        let e = estimate(&s, &inputs("f16", "f16", 4096, &empty));
        // per-layer: 2 layers × 3 tensors × 1 MiB = 6 MiB weights from layers.
        // non-layer: 2 MiB output + 4 MiB token_embd = 6 MiB.
        assert_eq!(e.weights_bytes, 12 * 1024 * 1024);
        assert_eq!(e.per_layer_bytes.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn kv_uses_arch_metadata() {
        let s = fake_summary();
        let empty: Vec<String> = Vec::new();
        let e = estimate(&s, &inputs("f16", "f16", 4096, &empty));
        // n_layers=2, n_kv=4, k=v=128, 2 bytes/element (f16).
        // per_layer_kv = 4 × (128*2 + 128*2) = 4 × 512 = 2048 bytes.
        // kv_per_token = 2 × 2048 = 4096 bytes.
        assert_eq!(e.kv_per_token, 4096);
    }

    #[test]
    fn kv_quantised_shrinks() {
        let s = fake_summary();
        let empty: Vec<String> = Vec::new();
        let e_q8 = estimate(&s, &inputs("q8_0", "q8_0", 4096, &empty));
        let e_f16 = estimate(&s, &inputs("f16", "f16", 4096, &empty));
        assert!(e_q8.kv_per_token < e_f16.kv_per_token);
    }

    #[test]
    fn layer_index_extracts() {
        assert_eq!(layer_index("blk.0.attn_q.weight"), Some(0));
        assert_eq!(layer_index("blk.42.ffn_down.weight"), Some(42));
        assert_eq!(layer_index("output.weight"), None);
        assert_eq!(layer_index("token_embd.weight"), None);
    }

    #[test]
    fn deci_is_llama_family() {
        // Regression: Nemotron-49B (arch = "deci") used to fall through to the
        // fallback estimator which returns `per_layer_bytes = None`, breaking
        // multi-GPU layer splits. It must be recognised as llama-family so the
        // per-layer walk runs.
        assert!(is_llama_family("deci"));
    }
}
