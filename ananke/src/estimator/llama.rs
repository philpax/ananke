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

    let cache_k = inputs.cache_type_k.unwrap_or("f16");
    let cache_v = inputs.cache_type_v.unwrap_or("f16");

    let bytes_k = kv::kv_bytes_per_element(cache_k);
    let bytes_v = kv::kv_bytes_per_element(cache_v);

    let kv_per_token = if n_layers > 0 && n_kv_heads > 0 {
        let per_layer_bytes_kv =
            n_kv_heads * ((key_length as f64 * bytes_k) + (value_length as f64 * bytes_v)) as u64;
        n_layers as u64 * per_layer_bytes_kv
    } else {
        0
    };

    Estimate {
        weights_bytes,
        kv_per_token,
        compute_buffer_mb: inputs.compute_buffer_mb.unwrap_or(400),
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
