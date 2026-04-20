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

/// Hardcoded sliding-window group length for architectures where llama.cpp
/// knows the pattern but the GGUF doesn't expose it. Gemma 2 / Gemma 3 use
/// 1 global-attention layer per every 6 layers (5 SWA + 1 global); the HF
/// config isn't round-tripped, so this constant mirrors the value baked
/// into llama.cpp's `LLM_ARCH_GEMMA*` handling.
///
/// Architectures that ship the pattern as a per-layer bool mask (e.g.
/// `gemma4.attention.sliding_window_pattern`) do *not* go through this
/// function — they use the mask directly.
pub fn hardcoded_swa_group_size(arch: &str) -> Option<u32> {
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
    // Gemma 4: attention.head_count_kv is a per-layer array (like deci),
    // attention.sliding_window_pattern is a per-layer bool mask, and
    // SWA layers use `*_length_swa` head dims distinct from the full-
    // attention layers' `*_length`. All of this is handled below; the
    // tensor layout is unchanged from llama-family.
    "gemma4",
    // Gemma 3n: MatFormer + PLE variant sharing gemma4's metadata schema
    // (per-layer `sliding_window_pattern` bool mask, `shared_kv_layers`).
    // head_count_kv is a scalar here rather than a per-layer array, but
    // `compute_kv_per_token` already handles the scalar case uniformly.
    // The architecture's 2+ GiB `per_layer_token_embd.weight` (PLE table)
    // is routed to CPU by the existing gemma4 non-layer special case, so
    // the packer's GPU pledge matches llama.cpp's actual placement.
    // MatFormer altup/laurel tensors live under `blk.N.*` and are picked
    // up by `collect_per_layer` automatically.
    "gemma3n",
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

    let kv_per_token = compute_kv_per_token(summary, arch, n_layers, inputs);

    Estimate {
        weights_bytes,
        kv_per_token,
        compute_buffer_mb: inputs
            .compute_buffer_mb
            .unwrap_or_else(|| super::compute_buffer::default_for(summary, inputs.context)),
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

/// Compute the `kv_per_token` term for a llama-family model, handling
/// three independent knobs that can all appear together:
///
/// - **Scalar vs per-layer `head_count_kv`.** Nvidia's `deci` and Gemma 4
///   both store `attention.head_count_kv` as a length-`n_layer` array;
///   every other family we've seen stores a scalar.
/// - **Sliding-window attention.** Gemma 2 / Gemma 3 use a hardcoded 1:5
///   global:SWA pattern (see `hardcoded_swa_group_size`); Gemma 4 stores
///   a per-layer bool mask in `attention.sliding_window_pattern` plus
///   separate K/V head dims for its SWA layers. On SWA layers the KV
///   cost caps at `min(context, sliding_window)` tokens.
/// - **Shared-KV layers.** Gemma 4 exposes `attention.shared_kv_layers`
///   N: the last N layers reuse the preceding layers' KV and therefore
///   don't contribute additional cache bytes.
///
/// The result is folded back into `kv_per_token × context = total KV
/// bytes` so the packer's downstream math stays identical.
fn compute_kv_per_token(
    summary: &GgufSummary,
    arch: &str,
    n_layers: u32,
    inputs: &super::types::EstimatorInputs<'_>,
) -> u64 {
    let cache_k = inputs.cache_type_k.unwrap_or("f16");
    let cache_v = inputs.cache_type_v.unwrap_or("f16");
    let bytes_k = kv::kv_bytes_per_element(cache_k);
    let bytes_v = kv::kv_bytes_per_element(cache_v);

    // Head-count-KV can be a scalar (broadcast across all layers) or a
    // per-layer array. Materialise a vector of length `n_layers` so the
    // loop below treats both uniformly.
    let kv_heads_raw: Vec<u32> = summary
        .metadata
        .get(&*format!("{arch}.attention.head_count_kv"))
        .and_then(|v| v.as_u32_array())
        .unwrap_or_default();
    let kv_heads_per_layer: Vec<u32> = if kv_heads_raw.len() == 1 {
        vec![kv_heads_raw[0]; n_layers as usize]
    } else {
        kv_heads_raw
    };
    if kv_heads_per_layer.is_empty() || kv_heads_per_layer.len() != n_layers as usize {
        return 0;
    }

    // Per-head byte widths. Gemma 4 uses different head dims for SWA
    // layers (`key_length_swa` / `value_length_swa`); everyone else
    // reuses the same pair for both.
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
    let key_length_swa = summary
        .metadata
        .get(&*format!("{arch}.attention.key_length_swa"))
        .and_then(|v| v.as_u32())
        .map(|v| v as u64)
        .unwrap_or(key_length);
    let value_length_swa = summary
        .metadata
        .get(&*format!("{arch}.attention.value_length_swa"))
        .and_then(|v| v.as_u32())
        .map(|v| v as u64)
        .unwrap_or(value_length);
    let per_head_full = ((key_length as f64 * bytes_k) + (value_length as f64 * bytes_v)) as u64;
    let per_head_swa =
        ((key_length_swa as f64 * bytes_k) + (value_length_swa as f64 * bytes_v)) as u64;

    // Build the per-layer SWA mask. Three sources, tried in order:
    //   1. An explicit per-layer bool array (gemma4).
    //   2. A hardcoded group size for architectures where llama.cpp
    //      bakes the pattern in (gemma2 / gemma3).
    //   3. No SWA — every layer is full attention.
    let sliding_window = summary
        .metadata
        .get(&*format!("{arch}.attention.sliding_window"))
        .and_then(|v| v.as_u32())
        .map(|v| v as u64);
    let pattern_mask: Option<Vec<bool>> = summary
        .metadata
        .get(&*format!("{arch}.attention.sliding_window_pattern"))
        .and_then(|v| v.as_bool_array())
        .filter(|m| m.len() == n_layers as usize);
    let is_swa_layer: Vec<bool> = match (&pattern_mask, hardcoded_swa_group_size(arch)) {
        (Some(mask), _) => mask.clone(),
        (None, Some(group)) if group > 0 => (0..n_layers)
            // Pattern `g` means every `g`-th layer (1-indexed) is global
            // — matches both llama.cpp's gemma3 handling and the "1
            // global per 6" HF config.
            .map(|i| (i + 1) % group != 0)
            .collect(),
        _ => vec![false; n_layers as usize],
    };

    // `shared_kv_layers = N` means the last N layers reuse earlier
    // layers' KV and contribute no additional cache bytes. Absent key =
    // 0 shared = every layer unique.
    let shared_kv_layers = summary
        .metadata
        .get(&*format!("{arch}.attention.shared_kv_layers"))
        .and_then(|v| v.as_u32())
        .unwrap_or(0);
    let unique_kv_count = (n_layers as u64).saturating_sub(shared_kv_layers as u64);

    // Walk the first `unique_kv_count` layers, summing the per-layer
    // cache bytes. SWA layers cap at the window size; full-attention
    // layers scale with the full context. We return `kv_per_token` so
    // the packer can multiply by context and recover the total.
    let context = inputs.context as u64;
    if context == 0 {
        return 0;
    }
    let mut total_kv_bytes = 0u64;
    for i in 0..unique_kv_count as usize {
        let kv_heads = kv_heads_per_layer[i] as u64;
        let (per_head, tokens) = if is_swa_layer[i] {
            let window = sliding_window.unwrap_or(context);
            (per_head_swa, context.min(window))
        } else {
            (per_head_full, context)
        };
        total_kv_bytes += kv_heads * per_head * tokens;
    }
    total_kv_bytes / context
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
            // Gemma 4's `per_layer_token_embd.weight` is a 42-slot embedding
            // stack (one per transformer block) that llama.cpp keeps on CPU
            // alongside `token_embd.weight`. For the E4B quant it's ~2.8 GiB
            // — bucketing it as GPU-resident caused the packer to over-
            // reserve a small single-GPU fit by ~3 GiB.
            "token_embd.weight" | "per_layer_token_embd.weight" => {
                nl.token_embd_bytes += tensor.byte_size
            }
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

    #[test]
    fn gemma4_is_llama_family() {
        // Gemma 4 reuses the llama-family tensor layout. It's handled via
        // compute_kv_per_token's per-layer bool mask + separate SWA head
        // dim paths, not a distinct estimator.
        assert!(is_llama_family("gemma4"));
    }

    #[test]
    fn gemma3n_is_llama_family() {
        // Gemma 3n (MatFormer / PLE variant) shares gemma4's metadata
        // schema. The estimator needs to recognise it so that the service
        // doesn't flip to `Disabled { ConfigError }` on first-Ensure.
        assert!(is_llama_family("gemma3n"));
    }

    #[test]
    fn gemma3n_ple_tensor_is_cpu_resident() {
        // The E4B quant carries a ~2.3 GiB `per_layer_token_embd.weight`
        // PLE table plus MatFormer `altup_*` / `per_layer_*_proj.weight`
        // tensors. The PLE must land on CPU (same rule as gemma4 above);
        // altup/proj are tiny and fall through to `other_bytes` which is
        // fine — they genuinely do live on GPU at runtime.
        let mut tensors = std::collections::BTreeMap::new();
        tensors.insert(
            SmolStr::new("token_embd.weight"),
            tensor("token_embd.weight", 352 * 1024 * 1024),
        );
        tensors.insert(
            SmolStr::new("per_layer_token_embd.weight"),
            tensor("per_layer_token_embd.weight", 2380 * 1024 * 1024),
        );
        tensors.insert(
            SmolStr::new("altup_proj.weight"),
            tensor("altup_proj.weight", 24 * 1024 * 1024),
        );
        tensors.insert(
            SmolStr::new("altup_unembd_proj.weight"),
            tensor("altup_unembd_proj.weight", 24 * 1024 * 1024),
        );
        tensors.insert(
            SmolStr::new("per_layer_model_proj.weight"),
            tensor("per_layer_model_proj.weight", 35 * 1024 * 1024),
        );
        let summary = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors,
            metadata: std::collections::BTreeMap::new(),
            block_count: Some(0),
            architecture: SmolStr::new("gemma3n"),
            shards: vec!["/fake".into()],
        };
        let nl = collect_non_layer(&summary);
        // PLE + token_embd both route to CPU.
        assert_eq!(nl.token_embd_bytes, (352 + 2380) * 1024 * 1024);
        // altup + per_layer_model_proj are GPU-resident (together < 100 MiB).
        assert_eq!(nl.other_bytes, (24 + 24 + 35) * 1024 * 1024);
        // No explicit output head — gemma3n uses weight-tied output via
        // token_embd, so `output_head_bytes` stays zero.
        assert_eq!(nl.output_head_bytes, 0);
    }

    #[test]
    fn per_layer_token_embd_is_cpu_resident() {
        // Gemma 4 E-variants carry a large `per_layer_token_embd.weight`
        // tensor (2.8 GiB for E4B) that llama.cpp keeps on CPU alongside
        // `token_embd.weight`. Bucketing it as GPU-resident caused the
        // packer to over-reserve a single-GPU fit by ~3 GiB.
        let mut tensors = std::collections::BTreeMap::new();
        tensors.insert(
            SmolStr::new("token_embd.weight"),
            tensor("token_embd.weight", 100 * 1024 * 1024),
        );
        tensors.insert(
            SmolStr::new("per_layer_token_embd.weight"),
            tensor("per_layer_token_embd.weight", 300 * 1024 * 1024),
        );
        tensors.insert(
            SmolStr::new("output_norm.weight"),
            tensor("output_norm.weight", 1024),
        );
        let summary = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors,
            metadata: std::collections::BTreeMap::new(),
            block_count: Some(0),
            architecture: SmolStr::new("gemma4"),
            shards: vec!["/fake".into()],
        };
        let nl = collect_non_layer(&summary);
        assert_eq!(nl.token_embd_bytes, 400 * 1024 * 1024);
        assert_eq!(nl.other_bytes, 1024);
    }

    /// Build a gemma4-shaped summary with a given per-layer SWA mask and
    /// head-count-KV array so the KV computation can be exercised end-to-end.
    fn gemma4_summary(is_swa: &[bool], kv_heads: &[u32], sliding_window: u32) -> GgufSummary {
        assert_eq!(is_swa.len(), kv_heads.len());
        let n_layers = is_swa.len() as u32;
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new("general.architecture"),
            GgufValue::String("gemma4".into()),
        );
        metadata.insert(SmolStr::new("gemma4.block_count"), GgufValue::U32(n_layers));
        metadata.insert(
            SmolStr::new("gemma4.attention.head_count_kv"),
            GgufValue::Array(kv_heads.iter().map(|h| GgufValue::U32(*h)).collect()),
        );
        metadata.insert(
            SmolStr::new("gemma4.attention.key_length"),
            GgufValue::U32(512),
        );
        metadata.insert(
            SmolStr::new("gemma4.attention.value_length"),
            GgufValue::U32(512),
        );
        metadata.insert(
            SmolStr::new("gemma4.attention.key_length_swa"),
            GgufValue::U32(256),
        );
        metadata.insert(
            SmolStr::new("gemma4.attention.value_length_swa"),
            GgufValue::U32(256),
        );
        metadata.insert(
            SmolStr::new("gemma4.attention.sliding_window"),
            GgufValue::U32(sliding_window),
        );
        metadata.insert(
            SmolStr::new("gemma4.attention.sliding_window_pattern"),
            GgufValue::Array(is_swa.iter().map(|b| GgufValue::Bool(*b)).collect()),
        );
        GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors: std::collections::BTreeMap::new(),
            metadata,
            block_count: Some(n_layers),
            architecture: SmolStr::new("gemma4"),
            shards: vec!["/fake".into()],
        }
    }

    #[test]
    fn gemma4_swa_uses_per_layer_mask_and_swa_head_dims() {
        // 4 layers: layer 2 is full-attention, the rest are SWA. Each layer
        // has 8 KV heads. Context = 4096, window = 1024 — SWA layers cap
        // their cache at the window; full layer uses the entire context.
        let mask = [true, true, false, true];
        let heads = [8u32, 8, 8, 8];
        let s = gemma4_summary(&mask, &heads, 1024);
        let empty: Vec<String> = Vec::new();
        let e = estimate(&s, &inputs("f16", "f16", 4096, &empty));

        // Per-head bytes: f16 (2 b) × (512+512) = 2048 for full, × (256+256) = 1024 for SWA.
        // Layer cost (K+V bytes per layer at this context):
        //   SWA layer: 8 × 1024 × 1024 = 8_388_608 bytes
        //   Full layer: 8 × 2048 × 4096 = 67_108_864 bytes
        // Total: 3 × 8_388_608 + 1 × 67_108_864 = 92_274_688 bytes
        // kv_per_token × context = total bytes, so kv_per_token = 22_528.
        let total_kv = e.kv_per_token * e.context as u64;
        assert_eq!(total_kv, 92_274_688);
    }

    #[test]
    fn gemma4_shared_kv_layers_skip_cache() {
        // Same 4-layer model as above, but the last layer is marked as a
        // shared-KV slot — it must contribute zero KV bytes.
        let mask = [true, true, false, true];
        let heads = [8u32, 8, 8, 8];
        let mut s = gemma4_summary(&mask, &heads, 1024);
        s.metadata.insert(
            SmolStr::new("gemma4.attention.shared_kv_layers"),
            GgufValue::U32(1),
        );
        let empty: Vec<String> = Vec::new();
        let e = estimate(&s, &inputs("f16", "f16", 4096, &empty));
        // Total must drop by one SWA layer's worth (8_388_608 bytes).
        let total_kv = e.kv_per_token * e.context as u64;
        assert_eq!(total_kv, 92_274_688 - 8_388_608);
    }
}
