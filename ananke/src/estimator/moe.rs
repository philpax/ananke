//! MoE estimator.
//!
//! Applies to: llama4, qwen3moe, qwen3vlmoe, deepseek2, mixtral, gpt-oss,
//! glm4moe, qwen35moe, deepseek4, glm-dsa, laguna.
//!
//! Identifies expert tensors by the `_exps` suffix on
//! `blk.N.ffn_{gate,up,down}_exps.weight` and itemises them into
//! `Estimate::expert_tensors`. The estimate keeps every expert's bytes in the
//! full `per_layer_bytes` total; the packer decides which experts to offload
//! (from live VRAM) and synthesises the matching `-ot` rules.

use std::collections::BTreeMap;

use smol_str::SmolStr;

use super::{
    kv,
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
    // DeepSeek-V4-Flash (deepseek4) uses the standard fused-expert tensor
    // layout — `blk.N.ffn_{gate,up,down}_exps.weight` plus a `_shexp`
    // shared expert — so the weight accounting and expert itemisation
    // apply unchanged. Its KV cache does *not*, though: only the ~half of
    // layers whose `attention.compress_ratios` entry is `4` keep a
    // key-only "CSA" (compressed sparse attention) cache of `n_ctx / 4`
    // cells; the rest use a far smaller HCA cache (ratio 128) plus a
    // fixed sliding-window cache. The generic `kv_for_hybrid` would price
    // it at `head_count_kv × (key_length + value_length) × n_layers` ≈ 88
    // KiB/token (11.5 GiB at 128k) versus the measured ~6.65 KiB/token
    // (0.84 GiB at 128k), a 13× over-reservation, so deepseek4 routes to
    // `deepseek4_kv_per_token` below instead.
    "deepseek4",
    // GLM-5 (glm-dsa) pairs the standard fused-expert layout —
    // `blk.N.ffn_{gate,up,down}_exps.weight` plus a `_shexp` shared
    // expert — with DeepSeek-style MLA attention, so the weight
    // accounting applies unchanged. Its KV cache does not: llama.cpp
    // stores no V cache at all for MLA architectures (`has_v =
    // !is_mla`), so the cache is a single K tensor of
    // `attention.key_length` (kv_lora_rank + rope dims, 576 for
    // GLM-5.2) elements per token per layer, and the trailing
    // `nextn_predict_layers` MTP block carries no main-context KV. The
    // generic `kv_for_hybrid` would add a phantom `value_length` V term
    // (a ~1.9× over-reservation), so glm-dsa routes to
    // `mla_kv_per_token` below. Despite the "dsa" in the name, the
    // pinned llama.cpp runs this arch as dense MLA (the deepseek2
    // graph, plain KV cache); the sparse-attention indexer tensors are
    // loaded but only the deepseek32 arch gets the DSA indexer cache.
    "glm-dsa",
    // Laguna MoE: fused-expert layout (`ffn_{gate,up,down}_exps` + `_shexp`
    // shared experts), plain GQA KV (scalar `head_count_kv`, constant
    // `key_length`/`value_length`). The per-layer `attention.head_count`
    // array only sizes Q projections and is irrelevant to KV, so the generic
    // `kv_for_hybrid` path is correct. Advertises `sliding_window` but
    // `kv_for_hybrid` doesn't model SWA eviction — safe over-estimation.
    "laguna",
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

    // KV cost per token. deepseek4's compressed caches need bespoke
    // handling (see `deepseek4_kv_per_token`); every other MoE arch either
    // has plain per-layer KV or the qwen35moe `full_attention_interval`
    // pattern, both covered by the shared hybrid logic.
    let kv_per_token = if arch == "deepseek4" {
        deepseek4_kv_per_token(summary, arch, n_layers, inputs)
    } else if arch == "glm-dsa" {
        mla_kv_per_token(summary, arch, n_layers, inputs)
    } else {
        super::hybrid::kv_for_hybrid(summary, arch, n_layers, inputs)
    };

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
        compute_buffer_mb: inputs.compute_buffer_mb.unwrap_or_else(|| {
            super::compute_buffer::default_for(summary, inputs.context, inputs.ubatch)
        }),
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

/// The `attention.compress_ratios` value that marks a CSA (compressed
/// sparse attention) layer — the only layers whose KV cache scales with
/// context. Layers with the ratio-128 HCA value or the leading `0` full-
/// attention value do not carry a context-scaling cache worth modelling.
const DEEPSEEK4_CSA_RATIO: u32 = 4;

/// f16 KV bytes per context token, per CSA layer, for deepseek4.
///
/// Calibrated, not derived from head dims: a 2×3090 f16 sweep at np=1
/// (2026-07-15) measured total KV of 836 MiB at 131072 context and 1655
/// MiB at 262144 — linear in context (≈ 6.65 KiB/token) across this
/// model's 21 CSA layers, i.e. ~317 bytes/token/layer. The q8_0 sweep
/// reconciled to the same figure once scaled by the element width, which
/// is why the per-token cost is priced off the K-cache element size below.
const DEEPSEEK4_CSA_KV_BYTES_PER_TOKEN_LAYER_F16: f64 = 317.0;

/// KV bytes per context token for deepseek4 (DeepSeek-V4-Flash).
///
/// Only the CSA layers keep a context-scaling cache; it is key-only (the
/// value projection is absorbed into the compressed latent, so llama.cpp
/// reports `V (f16): 0.00 MiB`), so the per-token cost tracks the
/// `cache_type_k` element width alone. The sibling HCA cache (ratio 128 →
/// `n_ctx / 128` cells) and the fixed sliding-window cache are small and
/// context-flat, so they fall into the compute-buffer headroom rather than
/// this per-token term. Returns `kv_per_token` so the packer multiplies by
/// context exactly as it does for every other family.
fn deepseek4_kv_per_token(
    summary: &GgufSummary,
    arch: &str,
    n_layers: u32,
    inputs: &EstimatorInputs<'_>,
) -> u64 {
    if inputs.context == 0 || n_layers == 0 {
        return 0;
    }
    let bytes_k = kv::kv_bytes_per_element(inputs.cache_type_k.unwrap_or("f16"));

    // Count the CSA layers from `attention.compress_ratios` when present;
    // fall back to the observed "roughly half the layers are CSA" ratio so
    // a quant that drops the array still gets a sane, non-zero estimate.
    let csa_layers = summary
        .metadata
        .get(&*format!("{arch}.attention.compress_ratios"))
        .and_then(|v| v.as_u32_array())
        .map(|ratios| ratios.iter().filter(|&&r| r == DEEPSEEK4_CSA_RATIO).count() as u64)
        .filter(|&n| n > 0)
        .unwrap_or((n_layers / 2) as u64);

    // f16 baseline scaled to the actual K-cache element width (f16 = 2.0).
    let per_layer = DEEPSEEK4_CSA_KV_BYTES_PER_TOKEN_LAYER_F16 * (bytes_k / 2.0);
    (csa_layers as f64 * per_layer) as u64
}

/// Fallback `attention.key_length` for MLA archs whose quant dropped the
/// key: kv_lora_rank (512) + rope dims (64), the GLM-5 / DeepSeek-V3
/// compressed-cache width.
const MLA_DEFAULT_KEY_LENGTH: u64 = 576;

/// KV bytes per context token for MLA architectures (glm-dsa).
///
/// llama.cpp allocates no V cache for MLA (`has_v = !is_mla` in
/// `llama-kv-cache.cpp`) — the value states are recovered from the
/// compressed latent via `attn_v_b` at compute time — so the per-token
/// cost is a single K tensor of `attention.key_length` elements per
/// layer, priced off `cache_type_k` alone. The trailing
/// `nextn_predict_layers` MTP block is excluded: llama.cpp's
/// `hparams.n_layer()` (which sizes the main-context cache) subtracts it.
fn mla_kv_per_token(
    summary: &GgufSummary,
    arch: &str,
    n_layers: u32,
    inputs: &EstimatorInputs<'_>,
) -> u64 {
    if inputs.context == 0 || n_layers == 0 {
        return 0;
    }
    let bytes_k = kv::kv_bytes_per_element(inputs.cache_type_k.unwrap_or("f16"));

    let key_length = summary
        .metadata
        .get(&*format!("{arch}.attention.key_length"))
        .and_then(|v| v.as_u32())
        .map(u64::from)
        .unwrap_or(MLA_DEFAULT_KEY_LENGTH);
    let nextn_layers = summary
        .metadata
        .get(&*format!("{arch}.nextn_predict_layers"))
        .and_then(|v| v.as_u32())
        .unwrap_or(0);
    let kv_layers = n_layers.saturating_sub(nextn_layers) as u64;

    (kv_layers as f64 * key_length as f64 * bytes_k) as u64
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
            ubatch: None,
            cache_type_k: None,
            cache_type_v: None,
            override_tensor: &empty,
            compute_buffer_mb: None,
            allow_fallback: false,
            mtp: false,
            draft_model: None,
        };

        let e = estimate(&summary, &inputs);
        // Per-layer KV: 4 heads × (128 + 128) × 2 bytes (f16) = 2048 bytes/token.
        // With interval=4 we only count 2 layers → kv_per_token = 4096.
        // Without the interval handling we'd naively multiply by 8 → 16384.
        assert_eq!(e.kv_per_token, 4096);
    }

    #[test]
    fn deepseek4_kv_uses_csa_layer_count_not_naive_mla() {
        use std::path::Path;

        use crate::{
            estimator::types::EstimatorInputs,
            gguf::types::{GgufSummary, GgufTensor, GgufType, GgufValue},
        };

        // 43-layer deepseek4 shape: layers 0-1 are ratio 0 (full attn),
        // then the rest alternate 4 (CSA) / 128 (HCA) → 21 CSA layers.
        let n_layers = 43u32;
        let compress_ratios: Vec<GgufValue> = (0..n_layers)
            .map(|i| {
                GgufValue::U32(match i {
                    0 | 1 => 0,
                    i if i % 2 == 0 => 4,
                    _ => 128,
                })
            })
            .collect();
        let csa_layers = compress_ratios
            .iter()
            .filter(|v| matches!(v, GgufValue::U32(4)))
            .count();
        assert_eq!(csa_layers, 21, "sanity: fixture has 21 CSA layers");

        let mut tensors = std::collections::BTreeMap::new();
        for layer in 0..n_layers {
            for kind in ["attn_kv", "ffn_gate_exps", "ffn_gate_shexp"] {
                let name = format!("blk.{layer}.{kind}.weight");
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
        }
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new("general.architecture"),
            GgufValue::String("deepseek4".into()),
        );
        metadata.insert(
            SmolStr::new("deepseek4.block_count"),
            GgufValue::U32(n_layers),
        );
        // Naive-MLA metadata that `kv_for_hybrid` would otherwise consume.
        metadata.insert(
            SmolStr::new("deepseek4.attention.head_count_kv"),
            GgufValue::U32(1),
        );
        metadata.insert(
            SmolStr::new("deepseek4.attention.key_length"),
            GgufValue::U32(512),
        );
        metadata.insert(
            SmolStr::new("deepseek4.attention.value_length"),
            GgufValue::U32(512),
        );
        metadata.insert(
            SmolStr::new("deepseek4.attention.compress_ratios"),
            GgufValue::Array(compress_ratios),
        );

        let summary = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors,
            metadata,
            block_count: Some(n_layers),
            architecture: SmolStr::new("deepseek4"),
            shards: vec!["/fake".into()],
        };

        let empty: Vec<String> = Vec::new();
        let inputs = EstimatorInputs {
            name: "demo",
            model: Path::new("/fake"),
            mmproj: None,
            context: 131072,
            ubatch: None,
            cache_type_k: Some("f16"),
            cache_type_v: Some("f16"),
            override_tensor: &empty,
            compute_buffer_mb: None,
            allow_fallback: false,
            mtp: false,
            draft_model: None,
        };

        let e = estimate(&summary, &inputs);
        // 21 CSA layers × 317 B/token (f16) = 6657 B/token — matches the
        // measured ~6.65 KiB/token, an order of magnitude below the naive
        // MLA formula (1 kv-head × (512+512) × 2 × 43 = 88_064 B/token).
        assert_eq!(e.kv_per_token, 6657);
        assert!(
            e.kv_per_token < 88_064 / 10,
            "deepseek4 KV must be far below the naive MLA estimate; got {}",
            e.kv_per_token
        );
        // Expert itemisation still works: 43 fused gate tensors, shared
        // experts (`_shexp`) excluded.
        assert_eq!(e.expert_tensors.as_ref().unwrap().len(), 43);
    }

    #[test]
    fn deepseek4_kv_scales_with_cache_type() {
        use std::path::Path;

        use crate::{
            estimator::types::EstimatorInputs,
            gguf::types::{GgufSummary, GgufValue},
        };

        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new("general.architecture"),
            GgufValue::String("deepseek4".into()),
        );
        metadata.insert(SmolStr::new("deepseek4.block_count"), GgufValue::U32(43));
        // No compress_ratios → falls back to n_layers / 2 = 21 CSA layers.
        let summary = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors: std::collections::BTreeMap::new(),
            metadata,
            block_count: Some(43),
            architecture: SmolStr::new("deepseek4"),
            shards: vec!["/fake".into()],
        };
        let empty: Vec<String> = Vec::new();
        let mk = |ctk: &'static str| EstimatorInputs {
            name: "demo",
            model: Path::new("/fake"),
            mmproj: None,
            context: 131072,
            ubatch: None,
            cache_type_k: Some(ctk),
            cache_type_v: Some(ctk),
            override_tensor: &empty,
            compute_buffer_mb: None,
            allow_fallback: false,
            mtp: false,
            draft_model: None,
        };
        // Fallback layer count (21) at f16 reproduces the 6657 figure.
        assert_eq!(estimate(&summary, &mk("f16")).kv_per_token, 6657);
        // q8_0 K-cache (1.0625 B/elem vs 2.0) shrinks the per-token cost.
        assert!(estimate(&summary, &mk("q8_0")).kv_per_token < 6657);
    }

    #[test]
    fn glm_dsa_kv_is_key_only_and_excludes_nextn_layers() {
        use std::path::Path;

        use crate::{
            estimator::types::EstimatorInputs,
            gguf::types::{GgufSummary, GgufValue},
        };

        // GLM-5.2 shape: 79 blocks, 1 NextN layer, MLA cache of 576
        // K elements per token per layer, phantom value_length of 512
        // that must NOT be priced.
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new("general.architecture"),
            GgufValue::String("glm-dsa".into()),
        );
        metadata.insert(SmolStr::new("glm-dsa.block_count"), GgufValue::U32(79));
        metadata.insert(
            SmolStr::new("glm-dsa.attention.head_count_kv"),
            GgufValue::U32(1),
        );
        metadata.insert(
            SmolStr::new("glm-dsa.attention.key_length"),
            GgufValue::U32(576),
        );
        metadata.insert(
            SmolStr::new("glm-dsa.attention.value_length"),
            GgufValue::U32(512),
        );
        metadata.insert(
            SmolStr::new("glm-dsa.nextn_predict_layers"),
            GgufValue::U32(1),
        );
        let summary = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors: std::collections::BTreeMap::new(),
            metadata,
            block_count: Some(79),
            architecture: SmolStr::new("glm-dsa"),
            shards: vec!["/fake".into()],
        };
        let empty: Vec<String> = Vec::new();
        let mk = |ctk: Option<&'static str>| EstimatorInputs {
            name: "demo",
            model: Path::new("/fake"),
            mmproj: None,
            context: 32768,
            ubatch: None,
            cache_type_k: ctk,
            cache_type_v: None,
            override_tensor: &empty,
            compute_buffer_mb: None,
            allow_fallback: false,
            mtp: false,
            draft_model: None,
        };
        // 78 KV layers × 576 elems × 2 bytes (f16) = 89856 bytes/token.
        // The naive K+V formula would give 79 × (576 + 512) × 2 = 171904.
        assert_eq!(estimate(&summary, &mk(None)).kv_per_token, 89_856);
        // q8_0 K-cache shrinks it by the element-width ratio; the V type
        // is irrelevant because no V cache exists.
        assert_eq!(
            estimate(&summary, &mk(Some("q8_0"))).kv_per_token,
            (78.0f64 * 576.0 * 1.0625) as u64
        );
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
            ubatch: None,
            cache_type_k: None,
            cache_type_v: None,
            override_tensor: &empty_override,
            compute_buffer_mb: None,
            allow_fallback: false,
            mtp: false,
            draft_model: None,
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

    #[test]
    fn laguna_kv_uses_scalar_head_count_kv_not_variable_head_count() {
        // Laguna's Q-projection head count is a per-layer array, but KV uses
        // a scalar `head_count_kv` — the array must not leak into the KV term.
        use std::path::Path;

        use crate::{
            estimator::types::EstimatorInputs,
            gguf::types::{GgufSummary, GgufTensor, GgufType, GgufValue},
        };

        let n_layers = 48u32;
        let mut tensors = std::collections::BTreeMap::new();
        // Layer 0 is dense (no experts); layers 1..47 are MoE.
        for layer in 0..n_layers {
            for kind in ["attn_q", "attn_k", "attn_v", "attn_output"] {
                let name = format!("blk.{layer}.{kind}.weight");
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
            if layer > 0 {
                // Fused routed experts + shared expert per MoE layer.
                for kind in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
                    let name = format!("blk.{layer}.{kind}.weight");
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
                for kind in ["ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"] {
                    let name = format!("blk.{layer}.{kind}.weight");
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
            }
        }

        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new("general.architecture"),
            GgufValue::String("laguna".into()),
        );
        metadata.insert(SmolStr::new("laguna.block_count"), GgufValue::U32(n_layers));
        // The variable Q head count — must not affect KV.
        let head_count: Vec<GgufValue> = (0..n_layers)
            .map(|i| GgufValue::U32(if i % 4 == 0 { 48 } else { 72 }))
            .collect();
        metadata.insert(
            SmolStr::new("laguna.attention.head_count"),
            GgufValue::Array(head_count),
        );
        // Scalar KV head count — this is what kv_for_hybrid reads.
        metadata.insert(
            SmolStr::new("laguna.attention.head_count_kv"),
            GgufValue::U32(8),
        );
        metadata.insert(
            SmolStr::new("laguna.attention.key_length"),
            GgufValue::U32(128),
        );
        metadata.insert(
            SmolStr::new("laguna.attention.value_length"),
            GgufValue::U32(128),
        );
        metadata.insert(
            SmolStr::new("laguna.attention.sliding_window"),
            GgufValue::U32(512),
        );

        let summary = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors,
            metadata,
            block_count: Some(n_layers),
            architecture: SmolStr::new("laguna"),
            shards: vec!["/fake".into()],
        };

        let empty: Vec<String> = Vec::new();
        let inputs = EstimatorInputs {
            name: "demo",
            model: Path::new("/fake"),
            mmproj: None,
            context: 32768,
            ubatch: None,
            cache_type_k: Some("f16"),
            cache_type_v: Some("f16"),
            override_tensor: &empty,
            compute_buffer_mb: None,
            allow_fallback: false,
            mtp: false,
            draft_model: None,
        };

        let e = estimate(&summary, &inputs);

        // KV must use the scalar head_count_kv (8), not the variable
        // head_count array (48/72). 48 layers × 8 heads × (128+128) × 2
        // bytes (f16) = 196_608 bytes/token.
        assert_eq!(e.kv_per_token, 196_608);

        // 47 MoE layers × 3 fused expert projections (gate/up/down) =
        // 141 itemised expert tensors. The dense layer 0 has none, and
        // the `_shexp` shared experts are excluded (always-on, not
        // offloadable).
        let experts = e.expert_tensors.expect("MoE arch must itemise experts");
        assert_eq!(experts.len(), 141);
        assert_eq!(e.expert_layers.len(), 47);
        // Layer 0 (dense) must not appear in the expert layer list.
        assert!(!e.expert_layers.contains(&0u32));

        // q8_0 KV shrinks by the element-width ratio (1.0625 / 2.0).
        let inputs_q8 = EstimatorInputs {
            cache_type_k: Some("q8_0"),
            cache_type_v: Some("q8_0"),
            ..inputs
        };
        let e_q8 = estimate(&summary, &inputs_q8);
        assert_eq!(e_q8.kv_per_token, 104_448);
        assert!(e_q8.kv_per_token < e.kv_per_token);
    }
}
