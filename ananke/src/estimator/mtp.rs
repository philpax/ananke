//! Multi-token-prediction (MTP / NextN) draft-context overhead.
//!
//! MTP ships in two shapes, and the estimator models both:
//!
//! **Embedded head (Qwen 3.6).** With `--spec-type draft-mtp` and no separate
//! draft GGUF, llama.cpp creates a second context against the *same* target
//! model. Its KV cache covers only the trailing `nextn_predict_layers` blocks
//! — the dense-attention MTP head — and uses the *draft* cache types, which
//! default to f16 regardless of the main context's `--cache-type-*`. No extra
//! weights load: the nextn-layer tensors live in the target GGUF and are
//! resident even without MTP. So the cost is `nextn KV (f16) + a roughly
//! constant compute buffer`. Calibrated against llama.cpp's own `[spec]
//! estimated memory usage of MTP context` figure on Qwen 3.6 27B (`qwen35`,
//! 4 KV heads) and 35B-A3B (`qwen35moe`, 2 KV heads) across context
//! (262144/524288) and parallelism (np 1/2): the KV term tracks `kv_heads ×
//! context` exactly, and the compute term sits at ~1.55–1.61 GiB independent
//! of both knobs (it is driven by the shared-tokenizer logit buffer at
//! `n_ubatch`, not model width).
//!
//! **Separate draft model (Gemma 4).** With `-md <file>` the MTP head is a
//! standalone GGUF (Gemma 4's `gemma4-assistant`, a 4-block model). Its
//! attention layers *share the target model's KV cache* — confirmed in the
//! load log (`llama_kv_cache: layer 3: sharing with layer 59`) — so it adds
//! no context-scaling KV. The whole cost is its GPU-resident weights
//! (everything but the CPU-side token embeddings) plus a small, roughly
//! constant draft compute/logit buffer. Calibrated against the production
//! 2×3090 run (peak 40858 MiB total) minus the target+mmproj estimate: the
//! draft contributes ~400 MiB, of which ~108 MiB is weights.

use super::types::EstimatorInputs;
use crate::gguf::GgufSummary;

/// Per-MTP-context compute buffer, MiB, for the *embedded* head. Rounded up to
/// stay above every observed datapoint (1553 MiB on 35B, 1609 MiB on 27B) with
/// headroom.
const MTP_COMPUTE_MIB: u64 = 1700;

/// Compute/logit buffer, MiB, for a *separate* draft model loaded via `-md`.
/// Far smaller than the embedded head's buffer: the separate draft reuses the
/// target's KV and its sampler runs on the CPU (`backend offload failed; using
/// CPU sampler`), so only a modest logit/workspace buffer lands on GPU.
/// Calibrated so `draft weights (~108 MiB) + this ≈ 400 MiB` matches the
/// measured Gemma 4 draft contribution, with a little headroom.
const DRAFT_MODEL_COMPUTE_MIB: u64 = 300;

/// Bytes per KV element for the MTP draft cache. llama.cpp leaves the draft
/// cache type at f16 unless `--spec-draft-cache-type-*` is set (ananke never
/// sets it), so this is independent of the main `cache_type_k`/`_v`.
const MTP_KV_BYTES_PER_ELEMENT: u64 = 2;

/// Default per-head K/V dimension when the GGUF omits the attention length
/// keys — matches the llama-family estimator's fallback.
const DEFAULT_HEAD_DIM: u64 = 128;

/// GPU-resident weight bytes for a separate draft model: every tensor except
/// the token embeddings, which llama.cpp keeps on CPU (same rule as the target
/// model's `token_embd.weight`).
fn draft_model_gpu_weight_bytes(draft: &GgufSummary) -> u64 {
    let token_embd = draft
        .tensors
        .get("token_embd.weight")
        .map(|t| t.byte_size)
        .unwrap_or(0);
    draft.total_tensor_bytes.saturating_sub(token_embd)
}

/// Extra VRAM (bytes) a separate draft model (`-md`) adds: its GPU-resident
/// weights plus a fixed draft compute buffer. The draft's attention layers
/// reuse the target's KV cache, so there is no context-scaling KV term.
fn separate_draft_overhead_bytes(draft: &GgufSummary) -> u64 {
    draft_model_gpu_weight_bytes(draft) + DRAFT_MODEL_COMPUTE_MIB * 1024 * 1024
}

/// Extra VRAM (bytes) the MTP draft context adds, or 0 when MTP is off or the
/// model carries no MTP head (`{arch}.nextn_predict_layers` absent or zero and
/// no separate draft model).
///
/// When `draft` is `Some`, the service runs with a separate draft GGUF (`-md`)
/// and the overhead is read from that file. Otherwise the target model's
/// embedded MTP head is modelled.
///
/// `inputs.context` is the configured total context. The embedded-head KV
/// scales with it linearly the same way the main KV does — total KV tokens
/// equal the context budget whether the main cache is unified (np auto) or
/// split per-slot (np > 1), so the estimator does not need to know the
/// parallelism.
pub fn mtp_overhead_bytes(
    summary: &GgufSummary,
    draft: Option<&GgufSummary>,
    inputs: &EstimatorInputs<'_>,
) -> u64 {
    if !inputs.mtp {
        return 0;
    }
    if let Some(draft) = draft {
        return separate_draft_overhead_bytes(draft);
    }
    let arch = summary.architecture.as_str();
    let nextn = meta_u32(summary, arch, "nextn_predict_layers").unwrap_or(0) as u64;
    if nextn == 0 {
        // `--spec-type draft-mtp` was requested but this model has no MTP
        // head; llama.cpp would refuse to draft, so there is no extra cost.
        return 0;
    }
    // The MTP head is a full-attention layer; `head_count_kv` is a scalar on
    // the qwen35 / qwen35moe families that ship MTP heads today.
    let n_kv_heads = meta_attn_u32(summary, arch, "head_count_kv").unwrap_or(0) as u64;
    if n_kv_heads == 0 {
        return 0;
    }
    let key_length =
        meta_attn_u32(summary, arch, "key_length").unwrap_or(DEFAULT_HEAD_DIM as u32) as u64;
    let value_length =
        meta_attn_u32(summary, arch, "value_length").unwrap_or(DEFAULT_HEAD_DIM as u32) as u64;
    let context = inputs.context as u64;

    let kv_bytes =
        nextn * n_kv_heads * (key_length + value_length) * MTP_KV_BYTES_PER_ELEMENT * context;
    kv_bytes + MTP_COMPUTE_MIB * 1024 * 1024
}

fn meta_u32(summary: &GgufSummary, arch: &str, key: &str) -> Option<u32> {
    summary
        .metadata
        .get(&*format!("{arch}.{key}"))
        .and_then(|v| v.as_u32())
}

fn meta_attn_u32(summary: &GgufSummary, arch: &str, key: &str) -> Option<u32> {
    summary
        .metadata
        .get(&*format!("{arch}.attention.{key}"))
        .and_then(|v| v.as_u32())
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, path::Path};

    use smol_str::SmolStr;

    use super::*;
    use crate::gguf::types::{GgufSummary, GgufValue};

    fn qwen35_summary(arch: &str, nextn: u32, kv_heads: u32) -> GgufSummary {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            SmolStr::new("general.architecture"),
            GgufValue::String(arch.into()),
        );
        metadata.insert(
            SmolStr::new(format!("{arch}.nextn_predict_layers")),
            GgufValue::U32(nextn),
        );
        metadata.insert(
            SmolStr::new(format!("{arch}.attention.head_count_kv")),
            GgufValue::U32(kv_heads),
        );
        metadata.insert(
            SmolStr::new(format!("{arch}.attention.key_length")),
            GgufValue::U32(256),
        );
        metadata.insert(
            SmolStr::new(format!("{arch}.attention.value_length")),
            GgufValue::U32(256),
        );
        GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors: BTreeMap::new(),
            metadata,
            block_count: Some(65),
            architecture: SmolStr::new(arch),
            shards: vec!["/fake".into()],
        }
    }

    fn inputs(context: u32, mtp: bool, empty: &[String]) -> EstimatorInputs<'_> {
        EstimatorInputs {
            name: "demo",
            model: Path::new("/fake"),
            mmproj: None,
            context,
            ubatch: None,
            cache_type_k: Some("q8_0"),
            cache_type_v: Some("q8_0"),
            override_tensor: empty,
            compute_buffer_mb: None,
            allow_fallback: false,
            mtp,
            draft_model: None,
        }
    }

    #[test]
    fn zero_when_mtp_disabled() {
        let s = qwen35_summary("qwen35", 1, 4);
        let empty: Vec<String> = Vec::new();
        assert_eq!(
            mtp_overhead_bytes(&s, None, &inputs(262144, false, &empty)),
            0
        );
    }

    #[test]
    fn zero_when_no_mtp_head() {
        // MTP requested but the model has nextn_predict_layers = 0.
        let s = qwen35_summary("qwen35", 0, 4);
        let empty: Vec<String> = Vec::new();
        assert_eq!(
            mtp_overhead_bytes(&s, None, &inputs(262144, true, &empty)),
            0
        );
    }

    #[test]
    fn qwen35_27b_matches_measured() {
        // 27B: nextn=1, 4 KV heads, 256+256, f16 draft cache, ctx 262144.
        // KV = 1 × 4 × 512 × 2 × 262144 = 1024 MiB; + 1700 MiB compute.
        let s = qwen35_summary("qwen35", 1, 4);
        let empty: Vec<String> = Vec::new();
        let got = mtp_overhead_bytes(&s, None, &inputs(262144, true, &empty));
        let mib = got / (1024 * 1024);
        assert_eq!(mib, 1024 + 1700);
    }

    #[test]
    fn uses_f16_draft_cache_not_main_cache_type() {
        // The inputs declare q8_0 for the main cache, but the MTP KV must be
        // sized in f16 (the draft cache default). 1024 MiB KV proves f16
        // (q8_0 would give ~544 MiB).
        let s = qwen35_summary("qwen35", 1, 4);
        let empty: Vec<String> = Vec::new();
        let kv_mib =
            mtp_overhead_bytes(&s, None, &inputs(262144, true, &empty)) / (1024 * 1024) - 1700;
        assert_eq!(kv_mib, 1024);
    }

    #[test]
    fn qwen35moe_35b_doubled_context() {
        // 35B-A3B: nextn=1, 2 KV heads, ctx 524288 (the doubled deploy).
        // KV = 1 × 2 × 512 × 2 × 524288 = 1024 MiB; + 1700 compute.
        let s = qwen35_summary("qwen35moe", 1, 2);
        let empty: Vec<String> = Vec::new();
        let mib = mtp_overhead_bytes(&s, None, &inputs(524288, true, &empty)) / (1024 * 1024);
        assert_eq!(mib, 1024 + 1700);
    }

    /// Build a separate-draft GGUF summary (Gemma 4's `gemma4-assistant`
    /// shape): a `token_embd.weight` kept on CPU plus the GPU-resident
    /// remainder, with `total_tensor_bytes` summing both.
    fn draft_summary(token_embd_mib: u64, gpu_weight_mib: u64) -> GgufSummary {
        use crate::gguf::types::{GgufTensor, GgufType};
        let mut tensors = BTreeMap::new();
        let mk = |name: &str, bytes: u64| GgufTensor {
            name: SmolStr::new(name),
            dtype: GgufType::F16,
            shape: vec![bytes / 2],
            byte_size: bytes,
            shard_idx: 0,
            offset: 0,
        };
        tensors.insert(
            SmolStr::new("token_embd.weight"),
            mk("token_embd.weight", token_embd_mib * 1024 * 1024),
        );
        tensors.insert(
            SmolStr::new("blk.0.attn_q.weight"),
            mk("blk.0.attn_q.weight", gpu_weight_mib * 1024 * 1024),
        );
        GgufSummary {
            path: "/fake-draft".into(),
            total_tensor_bytes: (token_embd_mib + gpu_weight_mib) * 1024 * 1024,
            tensors,
            metadata: BTreeMap::new(),
            block_count: Some(4),
            architecture: SmolStr::new("gemma4-assistant"),
            shards: vec!["/fake-draft".into()],
        }
    }

    #[test]
    fn separate_draft_counts_gpu_weights_plus_compute_not_kv() {
        // The target carries no embedded MTP head (gemma4, nextn = 0), so
        // without a draft model the overhead would be zero. With a separate
        // draft, the overhead is `(total - token_embd) + DRAFT_MODEL_COMPUTE`
        // and does NOT scale with context (the draft shares the target's KV).
        let target = qwen35_summary("gemma4", 0, 4);
        let draft = draft_summary(144, 108);
        let empty: Vec<String> = Vec::new();
        let mib = mtp_overhead_bytes(&target, Some(&draft), &inputs(204800, true, &empty))
            / (1024 * 1024);
        assert_eq!(mib, 108 + 300);

        // Doubling the context must not change the draft overhead.
        let mib_2x = mtp_overhead_bytes(&target, Some(&draft), &inputs(409600, true, &empty))
            / (1024 * 1024);
        assert_eq!(mib_2x, 108 + 300);
    }

    #[test]
    fn separate_draft_ignored_when_mtp_disabled() {
        let target = qwen35_summary("gemma4", 0, 4);
        let draft = draft_summary(144, 108);
        let empty: Vec<String> = Vec::new();
        assert_eq!(
            mtp_overhead_bytes(&target, Some(&draft), &inputs(204800, false, &empty)),
            0
        );
    }
}
