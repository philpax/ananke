//! Multi-token-prediction (MTP / NextN) draft-context overhead.
//!
//! When a service runs with `--spec-type draft-mtp`, llama.cpp creates a
//! second context against the *same* target model. Its KV cache covers only
//! the trailing `nextn_predict_layers` blocks — the dense-attention MTP head
//! — and uses the *draft* cache types, which default to f16 regardless of the
//! main context's `--cache-type-*` quantisation. No extra weights load: the
//! nextn-layer tensors live in the target GGUF and are resident even without
//! MTP. So the whole MTP cost is `nextn KV (f16) + a roughly constant compute
//! buffer`.
//!
//! Calibrated against llama.cpp's own `[spec] estimated memory usage of MTP
//! context` figure on Qwen 3.6 27B (`qwen35`, 4 KV heads) and 35B-A3B
//! (`qwen35moe`, 2 KV heads) across context (262144/524288) and parallelism
//! (np 1/2): the KV term tracks `kv_heads × context` exactly, and the compute
//! term sits at ~1.55–1.61 GiB independent of both knobs (it is driven by the
//! shared-tokenizer logit buffer at `n_ubatch`, not model width).

use super::types::EstimatorInputs;
use crate::gguf::GgufSummary;

/// Per-MTP-context compute buffer, MiB. Rounded up to stay above every
/// observed datapoint (1553 MiB on 35B, 1609 MiB on 27B) with headroom.
const MTP_COMPUTE_MIB: u64 = 1700;

/// Bytes per KV element for the MTP draft cache. llama.cpp leaves the draft
/// cache type at f16 unless `--spec-draft-cache-type-*` is set (ananke never
/// sets it), so this is independent of the main `cache_type_k`/`_v`.
const MTP_KV_BYTES_PER_ELEMENT: u64 = 2;

/// Default per-head K/V dimension when the GGUF omits the attention length
/// keys — matches the llama-family estimator's fallback.
const DEFAULT_HEAD_DIM: u64 = 128;

/// Extra VRAM (bytes) the MTP draft context adds, or 0 when MTP is off or the
/// model carries no MTP head (`{arch}.nextn_predict_layers` absent or zero).
///
/// `inputs.context` is the configured total context. The MTP KV scales with
/// it linearly the same way the main KV does — total KV tokens equal the
/// context budget whether the main cache is unified (np auto) or split
/// per-slot (np > 1), so the estimator does not need to know the parallelism.
pub fn mtp_overhead_bytes(summary: &GgufSummary, inputs: &EstimatorInputs<'_>) -> u64 {
    if !inputs.mtp {
        return 0;
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
            cache_type_k: Some("q8_0"),
            cache_type_v: Some("q8_0"),
            override_tensor: empty,
            n_cpu_moe: None,
            compute_buffer_mb: None,
            allow_fallback: false,
            mtp,
        }
    }

    #[test]
    fn zero_when_mtp_disabled() {
        let s = qwen35_summary("qwen35", 1, 4);
        let empty: Vec<String> = Vec::new();
        assert_eq!(mtp_overhead_bytes(&s, &inputs(262144, false, &empty)), 0);
    }

    #[test]
    fn zero_when_no_mtp_head() {
        // MTP requested but the model has nextn_predict_layers = 0.
        let s = qwen35_summary("qwen35", 0, 4);
        let empty: Vec<String> = Vec::new();
        assert_eq!(mtp_overhead_bytes(&s, &inputs(262144, true, &empty)), 0);
    }

    #[test]
    fn qwen35_27b_matches_measured() {
        // 27B: nextn=1, 4 KV heads, 256+256, f16 draft cache, ctx 262144.
        // KV = 1 × 4 × 512 × 2 × 262144 = 1024 MiB; + 1700 MiB compute.
        let s = qwen35_summary("qwen35", 1, 4);
        let empty: Vec<String> = Vec::new();
        let got = mtp_overhead_bytes(&s, &inputs(262144, true, &empty));
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
        let kv_mib = mtp_overhead_bytes(&s, &inputs(262144, true, &empty)) / (1024 * 1024) - 1700;
        assert_eq!(kv_mib, 1024);
    }

    #[test]
    fn qwen35moe_35b_doubled_context() {
        // 35B-A3B: nextn=1, 2 KV heads, ctx 524288 (the doubled deploy).
        // KV = 1 × 2 × 512 × 2 × 524288 = 1024 MiB; + 1700 compute.
        let s = qwen35_summary("qwen35moe", 1, 2);
        let empty: Vec<String> = Vec::new();
        let mib = mtp_overhead_bytes(&s, &inputs(524288, true, &empty)) / (1024 * 1024);
        assert_eq!(mib, 1024 + 1700);
    }
}
