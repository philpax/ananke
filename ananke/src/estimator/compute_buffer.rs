//! Per-architecture compute-buffer sizing.
//!
//! The packer multiplies this number by the count of active devices
//! (GPUs + CPU when token embeddings land there) and adds it to the
//! per-device reservation. It's the only term in the estimate that
//! reflects *unmodelled* overhead — CUDA context + cuBLAS workspace +
//! attention scratch + KV ring buffers, all lumped together.
//!
//! Per-architecture tuning exists because the overhead scales with
//! different knobs on different architectures:
//!
//! - Dense large-hidden (gemma3/gemma4 with hidden ≥ 4k): attention
//!   scratch grows quickly with context.
//! - Dense standard-hidden (llama, qwen3 < 5k): slower context scaling.
//! - MoE (gpt-oss, mixtral, qwen3moe): compute buffer is almost flat
//!   because only a few experts run per token.
//! - Hybrid MoE + SSM (qwen35moe): SSM state scales with `ssm_d_*`
//!   constants, not context, so the slope is small.
//! - Gemma 4 E-variants (detected by `per_layer_token_embd.weight`):
//!   small hidden + per-layer embeddings on CPU; fits under a much
//!   lower curve than the fat-model gemma4 default.
//!
//! Operators can still override per service via
//! `estimation.compute_buffer_mb`, which short-circuits this table.

use crate::gguf::GgufSummary;

/// Per-architecture knobs: `base + slope × (ctx / 1024)` MiB per device.
#[derive(Debug, Clone, Copy)]
struct Tuning {
    base: u32,
    slope: u32,
}

/// deepseek4 compute-buffer curve, calibrated against a 2×3090 sweep at
/// np=1, all experts on CPU (2026-07-15). The NSA indexer's prompt scratch
/// dominates and scales steeply with *both* context and ubatch: it scores
/// each of the `ubatch` query tokens against the whole sequence, so the
/// residual (VRAM − GPU non-expert weights − KV) is ≈ `k × ubatch ×
/// context`. Measured per-card residuals landed on `k ≈ 1.25e-4 MiB` —
/// equivalently a slope of **66 MiB per 1024 tokens at the default ubatch
/// of 512** (~9.3 GiB at 131072, ~17.5 GiB at 262144), scaling linearly
/// with ubatch. This is by far the steepest curve in the table because it
/// is the only arch whose scratch grows with the whole context on every
/// prompt token; the fixed sliding-window and HCA caches (not modelled as
/// per-token KV) ride in the base. The linear-in-ubatch scaling was
/// confirmed against a ubatch-1024 point (residual ~17.6 GiB at 131072,
/// vs the model's ~18.3 GiB — over-reserving by ~0.7 GiB).
const DEEPSEEK4_CB_BASE: u32 = 1400;
const DEEPSEEK4_CB_SLOPE_AT_UB512: u32 = 66;

/// llama.cpp's default `--ubatch-size`, and the ubatch the compute-buffer
/// curves are calibrated at. Also the fallback when a service leaves
/// `ubatch_size` unset.
const DEFAULT_UBATCH: u32 = 512;

/// deepseek4's per-1024-token slope at a given ubatch. Linear in ubatch off
/// the [`DEEPSEEK4_CB_SLOPE_AT_UB512`] calibration point, floored at 1 so a
/// tiny ubatch still reserves a non-zero context term.
fn deepseek4_cb_slope(ubatch: u32) -> u32 {
    let scaled =
        (DEEPSEEK4_CB_SLOPE_AT_UB512 as u64 * ubatch.max(1) as u64) / DEFAULT_UBATCH as u64;
    scaled.max(1) as u32
}

/// Lookup table for arch-specific tuning. The `_` arm is the llama-family
/// default, empirically the most conservative across the sweep. Add a row
/// here when calibration shows a given arch needs a different curve.
fn tuning_for(summary: &GgufSummary, ubatch: u32) -> Tuning {
    let arch = summary.architecture.as_str();
    match arch {
        // Gemma 4 E-variants (detected by `per_layer_token_embd.weight`):
        // small hidden + per-layer embeddings on CPU, so neither the
        // attention scratch nor the CUDA context take up as much room
        // as the fat-model gemma4 default. The full (2000, 7) curve
        // over-reserves E4B by ~2 GiB at 262k; (1100, 7) stays safely
        // above every observed datapoint (worst-case -176 MiB over).
        "gemma4" if is_gemma_e_variant(summary) => Tuning {
            base: 1100,
            slope: 7,
        },

        // Gemma family has large hidden, a few full-attention layers
        // buried under an SWA-heavy majority, and big attention scratch
        // even at 2k context. gemma-4-31B-it under-reserved by 2.4 GiB
        // at 8k with the old (800, 12) curve — base went up hard and
        // slope shrunk since the full-attention minority drives less
        // growth than a dense-everything model would.
        "gemma2" | "gemma3" | "gemma4" => Tuning {
            base: 2000,
            slope: 7,
        },

        // Vision-language MoE: same _exps pattern as the pure MoE
        // archs but a heavier attention stack + cross-attention to
        // vision tokens. qwen3-vl-30b-a3b-instruct under-reserved by
        // +1365 MiB at 131k with the plain MoE curve, so it gets its
        // own (700, 7) — base slightly higher, slope much higher.
        "qwen3vlmoe" => Tuning {
            base: 700,
            slope: 7,
        },

        // MoE-only (no SSM component). Attention scratch per active
        // expert is small; compute buffer barely grows with context.
        // Sweep showed gpt-oss 32k delta going *negative* at the 400/0
        // baseline, so this stays conservative at 600/2.
        "gpt-oss" | "mixtral" | "qwen3moe" | "llama4" | "deepseek2" | "glm4moe" => Tuning {
            base: 600,
            slope: 2,
        },

        // GLM-5 (glm-dsa). Dense MLA attention (the pinned llama.cpp runs
        // it on the deepseek2 graph; the DSA indexer path never engages),
        // so the residual is near-flat like the pure-MoE curve but with a
        // much fatter base: a single-3090 sweep of GLM-5.2 UD-IQ1_S with
        // all experts on CPU (2026-07-21) measured residuals of 1849 MiB
        // at 8k → 1899 at 32k → 1967 at 64k (slope ≈ 2 MiB per 1024
        // tokens), with a q8_0 K-cache point reconciling the MLA KV term
        // to within 16 MiB. Doubling ubatch to 1024 added only ~186 MiB,
        // so the ubatch sensitivity rides in the base rather than scaling
        // the slope. Base covers the worst measured point plus the ubatch
        // delta with headroom.
        "glm-dsa" => Tuning {
            base: 2300,
            slope: 3,
        },

        // Laguna MoE. Recalibrated 2026-07-22 against ik_llama's own
        // per-device buffer report under ananke-planned placement (not
        // --fit), 2×3090, ub2048, q8_0 KV: the CUDA0 compute buffer measured
        // 2058 MiB at 131072 context and is effectively flat in context
        // (sliding window 512 caps the attention scratch), so the old
        // (2400, 28) — a single --fit anchor — over-reserved by ~3.9 GiB per
        // card, needlessly spilling ~7.8 GiB of experts to CPU. The base
        // covers the 2058 measurement plus the active-prefill transient and
        // CUDA-context/fragmentation overhead (~1 GiB observed as
        // nvidia-smi minus ik's reported buffers), which the compute-buffer
        // term must absorb now that there is no --fit margin backstopping it.
        "laguna" => Tuning {
            base: 2800,
            slope: 2,
        },

        // DeepSeek-V4-Flash (deepseek4). Unlike the near-flat pure-MoE
        // curve, deepseek4's NSA "lightning indexer" builds a prompt-phase
        // scratch buffer that scales hard with context (it scores each
        // query against the whole sequence before the top-k gather), so
        // this arch needs a much steeper slope than any other MoE. The
        // fixed sliding-window + HCA caches, which this estimator folds
        // into the compute buffer rather than the per-token KV term, ride
        // along in the base. Calibrated on the UD-IQ3_XXS quant at the
        // default ubatch (512), 2×3090, np=1 (see below).
        "deepseek4" => Tuning {
            base: DEEPSEEK4_CB_BASE,
            slope: deepseek4_cb_slope(ubatch),
        },

        // Hybrid MoE + SSM (Qwen 3.5+). Most layers are SSM with fixed
        // state; only the 1-in-N full-attention layers contribute to
        // context-scaled overhead. qwen3.6-35b-a3b at 2k-262k showed
        // the old (600, 4) under-reserving by up to 1.8 GiB — both
        // knobs went up.
        "qwen35moe" => Tuning {
            base: 900,
            slope: 7,
        },

        // Hybrid dense + SSM (Qwen 3.5+ dense, e.g. qwen3.6-27b).
        // Same 1-in-N full-attention pattern as qwen35moe but with
        // standard dense FFNs and smaller KV (256-dim K/V vs 128+128
        // in the MoE variant). The SSM state is absorbed by the base.
        "qwen35" => Tuning {
            base: 800,
            slope: 6,
        },

        // Talkie: dense 13B with full MHA and a native 2048 context. A
        // single-GPU sweep at 2048/4096/8192/16384 (f16 KV, llama-server
        // defaults) put the residual compute buffer — real nvidia-smi
        // usage minus modelled GPU weights minus KV — at a near-flat
        // 414→428 MiB (warmup adds ~10). Slope is ~1 MiB per 1024 tokens,
        // so the dense default's (700, 8) over-reserves by ~290 MiB at
        // 2048. (500, 2) tracks the data with ~80 MiB of headroom and
        // never under-reserves across the swept range.
        "talkie" => Tuning {
            base: 500,
            slope: 2,
        },

        // Pure Mamba / SSM: no standard KV cache, so the compute buffer
        // is almost flat in context. Recurrent state lives in the per-
        // layer tensors, which the weights accounting already covers.
        "mamba" => Tuning {
            base: 500,
            slope: 2,
        },

        // Jamba-style hybrid (SSM + attention, no MoE). Similar to
        // qwen35moe but with fewer knobs exposed in GGUF metadata.
        "jamba" => Tuning {
            base: 600,
            slope: 4,
        },

        // LFM2/LFM2.5: hybrid shortconv + sparse-attention with a tiny
        // hidden size, so the scratch is dominated by the CUDA context and
        // nearly flat in context length. Calibrated on
        // LFM2.5-Embedding-350M-Q8_0 (2026-07-12, single 3090, --embeddings):
        // residuals 397/403/411/427 MiB at 2k/8k/16k/32k → (420, 1) covers
        // the worst case with ~25 MiB of headroom. The flat residual across
        // the sweep also confirms the per-layer-array KV term in the
        // llama-family estimator.
        "lfm2" => Tuning {
            base: 420,
            slope: 1,
        },

        // Llama-family default: llama / qwen2 / qwen3 / mistral / phi3 /
        // glm4 / deci. Covers everything that doesn't match above and
        // falls through the fallback estimator too. qwen3-4b sweep
        // showed the old (800, 12) curve over-reserving by 3 GiB at
        // 262k; (700, 8) stays safely above observed without wasting.
        _ => Tuning {
            base: 700,
            slope: 8,
        },
    }
}

/// Default per-device compute-buffer reservation for `summary` at
/// `context` tokens and the service's physical batch size. `ubatch = None`
/// (or an unset config) means llama.cpp's [`DEFAULT_UBATCH`]. Operators can
/// override the whole term per service via `estimation.compute_buffer_mb`.
/// `ubatch` only affects the deepseek4 curve.
pub fn default_for(summary: &GgufSummary, context: u32, ubatch: Option<u32>) -> u32 {
    let t = tuning_for(summary, ubatch.unwrap_or(DEFAULT_UBATCH));
    t.base
        .saturating_add(t.slope.saturating_mul(context / 1024))
}

/// Does `summary` look like a Gemma 4 E-variant (E4B and siblings)?
/// Detection is keyed on `per_layer_token_embd.weight`, the per-block
/// input-embedding stack that only E-variants carry.
fn is_gemma_e_variant(summary: &GgufSummary) -> bool {
    summary.tensors.contains_key("per_layer_token_embd.weight")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use smol_str::SmolStr;

    use super::*;
    use crate::gguf::types::{GgufSummary, GgufTensor, GgufType};

    fn summary_for(arch: &str) -> GgufSummary {
        GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors: BTreeMap::new(),
            metadata: BTreeMap::new(),
            block_count: None,
            architecture: SmolStr::new(arch),
            shards: vec!["/fake".into()],
        }
    }

    fn gemma4_e_variant_summary() -> GgufSummary {
        let mut s = summary_for("gemma4");
        s.tensors.insert(
            SmolStr::new("per_layer_token_embd.weight"),
            GgufTensor {
                name: SmolStr::new("per_layer_token_embd.weight"),
                dtype: GgufType::F32,
                shape: vec![1, 1],
                byte_size: 1024 * 1024,
                shard_idx: 0,
                offset: 0,
            },
        );
        s
    }

    #[test]
    fn llama_family_default_tuning() {
        let s = summary_for("qwen3");
        assert_eq!(default_for(&s, 2048, None), 700 + 8 * 2);
        assert_eq!(default_for(&s, 32768, None), 700 + 8 * 32);
    }

    #[test]
    fn gemma_family_has_higher_base_than_llama_default() {
        // gemma-4-31B's full-attention layers drive a big attention
        // scratch allocation even at small context — the gemma tuning
        // has to start well above the llama default to cover it.
        let gemma_2k = default_for(&summary_for("gemma4"), 2048, None);
        let llama_2k = default_for(&summary_for("qwen3"), 2048, None);
        assert!(
            gemma_2k > llama_2k,
            "gemma base should exceed llama default at 2k (gemma={gemma_2k} llama={llama_2k})"
        );
    }

    #[test]
    fn gemma4_e_variant_uses_smaller_curve() {
        // E-variants ship a `per_layer_token_embd.weight` tensor and
        // have a small hidden size; the fat-model gemma4 tuning over-
        // reserves them by ~2 GiB at 262k otherwise.
        let regular = default_for(&summary_for("gemma4"), 262144, None);
        let e_variant = default_for(&gemma4_e_variant_summary(), 262144, None);
        assert!(
            e_variant < regular,
            "E-variant cb should be strictly lower than regular gemma4 \
             (e={e_variant} regular={regular})"
        );
    }

    #[test]
    fn moe_tuning_is_flatter_than_dense() {
        let dense_32k = default_for(&summary_for("qwen3"), 32768, None);
        let moe_32k = default_for(&summary_for("gpt-oss"), 32768, None);
        assert!(
            moe_32k < dense_32k,
            "MoE compute buffer should be flatter than dense at long context; \
             dense={dense_32k} moe={moe_32k}"
        );
    }

    #[test]
    fn qwen35moe_sits_between_moe_only_and_dense() {
        // Hybrid SSM+MoE: full-attention layers are a minority but they
        // do cost more per 1k context than pure-MoE's near-flat curve.
        let moe_only_262k = default_for(&summary_for("gpt-oss"), 262144, None);
        let qwen35moe_262k = default_for(&summary_for("qwen35moe"), 262144, None);
        let dense_262k = default_for(&summary_for("qwen3"), 262144, None);
        assert!(
            moe_only_262k <= qwen35moe_262k && qwen35moe_262k <= dense_262k,
            "qwen35moe should land between MoE-only and dense at 262k \
             (moe={moe_only_262k} qwen35moe={qwen35moe_262k} dense={dense_262k})"
        );
    }

    #[test]
    fn talkie_is_tighter_than_llama_default_and_covers_measured() {
        // The talkie curve was calibrated against a single-GPU sweep whose
        // residual compute buffer stayed ~414-428 MiB across 2048..16384.
        // It must (a) stay strictly below the conservative dense default it
        // would otherwise inherit, and (b) still cover the measured peak
        // (~428 MiB warmed) at the model's native 2048 context.
        let talkie_2k = default_for(&summary_for("talkie"), 2048, None);
        let llama_2k = default_for(&summary_for("qwen3"), 2048, None);
        assert!(
            talkie_2k < llama_2k,
            "talkie cb should be tighter than the dense default \
             (talkie={talkie_2k} llama={llama_2k})"
        );
        assert!(
            talkie_2k >= 440,
            "talkie cb at 2048 must cover the measured ~428 MiB peak with headroom \
             (got {talkie_2k})"
        );
    }

    #[test]
    fn talkie_floors_to_base() {
        assert_eq!(default_for(&summary_for("talkie"), 0, None), 500);
    }

    #[test]
    fn glm_dsa_covers_measured_and_stays_flat() {
        // Calibrated on the GLM-5.2 UD-IQ1_S single-3090 sweep
        // (2026-07-21): residuals 1849 MiB at 8k, 1899 at 32k, 1967 at
        // 64k, plus ~186 MiB when ubatch doubles to 1024. The curve must
        // cover the worst point plus the ubatch delta (~2150) and must
        // not inherit deepseek4's ubatch-scaled slope.
        let glm = summary_for("glm-dsa");
        assert!(
            default_for(&glm, 65536, None) >= 2150,
            "must cover the measured 64k residual plus the ubatch delta (got {})",
            default_for(&glm, 65536, None)
        );
        // Near-flat: dense-MLA scratch grows ~2 MiB/1k, nothing like the
        // deepseek4 indexer curve at the same context.
        assert!(
            default_for(&glm, 131072, None)
                < default_for(&summary_for("deepseek4"), 131072, None) / 2
        );
        // ubatch is ignored for this arch.
        assert_eq!(
            default_for(&glm, 131072, Some(2048)),
            default_for(&glm, 131072, Some(512))
        );
    }

    #[test]
    fn deepseek4_covers_measured_indexer_buffer() {
        // The NSA indexer's prompt scratch is the steepest curve in the
        // table. It must (a) grow faster than every other MoE arch and
        // (b) cover the measured per-card residuals (~9.3 GiB at 131072,
        // ~17.5 GiB at 262144) with a little headroom.
        let ds4 = summary_for("deepseek4");
        let moe = summary_for("gpt-oss");
        assert!(
            default_for(&ds4, 262144, None) > default_for(&moe, 262144, None) * 4,
            "deepseek4 cb must dwarf the flat MoE curve at long context"
        );
        assert!(
            default_for(&ds4, 131072, None) >= 9297,
            "must cover the measured ~9.3 GiB residual at 131072 (got {})",
            default_for(&ds4, 131072, None)
        );
        assert!(
            default_for(&ds4, 262144, None) >= 17519,
            "must cover the measured ~17.5 GiB residual at 262144 (got {})",
            default_for(&ds4, 262144, None)
        );
    }

    #[test]
    fn deepseek4_compute_buffer_scales_with_ubatch() {
        let ds4 = summary_for("deepseek4");
        // Unset ubatch (None) resolves to llama.cpp's default of 512.
        assert_eq!(
            default_for(&ds4, 131072, None),
            default_for(&ds4, 131072, Some(512))
        );
        let slope_at = |ub| default_for(&ds4, 131072, Some(ub)) - DEEPSEEK4_CB_BASE;
        // The context-scaling term is linear in ubatch off the 512 baseline.
        assert_eq!(slope_at(1024), slope_at(512) * 2);
        assert_eq!(slope_at(2048), slope_at(512) * 4);
        assert_eq!(slope_at(256), slope_at(512) / 2);
        // Every other arch ignores ubatch entirely.
        let qwen = summary_for("qwen3");
        assert_eq!(
            default_for(&qwen, 131072, Some(2048)),
            default_for(&qwen, 131072, None)
        );
    }

    #[test]
    fn unknown_arch_falls_back_to_llama_default() {
        // Matches the conservative dense-family curve so unknown archs
        // that slip through the fallback still over-reserve safely.
        assert_eq!(
            default_for(&summary_for("brand-new-arch"), 8192, None),
            700 + 8 * 8
        );
    }

    #[test]
    fn absent_context_floors_to_base() {
        assert_eq!(default_for(&summary_for("qwen3"), 0, None), 700);
        assert_eq!(default_for(&summary_for("gpt-oss"), 512, None), 600);
        assert_eq!(default_for(&summary_for("gemma4"), 0, None), 2000);
        assert_eq!(default_for(&summary_for("qwen35moe"), 0, None), 900);
        assert_eq!(default_for(&gemma4_e_variant_summary(), 0, None), 1100);
    }
}
