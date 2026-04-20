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
//! - Tiny models (gemma-4-E4B): fixed CUDA+cuBLAS overhead dominates;
//!   base can be smaller than the fat-model default.
//!
//! Operators can still override per service via
//! `estimation.compute_buffer_mb`, which short-circuits this table.

/// Per-architecture knobs: `base + slope × (ctx / 1024)` MiB per device.
#[derive(Debug, Clone, Copy)]
struct Tuning {
    base: u32,
    slope: u32,
}

/// Lookup table for arch-specific tuning. The `_` arm is the llama-family
/// default, empirically the most conservative across the sweep. Add a row
/// here when calibration shows a given arch needs a different curve.
fn tuning_for(arch: &str) -> Tuning {
    match arch {
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

        // MoE-only (no SSM component). Attention scratch per active
        // expert is small; compute buffer barely grows with context.
        // Sweep showed gpt-oss 32k delta going *negative* at the 400/0
        // baseline, so this stays conservative at 600/2.
        "gpt-oss" | "mixtral" | "qwen3moe" | "qwen3vlmoe" | "llama4" | "deepseek2" | "glm4moe" => {
            Tuning {
                base: 600,
                slope: 2,
            }
        }

        // Hybrid MoE + SSM (Qwen 3.5+). Most layers are SSM with fixed
        // state; only the 1-in-N full-attention layers contribute to
        // context-scaled overhead. qwen3.6-35b-a3b at 2k-262k showed
        // the old (600, 4) under-reserving by up to 1.8 GiB — both
        // knobs went up.
        "qwen35moe" => Tuning {
            base: 900,
            slope: 7,
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

/// Default per-device compute-buffer reservation for `arch` at `context`
/// tokens. Operators can override per service via
/// `estimation.compute_buffer_mb`.
pub fn default_for(arch: &str, context: u32) -> u32 {
    let t = tuning_for(arch);
    t.base
        .saturating_add(t.slope.saturating_mul(context / 1024))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_family_default_tuning() {
        let cb = default_for("qwen3", 2048);
        assert_eq!(cb, 700 + 8 * 2);
        assert_eq!(default_for("qwen3", 32768), 700 + 8 * 32);
    }

    #[test]
    fn gemma_family_has_higher_base_than_llama_default() {
        // gemma-4-31B's full-attention layers drive a big attention
        // scratch allocation even at small context — the gemma tuning
        // has to start well above the llama default to cover it.
        let gemma_2k = default_for("gemma4", 2048);
        let llama_2k = default_for("qwen3", 2048);
        assert!(
            gemma_2k > llama_2k,
            "gemma base should exceed llama default at 2k (gemma={gemma_2k} llama={llama_2k})"
        );
    }

    #[test]
    fn moe_tuning_is_flatter_than_dense() {
        let dense_32k = default_for("qwen3", 32768);
        let moe_32k = default_for("gpt-oss", 32768);
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
        let moe_only_262k = default_for("gpt-oss", 262144);
        let qwen35moe_262k = default_for("qwen35moe", 262144);
        let dense_262k = default_for("qwen3", 262144);
        assert!(
            moe_only_262k <= qwen35moe_262k && qwen35moe_262k <= dense_262k,
            "qwen35moe should land between MoE-only and dense at 262k \
             (moe={moe_only_262k} qwen35moe={qwen35moe_262k} dense={dense_262k})"
        );
    }

    #[test]
    fn unknown_arch_falls_back_to_llama_default() {
        // Matches the conservative dense-family curve so unknown archs
        // that slip through the fallback still over-reserve safely.
        assert_eq!(default_for("brand-new-arch", 8192), 700 + 8 * 8);
    }

    #[test]
    fn absent_context_floors_to_base() {
        assert_eq!(default_for("qwen3", 0), 700);
        assert_eq!(default_for("gpt-oss", 512), 600);
        assert_eq!(default_for("gemma4", 0), 2000);
        assert_eq!(default_for("qwen35moe", 0), 900);
    }
}
