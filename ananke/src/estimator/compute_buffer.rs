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
        // Dense gemma family has large hidden + many attention layers.
        // Compute buffer grows noticeably with context.
        "gemma2" | "gemma3" | "gemma4" => Tuning {
            base: 800,
            slope: 12,
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
        // context-scaled overhead. Baseline matches MoE with a slight
        // slope bump to cover the minority of attention layers.
        "qwen35moe" => Tuning {
            base: 600,
            slope: 4,
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
        // falls through the fallback estimator too.
        _ => Tuning {
            base: 800,
            slope: 12,
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
        assert_eq!(cb, 800 + 12 * 2);
        assert_eq!(default_for("qwen3", 32768), 800 + 12 * 32);
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
    fn unknown_arch_falls_back_to_llama_default() {
        // Matches the conservative dense-family curve so unknown archs
        // that slip through the fallback still over-reserve safely.
        assert_eq!(default_for("brand-new-arch", 8192), 800 + 12 * 8);
    }

    #[test]
    fn absent_context_floors_to_base() {
        assert_eq!(default_for("qwen3", 0), 800);
        assert_eq!(default_for("gpt-oss", 512), 600);
    }
}
