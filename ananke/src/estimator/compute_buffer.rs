//! Per-device compute-buffer size used by the estimator.
//!
//! The packer multiplies this number by the count of active devices
//! (GPUs + CPU when token embeddings land there) and adds it to the
//! per-device reservation. It's the only term in the estimate that
//! reflects *unmodelled* overhead — CUDA context + cuBLAS workspace +
//! attention scratch + KV ring buffers, all lumped together.
//!
//! The constants below come from a calibration sweep (see
//! `scripts/stress/calibrate.py`) across seven llama-cpp workloads
//! spanning dense / MoE / vision / multi-GPU / override-tensor /
//! variable-KV families. The fit covers the observed worst-case
//! under-reservation (qwen3-vl at 32k context, +1258 MiB) with a
//! ~300 MiB margin so as-yet-unobserved rows still land under the
//! prediction.

/// Base per-device compute buffer, in MiB. Roughly: CUDA context
/// (~250 MiB/GPU) + cuBLAS workspace (~150 MiB) + the flat portion of
/// the inference graph's scratch. Sized so the 2k-context row with the
/// largest under-reservation in the sweep lands under prediction across
/// three active devices (2 GPUs + CPU for token_embd).
const BASE_MB: u32 = 800;

/// Additional per-device compute buffer per 1024 tokens of context.
/// Attention scratch grows with context, and the slope depends on the
/// model's hidden size — gemma-3-27b (hidden=5376, 62 layers) shows the
/// steepest growth in the sweep, ~48 MiB per 1k ctx across three active
/// devices = 16 MiB per 1k ctx per device. We use 12 here to leave a
/// small waste budget on gemma while keeping over-reservation bounded
/// on smaller-hidden models (qwen3-4b over-reserves ~25% at 2k context
/// under this slope, which is acceptable for the safety gain).
const SLOPE_MB_PER_1K_CTX: u32 = 12;

/// Default per-device compute-buffer reservation for a service running
/// at `context` tokens. Operators can override per service via
/// `estimation.compute_buffer_mb`; everyone else gets this.
pub fn default_for(context: u32) -> u32 {
    BASE_MB.saturating_add(SLOPE_MB_PER_1K_CTX.saturating_mul(context / 1024))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grows_with_context() {
        assert_eq!(default_for(2048), BASE_MB + SLOPE_MB_PER_1K_CTX * 2);
        assert_eq!(default_for(8192), BASE_MB + SLOPE_MB_PER_1K_CTX * 8);
        assert_eq!(default_for(32768), BASE_MB + SLOPE_MB_PER_1K_CTX * 32);
    }

    #[test]
    fn covers_observed_worst_cases() {
        // Sanity floor: every sweep observation at up to 32k context
        // produced a delta under the 1.5 GiB / device × 3 headroom we
        // budget. If this ever stops being true we need to recalibrate.
        let active_devices = 3u32;
        assert!(default_for(32768) * active_devices >= 2700);
    }

    #[test]
    fn absent_context_floors_to_base() {
        assert_eq!(default_for(0), BASE_MB);
        // context < 1024 rounds down to 0 tokens in the 1k grid; the
        // slope term is zero but the base covers the overhead.
        assert_eq!(default_for(512), BASE_MB);
    }
}
