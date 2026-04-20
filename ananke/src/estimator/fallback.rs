//! Fallback estimator for unknown architectures (spec §8.3).

use std::collections::BTreeMap;

use smol_str::SmolStr;
use tracing::warn;

use super::types::{Estimate, NonLayer};
use crate::gguf::GgufSummary;

/// Multiplier applied to the GGUF's on-disk tensor bytes as a rough
/// headroom factor for the unmodelled non-tensor overhead (KV, compute
/// buffer, context scratch).
const FALLBACK_WEIGHTS_SCALE: f64 = 1.15;

/// Flat headroom added on top of the scaled weights.
const FALLBACK_WEIGHTS_HEADROOM_BYTES: u64 = 512 * 1024 * 1024;

/// Produce a coarse estimate for any GGUF: scaled tensor bytes plus a flat
/// headroom landing in `weights_bytes`; no KV modelling; no per-layer
/// split. Emits a warning so the operator knows rolling correction is
/// the only tuning they'll get.
pub fn estimate_fallback(summary: &GgufSummary, context: u32) -> Estimate {
    warn!(
        architecture = %summary.architecture,
        "unknown architecture — using fallback estimator"
    );
    let weights = ((summary.total_tensor_bytes as f64) * FALLBACK_WEIGHTS_SCALE) as u64
        + FALLBACK_WEIGHTS_HEADROOM_BYTES;
    Estimate {
        weights_bytes: weights,
        kv_per_token: 0,
        compute_buffer_mb: super::compute_buffer::default_for(context),
        per_layer_bytes: None,
        attention_layers: None,
        non_layer: NonLayer {
            output_head_bytes: 0,
            token_embd_bytes: 0,
            other_bytes: 0,
        },
        override_tensor_bytes: BTreeMap::new(),
        expert_layers: Vec::new(),
        expert_layer_cpu_bytes: BTreeMap::new(),
        context,
        architecture: SmolStr::new(summary.architecture.as_str()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary_with(total_bytes: u64, arch: &str) -> GgufSummary {
        GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: total_bytes,
            tensors: Default::default(),
            metadata: Default::default(),
            block_count: None,
            architecture: SmolStr::new(arch),
            shards: vec!["/fake".into()],
        }
    }

    #[test]
    fn fallback_applies_declared_scale_and_headroom() {
        let s = summary_with(1_000_000_000, "nonsense-arch");
        let e = estimate_fallback(&s, 4096);
        // Assert against the named constants so the test tracks any
        // future re-tuning without silently drifting.
        assert_eq!(
            e.weights_bytes,
            (1_000_000_000f64 * FALLBACK_WEIGHTS_SCALE) as u64 + FALLBACK_WEIGHTS_HEADROOM_BYTES
        );
        assert_eq!(e.kv_per_token, 0);
    }
}
