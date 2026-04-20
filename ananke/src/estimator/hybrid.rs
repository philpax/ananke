//! Hybrid-architecture estimator (jamba and similar).

use std::collections::BTreeMap;

use smol_str::SmolStr;

use super::{
    llama::{collect_non_layer, collect_per_layer},
    types::{Estimate, EstimatorInputs},
};
use crate::gguf::GgufSummary;

pub const HYBRID_FAMILY: &[&str] = &["jamba"];

pub fn is_hybrid(arch: &str) -> bool {
    HYBRID_FAMILY.contains(&arch)
}

pub fn estimate(summary: &GgufSummary, inputs: &EstimatorInputs<'_>) -> Estimate {
    // For phase 3, treat hybrid like llama-family but with no KV cache
    // modelled (safer over-estimate side) and no per-layer type
    // differentiation (future work).
    let arch = summary.architecture.as_str();
    let n_layers = summary.block_count.unwrap_or(0);

    let per_layer = collect_per_layer(summary, n_layers);
    let non_layer = collect_non_layer(summary);
    let weights_bytes = per_layer.iter().sum::<u64>()
        + non_layer.output_head_bytes
        + non_layer.token_embd_bytes
        + non_layer.other_bytes;

    Estimate {
        weights_bytes,
        kv_per_token: 0, // conservative; refined when real jamba metadata arrives.
        compute_buffer_mb: inputs
            .compute_buffer_mb
            .unwrap_or_else(|| super::compute_buffer::default_for(summary, inputs.context)),
        per_layer_bytes: Some(per_layer),
        attention_layers: None,
        non_layer,
        override_tensor_bytes: BTreeMap::new(),
        expert_layers: Vec::new(),
        expert_layer_cpu_bytes: BTreeMap::new(),
        context: inputs.context,
        architecture: SmolStr::new(arch),
    }
}

#[cfg(test)]
mod tests {
    // Smoke-level coverage provided by the dispatcher test in estimator::mod.
}
