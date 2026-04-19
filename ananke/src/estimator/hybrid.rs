//! Hybrid-architecture estimator (jamba and similar).

use std::collections::BTreeMap;

use smol_str::SmolStr;

use super::{
    llama::{collect_non_layer, collect_per_layer},
    types::Estimate,
};
use crate::{config::ServiceConfig, gguf::GgufSummary};

pub fn is_hybrid(arch: &str) -> bool {
    arch == "jamba"
}

pub fn estimate(summary: &GgufSummary, svc: &ServiceConfig) -> Estimate {
    // For phase 3, treat hybrid like llama-family but with no KV cache
    // modelled (safer over-estimate side) and no per-layer type
    // differentiation (future work).
    let lc = svc
        .llama_cpp()
        .expect("hybrid::estimate on non-llama-cpp service");
    let arch = summary.architecture.as_str();
    let context = lc.context.unwrap_or(4096);
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
        compute_buffer_mb: lc.estimation.compute_buffer_mb.unwrap_or(400),
        per_layer_bytes: Some(per_layer),
        attention_layers: None,
        non_layer,
        override_tensor_bytes: BTreeMap::new(),
        expert_layers: Vec::new(),
        expert_layer_cpu_bytes: BTreeMap::new(),
        context,
        architecture: SmolStr::new(arch),
    }
}

#[cfg(test)]
mod tests {
    // Smoke-level coverage provided by the dispatcher test in estimator::mod.
}
