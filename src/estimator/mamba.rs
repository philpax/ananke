//! SSM/Mamba estimator.
//!
//! No conventional KV cache. State cost derived from mamba.ssm.*
//! metadata. flash_attn and cache_type_* don't apply and validation
//! rejects them for this architecture (spec §6.5).

use std::collections::BTreeMap;

use smol_str::SmolStr;

use super::{
    llama::{collect_non_layer, layer_index},
    types::Estimate,
};
use crate::{config::ServiceConfig, gguf::GgufSummary};

pub fn is_mamba(arch: &str) -> bool {
    arch == "mamba"
}

pub fn estimate(summary: &GgufSummary, svc: &ServiceConfig) -> Estimate {
    let arch = summary.architecture.as_str();
    let context = svc.raw.context.unwrap_or(4096);
    let n_layers = summary.block_count.unwrap_or(0);

    let per_layer = super::llama::collect_per_layer(summary, n_layers);
    let non_layer = collect_non_layer(summary);

    let weights_bytes = per_layer.iter().sum::<u64>()
        + non_layer.output_head_bytes
        + non_layer.token_embd_bytes
        + non_layer.other_bytes;

    // State cost: state_size × conv_kernel × inner_size × 4 bytes (f32), per layer.
    let state_size = summary
        .metadata
        .get("mamba.ssm.state_size")
        .and_then(|v| v.as_u32())
        .unwrap_or(16) as u64;
    let conv_kernel = summary
        .metadata
        .get("mamba.ssm.conv_kernel")
        .and_then(|v| v.as_u32())
        .unwrap_or(4) as u64;
    let inner_size = summary
        .metadata
        .get("mamba.ssm.inner_size")
        .and_then(|v| v.as_u32())
        .unwrap_or(0) as u64;

    let state_per_layer = state_size * conv_kernel * inner_size * 4;
    let kv_per_token = n_layers as u64 * state_per_layer;

    Estimate {
        weights_bytes,
        kv_per_token,
        compute_buffer_mb: svc
            .raw
            .estimation
            .as_ref()
            .and_then(|e| e.compute_buffer_mb)
            .unwrap_or(400),
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

// Suppress dead-code lint on layer_index re-export path used by mamba.
#[allow(dead_code)]
fn _use_layer_index(name: &str) -> Option<u32> {
    layer_index(name)
}

#[cfg(test)]
mod tests {
    // Mamba fixtures are nontrivial; smoke-level coverage via the
    // dispatcher integration test in Task 8. A dedicated fixture file
    // would duplicate estimator::llama's wiring without meaningful
    // extra coverage for this feature.
}
