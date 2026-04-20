//! SSM/Mamba estimator.
//!
//! No conventional KV cache. State cost derived from mamba.ssm.*
//! metadata. flash_attn and cache_type_* don't apply and validation
//! rejects them for this architecture (spec §6.5).

use std::collections::BTreeMap;

use smol_str::SmolStr;

use super::{
    llama::collect_non_layer,
    types::{Estimate, EstimatorInputs},
};
use crate::gguf::GgufSummary;

/// Default mamba.ssm.state_size when the model metadata omits it. Matches the
/// Mamba-1 reference implementation.
const DEFAULT_STATE_SIZE: u64 = 16;

/// Default mamba.ssm.conv_kernel width when metadata omits it. Matches the
/// Mamba-1 reference implementation.
const DEFAULT_CONV_KERNEL: u64 = 4;

/// Size in bytes of one SSM state element (f32).
const STATE_ELEMENT_BYTES: u64 = std::mem::size_of::<f32>() as u64;

/// Default context length when the service config does not set one. Mirrors
/// llama.cpp's default.
const DEFAULT_CONTEXT: u32 = 4096;

pub const MAMBA_FAMILY: &[&str] = &["mamba"];

pub fn is_mamba(arch: &str) -> bool {
    MAMBA_FAMILY.contains(&arch)
}

pub fn estimate(summary: &GgufSummary, inputs: &EstimatorInputs<'_>) -> Estimate {
    let arch = summary.architecture.as_str();
    let context = if inputs.context == 0 {
        DEFAULT_CONTEXT
    } else {
        inputs.context
    };
    let n_layers = summary.block_count.unwrap_or(0);

    let per_layer = super::llama::collect_per_layer(summary, n_layers);
    let non_layer = collect_non_layer(summary);

    let weights_bytes = per_layer.iter().sum::<u64>()
        + non_layer.output_head_bytes
        + non_layer.token_embd_bytes
        + non_layer.other_bytes;

    // State cost: state_size × conv_kernel × inner_size × sizeof(f32), per layer.
    let state_size = summary
        .metadata
        .get("mamba.ssm.state_size")
        .and_then(|v| v.as_u32())
        .map_or(DEFAULT_STATE_SIZE, u64::from);
    let conv_kernel = summary
        .metadata
        .get("mamba.ssm.conv_kernel")
        .and_then(|v| v.as_u32())
        .map_or(DEFAULT_CONV_KERNEL, u64::from);
    let inner_size = summary
        .metadata
        .get("mamba.ssm.inner_size")
        .and_then(|v| v.as_u32())
        .map_or(0u64, u64::from);

    let state_per_layer = state_size * conv_kernel * inner_size * STATE_ELEMENT_BYTES;
    let kv_per_token = n_layers as u64 * state_per_layer;

    Estimate {
        weights_bytes,
        kv_per_token,
        compute_buffer_mb: inputs
            .compute_buffer_mb
            .unwrap_or_else(|| super::compute_buffer::default_for(arch, context)),
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
    // Mamba fixtures are nontrivial; smoke-level coverage via the
    // dispatcher integration test in Task 8. A dedicated fixture file
    // would duplicate estimator::llama's wiring without meaningful
    // extra coverage for this feature.
}
