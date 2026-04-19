//! VRAM estimator — architecture-aware dispatch.

pub mod fallback;
pub mod hybrid;
pub mod kv;
pub mod llama;
pub mod mamba;
pub mod moe;
pub mod override_tensor;
pub mod types;

use tracing::{info, warn};
pub use types::{Estimate, EstimatorInputs, NonLayer};

use crate::{
    gguf::{self, GgufSummary},
    system::Fs,
};

/// Produce a base estimate for the model described by `inputs`. Reads the
/// GGUF (including any mmproj) through `fs` and dispatches on
/// `general.architecture`. Pure function over `inputs` + the bytes on
/// disk; caller applies rolling correction + safety factor afterward.
pub fn estimate_from_path(fs: &dyn Fs, inputs: &EstimatorInputs<'_>) -> Result<Estimate, String> {
    let summary = gguf::read(fs, inputs.model).map_err(|e| e.to_string())?;

    info!(
        service = %inputs.name,
        architecture = %summary.architecture,
        block_count = ?summary.block_count,
        tensor_count = summary.tensors.len(),
        total_tensor_gb = summary.total_tensor_bytes / (1024 * 1024 * 1024),
        shard_count = summary.shards.len(),
        "gguf summary",
    );

    let mut est = dispatch(&summary, inputs);

    info!(
        service = %inputs.name,
        weights_gb = est.weights_bytes / (1024 * 1024 * 1024),
        per_layer_len = est.per_layer_bytes.as_ref().map(|v| v.len()).unwrap_or(0),
        kv_per_token = est.kv_per_token,
        "post-dispatch estimate",
    );

    // Apply user-declared override_tensor rules BEFORE mmproj so matched
    // tensors leave the layer/non-layer budget cleanly (spec §8.2.4).
    if !inputs.override_tensor.is_empty() {
        override_tensor::parse_and_apply(&mut est, &summary, inputs.override_tensor);
    }

    // Add mmproj bytes to GPU 0 weights (per spec §8.3).
    if let Some(mmproj) = inputs.mmproj {
        match gguf::read(fs, mmproj) {
            Ok(proj) => {
                est.weights_bytes = est.weights_bytes.saturating_add(proj.total_tensor_bytes);
                est.non_layer.other_bytes = est
                    .non_layer
                    .other_bytes
                    .saturating_add(proj.total_tensor_bytes);
            }
            Err(e) => warn!(error = %e, path = %mmproj.display(), "mmproj read failed"),
        }
    }

    Ok(est)
}

pub fn dispatch(summary: &GgufSummary, inputs: &EstimatorInputs<'_>) -> Estimate {
    let arch = summary.architecture.as_str();
    if llama::is_llama_family(arch) {
        return llama::estimate(summary, inputs);
    }
    if moe::is_moe(arch) {
        return moe::estimate(summary, inputs);
    }
    if mamba::is_mamba(arch) {
        return mamba::estimate(summary, inputs);
    }
    if hybrid::is_hybrid(arch) {
        return hybrid::estimate(summary, inputs);
    }
    fallback::estimate_fallback(summary, inputs.context)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use smol_str::SmolStr;

    use super::*;
    use crate::gguf::types::{GgufSummary, GgufValue};

    fn inputs_for<'a>(empty_override: &'a [String]) -> EstimatorInputs<'a> {
        EstimatorInputs {
            name: "demo",
            model: Path::new("/fake"),
            mmproj: None,
            context: 4096,
            cache_type_k: Some("f16"),
            cache_type_v: Some("f16"),
            override_tensor: empty_override,
            n_cpu_moe: None,
            compute_buffer_mb: None,
        }
    }

    #[test]
    fn dispatch_recognises_known_families() {
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new("general.architecture"),
            GgufValue::String("qwen3".into()),
        );
        metadata.insert(SmolStr::new("qwen3.block_count"), GgufValue::U32(1));
        let summary = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors: Default::default(),
            metadata,
            block_count: Some(1),
            architecture: SmolStr::new("qwen3"),
            shards: vec!["/fake".into()],
        };
        let empty: Vec<String> = Vec::new();
        let e = dispatch(&summary, &inputs_for(&empty));
        assert_eq!(e.architecture, "qwen3");
    }

    #[test]
    fn dispatch_unknown_goes_to_fallback() {
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new("general.architecture"),
            GgufValue::String("novel-arch".into()),
        );
        let summary = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 1_000_000,
            tensors: Default::default(),
            metadata,
            block_count: None,
            architecture: SmolStr::new("novel-arch"),
            shards: vec!["/fake".into()],
        };
        let empty: Vec<String> = Vec::new();
        let e = dispatch(&summary, &inputs_for(&empty));
        // Fallback uses 1.15 × total + 512 MB.
        assert!(e.weights_bytes >= 512 * 1024 * 1024);
    }
}
