//! VRAM estimator — architecture-aware dispatch.

pub mod fallback;
pub mod hybrid;
pub mod kv;
pub mod llama;
pub mod mamba;
pub mod moe;
pub mod override_tensor;
pub mod types;

use smol_str::SmolStr;
use tracing::{info, warn};
pub use types::{Estimate, EstimatorInputs, NonLayer};

use crate::{
    gguf::{self, GgufSummary},
    system::Fs,
};

/// One per-architecture family, paired with the `general.architecture`
/// values it accepts and the function that produces an `Estimate` for
/// them. Dispatch walks this table top-to-bottom; error formatting
/// enumerates the same table so "recognised" never drifts from "actually
/// dispatched".
struct Family {
    name: &'static str,
    arches: &'static [&'static str],
    estimate: fn(&GgufSummary, &EstimatorInputs<'_>) -> Estimate,
}

const FAMILIES: &[Family] = &[
    Family {
        name: "llama",
        arches: llama::LLAMA_FAMILY,
        estimate: llama::estimate,
    },
    Family {
        name: "moe",
        arches: moe::MOE_FAMILY,
        estimate: moe::estimate,
    },
    Family {
        name: "mamba",
        arches: mamba::MAMBA_FAMILY,
        estimate: mamba::estimate,
    },
    Family {
        name: "hybrid",
        arches: hybrid::HYBRID_FAMILY,
        estimate: hybrid::estimate,
    },
];

/// Failure modes from [`estimate_from_path`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EstimatorError {
    /// `gguf::read` failed (bad magic, IO error, unsupported dtype, …).
    /// The inner string is the reader's own diagnostic.
    GgufRead { path: std::path::PathBuf, cause: String },
    /// The GGUF parsed cleanly but its `general.architecture` is not in
    /// any per-family estimator's allowlist and the service config
    /// hasn't opted into the coarse fallback.
    UnknownArchitecture { architecture: SmolStr },
}

impl std::fmt::Display for EstimatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GgufRead { path, cause } => {
                write!(f, "read gguf at {}: {cause}", path.display())
            }
            Self::UnknownArchitecture { architecture } => {
                write!(
                    f,
                    "architecture {architecture:?} is not recognised. Set \
                     `estimation.allow_fallback = true` on the service to \
                     accept the coarse fallback (no KV modelling, weights-only). \
                     Recognised families:"
                )?;
                for fam in FAMILIES {
                    write!(f, " {}=[", fam.name)?;
                    for (i, arch) in fam.arches.iter().enumerate() {
                        if i > 0 {
                            f.write_str(",")?;
                        }
                        f.write_str(arch)?;
                    }
                    f.write_str("]")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for EstimatorError {}

/// Produce a base estimate for the model described by `inputs`. Reads the
/// GGUF (including any mmproj) through `fs` and dispatches on
/// `general.architecture`. Pure function over `inputs` + the bytes on
/// disk; caller applies rolling correction + safety factor afterward.
pub fn estimate_from_path(
    fs: &dyn Fs,
    inputs: &EstimatorInputs<'_>,
) -> Result<Estimate, EstimatorError> {
    let summary = gguf::read(fs, inputs.model).map_err(|e| EstimatorError::GgufRead {
        path: inputs.model.to_path_buf(),
        cause: e.to_string(),
    })?;

    info!(
        service = %inputs.name,
        architecture = %summary.architecture,
        block_count = ?summary.block_count,
        tensor_count = summary.tensors.len(),
        total_tensor_gb = summary.total_tensor_bytes / (1024 * 1024 * 1024),
        shard_count = summary.shards.len(),
        "gguf summary",
    );

    let mut est = dispatch(&summary, inputs)?;

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

pub fn dispatch(
    summary: &GgufSummary,
    inputs: &EstimatorInputs<'_>,
) -> Result<Estimate, EstimatorError> {
    let arch = summary.architecture.as_str();
    for fam in FAMILIES {
        if fam.arches.contains(&arch) {
            return Ok((fam.estimate)(summary, inputs));
        }
    }
    if inputs.allow_fallback {
        return Ok(fallback::estimate_fallback(summary, inputs.context));
    }
    Err(EstimatorError::UnknownArchitecture {
        architecture: summary.architecture.clone(),
    })
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
            allow_fallback: true,
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
        let e = dispatch(&summary, &inputs_for(&empty)).unwrap();
        assert_eq!(e.architecture, "qwen3");
    }

    #[test]
    fn dispatch_unknown_goes_to_fallback_when_opted_in() {
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
        let mut inputs = inputs_for(&empty);
        inputs.allow_fallback = true;
        let e = dispatch(&summary, &inputs).unwrap();
        // Fallback returns a non-zero weights estimate (the exact formula
        // lives in `fallback::estimate_fallback`; asserting shape here
        // keeps this test decoupled from the specific coefficients).
        assert!(e.weights_bytes > 0);
    }

    /// Regression: an unknown architecture with `allow_fallback = false`
    /// must return `UnknownArchitecture` rather than silently producing
    /// the coarse fallback guess. Guards against another glm4moe-style
    /// silent 67× under-reservation.
    #[test]
    fn unknown_architecture_rejects_without_opt_in() {
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
        let mut inputs = inputs_for(&empty);
        inputs.allow_fallback = false;
        match dispatch(&summary, &inputs) {
            Err(EstimatorError::UnknownArchitecture { architecture }) => {
                assert_eq!(architecture.as_str(), "novel-arch");
            }
            other => panic!("expected UnknownArchitecture; got {other:?}"),
        }
    }
}
