//! VRAM estimator — architecture-aware dispatch.

pub mod compute_buffer;
pub mod compute_model;
pub mod host_buffer;
pub mod hybrid;
mod ik_assignment;
pub mod kv;
pub mod llama;
pub mod mamba;
pub mod moe;
pub mod mtp;
pub mod override_tensor;
pub mod recurrent;
mod replicated;
pub mod service_inputs;
pub mod tuning;
pub mod types;

use ananke_fs::Fs;
use ananke_gguf::{self, Architecture, GgufSummary};
use tracing::{info, warn};
pub use types::{
    Buffers, Estimate, EstimatorInputs, ExpertKind, ExpertTensor, Fork, HostBytes, IkAssignment,
    Layout, Mtp, NonLayer, Speculation,
};

/// One per-architecture family, paired with the `general.architecture`
/// values it accepts and the function that produces an `Estimate` for
/// them. Dispatch walks this table top-to-bottom; error formatting
/// enumerates the same table so "recognised" never drifts from "actually
/// dispatched".
struct Family {
    name: &'static str,
    arches: &'static [Architecture],
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
    /// `ananke_gguf::read` failed (bad magic, IO error, unsupported dtype, …).
    /// The inner error is the reader's own diagnostic.
    GgufRead {
        path: std::path::PathBuf,
        cause: ananke_gguf::ReadError,
    },
    /// The GGUF parsed cleanly but no per-family estimator covers its
    /// `general.architecture`. Nothing here can price the model's KV cache or
    /// graph, so the estimator refuses rather than returning a number it has
    /// no basis for; the operator declares the reservation instead.
    UnknownArchitecture { architecture: Architecture },
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
                    "architecture {architecture} is not recognised, so its memory \
                     cannot be estimated. Give the service an explicit reservation \
                     (`mode` plus `reserve_gb`, or `min_reserve_gb`/`max_reserve_gb`) \
                     and it is placed from that instead. Recognised families:"
                )?;
                for fam in FAMILIES {
                    write!(f, " {}=[", fam.name)?;
                    for (i, arch) in fam.arches.iter().enumerate() {
                        if i > 0 {
                            f.write_str(",")?;
                        }
                        f.write_str(arch.as_str())?;
                    }
                    f.write_str("]")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for EstimatorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::GgufRead { cause, .. } => Some(cause),
            Self::UnknownArchitecture { .. } => None,
        }
    }
}

/// Produce a base estimate for the model described by `inputs`. Reads the
/// GGUF (including any mmproj) through `fs` and dispatches on
/// `general.architecture`. Pure function over `inputs` + the bytes on
/// disk; caller applies rolling correction + safety factor afterward.
///
/// Thin wrapper around [`estimate_with_summary`] for callers that don't
/// need the GGUF summary back; new code that wants both should call
/// `estimate_with_summary` directly so the file is parsed only once.
pub fn estimate_from_path(
    fs: &dyn Fs,
    inputs: &EstimatorInputs<'_>,
) -> Result<Estimate, EstimatorError> {
    estimate_with_summary(fs, inputs).map(|(_summary, est)| est)
}

/// Same as [`estimate_from_path`] but also returns the parsed
/// [`GgufSummary`] so the caller can derive `ModelInfo`-style facts
/// (architecture, block count, metadata keys) without re-parsing the
/// file. Used by the management `ServiceDetail` cache and by the
/// supervisor's spawn-time cache warming so the two paths share one
/// GGUF read.
pub fn estimate_with_summary(
    fs: &dyn Fs,
    inputs: &EstimatorInputs<'_>,
) -> Result<(GgufSummary, Estimate), EstimatorError> {
    let summary = ananke_gguf::read(fs, inputs.model).map_err(|e| EstimatorError::GgufRead {
        path: inputs.model.to_path_buf(),
        cause: e,
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

    // MTP / NextN draft-context overhead is architecture-independent
    // post-processing: it reads `nextn_predict_layers` + the full-attention
    // head dims straight from the GGUF (embedded head), or — when a separate
    // draft GGUF is configured via `-md` — that file's resident weights. It
    // applies uniformly to whichever family dispatched above rather than
    // living in each one.
    let draft_summary = match inputs.speculation.draft_model() {
        Some(path) => match ananke_gguf::read(fs, path) {
            Ok(s) => Some(s),
            Err(e) => {
                warn!(
                    service = %inputs.name,
                    error = %e,
                    path = %path.display(),
                    "draft model read failed; MTP overhead will be under-estimated",
                );
                None
            }
        },
        _ => None,
    };
    est.mtp.bytes = mtp::mtp_overhead_bytes(&summary, draft_summary.as_ref(), inputs);
    est.mtp.weight_bytes = mtp::mtp_weight_bytes(draft_summary.as_ref(), inputs);

    // What the head GPU holds beyond every other card, which the packer trims
    // off the secondaries. The compute model derives it — the logits buffer, the
    // head card's flat graph cost, and the expert-staging buffers a hybrid
    // places on the primary — rather than approximating it with a bare
    // `n_vocab x ubatch` term, which can only be a deliberate under-estimate.
    est.buffers.output_bytes =
        u64::from(compute_model::head_extra_mib(&summary, inputs)).saturating_mul(1024 * 1024);

    // Host-side overhead: the pinned graph arena and the server's prompt
    // cache. Architecture-independent apart from the sliding-window lookup,
    // and charged to the `Cpu` slot regardless of where the layers land, so
    // it is computed here rather than in each family estimate.
    est.host.overhead_bytes = host_buffer::host_overhead_bytes(&summary, inputs);
    est.host.cache_bytes = host_buffer::prompt_cache_bytes(inputs);
    est.host.slot_bytes = host_buffer::slot_host_bytes(&summary.architecture, inputs);
    est.host.checkpoint_bytes = host_buffer::checkpoint_headroom_bytes(&summary, inputs);

    info!(
        service = %inputs.name,
        weights_gb = est.weights_bytes / (1024 * 1024 * 1024),
        per_layer_len = est.layout.per_layer_bytes.as_ref().map(|v| v.len()).unwrap_or(0),
        kv_per_token = est.kv_per_token,
        mtp_mb = est.mtp.bytes / (1024 * 1024),
        "post-dispatch estimate",
    );

    if inputs.fork.is_ik() {
        drop_mtp_head_blocks(&mut est, &summary);
    }

    // Apply user-declared override_tensor rules BEFORE mmproj so matched
    // tensors leave the layer/non-layer budget cleanly.
    if !inputs.override_tensor.is_empty() {
        override_tensor::parse_and_apply(&mut est, &summary, inputs.override_tensor);
    }

    // Add mmproj bytes to GPU 0 weights, and its CLIP graph buffer beside
    // them. llama.cpp reserves both together on one device and reports the sum
    // (`[mtmd] adding N MiB to fit_params_target for device CUDA0`), but they
    // are charged apart: the weights are file-backed and the graph buffer is
    // not, and only the former belongs in the tally the host-pool observation
    // subtracts.
    if let Some(mmproj) = inputs.mmproj {
        match ananke_gguf::read(fs, mmproj) {
            Ok(proj) => {
                est.weights_bytes = est.weights_bytes.saturating_add(proj.total_tensor_bytes);
                est.layout.non_layer.other_bytes = est
                    .layout
                    .non_layer
                    .other_bytes
                    .saturating_add(proj.total_tensor_bytes);
                est.buffers.mmproj_graph_bytes = tuning::MMPROJ_GRAPH_BYTES;
            }
            Err(e) => warn!(error = %e, path = %mmproj.display(), "mmproj read failed"),
        }
    }

    // Weights a tensor split holds on every card rather than dividing. Read from
    // the tensor table, so it is zero for an architecture without them — which is
    // every dense model measured. See [`crate::replicated`].
    est.layout.tensor_split_replicated_bytes = replicated::tensor_split_replicated_bytes(&summary);

    Ok((summary, est))
}

/// Drop an MTP head's trailing blocks from the weight accounting.
///
/// ik_llama does not load them at all — it says so for the tensors it can name
/// (`model has unused tensor blk.78.indexer.k_norm.weight -- ignoring`) and its
/// buffer sizes show the whole block missing. Qwen3.6-35B-A3B loads 25288 MiB
/// across its GPU and host buffers on the fork against the GGUF's 25890, and the
/// 602 MiB difference is block 40, its nextn head. The same model on mainline
/// loads 25375 MiB onto the card — the GGUF less its CPU-mapped embedding table
/// — so mainline does load the head, and this correction is fork-only.
///
/// The blocks are removed from every view the packer reads, not just the total:
/// leaving them in `expert_tensors` would have it offload experts belonging to a
/// block whose per-layer cost is zero.
fn drop_mtp_head_blocks(est: &mut Estimate, summary: &GgufSummary) {
    let span = recurrent::context_layer_span(summary);
    let Some(per_layer) = est.layout.per_layer_bytes.as_mut() else {
        return;
    };
    if span as usize >= per_layer.len() {
        return;
    }
    for bytes in per_layer.iter_mut().skip(span as usize) {
        est.weights_bytes = est.weights_bytes.saturating_sub(*bytes);
        *bytes = 0;
    }
    let before = est.layout.expert_layers.len();
    est.layout.expert_layers.retain(|&l| l < span);
    // How many expert layers the head accounted for. ik's `-ncmoe` window is
    // taken over the *full* block range, so a trailing window swallows these —
    // it logs an override for them and then never loads them, wasting the slot.
    est.mtp.head_expert_layers = (before - est.layout.expert_layers.len()) as u32;
    if let Some(tensors) = est.layout.expert_tensors.as_mut() {
        tensors.retain(|t| t.layer < span);
    }
}

pub fn dispatch(
    summary: &GgufSummary,
    inputs: &EstimatorInputs<'_>,
) -> Result<Estimate, EstimatorError> {
    for fam in FAMILIES {
        if fam.arches.contains(&summary.architecture) {
            let mut estimate = (fam.estimate)(summary, inputs);
            estimate.layout.ik_assignment =
                ik_assignment::assignment(summary, inputs, &estimate.layout);
            return Ok(estimate);
        }
    }
    Err(EstimatorError::UnknownArchitecture {
        architecture: summary.architecture.clone(),
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use ananke_gguf::{
        Architecture, GgufType, keys,
        types::{GgufSummary, GgufValue},
    };
    use smol_str::SmolStr;

    use super::*;

    fn inputs_for() -> EstimatorInputs<'static> {
        EstimatorInputs {
            name: "demo",
            context: 4096,
            cache_type_k: Some(GgufType::F16),
            cache_type_v: Some(GgufType::F16),
            ..EstimatorInputs::empty(Path::new("/fake"))
        }
    }

    #[test]
    fn dispatch_recognises_known_families() {
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new(keys::ARCHITECTURE),
            GgufValue::String("qwen3".into()),
        );
        metadata.insert(SmolStr::new("qwen3.block_count"), GgufValue::U32(1));
        let summary = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors: Default::default(),
            metadata,
            block_count: Some(1),
            architecture: Architecture::Qwen3,
            shards: vec!["/fake".into()],
        };
        let e = dispatch(&summary, &inputs_for()).unwrap();
        assert_eq!(e.architecture, Architecture::Qwen3);
    }

    /// An architecture no family covers is refused, never estimated. A guess
    /// here reserved 400 MiB against glm4moe's real 27 GiB before the
    /// architecture was recognised; the operator gives these services an
    /// explicit reservation instead.
    #[test]
    fn an_unrecognised_architecture_is_refused() {
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new(keys::ARCHITECTURE),
            GgufValue::String("novel-arch".into()),
        );
        let summary = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 1_000_000,
            tensors: Default::default(),
            metadata,
            block_count: None,
            architecture: Architecture::from("novel-arch"),
            shards: vec!["/fake".into()],
        };
        match dispatch(&summary, &inputs_for()) {
            Err(EstimatorError::UnknownArchitecture { architecture }) => {
                assert_eq!(architecture, Architecture::from("novel-arch"));
            }
            other => panic!("expected UnknownArchitecture; got {other:?}"),
        }
    }

    /// The diagnostic has to name the architectures that would have worked —
    /// it is the operator's only list of what ananke can estimate.
    #[test]
    fn the_refusal_names_every_recognised_architecture() {
        let message = EstimatorError::UnknownArchitecture {
            architecture: Architecture::from("novel-arch"),
        }
        .to_string();
        for fam in FAMILIES {
            for arch in fam.arches {
                assert!(
                    message.contains(arch.as_str()),
                    "{} missing from the refusal message",
                    arch.as_str()
                );
            }
        }
    }
}
