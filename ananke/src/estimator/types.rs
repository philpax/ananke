//! Estimator output types.

use std::{collections::BTreeMap, path::Path};

use smol_str::SmolStr;

use crate::config::{DeviceSlot, ServiceConfig};

/// Pure inputs the estimator reads. The daemon builds one of these from a
/// `ServiceConfig` on each spawn; standalone callers (calibration tools,
/// model-size inspection examples) construct one directly without having
/// to fabricate an entire `ServiceConfig`.
///
/// Lifetimes keep it borrow-only — a `&EstimatorInputs<'_>` never owns its
/// strings or path references.
#[derive(Debug, Clone)]
pub struct EstimatorInputs<'a> {
    /// Service name — used for log context only; the estimate itself is
    /// identical regardless of what this says.
    pub name: &'a str,
    /// Path to the first GGUF shard.
    pub model: &'a Path,
    /// Optional vision projector (adds its tensor bytes to the weights
    /// total and the first-GPU non-layer bucket).
    pub mmproj: Option<&'a Path>,
    /// Context window the child will be launched with. Absent means 4096.
    pub context: u32,
    /// Physical batch size (`--ubatch-size` / `-ub`) the child will launch
    /// with. Absent means llama.cpp's default of 512. Only the deepseek4
    /// NSA-indexer compute buffer scales with it (∝ `ubatch × context`);
    /// every other architecture's compute buffer is ~ubatch-independent, so
    /// the estimator ignores this outside that arch.
    pub ubatch: Option<u32>,
    /// K-cache quantisation (f16, q8_0, etc.). Absent means f16.
    pub cache_type_k: Option<&'a str>,
    /// V-cache quantisation. Absent means f16.
    pub cache_type_v: Option<&'a str>,
    /// `override_tensor` regex rules to pin specific tensors to CPU / a GPU.
    pub override_tensor: &'a [String],
    /// Override for the compute-buffer reservation (MB per active device).
    /// Absent means the estimator's 400 MB default.
    pub compute_buffer_mb: Option<u32>,
    /// Whether the operator has opted into the coarse fallback when the
    /// GGUF's architecture isn't recognised by any per-family estimator.
    /// `false` by default — unknown architectures return an error instead
    /// of silently producing a guess that may be badly wrong.
    pub allow_fallback: bool,
    /// Whether the service runs with `--spec-type draft-mtp`. When set and
    /// the model carries an MTP head (`nextn_predict_layers > 0`), the
    /// estimator adds the MTP draft context's KV + compute overhead. See
    /// [`crate::estimator::mtp`].
    pub mtp: bool,
    /// Optional separate draft-model GGUF (`-md`). When `mtp` is set and
    /// this is present, the estimator reads this file's resident weights
    /// plus a draft compute buffer rather than the target model's embedded
    /// MTP head. See [`crate::estimator::mtp`].
    pub draft_model: Option<&'a Path>,
}

impl<'a> EstimatorInputs<'a> {
    /// Distil the estimator-relevant fields out of a `ServiceConfig`.
    /// Returns `None` if `svc` is a command-template service — the
    /// estimator only applies to llama-cpp workloads.
    pub fn from_service(svc: &'a ServiceConfig) -> Option<Self> {
        let lc = svc.llama_cpp()?;
        Some(Self {
            name: svc.name.as_str(),
            model: lc.model.as_path(),
            mmproj: lc.mmproj.as_deref(),
            context: lc.context.unwrap_or(4096),
            ubatch: lc.ubatch_size,
            cache_type_k: lc.cache_type_k.as_deref(),
            cache_type_v: lc.cache_type_v.as_deref(),
            override_tensor: &lc.override_tensor,
            compute_buffer_mb: lc.estimation.compute_buffer_mb,
            allow_fallback: lc.estimation.allow_fallback.unwrap_or(false),
            mtp: lc.spec_type.as_deref() == Some("draft-mtp"),
            draft_model: lc.draft_model.as_deref(),
        })
    }

    /// Stable hash of every field that would change the estimate's
    /// numbers. Cache layers (currently the daemon-side
    /// `EstimateCache`) compare this against the value stored
    /// alongside a cached entry to detect "the operator edited
    /// `context` / `override_tensor` / `cache_type_*` / … without
    /// changing the GGUF path" — the model on disk is the same but
    /// the estimate isn't.
    ///
    /// `model` and `mmproj` paths are deliberately excluded because
    /// the cache keys on them separately (any path change is a
    /// different model, not a different config of the same model).
    /// `draft_model` *is* hashed here because the cache does not key on
    /// it separately, yet swapping the draft GGUF changes the estimate.
    /// `name` is excluded because it's a log-context-only field.
    pub fn config_fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.context.hash(&mut hasher);
        self.ubatch.hash(&mut hasher);
        self.cache_type_k.hash(&mut hasher);
        self.cache_type_v.hash(&mut hasher);
        self.override_tensor.hash(&mut hasher);
        self.compute_buffer_mb.hash(&mut hasher);
        self.allow_fallback.hash(&mut hasher);
        self.mtp.hash(&mut hasher);
        self.draft_model.hash(&mut hasher);
        hasher.finish()
    }
}

/// Base estimate for a service's VRAM footprint, pre-safety-factor and
/// pre-rolling-correction.
#[derive(Debug, Clone)]
pub struct Estimate {
    /// Static weight bytes (including mmproj if present).
    pub weights_bytes: u64,
    /// KV cache bytes per context token (zero for architectures without KV).
    pub kv_per_token: u64,
    /// Compute buffer per device in MB (default 400).
    pub compute_buffer_mb: u32,
    /// Extra VRAM (bytes) for the MTP / NextN draft context when the
    /// service runs `--spec-type draft-mtp`. Zero when MTP is off or the
    /// model carries no MTP head. Reserved as a single lump on the
    /// primary GPU by the packer. See [`super::mtp`].
    pub mtp_bytes: u64,
    /// Per-layer weight bytes for index-ordered packing. `None` for
    /// architectures where layer-aware placement isn't applicable
    /// (currently SSM/Mamba; in that case `placement` uses single-device
    /// best-fit on `total = weights + compute_buffer`).
    pub per_layer_bytes: Option<Vec<u64>>,
    /// Layer indices that are attention-bearing (used to scope KV
    /// cost to those layers). `None` = all layers carry KV.
    pub attention_layers: Option<Vec<u32>>,
    /// Non-layer tensors: output head, token embeddings, norms.
    pub non_layer: NonLayer,
    /// Tensor-level overrides (from `override_tensor` rules) already
    /// resolved to per-device byte attributions by the estimator.
    pub override_tensor_bytes: BTreeMap<DeviceSlot, u64>,
    /// Layer indices that carry expert (`_exps`) tensors — diagnostic only.
    /// Empty for non-MoE architectures.
    pub expert_layers: Vec<u32>,
    /// The offloadable expert tensors (fused `blk.N.ffn_{gate,up,down}_exps`),
    /// `Some` for MoE architectures. The packer chooses which of these to move
    /// off-GPU (to CPU or a secondary GPU) to make the model fit, and
    /// synthesises the matching `-ot` rules. These bytes are *also* counted in
    /// `per_layer_bytes[i]` (the full per-layer cost); when the packer offloads
    /// an expert it subtracts that tensor's bytes from the layer's GPU share.
    /// Keeping `per_layer_bytes` full means every non-expert-aware code path
    /// (the plain layer walk, the sharded/tensor-split path, `override_tensor`
    /// accounting) stays correct without special-casing MoE. `None` for non-MoE
    /// architectures.
    pub expert_tensors: Option<Vec<ExpertTensor>>,
    /// `context` that was used to compute `kv_per_token × context`.
    pub context: u32,
    /// Architecture string for diagnostics.
    pub architecture: SmolStr,
}

/// One offloadable fused expert tensor on a MoE layer. llama.cpp stacks every
/// expert of a given projection into a single tensor per layer
/// (`blk.N.ffn_gate_exps.weight`, …), so there are at most three per layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertTensor {
    /// Block/layer index this expert tensor belongs to.
    pub layer: u32,
    /// Which projection (gate / up / down) this is.
    pub kind: ExpertKind,
    /// Tensor weight bytes — the amount freed from GPU when offloaded.
    pub bytes: u64,
}

/// The three expert projections a MoE layer can carry. Used to build precise
/// `-ot blk.N.ffn_<kind>_exps.=<device>` rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExpertKind {
    Gate,
    Up,
    Down,
}

impl ExpertKind {
    /// The `ffn_<…>_exps` token as it appears in the GGUF tensor name and the
    /// `-ot` regex.
    pub fn tensor_token(self) -> &'static str {
        match self {
            ExpertKind::Gate => "gate",
            ExpertKind::Up => "up",
            ExpertKind::Down => "down",
        }
    }
}

/// Non-layer tensor footprint (matches llama.cpp's behaviour).
#[derive(Debug, Clone, Default)]
pub struct NonLayer {
    /// Output head — attributed to GPU 0 if any GPU used, else CPU.
    pub output_head_bytes: u64,
    /// Token embeddings — always on CPU.
    pub token_embd_bytes: u64,
    /// Small tensors (norms, rope tables) lumped together.
    pub other_bytes: u64,
}
