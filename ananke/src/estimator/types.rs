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
    /// K-cache quantisation (f16, q8_0, etc.). Absent means f16.
    pub cache_type_k: Option<&'a str>,
    /// V-cache quantisation. Absent means f16.
    pub cache_type_v: Option<&'a str>,
    /// `override_tensor` regex rules to pin specific tensors to CPU / a GPU.
    pub override_tensor: &'a [String],
    /// MoE-specific: how many expert layers to offload to CPU via
    /// `--n-cpu-moe`. Absent means 0.
    pub n_cpu_moe: Option<u32>,
    /// Override for the compute-buffer reservation (MB per active device).
    /// Absent means the estimator's 400 MB default.
    pub compute_buffer_mb: Option<u32>,
    /// Whether the operator has opted into the coarse fallback when the
    /// GGUF's architecture isn't recognised by any per-family estimator.
    /// `false` by default — unknown architectures return an error instead
    /// of silently producing a guess that may be badly wrong.
    pub allow_fallback: bool,
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
            cache_type_k: lc.cache_type_k.as_deref(),
            cache_type_v: lc.cache_type_v.as_deref(),
            override_tensor: &lc.override_tensor,
            n_cpu_moe: lc.n_cpu_moe,
            compute_buffer_mb: lc.estimation.compute_buffer_mb,
            allow_fallback: lc.estimation.allow_fallback.unwrap_or(false),
        })
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
    /// Number of expert-bearing layers eligible for `n_cpu_moe` offload
    /// (only meaningful for MoE).
    pub expert_layers: Vec<u32>,
    /// Per expert-layer, the bytes saved by offloading those experts to CPU.
    pub expert_layer_cpu_bytes: BTreeMap<u32, u64>,
    /// `context` that was used to compute `kv_per_token × context`.
    pub context: u32,
    /// Architecture string for diagnostics.
    pub architecture: SmolStr,
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
