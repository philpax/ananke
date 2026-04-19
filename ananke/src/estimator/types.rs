//! Estimator output types.

use std::collections::BTreeMap;

use smol_str::SmolStr;

use crate::config::DeviceSlot;

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
