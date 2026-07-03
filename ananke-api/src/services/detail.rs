//! `GET /api/services/{name}` — service detail.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    internal::{fit_verdict::FitVerdict, log_line::LogLine},
    shared::{metadata::AnankeMetadata, modality::Modality},
};

/// `GET /api/services/{name}` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ServiceDetail {
    /// Service name.
    pub name: String,
    /// Current state name.
    pub state: String,
    /// `"persistent"` or `"ondemand"`.
    pub lifecycle: String,
    /// Eviction priority.
    pub priority: u8,
    /// Public port.
    pub port: u16,
    /// Private (upstream) port.
    pub private_port: u16,
    /// Template name, e.g. `"llamacpp"` or `"command"`.
    pub template: String,
    /// Manual placement override, keyed by device slot string.
    pub placement_override: BTreeMap<String, u64>,
    /// Idle-before-drain timeout.
    pub idle_timeout_ms: u64,
    /// Active run id if any.
    pub run_id: Option<i64>,
    /// Child PID if any.
    pub pid: Option<i32>,
    /// Most recent log lines for a frontend's first-paint context.
    pub recent_logs: Vec<LogLine>,
    /// Rolling estimator correction factor.
    pub rolling_mean: Option<f32>,
    /// Sample count backing the rolling mean.
    pub rolling_samples: u64,
    /// Observed memory peak (VRAM + RSS) across the service's lifetime.
    pub observed_peak_bytes: u64,
    /// Placeholder for elastic-borrower tracking.
    pub elastic_borrower: Option<String>,
    /// GGUF-derived facts about the model file. `None` for command-
    /// template services and for llama-cpp services whose GGUF couldn't
    /// be read (file missing, unrecognised, etc.). Cached in the daemon
    /// after the first successful read so repeated detail polls don't
    /// re-parse the file.
    pub model_info: Option<ModelInfo>,
    /// Memory estimate computed against the service's configured context
    /// and KV cache settings. Same caching rules as `model_info`.
    pub estimate: Option<EstimateSummary>,
    /// Per-device placement the service would take under current conditions,
    /// plus whether it fits without eviction. `None` for command-template
    /// services, services with a manual `placement_override`, and llama-cpp
    /// services whose GGUF couldn't be read.
    pub placement_preview: Option<PlacementPreview>,
    /// What pledge the service is currently holding on each device.
    /// Empty when idle. Keys are slot strings (`"cpu"`, `"gpu:0"`, …),
    /// values are MiB.
    pub current_allocation: BTreeMap<String, u64>,
    /// What kind of OpenAI endpoint the service serves. See
    /// [`crate::services::list::ServiceSummary::modality`] for the rendering rule.
    #[serde(default, skip_serializing_if = "Modality::is_chat")]
    pub modality: Modality,
    /// Passthrough entries from `[[service]] metadata.*`. See
    /// [`crate::services::list::ServiceSummary::ananke_metadata`].
    #[serde(default, skip_serializing_if = "AnankeMetadata::is_empty")]
    #[schema(value_type = Object)]
    pub ananke_metadata: AnankeMetadata,
    /// Wall-clock timestamp (ms since epoch) of the last time the
    /// service was provisioned or received a request. `None` if the
    /// service has never been started.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_used_ms: Option<i64>,
}

/// GGUF-derived facts about a model file. Read once per service per
/// daemon run and cached; the file isn't re-parsed on every detail
/// poll.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ModelInfo {
    /// `general.architecture` from the GGUF (`"qwen2"`, `"llama"`,
    /// `"qwen3moe"`, …).
    pub architecture: String,
    /// Friendly name declared in `general.name`, when the publisher
    /// set one. Typically the upstream model identifier ("Qwen3 32B",
    /// "Llama 3.3 70B Instruct", …) rather than ananke's service
    /// name. Absent for hand-built GGUFs that left the field blank.
    pub model_name: Option<String>,
    /// `general.license` if declared (`"apache-2.0"`, `"llama3.3"`,
    /// `"gemma"`, …). Surfaced so operators can see the licence
    /// without leaving the dashboard.
    pub license: Option<String>,
    /// `general.parameter_count` if declared by the publisher. Modern
    /// converter tools (Unsloth, bartowski's repacks, …) write this;
    /// older GGUFs may not. The field is the model's *parameter*
    /// count, not the on-disk byte size — quantisation makes the two
    /// diverge by a large factor.
    pub parameter_count: Option<u64>,
    /// Sum of every tensor's byte size across every shard. The
    /// reference number when sizing the model on disk.
    pub total_tensor_bytes: u64,
    /// Layer count exposed by the architecture's `{arch}.block_count`
    /// metadata key, when present.
    pub block_count: Option<u32>,
    /// Number of shards discovered. 1 for a single-file GGUF.
    pub shard_count: u32,
    /// The model's *trained* context window, read from
    /// `{arch}.context_length`. The service's configured context may
    /// be lower.
    pub trained_context_length: Option<u32>,
    /// File name (basename of shard 0). Useful for the UI to show the
    /// underlying filename without the full path.
    pub file_name: String,
    /// `true` when the service has a configured `mmproj` GGUF — the
    /// standard signal for vision / multimodal support.
    pub has_mmproj: bool,
}

/// Estimator output projected to the wire. Carries the components a
/// reader needs to answer "how much VRAM will this service take?"
/// without having to re-derive any of them client-side.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct EstimateSummary {
    /// Static weight bytes (including mmproj if configured).
    pub weights_bytes: u64,
    /// KV cache bytes per context token. Zero for SSM/Mamba
    /// architectures with no KV cache.
    pub kv_per_token: u64,
    /// Configured context window the estimate is sized against.
    pub configured_context: u32,
    /// `kv_per_token × configured_context`. Precomputed so the
    /// frontend can render the total directly.
    pub kv_bytes_for_context: u64,
    /// Compute-buffer reservation per active device, in bytes.
    pub compute_buffer_bytes_per_device: u64,
}

/// Where a service's memory would land per device under current conditions, and
/// whether it fits without the daemon having to evict or reclaim. Computed by
/// running the placement engine against the live snapshot and pledge book.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct PlacementPreview {
    /// Per-device memory the service would occupy, sorted by device. Keys are
    /// slot strings (`"cpu"`, `"gpu:0"`, …); values are bytes.
    pub devices: Vec<DevicePlacement>,
    /// Whether the placement fits right now without eviction or reclaim.
    pub verdict: FitVerdict,
    /// Total expert-tensor bytes the packer offloaded to the CPU for a MoE
    /// model (auto/manual `expert_offload`). Zero for non-MoE services or when
    /// nothing was offloaded.
    pub expert_offload_bytes: u64,
    /// Number of distinct layers with at least one expert offloaded to the CPU.
    pub expert_offload_layers: u32,
}

/// One device's share of a [`PlacementPreview`], with enough context to draw a
/// utilisation bar: this service's share, what is already in use by everything
/// else, and the device's total capacity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct DevicePlacement {
    /// Slot string: `"cpu"` or `"gpu:N"`.
    pub device: String,
    /// Memory bytes this service reserves on the device — the pledge floor for a
    /// dynamic service.
    pub bytes: u64,
    /// Upper bound this service could grow to on the device. Equals `bytes`
    /// for fixed-size services; larger for a dynamic command service that may
    /// borrow up to its configured maximum.
    pub max_bytes: u64,
    /// Memory bytes already in use on the device by everything except this
    /// service (for a running service, its own resident memory is excluded so
    /// `used_by_others_bytes + bytes` doesn't double-count it).
    pub used_by_others_bytes: u64,
    /// Total memory capacity of the device, in bytes. Zero if unknown.
    pub total_bytes: u64,
}
