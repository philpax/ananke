//! Service summary and detail views.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{logs::LogLine, metadata::AnankeMetadata};

/// Response from `GET /api/services`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ServicesResponse {
    /// Registered services.
    pub services: Vec<ServiceSummary>,
    /// Port the OpenAI-compatible API is listening on.
    pub openai_api_port: u16,
}

/// One entry in `GET /api/services`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ServiceSummary {
    /// Service name (matches `[[service]]` table in config).
    pub name: String,
    /// State like `"idle"`, `"running"`, `"disabled_user_disabled"`.
    pub state: String,
    /// `"persistent"` or `"ondemand"`.
    pub lifecycle: String,
    /// Eviction priority.
    pub priority: u8,
    /// Public port the proxy listens on.
    pub port: u16,
    /// Active run id if currently running.
    pub run_id: Option<i64>,
    /// Child PID if currently running.
    pub pid: Option<i32>,
    /// Placeholder for elastic-borrower tracking (future work).
    pub elastic_borrower: Option<String>,
    /// `true` when the service's `[[service.llama_cpp]]` config has a
    /// `mmproj` entry — the standard signal that it supports vision /
    /// multimodal input. `None` for non-llama-cpp services. Cheap
    /// enough (config-only check) to ship on every list entry.
    pub has_mmproj: Option<bool>,
    /// Passthrough entries from `[[service]] metadata.*`. Empty when
    /// none were set; the field is elided from JSON when the map is
    /// empty so existing consumers see no change unless a service opts
    /// in to metadata.
    #[serde(default, skip_serializing_if = "AnankeMetadata::is_empty")]
    #[schema(value_type = Object)]
    pub ananke_metadata: AnankeMetadata,
}

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
    /// Observed VRAM peak across the service's lifetime.
    pub observed_peak_bytes: u64,
    /// Placeholder for elastic-borrower tracking.
    pub elastic_borrower: Option<String>,
    /// GGUF-derived facts about the model file. `None` for command-
    /// template services and for llama-cpp services whose GGUF couldn't
    /// be read (file missing, unrecognised, etc.). Cached in the daemon
    /// after the first successful read so repeated detail polls don't
    /// re-parse the file.
    pub model_info: Option<ModelInfo>,
    /// VRAM estimate computed against the service's configured context
    /// and KV cache settings. Same caching rules as `model_info`.
    pub estimate: Option<EstimateSummary>,
    /// What pledge the service is currently holding on each device.
    /// Empty when idle. Keys are slot strings (`"cpu"`, `"gpu:0"`, …),
    /// values are MiB.
    pub current_allocation: BTreeMap<String, u64>,
    /// Passthrough entries from `[[service]] metadata.*`. See
    /// [`ServiceSummary::ananke_metadata`].
    #[serde(default, skip_serializing_if = "AnankeMetadata::is_empty")]
    #[schema(value_type = Object)]
    pub ananke_metadata: AnankeMetadata,
}

/// GGUF-derived facts about a model file. Read once per service per
/// daemon run and cached; the file isn't re-parsed on every detail
/// poll.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ModelInfo {
    /// `general.architecture` from the GGUF (`"qwen2"`, `"llama"`,
    /// `"qwen3moe"`, …).
    pub architecture: String,
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
