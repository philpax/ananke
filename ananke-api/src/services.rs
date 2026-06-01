//! Service summary and detail views.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{logs::LogLine, metadata::AnankeMetadata};

/// What kind of model the service exposes through the OpenAI-compatible
/// API. Drives badge rendering in the frontend and lets clients (Discord
/// rotation, RAG indexers) filter the model list by purpose without
/// parsing `metadata.*` strings.
///
/// Defaults to `Chat` so existing configs and JSON payloads stay
/// byte-identical — the field is elided from the wire when it's `Chat`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    /// Text generation: `/v1/chat/completions` and `/v1/completions`.
    /// The default for backward compatibility.
    #[default]
    Chat,
    /// Vector embeddings: `/v1/embeddings`. Pooling-only models such as
    /// jina-embeddings-v5, BGE, E5, etc.
    Embedding,
}

impl Modality {
    /// Predicate for `#[serde(skip_serializing_if)]` so the default
    /// (`Chat`) is elided from JSON. Existing chat services then ship
    /// the exact same wire bytes they shipped before this field landed.
    pub fn is_chat(&self) -> bool {
        matches!(self, Modality::Chat)
    }
}

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
    /// What kind of OpenAI endpoint the service serves. Elided from
    /// JSON when [`Modality::Chat`] (the default) so existing chat
    /// services emit unchanged wire bytes; embedding services explicitly
    /// declare `modality = "embedding"` in their `[[service]]` block.
    #[serde(default, skip_serializing_if = "Modality::is_chat")]
    pub modality: Modality,
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
    /// What kind of OpenAI endpoint the service serves. See
    /// [`ServiceSummary::modality`] for the rendering rule.
    #[serde(default, skip_serializing_if = "Modality::is_chat")]
    pub modality: Modality,
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
