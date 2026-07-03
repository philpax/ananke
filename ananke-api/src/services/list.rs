//! `GET /api/services` — service list.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    internal::fit_verdict::FitVerdict,
    shared::{metadata::AnankeMetadata, modality::Modality},
};

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
    /// Number of requests currently in flight through the proxy for this
    /// service. Zero when the service is not running.
    #[serde(default)]
    pub inflight_count: u64,
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
    /// Whether the service's estimated placement fits under current
    /// device conditions. `None` when the verdict can't be computed
    /// (e.g. a llama-cpp service whose GGUF hasn't been read yet).
    /// Running services are always `Fits`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit_verdict: Option<FitVerdict>,
    /// Total VRAM bytes the service would reserve across all devices
    /// under current conditions (from the placement preview). Includes
    /// weights, KV cache, and compute buffer. `None` when the placement
    /// can't be computed (e.g. a command service that reserves no VRAM).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_bytes: Option<u64>,
    /// Wall-clock timestamp (ms since epoch) of the last time the
    /// service was provisioned or received a request. `None` if the
    /// service has never been started.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_used_ms: Option<i64>,
}
