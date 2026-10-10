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
    /// Whether the service supports the OpenAI multiplexer.
    #[serde(default)]
    pub openai_compat: bool,
    /// Published service UI URL from `metadata.web_ui_url`, when configured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_ui_url: Option<String>,
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
    /// JSON when [`Modality::Chat`] (the default), so a chat service's
    /// row carries no `modality` key; embedding services explicitly
    /// declare `modality = "embedding"` in their `[[service]]` block.
    #[serde(default, skip_serializing_if = "Modality::is_chat")]
    pub modality: Modality,
    /// Passthrough entries from `[[service]] metadata.*`. Empty when
    /// none are set, and the field is elided from JSON when the map is
    /// empty, so the key appears only for a service that opts in to
    /// metadata.
    #[serde(default, skip_serializing_if = "AnankeMetadata::is_empty")]
    #[schema(value_type = Object)]
    pub ananke_metadata: AnankeMetadata,
    /// Whether the service's estimated placement fits under current
    /// device conditions. `None` when the verdict can't be computed
    /// (e.g. a llama-cpp service whose GGUF hasn't been read yet).
    /// Running services are always `Fits`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit_verdict: Option<FitVerdict>,
    /// Total bytes the service would occupy across all devices — GPU VRAM and
    /// host RAM alike — under current conditions. Includes weights, KV cache,
    /// and compute buffer.
    ///
    /// Normally this is the placement preview's per-device sum: what the
    /// service would actually reserve. When `fit_verdict` is `does_not_fit`
    /// there is no placement to sum, so this falls back to the estimator's
    /// aggregate demand — how much the model *would need*. Read the two fields
    /// together; a `does_not_fit` verdict means this figure is a requirement,
    /// not a reservation.
    ///
    /// `None` when neither can be computed (e.g. a command service that
    /// reserves nothing, or a llama-cpp service whose GGUF hasn't been read).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub footprint_bytes: Option<u64>,
    /// Where [`Self::footprint_bytes`] would land, per device, sorted by device.
    ///
    /// The same figure broken out rather than a second one: the total is this
    /// list's sum, so a row can show both without the two disagreeing. Empty when
    /// there is no placement to describe, which is the same condition that leaves
    /// `footprint_bytes` `None`.
    ///
    /// Deliberately slimmer than the detail view's `DevicePlacement`, which also
    /// carries each device's capacity and what else is using it. Those support a
    /// utilisation bar; a list row only names where the memory goes, and this
    /// endpoint is polled every two seconds.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub footprint_devices: Vec<DeviceFootprint>,
    /// Wall-clock timestamp (ms since epoch) of the last time the
    /// service was provisioned or received a request. `None` if the
    /// service has never been started.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_used_ms: Option<i64>,
}

/// One device's share of a [`ServiceSummary`]'s footprint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct DeviceFootprint {
    /// Slot string: `"cpu"` or `"gpu:N"`, as the placement engine names it.
    pub device: String,
    /// Memory bytes the service would occupy on that device.
    pub bytes: u64,
}
