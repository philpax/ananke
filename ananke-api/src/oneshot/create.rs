//! `POST /api/oneshot` — oneshot request + response bodies.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// `POST /api/oneshot` request body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct OneshotRequest {
    /// Optional human-meaningful name; daemon auto-generates if absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// `"llama-cpp"` or `"command"`.
    pub template: String,
    /// Command + args for the `"command"` template.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<Vec<String>>,
    /// Working directory for the spawned child.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workdir: Option<String>,
    /// Allocation mode + sizes.
    pub allocation: OneshotAllocation,
    /// Device-placement hints.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub devices: Option<OneshotDevices>,
    /// Eviction priority.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<u8>,
    /// Time-to-live duration string (`"2h"`, `"30m"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
    /// Explicit port request; `None` means daemon picks from the pool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    /// Health check configuration. When `None` (the default), the oneshot
    /// transitions to Running immediately after spawn without waiting for
    /// a health probe. Set to enable HTTP-based readiness checks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub health: Option<OneshotHealth>,
    /// Free-form metadata passed through to responses.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

/// Health-check configuration for a oneshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct OneshotHealth {
    /// HTTP path to probe (e.g. `"/health"`, `"/system_stats"`).
    pub http: String,
    /// Timeout duration string (e.g. `"30s"`, `"2m"`). Defaults to 3 minutes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout: Option<String>,
    /// Probe interval duration string. Defaults to 5 seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probe_interval: Option<String>,
}

/// Allocation-mode knobs for [`OneshotRequest`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct OneshotAllocation {
    /// `"static"` or `"dynamic"`; `None` for llama-cpp template.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// Static allocation amount.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_gb: Option<f32>,
    /// Dynamic min.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_vram_gb: Option<f32>,
    /// Dynamic max.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_vram_gb: Option<f32>,
}

/// Device-placement hints for [`OneshotRequest`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct OneshotDevices {
    /// `"gpu-only"` / `"cpu-only"` / `"hybrid"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement: Option<String>,
}

/// `POST /api/oneshot` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct OneshotResponse {
    /// ULID identifier (prefixed `oneshot_`).
    pub id: String,
    /// Assigned name.
    pub name: String,
    /// Bound port.
    pub port: u16,
    /// Path component suitable for appending to `ANANKE_ENDPOINT` to open the log stream.
    pub logs_url: String,
}
