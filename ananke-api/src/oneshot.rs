//! Oneshot request + response bodies.

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
    /// Free-form metadata passed through to responses.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, serde_json::Value>,
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

/// `GET /api/oneshot/{id}` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct OneshotStatus {
    /// ULID id.
    pub id: String,
    /// Name.
    pub name: String,
    /// Current state (`"pending"`, `"running"`, `"ended"`, `"evicted"`, etc.).
    pub state: String,
    /// Port.
    pub port: u16,
    /// Submission timestamp.
    pub submitted_at_ms: i64,
    /// Start timestamp, if reached Running.
    pub started_at_ms: Option<i64>,
    /// End timestamp, if terminal.
    pub ended_at_ms: Option<i64>,
    /// Exit code when the child terminated with one.
    pub exit_code: Option<i32>,
    /// Path component for the WS log-stream endpoint.
    pub logs_url: String,
}
