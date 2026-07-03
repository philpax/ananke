//! `GET /api/services/{name}/command` — launch command preview.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Whether a [`LaunchCommand`] describes a live process or a what-if.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum LaunchCommandSource {
    /// The service is running; this configuration is what it was launched
    /// with (recomputed from the current config and placement, so it matches
    /// the live process unless the config was edited since it started).
    Running,
    /// The service is not running; this is the command it would launch with
    /// on the next start, given the current config and device state.
    Preview,
}

/// One environment variable ananke sets on the child process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct EnvVar {
    /// Variable name (e.g. `CUDA_VISIBLE_DEVICES`).
    pub key: String,
    /// Variable value.
    pub value: String,
}

/// Response from `GET /api/services/{name}/command`: the launch command
/// computed under two scenarios.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct LaunchCommandResponse {
    /// Command on an empty cluster — what the service would launch with
    /// if no other services held pledges. Always present when the service
    /// can fit on the hardware at all.
    pub on_empty: LaunchCommand,
    /// Command against the current device state and pledge book. `None`
    /// when the service can't fit alongside currently running services.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active: Option<LaunchCommand>,
}

/// One launch command — argv and environment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct LaunchCommand {
    /// Whether the service is running (`running`) or this is a preview of the
    /// next start (`preview`).
    pub source: LaunchCommandSource,
    /// The full argv. `argv[0]` is the binary; the rest are its arguments.
    /// Already split into tokens — no shell quoting is applied, so a client
    /// rendering a copy-pasteable line should quote as needed.
    pub argv: Vec<String>,
    /// Environment variables ananke sets or overrides for the child (notably
    /// `CUDA_VISIBLE_DEVICES`), sorted by key. Not the full inherited
    /// environment.
    pub env: Vec<EnvVar>,
}
