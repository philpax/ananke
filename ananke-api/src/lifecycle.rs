//! Service lifecycle POST response bodies.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// `POST /api/services/{name}/start` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
#[allow(missing_docs)]
pub enum StartResponse {
    AlreadyRunning,
    Started { run_id: i64 },
    QueueFull,
    Unavailable { reason: String },
}

/// `POST /api/services/{name}/stop` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
#[allow(missing_docs)]
pub enum StopResponse {
    NotRunning,
    Drained,
}

/// `POST /api/services/{name}/enable` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
#[allow(missing_docs)]
pub enum EnableResponse {
    Enabled,
    NotDisabled,
    AlreadyEnabled,
}

/// `POST /api/services/{name}/disable` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
#[allow(missing_docs)]
pub enum DisableResponse {
    Disabled,
    AlreadyDisabled,
}
