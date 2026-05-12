//! Service lifecycle POST response bodies.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::errors::ApiErrorBody;

/// `POST /api/services/{name}/start` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
#[allow(missing_docs)]
pub enum StartResponse {
    AlreadyRunning,
    Started {
        run_id: i64,
    },
    QueueFull,
    /// The supervisor declined to start: VRAM didn't fit, the service
    /// is disabled, etc. The embedded [`ApiErrorBody`] carries the
    /// same typed slug, message, and kind that a 503 `ApiError`
    /// response would (so clients can switch on `error.code` instead
    /// of pattern-matching a freeform `reason` string). The 202 status
    /// is preserved because this is a "controlled outcome" of the
    /// start request, not a server-side fault.
    Unavailable {
        error: ApiErrorBody,
    },
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
