//! OpenAI-shaped error envelope used by every `/api/*` error response.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// `{"error": {"code", "message", "type"}}`. The shared wire DTO; the
/// daemon-side `ApiErrorCode` enum is the source of truth for which
/// `code`, `message`, and `type` go together.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct ApiError {
    /// The nested body with the actual error metadata.
    pub error: ApiErrorBody,
}

/// Inner body of [`ApiError`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct ApiErrorBody {
    /// Short slug identifying the error class.
    pub code: String,
    /// Human-readable error message.
    pub message: String,
    /// OpenAI's taxonomy — `"invalid_request_error"` for things the
    /// client could have avoided, `"server_error"` for daemon-side
    /// problems. Previously hardcoded to `"server_error"` everywhere;
    /// now derived from the daemon's `ApiErrorCode` variant.
    #[serde(rename = "type")]
    pub kind: String,
}

impl ApiError {
    /// Build an error with `type: "server_error"`. Kept as the
    /// shorthand for the common case; new daemon-side code should
    /// go through `ApiErrorCode` and rely on `From<ApiErrorCode>
    /// for ApiError` so the kind is variant-accurate.
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::with_kind(code, message, "server_error")
    }

    /// Build an error with an explicit `type`. Used by the
    /// daemon-side `From<ApiErrorCode> for ApiError` impl so the
    /// projection honours each variant's actual taxonomy bucket.
    pub fn with_kind(
        code: impl Into<String>,
        message: impl Into<String>,
        kind: impl Into<String>,
    ) -> Self {
        Self {
            error: ApiErrorBody {
                code: code.into(),
                message: message.into(),
                kind: kind.into(),
            },
        }
    }
}
