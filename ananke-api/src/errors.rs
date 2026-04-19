//! OpenAI-shaped error envelope used by every `/api/*` error response.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// `{"error": {"code", "message", "type"}}` with `type` fixed to `"server_error"`.
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
    /// Always `"server_error"` for our API.
    #[serde(rename = "type")]
    pub kind: String,
}

impl ApiError {
    /// Build an error with `type: "server_error"`.
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            error: ApiErrorBody {
                code: code.into(),
                message: message.into(),
                kind: "server_error".to_string(),
            },
        }
    }
}
