//! `GET /api/info` response type.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// `GET /api/info` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct DaemonInfoResponse {
    /// The OpenAI-compatible API listen address (e.g. `"0.0.0.0:7070"`).
    pub openai_listen: String,
    /// The management API listen address (e.g. `"0.0.0.0:7071"`).
    pub management_listen: String,
}
