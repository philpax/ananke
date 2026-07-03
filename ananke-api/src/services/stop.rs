//! `POST /api/services/{name}/stop` response body.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// `POST /api/services/{name}/stop` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
#[allow(missing_docs)]
pub enum StopResponse {
    NotRunning,
    Drained,
}
