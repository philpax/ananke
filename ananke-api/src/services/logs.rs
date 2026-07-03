//! `GET /api/services/{name}/logs` — paginated log retrieval.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::internal::log_line::LogLine;

/// `GET /api/services/{name}/logs` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct LogsResponse {
    /// Newest-first page of log lines.
    pub logs: Vec<LogLine>,
    /// Opaque cursor for paging further back; `None` when exhausted.
    pub next_cursor: Option<String>,
}

/// Query parameters for `GET /api/services/{name}/logs`.
#[derive(Debug, Deserialize)]
pub struct LogsQuery {
    /// Earliest timestamp_ms, inclusive.
    pub since: Option<i64>,
    /// Latest timestamp_ms, inclusive.
    pub until: Option<i64>,
    /// Restrict to one run_id.
    pub run: Option<i64>,
    /// `"stdout"` or `"stderr"`.
    pub stream: Option<String>,
    /// Max rows to return (≤1000, default 200).
    pub limit: Option<u32>,
    /// Opaque cursor from a prior response.
    pub before: Option<String>,
}
