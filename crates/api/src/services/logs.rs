//! `GET /api/services/{name}/logs` — paginated log retrieval.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::internal::log_line::{LogLine, LogPosition};

/// `GET /api/services/{name}/logs` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct LogsResponse {
    /// Newest-first page of log lines.
    pub logs: Vec<LogLine>,
    /// Opaque cursor for paging further back; `None` when exhausted.
    pub next_cursor: Option<String>,
    /// The newest position in the service's whole log when the page was read,
    /// ignoring `run`, `stream`, and `since`: follow from here to continue the
    /// page without a gap. `None` when the request has `before` or `until`,
    /// since rows newer than the page would be skipped, or when the service
    /// has no lines.
    #[serde(default)]
    pub follow_after: Option<LogPosition>,
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
    /// `"stdout"`, `"stderr"`, or `"combined"` (container merged output).
    pub stream: Option<String>,
    /// Max rows to return (≤1000, default 200).
    pub limit: Option<u32>,
    /// Opaque cursor from a prior response.
    pub before: Option<String>,
}
