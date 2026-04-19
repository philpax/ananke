//! Log line + paginated logs response.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// One captured stdout/stderr line.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct LogLine {
    /// Millisecond UNIX timestamp the line was received.
    pub timestamp_ms: i64,
    /// `"stdout"` or `"stderr"`.
    pub stream: String,
    /// The line content (sans trailing newline).
    pub line: String,
    /// Owning run id.
    pub run_id: i64,
    /// Sequence number within `(service_id, run_id)`, monotonic per run.
    pub seq: i64,
}

/// `GET /api/services/{name}/logs` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct LogsResponse {
    /// Newest-first page of log lines.
    pub logs: Vec<LogLine>,
    /// Opaque cursor for paging further back; `None` when exhausted.
    pub next_cursor: Option<String>,
}

/// Frame sent over `/api/services/{name}/logs/stream`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LogStreamMessage {
    /// One captured log line.
    Line(LogLine),
    /// Subscriber lagged; this many frames were dropped.
    Overflow {
        /// Number of dropped frames.
        dropped: u64,
    },
}
