//! One captured stdout/stderr line.
//!
//! `LogLine` is both a wire type (returned by `GET /api/services/:name/logs`
//! and streamed over `WS /api/services/:name/logs/stream`) and a database
//! row type (used by `db/logs.rs`). Because it straddles the HTTP and DB
//! boundaries it lives in the internal module rather than under a single
//! endpoint.

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
