//! One captured stdout/stderr/combined line.
//!
//! `LogLine` is both a wire type (returned by `GET /api/services/:name/logs`
//! and followed over `/api/live`) and a database
//! row type (used by `db/logs.rs`). Because it straddles the HTTP and DB
//! boundaries it lives in the internal module rather than under a single
//! endpoint.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// One captured stdout/stderr/combined line.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct LogLine {
    /// Millisecond UNIX timestamp the line was received.
    pub timestamp_ms: i64,
    /// `"stdout"`, `"stderr"`, or `"combined"` (container merged output).
    pub stream: String,
    /// The line content (sans trailing newline).
    pub line: String,
    /// Owning run id.
    pub run_id: i64,
    /// Sequence number within `(service_id, run_id)`, monotonic per run.
    pub seq: i64,
}

impl LogLine {
    /// The line's place in its service's log.
    pub fn position(&self) -> LogPosition {
        LogPosition {
            run_id: self.run_id,
            seq: self.seq,
        }
    }
}

/// Where a line sits in its service's log. Positions order by `(run_id, seq)`:
/// a later run always has a larger id, and `seq` counts up within a run.
// Field order is the derived `Ord`; keep `run_id` first.
#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, ToSchema,
)]
pub struct LogPosition {
    /// The run the line belongs to.
    pub run_id: i64,
    /// The line's sequence number within the run.
    pub seq: i64,
}

/// Which output stream a line came from, as a filter.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum LogStream {
    /// The child's standard output.
    Stdout,
    /// The child's standard error.
    Stderr,
    /// Merged container output; see [`LogLine::stream`].
    Combined,
}

impl LogStream {
    /// The spelling a [`LogLine::stream`] carries.
    pub fn as_str(self) -> &'static str {
        match self {
            LogStream::Stdout => "stdout",
            LogStream::Stderr => "stderr",
            LogStream::Combined => "combined",
        }
    }
}
