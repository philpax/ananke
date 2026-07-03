//! `WS /api/services/{name}/logs/stream` — live log tail.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::internal::log_line::LogLine;

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
