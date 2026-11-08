//! `GET /api/events/history` response type.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::internal::event::Event;

/// The daemon's bounded window of recent events, oldest first.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct EventHistory {
    /// The start of the window, in unix milliseconds: the daemon's start time
    /// until the window first evicts, then the oldest retained event's
    /// timestamp. Earlier events may have happened but are not listed.
    pub since_ms: i64,
    /// The retained events, oldest first.
    pub events: Vec<Event>,
}
