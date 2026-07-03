//! `GET /api/oneshot/{id}` — oneshot status.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// `GET /api/oneshot/{id}` response body. Also used by `GET /api/oneshot`
/// (which returns `Vec<OneshotStatus>`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct OneshotStatus {
    /// ULID id.
    pub id: String,
    /// Name.
    pub name: String,
    /// Current state (`"pending"`, `"running"`, `"ended"`, `"evicted"`, etc.).
    pub state: String,
    /// Port.
    pub port: u16,
    /// Submission timestamp.
    pub submitted_at_ms: i64,
    /// Start timestamp, if reached Running.
    pub started_at_ms: Option<i64>,
    /// End timestamp, if terminal.
    pub ended_at_ms: Option<i64>,
    /// Exit code when the child terminated with one.
    pub exit_code: Option<i32>,
    /// Path component for the WS log-stream endpoint.
    pub logs_url: String,
}
