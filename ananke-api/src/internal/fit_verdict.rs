//! Whether a service's estimated placement fits under current conditions.
//!
//! Used by the supervisor's placement engine (`supervise/preview.rs`) as
//! well as the `/api/services` and `/api/services/:name` endpoints, so it
//! lives in the internal module rather than under a single endpoint.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Whether a service's estimated placement fits under current conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum FitVerdict {
    /// Starts now in currently-free memory — no eviction needed.
    Fits,
    /// Fits within the hardware, but currently-free memory is insufficient, so
    /// the daemon would reclaim or evict lower-priority peers to make room.
    NeedsEviction,
    /// Too large for the allowed GPUs even with everything else gone.
    DoesNotFit,
}
