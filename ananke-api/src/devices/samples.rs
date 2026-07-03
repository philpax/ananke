//! `GET /api/devices/samples` — device memory samples.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// `GET /api/devices/samples` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct DeviceSamplesResponse {
    /// Samples ordered by timestamp ascending.
    pub samples: Vec<DeviceSampleResponse>,
}

/// One device memory sample.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct DeviceSampleResponse {
    /// Device id (`"gpu:0"`, `"cpu"`, etc.).
    pub device: String,
    /// Sample timestamp (ms since epoch).
    pub timestamp_ms: i64,
    /// Total capacity in bytes.
    pub total_bytes: i64,
    /// Free bytes at sample time.
    pub free_bytes: i64,
    /// Used bytes at sample time.
    pub used_bytes: i64,
}
