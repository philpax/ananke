//! Device + reservation views.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// One entry in `GET /api/devices`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct DeviceSummary {
    /// Device id string, e.g. `"gpu:0"` or `"cpu"`.
    pub id: String,
    /// Human-readable device name.
    pub name: String,
    /// Total byte capacity.
    pub total_bytes: u64,
    /// Currently free bytes.
    pub free_bytes: u64,
    /// Per-service reservation breakdown.
    pub reservations: Vec<DeviceReservation>,
}

/// One reservation held against a device.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct DeviceReservation {
    /// Service holding the reservation.
    pub service: String,
    /// Bytes reserved.
    pub bytes: u64,
    /// Whether the reservation is elastic (dynamic service borrowing).
    pub elastic: bool,
}
