//! `GET /api/devices` — device + reservation views.

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
    /// Service holding the reservation. A manual reservation appears as
    /// `reservation:<id>`, with `reservation` set.
    pub service: String,
    /// The manual reservation's id when this row is one rather than a
    /// service's pledge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reservation: Option<i64>,
    /// Bytes reserved.
    pub bytes: u64,
    /// Whether the reservation is elastic (dynamic service borrowing).
    pub elastic: bool,
}
