//! `POST /api/reservations` — place a manual reservation.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::reservations::list::ManualReservation;

/// Request body for `POST /api/reservations`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct CreateReservationRequest {
    /// Bytes to reserve per device id (`"cpu"` or `"gpu:N"`). At least one
    /// device; each amount is rounded up to a whole MiB.
    pub devices: BTreeMap<String, u64>,
    /// Free-form label naming what the memory is for.
    #[serde(default)]
    pub label: Option<String>,
    /// Whether to answer only once evicted services have stopped and
    /// released their memory, so a job started on the reply finds it free.
    /// Defaults to `true`; with `false` the reply comes straight after the
    /// evictions begin, and the reservation already holds its memory
    /// against services either way.
    #[serde(default)]
    pub wait: Option<bool>,
}

/// Response from `POST /api/reservations`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct CreateReservationResponse {
    /// The reservation as stored.
    pub reservation: ManualReservation,
    /// Services evicted to make room. With `wait`, they have stopped by
    /// the time of the response; without it, they may still be stopping.
    pub evicted: Vec<String>,
}
