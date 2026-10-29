//! `GET /api/reservations` — every reservation, manual and service.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::shared::service_state::ServiceStateKind;

/// Response from `GET /api/reservations`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct ReservationsResponse {
    /// Operator-placed reservations, oldest first.
    pub manual: Vec<ManualReservation>,
    /// Pledges held by services ananke runs, by service name.
    pub services: Vec<ServiceReservation>,
}

/// A reservation an operator placed through the API or CLI.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct ManualReservation {
    /// The reservation's id, used to remove it.
    pub id: i64,
    /// Free-form label naming what the memory is for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// When it was placed, in unix milliseconds.
    pub created_at_ms: i64,
    /// Reserved bytes per device id (`"cpu"` or `"gpu:N"`).
    pub devices: BTreeMap<String, u64>,
}

/// The memory a service ananke runs has pledged.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct ServiceReservation {
    /// The service's name.
    pub service: String,
    /// The service's current state.
    pub state: ServiceStateKind,
    /// Requests in flight through the service's proxy.
    pub inflight_count: u64,
    /// Pledged bytes per device id (`"cpu"` or `"gpu:N"`).
    pub devices: BTreeMap<String, u64>,
}
