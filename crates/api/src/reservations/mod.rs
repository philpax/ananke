//! `/api/reservations*` endpoint types — manual and service reservations.
//!
//! A reservation holds memory on one or more devices. Service reservations
//! are the pledges ananke makes for the services it runs; manual
//! reservations are placed by an operator so that memory stays free for
//! work outside ananke. Placement treats a manual reservation as memory
//! already in use: ananke never places a service into it, but processes
//! outside ananke may use it.

pub mod create;
pub mod list;

pub use create::{CreateReservationRequest, CreateReservationResponse};
pub use list::{ManualReservation, ReservationsResponse, ServiceReservation};
