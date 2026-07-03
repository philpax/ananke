//! `/api/devices*` endpoint types — device list and samples.

pub mod list;
pub mod samples;

pub use list::{DeviceReservation, DeviceSummary};
pub use samples::{DeviceSampleResponse, DeviceSamplesResponse};
