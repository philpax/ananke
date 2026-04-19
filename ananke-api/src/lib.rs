//! Shared HTTP DTOs between the `ananke` daemon and the `anankectl` CLI.

#![deny(missing_docs)]

pub mod config;
pub mod devices;
pub mod errors;
pub mod events;
pub mod lifecycle;
pub mod logs;
pub mod oneshot;
pub mod services;

pub use config::{ConfigResponse, ConfigValidateRequest, ConfigValidateResponse, ValidationError};
pub use devices::{DeviceReservation, DeviceSummary};
pub use errors::ApiError;
pub use events::Event;
pub use lifecycle::{DisableResponse, EnableResponse, StartResponse, StopResponse};
pub use logs::{LogLine, LogsResponse};
pub use oneshot::{OneshotRequest, OneshotResponse, OneshotStatus};
pub use services::{ServiceDetail, ServiceSummary};
