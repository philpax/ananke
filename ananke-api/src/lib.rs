//! Shared HTTP DTOs between the `ananke` daemon and the `anankectl` CLI.

#![deny(missing_docs)]

pub mod config;
pub mod defaults;
pub mod devices;
pub mod errors;
pub mod events;
pub mod lifecycle;
pub mod logs;
pub mod metadata;
pub mod metrics;
pub mod oneshot;
pub mod services;

pub use config::{ConfigResponse, ConfigValidateRequest, ConfigValidateResponse, ValidationError};
pub use devices::{DeviceReservation, DeviceSummary};
pub use errors::{ApiError, ApiErrorBody};
pub use events::Event;
pub use lifecycle::{DisableResponse, EnableResponse, StartResponse, StopResponse};
pub use logs::{LogLine, LogStreamMessage, LogsResponse};
pub use metadata::AnankeMetadata;
pub use metrics::{
    DeviceSampleResponse, DeviceSamplesResponse, MetricBucketResponse, MetricsResponse,
};
pub use oneshot::{OneshotHealth, OneshotRequest, OneshotResponse, OneshotStatus};
pub use services::{
    DevicePlacement, EnvVar, EstimateSummary, FitVerdict, LaunchCommand, LaunchCommandSource,
    Modality, ModelInfo, PlacementPreview, ServiceDetail, ServiceSummary, ServicesResponse,
};
