//! `/api/services*` endpoint types — list, detail, command, lifecycle
//! (start/stop/restart/enable/disable), and logs.

pub mod command;
pub mod detail;
pub mod disable;
pub mod enable;
pub mod list;
pub mod logs;
pub mod restart;
pub mod start;
pub mod stop;

pub use command::{EnvVar, LaunchCommand, LaunchCommandResponse, LaunchCommandSource};
pub use detail::{DevicePlacement, EstimateSummary, ModelInfo, PlacementPreview, ServiceDetail};
pub use disable::DisableResponse;
pub use enable::EnableResponse;
pub use list::{ServiceSummary, ServicesResponse};
pub use logs::{LogsQuery, LogsResponse};
pub use restart::RestartResponse;
pub use start::StartResponse;
pub use stop::StopResponse;
