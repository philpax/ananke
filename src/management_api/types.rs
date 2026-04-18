//! Response shapes for read-only management endpoints.

use serde::Serialize;
use utoipa::ToSchema;

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ServiceSummary {
    pub name: String,
    pub state: String,
    pub lifecycle: String,
    pub priority: u8,
    pub port: u16,
    pub run_id: Option<i64>,
    pub pid: Option<i32>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ServiceDetail {
    pub name: String,
    pub state: String,
    pub lifecycle: String,
    pub priority: u8,
    pub port: u16,
    pub private_port: u16,
    pub template: String,
    pub placement_override: std::collections::BTreeMap<String, u64>,
    pub idle_timeout_ms: u64,
    pub run_id: Option<i64>,
    pub pid: Option<i32>,
    pub recent_logs: Vec<LogLine>,
    /// Rolling correction mean from the last N completed runs. `1.0` means
    /// the estimator is on target; values above `1.0` indicate under-estimation.
    pub rolling_mean: f64,
    /// Number of completed runs used to compute `rolling_mean`.
    pub rolling_samples: u32,
    /// Observed peak memory (GPU VRAM + CPU RSS) in bytes for the current or
    /// most recent run. `0` if no observation has been recorded yet.
    pub observed_peak_bytes: u64,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct LogLine {
    pub timestamp_ms: i64,
    pub stream: String,
    pub line: String,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct DeviceSummary {
    pub id: String,
    pub name: String,
    pub total_bytes: u64,
    pub free_bytes: u64,
    pub reservations: Vec<DeviceReservation>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct DeviceReservation {
    pub service: String,
    pub bytes: u64,
}
