//! Per-service runtime tracking: activity timestamps, in-flight counters,
//! live VRAM observations, and rolling safety factors.

pub mod activity;
pub mod inflight;
pub mod observation;
pub mod rolling;

use std::time::{SystemTime, UNIX_EPOCH};

/// Wall-clock milliseconds since the Unix epoch.
///
/// Returned as `i64` because that is the width we use for log timestamps,
/// DB rows (`service_logs.timestamp_ms`), and `ServiceConfig` durations.
pub fn now_unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

/// Unsigned variant of [`now_unix_ms`] for APIs that carry millis as `u64`.
pub fn now_unix_ms_u64() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
