//! `GET /api/metrics` and `GET /api/devices/samples` response types.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// `GET /api/metrics` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct MetricsResponse {
    /// Time-series buckets, ordered by `bucket_start` ascending.
    pub buckets: Vec<MetricBucketResponse>,
}

/// One time bucket of aggregated request metrics, scoped to a single service.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct MetricBucketResponse {
    /// Service name these metrics belong to. `None` if the service has been
    /// deleted from the database but metric rows remain.
    pub service: Option<String>,
    /// Start of the bucket (ms since epoch).
    pub bucket_start: i64,
    /// Number of requests in the bucket.
    pub request_count: i64,
    /// Total prompt tokens across all requests in the bucket.
    pub prompt_tokens: i64,
    /// Total completion tokens across all requests in the bucket.
    pub completion_tokens: i64,
    /// Average request duration in milliseconds, if any requests had timing data.
    pub avg_duration_ms: Option<f64>,
    /// Number of requests with a 4xx/5xx status code.
    pub error_count: i64,
}

/// `GET /api/devices/samples` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct DeviceSamplesResponse {
    /// Samples ordered by timestamp ascending.
    pub samples: Vec<DeviceSampleResponse>,
}

/// One device memory sample.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct DeviceSampleResponse {
    /// Device id (`"gpu:0"`, `"cpu"`, etc.).
    pub device: String,
    /// Sample timestamp (ms since epoch).
    pub timestamp_ms: i64,
    /// Total capacity in bytes.
    pub total_bytes: i64,
    /// Free bytes at sample time.
    pub free_bytes: i64,
    /// Used bytes at sample time.
    pub used_bytes: i64,
}
