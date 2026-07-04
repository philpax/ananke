//! `GET /api/metrics` response types.

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
    /// Average time-to-first-token in milliseconds (streaming requests only).
    pub avg_ttft_ms: Option<f64>,
    /// Output tokens per second during decode: completion tokens divided by
    /// total decode time. `None` if no timed requests in the bucket.
    pub output_tps: Option<f64>,
    /// Input tokens per second during prompt processing: prompt tokens
    /// divided by total TTFT. `None` if no timed requests in the bucket.
    pub input_tps: Option<f64>,
    /// End-to-end aggregate throughput: total tokens (prompt + completion)
    /// divided by total wall-clock duration. Available whenever the bucket
    /// has any request with a recorded duration, including non-streaming
    /// requests with no engine timings where no input/output split exists.
    /// This is end-to-end throughput, *not* a decode rate; the UI surfaces
    /// it as a fallback when the split figures are unavailable.
    pub aggregate_tps: Option<f64>,
}
