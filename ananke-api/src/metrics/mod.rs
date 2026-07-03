//! `/api/metrics` and `/metrics` endpoint types.

pub mod get;
pub mod prometheus;

pub use get::{MetricBucketResponse, MetricsResponse};
