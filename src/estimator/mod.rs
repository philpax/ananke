//! VRAM estimator.

pub mod fallback;
pub mod hybrid;
pub mod kv;
pub mod llama;
pub mod mamba;
pub mod moe;
pub mod types;

pub use types::{Estimate, NonLayer};

use crate::config::ServiceConfig;
use crate::gguf::GgufSummary;

/// Dispatch on `general.architecture`. Real dispatch is filled out in
/// Task 8; for now route everything to fallback.
pub fn estimate(summary: &GgufSummary, svc: &ServiceConfig) -> Estimate {
    let context = svc.raw.context.unwrap_or(4096);
    fallback::estimate_fallback(summary, context)
}
