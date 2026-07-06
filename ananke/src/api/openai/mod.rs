//! Unified OpenAI listener — Axum router factory.

pub mod errors;
pub mod filters;
pub mod handlers;
pub mod metrics;
pub mod schema;
pub mod stall;
pub mod unimplemented;

use axum::Router;

use crate::daemon::app_state::AppState;

pub fn router(state: AppState) -> Router {
    handlers::register(Router::new(), state)
}
