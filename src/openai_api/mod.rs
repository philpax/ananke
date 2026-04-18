//! Unified OpenAI listener — Axum router factory.

pub mod errors;
pub mod filters;
pub mod handlers;
pub mod schema;
pub mod unimplemented;

use axum::Router;

use crate::app_state::AppState;

pub fn router(state: AppState) -> Router {
    handlers::register(Router::new(), state)
}
