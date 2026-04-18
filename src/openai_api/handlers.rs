//! Handlers for /v1/models and the three POST body-rewriting endpoints.

use axum::Router;

use crate::app_state::AppState;

pub fn register(router: Router, _state: AppState) -> Router {
    router
}
