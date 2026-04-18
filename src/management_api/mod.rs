//! Read-only management API — `/api/services`, `/api/devices`, and
//! `/api/openapi.json`.

pub mod handlers;
pub mod types;

use axum::Router;

use crate::app_state::AppState;

pub fn router(state: AppState) -> Router {
    handlers::register(Router::new(), state.clone())
        .merge(crate::openapi::register(Router::new(), state))
}
