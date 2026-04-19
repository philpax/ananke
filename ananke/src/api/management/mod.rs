//! Read-only management API — `/api/services`, `/api/devices`, and
//! `/api/openapi.json`.

pub mod handlers;
pub mod lifecycle;
pub mod logs;

use axum::{
    Router,
    routing::{get, post},
};

use crate::daemon::app_state::AppState;

pub fn router(state: AppState) -> Router {
    let mgmt: Router = Router::new()
        .route("/api/services/:name/start", post(lifecycle::post_start))
        .route("/api/services/:name/stop", post(lifecycle::post_stop))
        .route("/api/services/:name/restart", post(lifecycle::post_restart))
        .route("/api/services/:name/enable", post(lifecycle::post_enable))
        .route("/api/services/:name/disable", post(lifecycle::post_disable))
        .route("/api/services/:name/logs", get(logs::get_logs))
        .with_state(state.clone());
    handlers::register(Router::new(), state.clone())
        .merge(crate::api::openapi::register(Router::new(), state.clone()))
        .merge(crate::oneshot::handlers::register(Router::new(), state))
        .merge(mgmt)
}
