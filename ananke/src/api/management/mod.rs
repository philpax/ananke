//! Read-only management API — `/api/services`, `/api/devices`, and
//! `/api/openapi.json`.

pub mod config;
pub mod events_ws;
pub mod handlers;
pub mod lifecycle;
pub mod logs;
pub mod logs_ws;

use axum::{
    Router,
    routing::{any, get, post},
};

use crate::daemon::app_state::AppState;

pub fn register(router: Router, state: AppState) -> Router {
    let mgmt: Router = Router::new()
        .route(
            "/api/config",
            get(config::get_config).put(config::put_config),
        )
        .route("/api/config/validate", post(config::post_validate))
        .route("/api/services/:name/start", post(lifecycle::post_start))
        .route("/api/services/:name/stop", post(lifecycle::post_stop))
        .route("/api/services/:name/restart", post(lifecycle::post_restart))
        .route("/api/services/:name/enable", post(lifecycle::post_enable))
        .route("/api/services/:name/disable", post(lifecycle::post_disable))
        .route("/api/services/:name/logs", get(logs::get_logs))
        .route("/api/services/:name/logs/stream", any(logs_ws::get_logs_ws))
        .route("/api/events", any(events_ws::get_events_ws))
        .with_state(state.clone());
    handlers::register(router, state.clone())
        .merge(crate::api::openapi::register(Router::new(), state.clone()))
        .merge(crate::oneshot::handlers::register(Router::new(), state))
        .merge(mgmt)
}

pub fn router(state: AppState) -> Router {
    // Mount the embedded frontend as the last-resort fallback so `/api/*`,
    // `/v1/*`, and the WebSocket routes keep first dibs on every request.
    crate::api::frontend::register(register(Router::new(), state))
}
