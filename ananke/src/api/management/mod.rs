//! Read-only management API — `/api/services`, `/api/devices`, `/api/metrics`,
//! and `/api/openapi.json`.

pub mod config;
pub mod events_history;
pub mod handlers;
pub mod info;
pub mod lifecycle;
pub mod live;
pub mod logs;
pub mod metrics;
mod origin;
pub mod reservations;

use axum::{
    Router,
    routing::{get, post},
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
        .route(
            "/api/events/history",
            get(events_history::get_events_history),
        )
        .route("/api/live", get(live::get_live_ws))
        .route("/api/info", get(info::get_info))
        .route(
            "/api/reservations",
            get(reservations::get_reservations).post(reservations::post_reservation),
        )
        .route(
            "/api/reservations/:id",
            axum::routing::delete(reservations::delete_reservation),
        )
        .with_state(state.clone());
    handlers::register(router, state.clone())
        .merge(metrics::register(Router::new(), state.clone()))
        .merge(crate::api::openapi::register(Router::new(), state.clone()))
        .merge(crate::api::prometheus::register(Router::new(), state))
        .merge(mgmt)
}

pub fn router(state: AppState) -> Router {
    // Mount the embedded frontend as the last-resort fallback so `/api/*`,
    // `/v1/*`, and the WebSocket routes keep first dibs on every request.
    crate::api::frontend::register(register(Router::new(), state))
}
