//! `GET /api/info` handler.

use ananke_api::info::get::DaemonInfoResponse;
use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response},
};

use crate::daemon::app_state::AppState;

#[utoipa::path(
    summary = "Get daemon listen addresses and uptime",
    get,
    path = "/api/info",
    responses((status = 200, body = DaemonInfoResponse))
)]
pub async fn get_info(State(state): State<AppState>) -> Response {
    Json(info_snapshot(&state)).into_response()
}

/// The `GET /api/info` body; the live `info` resource shares it.
///
/// The uptime makes every body differ, so the live resource resends it on
/// each config reload even when the addresses are unchanged. That is rare
/// enough not to matter, and a client gets a fresh uptime from it.
pub(crate) fn info_snapshot(state: &AppState) -> DaemonInfoResponse {
    let cfg = state.config.effective();
    let uptime_ms = u64::try_from(state.started_at.elapsed().as_millis()).unwrap_or(u64::MAX);
    DaemonInfoResponse {
        openai_listen: cfg.daemon.openai_listen.clone(),
        management_listen: cfg.daemon.management_listen.clone(),
        uptime_ms,
    }
}
