//! `GET /api/info` handler.

use ananke_api::info::get::DaemonInfoResponse;
use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response},
};

use crate::daemon::app_state::AppState;

#[utoipa::path(
    summary = "Get daemon listen addresses",
    get,
    path = "/api/info",
    responses((status = 200, body = DaemonInfoResponse))
)]
pub async fn get_info(State(state): State<AppState>) -> Response {
    let cfg = state.config.effective();
    Json(DaemonInfoResponse {
        openai_listen: cfg.daemon.openai_listen.clone(),
        management_listen: cfg.daemon.management_listen.clone(),
    })
    .into_response()
}
