//! `GET /api/live` WebSocket — multiplexed, read-only subscriptions to the
//! management API's live state.
//!
//! Each subscription gets a full snapshot of its resource, then another
//! whenever something the resource reads changes and the body differs from
//! the last one sent. The bodies are the HTTP `GET` DTOs, computed by the
//! same functions. A follow instead streams an append-only feed, such as a
//! service's log, from a client-held position; see `logs`. See `protocol`
//! for the frame handling, `resource` for what wakes each resource, and
//! `session` for the connection shell.

mod logs;
mod protocol;
mod resource;
mod session;
mod subscription;
mod triggers;

use axum::{
    extract::{State, WebSocketUpgrade},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};

use crate::{api::management::origin::same_origin, daemon::app_state::AppState};

/// Largest inbound frame accepted. Requests are tiny; this only bounds
/// what a misbehaving client can make the daemon buffer.
const MAX_INBOUND_BYTES: usize = 64 * 1024;

pub async fn get_live_ws(
    State(state): State<AppState>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    if !same_origin(&headers) {
        return (
            StatusCode::FORBIDDEN,
            "live: the WebSocket origin does not match the host",
        )
            .into_response();
    }
    ws.max_message_size(MAX_INBOUND_BYTES)
        .max_frame_size(MAX_INBOUND_BYTES)
        .on_upgrade(move |socket| session::serve(socket, state))
}
