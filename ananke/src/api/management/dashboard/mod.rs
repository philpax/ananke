//! One WebSocket carries dashboard reads, producer-driven subscriptions, mutations, and chat.

mod query;
mod session;

use axum::{
    extract::{State, WebSocketUpgrade},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};

use crate::daemon::app_state::AppState;

#[utoipa::path(get, path = "/api/dashboard", responses((status = 101, description = "Multiplexed dashboard WebSocket. See DashboardRequest and DashboardMessage."), (status = 403, description = "The browser Origin does not match the Host.")))]
pub async fn upgrade(
    State(state): State<AppState>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    // Cookie-authenticated dashboards must not accept commands from another site's socket.
    if let Some(origin) = headers.get("origin") {
        let allowed = origin
            .to_str()
            .ok()
            .and_then(|origin| reqwest::Url::parse(origin).ok())
            .is_some_and(|origin| {
                matches!(origin.scheme(), "http" | "https")
                    && headers
                        .get("host")
                        .and_then(|host| host.to_str().ok())
                        .and_then(|host| {
                            reqwest::Url::parse(&format!("{}://{host}", origin.scheme())).ok()
                        })
                        .is_some_and(|host| {
                            host.host() == origin.host()
                                && host.port_or_known_default() == origin.port_or_known_default()
                        })
            });
        if !allowed {
            return (
                StatusCode::FORBIDDEN,
                "dashboard: WebSocket origin does not match the host",
            )
                .into_response();
        }
    }
    ws.max_message_size(2 * 1024 * 1024)
        .on_upgrade(move |socket| session::serve(socket, state))
}
