//! `GET /api/services/{name}/logs/stream` WebSocket — live log tail for a single service.

use std::time::Duration;

use ananke_api::{ApiError, LogLine, LogStreamMessage};
use axum::{
    Json,
    extract::{Path, State, WebSocketUpgrade, ws::Message},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use tokio::{select, sync::broadcast::error::RecvError};
use tracing::warn;

use crate::daemon::app_state::AppState;

pub async fn get_logs_ws(
    State(state): State<AppState>,
    Path(name): Path<String>,
    ws: WebSocketUpgrade,
) -> Response {
    let service_id = match resolve_service_id(&state, &name).await {
        Some(id) => id,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(ApiError::new(
                    "service_not_found",
                    format!("service `{name}` not found"),
                )),
            )
                .into_response();
        }
    };

    let rx = state.batcher.subscribe();
    ws.on_upgrade(move |socket| serve_logs(socket, service_id, rx))
}

async fn serve_logs(
    mut socket: axum::extract::ws::WebSocket,
    service_id: i64,
    mut rx: tokio::sync::broadcast::Receiver<(i64, LogLine)>,
) {
    loop {
        select! {
            recv = rx.recv() => match recv {
                Ok((sid, line)) if sid == service_id => {
                    let msg = LogStreamMessage::Line(line);
                    let json = match serde_json::to_string(&msg) {
                        Ok(s) => s,
                        Err(e) => { warn!(error = %e, "log line serialise failed"); continue; }
                    };
                    if socket.send(Message::Text(json)).await.is_err() { return; }
                }
                Ok(_) => continue,
                Err(RecvError::Lagged(n)) => {
                    let msg = LogStreamMessage::Overflow { dropped: n };
                    if let Ok(s) = serde_json::to_string(&msg)
                        && socket.send(Message::Text(s)).await.is_err()
                    {
                        return;
                    }
                }
                Err(RecvError::Closed) => return,
            },
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Close(_))) | None => return,
                    Some(Ok(Message::Ping(p))) => {
                        let _ = socket.send(Message::Pong(p)).await;
                    }
                    _ => {}
                }
            }
            _ = tokio::time::sleep(Duration::from_secs(30)) => {
                // Heartbeat; keeps intermediaries from closing an idle WS.
                if socket.send(Message::Ping(vec![])).await.is_err() { return; }
            }
        }
    }
}

async fn resolve_service_id(state: &AppState, name: &str) -> Option<i64> {
    use crate::db::models::Service;
    let mut handle = state.db.handle();
    Service::filter_by_name(name.to_string())
        .first()
        .exec(&mut handle)
        .await
        .ok()
        .flatten()
        .map(|s| s.service_id as i64)
}
