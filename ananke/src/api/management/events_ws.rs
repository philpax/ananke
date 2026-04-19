//! `GET /api/events` WebSocket — system-wide event bus.

use std::time::Duration;

use ananke_api::Event;
use axum::{
    extract::{
        Query, State, WebSocketUpgrade,
        ws::{Message, WebSocket},
    },
    response::Response,
};
use serde::Deserialize;
use tokio::{select, sync::broadcast::error::RecvError};
use tracing::warn;

use crate::daemon::app_state::AppState;

#[derive(Debug, Deserialize)]
pub struct EventsQuery {
    pub service: Option<String>,
}

pub async fn get_events_ws(
    State(state): State<AppState>,
    Query(q): Query<EventsQuery>,
    ws: WebSocketUpgrade,
) -> Response {
    ws.on_upgrade(move |socket| serve_events(socket, state, q.service))
}

async fn serve_events(mut socket: WebSocket, state: AppState, service_filter: Option<String>) {
    let mut rx = state.events.subscribe();
    loop {
        select! {
            recv = rx.recv() => match recv {
                Ok(event) => {
                    if !passes_filter(&event, service_filter.as_deref()) { continue; }
                    let json = match serde_json::to_string(&event) {
                        Ok(s) => s,
                        Err(e) => { warn!(error = %e, "event serialise failed"); continue; }
                    };
                    if socket.send(Message::Text(json)).await.is_err() { return; }
                }
                Err(RecvError::Lagged(n)) => {
                    let overflow = Event::Overflow { dropped: n };
                    if let Ok(s) = serde_json::to_string(&overflow)
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

fn passes_filter(event: &Event, service_filter: Option<&str>) -> bool {
    let Some(filter) = service_filter else {
        return true;
    };
    match event {
        Event::StateChanged { service, .. }
        | Event::AllocationChanged { service, .. }
        | Event::EstimatorDrift { service, .. } => service.as_str() == filter,
        Event::ConfigReloaded { .. } | Event::Overflow { .. } => true,
    }
}
