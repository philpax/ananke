//! `GET /api/events/history` handler.

use ananke_api::events::EventHistory;
use axum::{Json, extract::State};

use crate::daemon::app_state::AppState;

#[utoipa::path(
    summary = "List recent events",
    get,
    path = "/api/events/history",
    responses((status = 200, body = EventHistory))
)]
pub async fn get_events_history(State(state): State<AppState>) -> Json<EventHistory> {
    Json(events_history(&state))
}

pub(crate) fn events_history(state: &AppState) -> EventHistory {
    state.events.history()
}
