//! HTTP handlers for the oneshot API: `POST /api/oneshot`, `GET /api/oneshot`,
//! `GET /api/oneshot/:id`, `DELETE /api/oneshot/:id`.

use ananke_api::oneshot::{OneshotRequest, OneshotResponse, OneshotStatus};
use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{Router, delete, get, post},
};
use smol_str::SmolStr;
use ulid::Ulid;

use crate::{
    config::validate::parse_duration_ms,
    daemon::app_state::AppState,
    oneshot::{OneshotId, OneshotRecord},
};

pub fn register(router: Router, state: AppState) -> Router {
    let oneshot_router: Router = Router::new()
        .route("/api/oneshot", post(post_oneshot))
        .route("/api/oneshot", get(list_oneshots))
        .route("/api/oneshot/:id", get(get_oneshot))
        .route("/api/oneshot/:id", delete(delete_oneshot))
        .with_state(state);
    router.merge(oneshot_router)
}

#[utoipa::path(
    post,
    path = "/api/oneshot",
    request_body = OneshotRequest,
    responses((status = 200, body = OneshotResponse), (status = 400), (status = 503))
)]
pub async fn post_oneshot(
    State(state): State<AppState>,
    Json(req): Json<OneshotRequest>,
) -> Response {
    // Parse TTL; treat absent TTL as an error — callers must set a finite lifetime.
    let ttl_str = match &req.ttl {
        Some(s) => s.as_str(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "ttl is required"})),
            )
                .into_response();
        }
    };
    let ttl_ms = match parse_duration_ms(ttl_str) {
        Ok(ms) => ms,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("invalid ttl: {e}")})),
            )
                .into_response();
        }
    };

    // Generate a stable id.
    let id: OneshotId = match &req.name {
        Some(n) => SmolStr::new(n),
        None => SmolStr::new(format!("oneshot_{}", Ulid::new())),
    };

    // Allocate a port.
    let port = match state.port_pool.lock().allocate() {
        Some(p) => p,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "port pool exhausted"})),
            )
                .into_response();
        }
    };

    let name = id.to_string();

    match crate::oneshot::spawn::spawn_oneshot(&state, id.clone(), req, port, ttl_ms).await {
        Ok(()) => {
            let resp = OneshotResponse {
                logs_url: format!("/api/services/{}/logs", id),
                id: id.to_string(),
                name,
                port,
            };
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => {
            // Release the port back to the pool on spawn failure.
            state.port_pool.lock().release(port);
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": e})),
            )
                .into_response()
        }
    }
}

#[utoipa::path(
    get,
    path = "/api/oneshot",
    responses((status = 200, body = Vec<OneshotStatus>))
)]
pub async fn list_oneshots(State(state): State<AppState>) -> Response {
    let records = state.oneshots.list();
    let out: Vec<OneshotStatus> = records.iter().map(record_to_status).collect();
    (StatusCode::OK, Json(out)).into_response()
}

#[utoipa::path(
    get,
    path = "/api/oneshot/{id}",
    responses((status = 200, body = OneshotStatus), (status = 404))
)]
pub async fn get_oneshot(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    match state.oneshots.get(&id) {
        Some(r) => (StatusCode::OK, Json(record_to_status(&r))).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "not found"})),
        )
            .into_response(),
    }
}

#[utoipa::path(
    delete,
    path = "/api/oneshot/{id}",
    responses((status = 204), (status = 404))
)]
pub async fn delete_oneshot(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let Some(record) = state.oneshots.remove(&id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "not found"})),
        )
            .into_response();
    };

    // If the TTL watcher has already drained and released the port, skip both
    // steps — the record was kept as a tombstone purely so callers could poll
    // the terminal state. Removing it here is the only side effect.
    if record.ended_at_ms.is_none() {
        if let Some(handle) = state.registry.get(&record.service_name) {
            handle
                .begin_drain(crate::supervise::drain::DrainReason::UserKilled)
                .await;
        }
        state.port_pool.lock().release(record.port);
    }

    StatusCode::NO_CONTENT.into_response()
}

fn record_to_status(r: &OneshotRecord) -> OneshotStatus {
    let state = if r.ended_at_ms.is_some() {
        "ended"
    } else {
        "running"
    };
    OneshotStatus {
        id: r.id.to_string(),
        name: r.service_name.to_string(),
        state: state.to_string(),
        port: r.port,
        submitted_at_ms: r.started_at_ms as i64,
        started_at_ms: Some(r.started_at_ms as i64),
        ended_at_ms: r.ended_at_ms.map(|v| v as i64),
        exit_code: r.exit_code,
        logs_url: format!("/api/services/{}/logs", r.id),
    }
}
