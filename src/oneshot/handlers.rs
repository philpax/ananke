//! HTTP handlers for the oneshot API: `POST /api/oneshot`, `GET /api/oneshot`,
//! `GET /api/oneshot/:id`, `DELETE /api/oneshot/:id`.

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{Router, delete, get, post},
};
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;
use ulid::Ulid;
use utoipa::ToSchema;

use crate::{
    config::validate::parse_duration_ms,
    daemon::app_state::AppState,
    oneshot::{OneshotId, OneshotRecord},
};

/// Memory allocation strategy for a oneshot request.
#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct OneshotAllocation {
    /// Allocation mode: `"static"`, `"dynamic"`, or absent for none.
    pub mode: Option<String>,
    /// Static mode: VRAM in GiB.
    pub vram_gb: Option<f32>,
    /// Dynamic mode: minimum VRAM in GiB.
    pub min_vram_gb: Option<f32>,
    /// Dynamic mode: maximum VRAM in GiB.
    pub max_vram_gb: Option<f32>,
}

/// Request body for `POST /api/oneshot`.
#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct OneshotRequest {
    /// Optional stable name. Auto-generated from ULID when absent.
    pub name: Option<String>,
    /// Service template: `"llama-cpp"` or `"command"`.
    pub template: String,
    /// Command argv for the `"command"` template.
    pub command: Option<Vec<String>>,
    /// Working directory for the `"command"` template.
    pub workdir: Option<String>,
    /// Memory allocation strategy.
    pub allocation: OneshotAllocation,
    /// Scheduling priority (0–255, higher wins). Defaults to 50.
    pub priority: Option<u8>,
    /// TTL as a human duration string: e.g. `"10m"`, `"30s"`, `"500ms"`, `"2h"`.
    pub ttl: String,
    /// Override public port. Allocated from the pool when absent.
    pub port: Option<u16>,
}

/// Response body for a successfully spawned oneshot.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct OneshotResponse {
    pub id: String,
    pub name: String,
    pub port: u16,
    pub logs_url: String,
}

/// Status of a live or recently completed oneshot.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct OneshotStatus {
    pub id: String,
    pub name: String,
    pub state: String,
    pub started_at_ms: Option<u64>,
    pub ended_at_ms: Option<u64>,
    pub exit_code: Option<i32>,
}

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
    // Parse TTL.
    let ttl_ms = match parse_duration_ms(&req.ttl) {
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

    match state.spawn_oneshot(id.clone(), req, port, ttl_ms).await {
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

    // Signal the supervisor to drain.
    if let Some(handle) = state.registry.get(&record.service_name) {
        handle
            .begin_drain(crate::supervise::drain::DrainReason::UserKilled)
            .await;
    }

    // Return the port to the pool.
    state.port_pool.lock().release(record.port);

    StatusCode::NO_CONTENT.into_response()
}

fn record_to_status(r: &OneshotRecord) -> OneshotStatus {
    OneshotStatus {
        id: r.id.to_string(),
        name: r.service_name.to_string(),
        // State tracking is deferred; all live records are "running".
        state: "running".to_string(),
        started_at_ms: Some(r.started_at_ms),
        ended_at_ms: None,
        exit_code: None,
    }
}
