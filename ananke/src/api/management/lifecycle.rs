//! POST /api/services/{name}/{start,stop,restart,enable,disable} handlers.

use std::time::Duration;

use ananke_api::{ApiError, DisableResponse, EnableResponse, StartResponse, StopResponse};
use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use tracing::{info, warn};

use crate::{
    daemon::app_state::AppState,
    supervise::{
        DisableResult, EnableResult, EnsureFailure, EnsureOutcome, SupervisorHandle, await_ensure,
        drain::DrainReason,
    },
};

pub async fn post_start(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    info!(service = %name, "management start requested");
    let Some((svc, handle)) = resolve(&state, &name) else {
        warn!(service = %name, "management start rejected: service not found");
        return not_found(&name);
    };
    let duration = Duration::from_millis(svc.max_request_duration_ms);
    let body = match await_ensure(&handle, duration).await {
        EnsureOutcome::Ready {
            was_already_running: true,
        } => StartResponse::AlreadyRunning,
        EnsureOutcome::Ready {
            was_already_running: false,
        } => {
            let run_id = handle.peek().run_id.unwrap_or(0);
            StartResponse::Started { run_id }
        }
        EnsureOutcome::Failed(EnsureFailure::StartQueueFull) => {
            warn!(service = %name, "management start rejected: start_queue_full");
            StartResponse::QueueFull
        }
        EnsureOutcome::Failed(EnsureFailure::InsufficientVram(reason)) => {
            warn!(service = %name, %reason, "management start rejected: insufficient_vram");
            StartResponse::Unavailable { reason }
        }
        EnsureOutcome::Failed(EnsureFailure::ServiceDisabled(reason)) => {
            warn!(service = %name, %reason, "management start rejected: service_disabled");
            StartResponse::Unavailable { reason }
        }
        EnsureOutcome::Failed(EnsureFailure::StartFailed(reason)) => {
            warn!(service = %name, %reason, "management start failed");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ApiError::new("start_failed", reason)),
            )
                .into_response();
        }
    };
    (StatusCode::ACCEPTED, Json(body)).into_response()
}

pub async fn post_stop(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    info!(service = %name, "management stop requested");
    let Some((_svc, handle)) = resolve(&state, &name) else {
        warn!(service = %name, "management stop rejected: service not found");
        return not_found(&name);
    };
    handle.begin_drain(DrainReason::UserKilled).await;
    (StatusCode::ACCEPTED, Json(StopResponse::Drained)).into_response()
}

pub async fn post_restart(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    info!(service = %name, "management restart requested");
    let Some((svc, handle)) = resolve(&state, &name) else {
        warn!(service = %name, "management restart rejected: service not found");
        return not_found(&name);
    };
    handle.begin_drain(DrainReason::UserKilled).await;
    let duration = Duration::from_millis(svc.max_request_duration_ms);
    let body = match await_ensure(&handle, duration).await {
        EnsureOutcome::Ready {
            was_already_running: true,
        } => StartResponse::AlreadyRunning,
        EnsureOutcome::Ready {
            was_already_running: false,
        } => {
            let run_id = handle.peek().run_id.unwrap_or(0);
            StartResponse::Started { run_id }
        }
        EnsureOutcome::Failed(EnsureFailure::StartQueueFull) => {
            warn!(service = %name, "management restart rejected: start_queue_full");
            StartResponse::QueueFull
        }
        EnsureOutcome::Failed(EnsureFailure::InsufficientVram(reason)) => {
            warn!(service = %name, %reason, "management restart rejected: insufficient_vram");
            StartResponse::Unavailable { reason }
        }
        EnsureOutcome::Failed(EnsureFailure::ServiceDisabled(reason)) => {
            warn!(service = %name, %reason, "management restart rejected: service_disabled");
            StartResponse::Unavailable { reason }
        }
        EnsureOutcome::Failed(EnsureFailure::StartFailed(reason)) => {
            warn!(service = %name, %reason, "management restart failed");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ApiError::new("start_failed", reason)),
            )
                .into_response();
        }
    };
    (StatusCode::ACCEPTED, Json(body)).into_response()
}

pub async fn post_enable(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    info!(service = %name, "management enable requested");
    let Some((_svc, handle)) = resolve(&state, &name) else {
        warn!(service = %name, "management enable rejected: service not found");
        return not_found(&name);
    };
    let body = match handle.enable().await {
        EnableResult::Enabled => EnableResponse::Enabled,
        EnableResult::NotDisabled => EnableResponse::AlreadyEnabled,
    };
    (StatusCode::OK, Json(body)).into_response()
}

pub async fn post_disable(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    info!(service = %name, "management disable requested");
    let Some((_svc, handle)) = resolve(&state, &name) else {
        warn!(service = %name, "management disable rejected: service not found");
        return not_found(&name);
    };
    let body = match handle.disable().await {
        DisableResult::Disabled => DisableResponse::Disabled,
        DisableResult::AlreadyDisabled => DisableResponse::AlreadyDisabled,
    };
    (StatusCode::OK, Json(body)).into_response()
}

fn resolve(
    state: &AppState,
    name: &str,
) -> Option<(
    crate::config::validate::ServiceConfig,
    std::sync::Arc<SupervisorHandle>,
)> {
    let svc = state
        .config
        .effective()
        .services
        .iter()
        .find(|s| s.name == name)
        .cloned()?;
    let handle = state.registry.get(name)?;
    Some((svc, handle))
}

fn not_found(name: &str) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(ApiError::new(
            "service_not_found",
            format!("service `{name}` not found"),
        )),
    )
        .into_response()
}
