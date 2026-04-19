//! POST /api/services/{name}/{start,stop,restart,enable,disable} handlers.

use std::time::Duration;

use ananke_api::{ApiError, DisableResponse, EnableResponse, StartResponse, StopResponse};
use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};

use crate::{
    daemon::app_state::AppState,
    supervise::{
        DisableResult, EnableResult, EnsureFailure, EnsureOutcome, SupervisorHandle, await_ensure,
        drain::DrainReason,
    },
};

pub async fn post_start(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    let Some((svc, handle)) = resolve(&state, &name) else {
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
            let run_id = handle.run_id().await.unwrap_or(0);
            StartResponse::Started { run_id }
        }
        EnsureOutcome::Failed(EnsureFailure::StartQueueFull) => StartResponse::QueueFull,
        EnsureOutcome::Failed(EnsureFailure::InsufficientVram(reason)) => {
            StartResponse::Unavailable { reason }
        }
        EnsureOutcome::Failed(EnsureFailure::ServiceDisabled(reason)) => {
            StartResponse::Unavailable { reason }
        }
        EnsureOutcome::Failed(EnsureFailure::StartFailed(reason)) => {
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
    let Some((_svc, handle)) = resolve(&state, &name) else {
        return not_found(&name);
    };
    handle.begin_drain(DrainReason::UserKilled).await;
    (StatusCode::ACCEPTED, Json(StopResponse::Drained)).into_response()
}

pub async fn post_restart(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    let Some((svc, handle)) = resolve(&state, &name) else {
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
            let run_id = handle.run_id().await.unwrap_or(0);
            StartResponse::Started { run_id }
        }
        EnsureOutcome::Failed(EnsureFailure::StartQueueFull) => StartResponse::QueueFull,
        EnsureOutcome::Failed(EnsureFailure::InsufficientVram(reason)) => {
            StartResponse::Unavailable { reason }
        }
        EnsureOutcome::Failed(EnsureFailure::ServiceDisabled(reason)) => {
            StartResponse::Unavailable { reason }
        }
        EnsureOutcome::Failed(EnsureFailure::StartFailed(reason)) => {
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
    let Some((_svc, handle)) = resolve(&state, &name) else {
        return not_found(&name);
    };
    let body = match handle.enable().await {
        EnableResult::Enabled => EnableResponse::Enabled,
        EnableResult::NotDisabled => EnableResponse::AlreadyEnabled,
    };
    (StatusCode::OK, Json(body)).into_response()
}

pub async fn post_disable(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    let Some((_svc, handle)) = resolve(&state, &name) else {
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
