//! POST /api/services/{name}/{start,stop,restart,enable,disable} handlers.

use std::time::Duration;

use ananke_api::{
    services::{
        disable::DisableResponse, enable::EnableResponse, start::StartResponse, stop::StopResponse,
    },
    shared::errors::ApiError,
};
use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use smol_str::SmolStr;
use tracing::{info, warn};

use crate::{
    api::errors::ApiErrorCode,
    daemon::app_state::AppState,
    supervise::{
        DisableResult, EnableResult, EnsureOutcome, SupervisorHandle, await_ensure,
        drain::DrainReason, provision::ensure_failure_to_api_code,
    },
};

#[utoipa::path(
    summary = "Start a service",
    post,
    path = "/api/services/{name}/start",
    params(("name" = String, Path, description = "Service name")),
    responses(
        (status = 202, body = StartResponse),
        (status = 404, body = ApiError, description = "service_not_found"),
        (status = 503, body = ApiError, description = "service_disabled, start_queue_full, start_failed, insufficient_vram, service_blocked")
    )
)]
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
        EnsureOutcome::Failed(failure) => {
            return management_failure_response(&name, "start", failure);
        }
    };
    (StatusCode::ACCEPTED, Json(body)).into_response()
}

#[utoipa::path(
    summary = "Stop a service",
    post,
    path = "/api/services/{name}/stop",
    params(("name" = String, Path, description = "Service name")),
    responses(
        (status = 202, body = StopResponse),
        (status = 404, body = ApiError, description = "service_not_found")
    )
)]
pub async fn post_stop(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    info!(service = %name, "management stop requested");
    let Some((_svc, handle)) = resolve(&state, &name) else {
        warn!(service = %name, "management stop rejected: service not found");
        return not_found(&name);
    };
    handle.begin_drain(DrainReason::UserKilled).await;
    (StatusCode::ACCEPTED, Json(StopResponse::Drained)).into_response()
}

#[utoipa::path(
    summary = "Restart a service",
    post,
    path = "/api/services/{name}/restart",
    params(("name" = String, Path, description = "Service name")),
    responses(
        (status = 202, body = StartResponse),
        (status = 404, body = ApiError, description = "service_not_found"),
        (status = 503, body = ApiError, description = "service_disabled, start_queue_full, start_failed, insufficient_vram, service_blocked")
    )
)]
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
        EnsureOutcome::Failed(failure) => {
            return management_failure_response(&name, "restart", failure);
        }
    };
    (StatusCode::ACCEPTED, Json(body)).into_response()
}

#[utoipa::path(
    summary = "Enable a disabled service",
    post,
    path = "/api/services/{name}/enable",
    params(("name" = String, Path, description = "Service name")),
    responses(
        (status = 200, body = EnableResponse),
        (status = 404, body = ApiError, description = "service_not_found")
    )
)]
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

#[utoipa::path(
    summary = "Disable a service",
    post,
    path = "/api/services/{name}/disable",
    params(("name" = String, Path, description = "Service name")),
    responses(
        (status = 200, body = DisableResponse),
        (status = 404, body = ApiError, description = "service_not_found")
    )
)]
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
    ApiErrorCode::ServiceNotFound {
        name: SmolStr::new(name),
    }
    .into_response()
}

/// Single place that turns an `EnsureFailure` from the supervisor into
/// the management API's response. Variants the management surface
/// elects to surface as "controlled outcome" 202s (queue full, vram
/// shortfall, disabled) project to the typed `StartResponse`; the
/// hard-error variants (start failed, blocked) go through
/// `ApiErrorCode`'s standard 503 projection. Shared between
/// `post_start` and `post_restart` so the two handlers can't drift.
///
/// `Unavailable` carries the same `ApiErrorBody` shape a 503 error
/// would (slug + message + kind), built through the same
/// `ApiErrorCode` pipeline — so a client switching on `error.code`
/// sees `insufficient_vram` / `service_disabled` here and at the 503
/// surfaces, without having to special-case 202-with-Unavailable.
fn management_failure_response(
    name: &str,
    op: &'static str,
    failure: crate::supervise::EnsureFailure,
) -> Response {
    use crate::supervise::EnsureFailure;
    match failure {
        EnsureFailure::StartQueueFull => {
            warn!(service = name, operation = op, "rejected: start_queue_full");
            (StatusCode::ACCEPTED, Json(StartResponse::QueueFull)).into_response()
        }
        EnsureFailure::InsufficientVram(_) | EnsureFailure::ServiceDisabled(_) => {
            let code = ensure_failure_to_api_code(&SmolStr::new(name), failure);
            warn!(service = name, operation = op, slug = %code.slug(), message = %code.message(), "rejected (controlled)");
            let body: ApiError = code.into();
            (
                StatusCode::ACCEPTED,
                Json(StartResponse::Unavailable { error: body.error }),
            )
                .into_response()
        }
        other => {
            let code = ensure_failure_to_api_code(&SmolStr::new(name), other);
            warn!(service = name, operation = op, slug = %code.slug(), message = %code.message(), "rejected");
            code.into_response()
        }
    }
}
