//! OpenAI-shaped error responses: `{error: {code, message, type}}`.

use std::fmt::{self, Display};

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;
use utoipa::ToSchema;

/// Every distinct `error.code` slug the proxy + OpenAI-compat layer can
/// emit. Having them as an enum means callers pick a variant rather than
/// hand-typing a string, and the (status, error-type) pair travels with
/// the code instead of being threaded through as two separate arguments.
#[derive(Debug, Clone, Copy)]
pub enum ProxyErrorCode {
    /// The client referenced a `model` name that isn't configured.
    ModelNotFound,
    /// The service exists but is disabled (OOM'd, hit max restarts, etc.).
    ServiceDisabled,
    /// The service's start queue is saturated.
    StartQueueFull,
    /// The service tried to start and the spawn itself failed.
    StartFailed,
    /// The service tried to start and the packer couldn't fit it.
    InsufficientVram,
    /// The client hit an OpenAI endpoint we haven't implemented.
    NotImplemented,
    /// The client sent a malformed request (bad JSON, missing field, …).
    InvalidRequest,
    /// The upstream llama-server rejected the connection, timed out, or
    /// otherwise failed to respond. Wire-level, before we ever saw its
    /// status line.
    UpstreamUnavailable,
    /// Something failed inside the proxy itself (URI parse, header build,
    /// body collect, …) — a programming or config bug rather than an
    /// upstream issue.
    ProxyInternal,
}

impl ProxyErrorCode {
    /// HTTP status that goes with this code on the wire.
    pub fn status(&self) -> StatusCode {
        match self {
            Self::ModelNotFound => StatusCode::NOT_FOUND,
            Self::ServiceDisabled
            | Self::StartQueueFull
            | Self::StartFailed
            | Self::InsufficientVram => StatusCode::SERVICE_UNAVAILABLE,
            Self::NotImplemented => StatusCode::NOT_IMPLEMENTED,
            Self::InvalidRequest => StatusCode::BAD_REQUEST,
            Self::UpstreamUnavailable => StatusCode::BAD_GATEWAY,
            Self::ProxyInternal => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// OpenAI's error-type taxonomy — `invalid_request_error` for anything
    /// the client could have avoided, `server_error` for our problems.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::ModelNotFound | Self::NotImplemented | Self::InvalidRequest => {
                "invalid_request_error"
            }
            Self::ServiceDisabled
            | Self::StartQueueFull
            | Self::StartFailed
            | Self::InsufficientVram
            | Self::UpstreamUnavailable
            | Self::ProxyInternal => "server_error",
        }
    }
}

impl Display for ProxyErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let slug = match self {
            Self::ModelNotFound => "model_not_found",
            Self::ServiceDisabled => "service_disabled",
            Self::StartQueueFull => "start_queue_full",
            Self::StartFailed => "start_failed",
            Self::InsufficientVram => "insufficient_vram",
            Self::NotImplemented => "not_implemented",
            Self::InvalidRequest => "invalid_request_error",
            Self::UpstreamUnavailable => "upstream_unavailable",
            Self::ProxyInternal => "proxy_internal",
        };
        f.write_str(slug)
    }
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ErrorDetail {
    pub code: String,
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
}

pub fn err(code: ProxyErrorCode, message: impl Into<String>) -> Response {
    let body = ErrorBody {
        error: ErrorDetail {
            code: code.to_string(),
            message: message.into(),
            kind: code.kind().into(),
        },
    };
    (code.status(), Json(body)).into_response()
}

pub fn not_found_model(name: &str) -> Response {
    err(
        ProxyErrorCode::ModelNotFound,
        format!("model `{name}` not found"),
    )
}

pub fn service_disabled(name: &str, reason: &str) -> Response {
    err(
        ProxyErrorCode::ServiceDisabled,
        format!("service `{name}` is disabled: {reason}"),
    )
}

pub fn start_queue_full(name: &str) -> Response {
    err(
        ProxyErrorCode::StartQueueFull,
        format!("start queue full for service `{name}`"),
    )
}

pub fn start_failed(name: &str, detail: &str) -> Response {
    err(
        ProxyErrorCode::StartFailed,
        format!("service `{name}` failed to start: {detail}"),
    )
}

pub fn insufficient_vram(name: &str, detail: &str) -> Response {
    err(
        ProxyErrorCode::InsufficientVram,
        format!("service `{name}` cannot fit: {detail}"),
    )
}

pub fn not_implemented(path: &str) -> Response {
    err(
        ProxyErrorCode::NotImplemented,
        format!("endpoint `{path}` is not implemented"),
    )
}

pub fn bad_request(msg: impl Into<String>) -> Response {
    err(ProxyErrorCode::InvalidRequest, msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_display_matches_openai_slug() {
        assert_eq!(ProxyErrorCode::ModelNotFound.to_string(), "model_not_found");
        assert_eq!(
            ProxyErrorCode::InsufficientVram.to_string(),
            "insufficient_vram"
        );
        assert_eq!(
            ProxyErrorCode::InvalidRequest.to_string(),
            "invalid_request_error"
        );
    }

    #[test]
    fn status_and_kind_match_openai_taxonomy() {
        assert_eq!(
            ProxyErrorCode::ModelNotFound.status(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            ProxyErrorCode::ModelNotFound.kind(),
            "invalid_request_error"
        );
        assert_eq!(
            ProxyErrorCode::InsufficientVram.status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(ProxyErrorCode::InsufficientVram.kind(), "server_error");
        assert_eq!(
            ProxyErrorCode::NotImplemented.status(),
            StatusCode::NOT_IMPLEMENTED,
        );
    }
}
