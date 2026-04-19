//! OpenAI-shaped error responses: `{error: {code, message, type}}`.

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;
use utoipa::ToSchema;

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

pub fn err(status: StatusCode, code: &str, message: impl Into<String>, kind: &str) -> Response {
    let body = ErrorBody {
        error: ErrorDetail {
            code: code.into(),
            message: message.into(),
            kind: kind.into(),
        },
    };
    (status, Json(body)).into_response()
}

pub fn not_found_model(name: &str) -> Response {
    err(
        StatusCode::NOT_FOUND,
        "model_not_found",
        format!("model `{name}` not found"),
        "invalid_request_error",
    )
}

pub fn service_disabled(name: &str, reason: &str) -> Response {
    err(
        StatusCode::SERVICE_UNAVAILABLE,
        "service_disabled",
        format!("service `{name}` is disabled: {reason}"),
        "server_error",
    )
}

pub fn start_queue_full(name: &str) -> Response {
    err(
        StatusCode::SERVICE_UNAVAILABLE,
        "start_queue_full",
        format!("start queue full for service `{name}`"),
        "server_error",
    )
}

pub fn start_failed(name: &str, detail: &str) -> Response {
    err(
        StatusCode::SERVICE_UNAVAILABLE,
        "start_failed",
        format!("service `{name}` failed to start: {detail}"),
        "server_error",
    )
}

pub fn insufficient_vram(name: &str, detail: &str) -> Response {
    err(
        StatusCode::SERVICE_UNAVAILABLE,
        "insufficient_vram",
        format!("service `{name}` cannot fit: {detail}"),
        "server_error",
    )
}

pub fn not_implemented(path: &str) -> Response {
    err(
        StatusCode::NOT_IMPLEMENTED,
        "not_implemented",
        format!("endpoint `{path}` is not implemented"),
        "invalid_request_error",
    )
}

pub fn bad_request(msg: impl Into<String>) -> Response {
    err(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        msg,
        "invalid_request_error",
    )
}
