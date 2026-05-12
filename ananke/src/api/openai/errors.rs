//! Helpers that build axum `Response`s for the OpenAI-compat surface.
//!
//! The error type itself lives one level up at [`crate::api::errors::
//! ApiErrorCode`] — that's the single source of truth for slug, status,
//! kind, and message. The thin wrappers here exist so the OpenAI
//! handlers stay readable (`return errors::insufficient_vram(&model,
//! &msg)`) without callers having to spell out the enum variant
//! inline. New error classes should be added by extending
//! `ApiErrorCode`, not by adding free functions here.

use axum::response::{IntoResponse, Response};
use smol_str::SmolStr;

use crate::api::errors::ApiErrorCode;

pub fn not_found_model(name: &str) -> Response {
    ApiErrorCode::ModelNotFound {
        name: SmolStr::new(name),
    }
    .into_response()
}

pub fn service_disabled(name: &str, reason: &str) -> Response {
    ApiErrorCode::ServiceDisabled {
        name: SmolStr::new(name),
        reason: reason.to_string(),
    }
    .into_response()
}

pub fn start_queue_full(name: &str) -> Response {
    ApiErrorCode::StartQueueFull {
        name: SmolStr::new(name),
    }
    .into_response()
}

pub fn start_failed(name: &str, reason: &str) -> Response {
    ApiErrorCode::StartFailed {
        name: SmolStr::new(name),
        reason: reason.to_string(),
    }
    .into_response()
}

pub fn insufficient_vram(name: &str, reason: &str) -> Response {
    ApiErrorCode::InsufficientVram {
        name: SmolStr::new(name),
        reason: reason.to_string(),
    }
    .into_response()
}

pub fn service_blocked(name: &str, busy_peers: &[SmolStr]) -> Response {
    ApiErrorCode::ServiceBlocked {
        name: SmolStr::new(name),
        busy_peers: busy_peers.to_vec(),
    }
    .into_response()
}

pub fn not_implemented(path: &str) -> Response {
    ApiErrorCode::NotImplemented {
        path: path.to_string(),
    }
    .into_response()
}

pub fn bad_request(reason: impl Into<String>) -> Response {
    ApiErrorCode::InvalidRequest {
        reason: reason.into(),
    }
    .into_response()
}
