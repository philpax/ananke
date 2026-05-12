//! Unified daemon-side error code with structured data.
//!
//! Every error the API surface can emit lives as one variant of
//! [`ApiErrorCode`]. The variant carries the structured fields its
//! message is built from (service name, busy peer list, persist
//! reason, …) — not pre-formatted strings — so the slug, status,
//! taxonomy kind, and human-readable message can all be derived from
//! one source of truth at projection time.
//!
//! Three projections live here:
//!
//! * [`ApiError`] (the shared wire envelope in `ananke-api`) via
//!   [`From<ApiErrorCode>`]. Used by management handlers.
//! * `axum::response::Response` via [`IntoResponse`]. Used everywhere
//!   we return an axum response directly.
//! * `Response<ProxyBody>` via [`proxy::error_response`] in the hyper
//!   proxy data plane. Built from `status()` + `slug()` + `message()`
//!   + `kind()` so it stays byte-identical to the axum surface.
//!
//! Adding a new error class is a one-place change: extend the enum,
//! cover it in `slug`, `status`, `kind`, and `message`, and every
//! surface picks it up automatically.

pub use ananke_api::ApiError;
use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use smol_str::SmolStr;

/// Every error class the daemon's HTTP/API surfaces can return.
///
/// Variant data is exactly what's needed to render `message()`; no
/// freeform `String` payloads unless the underlying failure mode is
/// itself freeform (estimator errors, IO errors, etc.).
#[derive(Debug, Clone)]
pub enum ApiErrorCode {
    /// Client referenced a model name that isn't configured.
    ModelNotFound { name: SmolStr },
    /// Management caller referenced a service name that isn't in the
    /// live config.
    ServiceNotFound { name: SmolStr },
    /// Service is administratively disabled (OOM auto-disable, hit max
    /// restarts, operator-disabled, …).
    ServiceDisabled { name: SmolStr, reason: String },
    /// Supervisor's start queue saturated.
    StartQueueFull { name: SmolStr },
    /// Spawn / health-probe / queue-bus failure during ensure.
    StartFailed { name: SmolStr, reason: String },
    /// Packer couldn't lay out the model.
    InsufficientVram { name: SmolStr, reason: String },
    /// Queued behind a busy non-elastic peer beyond `QUEUE_BLOCKED_GRACE`.
    ServiceBlocked {
        name: SmolStr,
        busy_peers: Vec<SmolStr>,
    },
    /// Upstream child rejected the wire or never replied.
    UpstreamUnavailable { reason: String },
    /// Bug inside the proxy itself (URI parse, header build, body
    /// collect, …) — a programming or config bug, not an upstream
    /// issue.
    ProxyInternal { reason: String },
    /// OpenAI endpoint the daemon hasn't implemented.
    NotImplemented { path: String },
    /// Client request was malformed (bad JSON, missing field, …).
    InvalidRequest { reason: String },
    /// Log-paging cursor failed to decode.
    InvalidCursor,
    /// Config PUT arrived without an `If-Match` precondition header.
    IfMatchRequired,
    /// Config PUT's `If-Match` didn't match the current on-disk hash.
    HashMismatch { server_hash: String },
    /// Config write failed at the IO layer.
    PersistFailed { reason: String },
}

impl ApiErrorCode {
    /// Stable wire slug. Clients may switch on these strings, so each
    /// value is treated as part of the public API surface.
    pub fn slug(&self) -> &'static str {
        match self {
            Self::ModelNotFound { .. } => "model_not_found",
            Self::ServiceNotFound { .. } => "service_not_found",
            Self::ServiceDisabled { .. } => "service_disabled",
            Self::StartQueueFull { .. } => "start_queue_full",
            Self::StartFailed { .. } => "start_failed",
            Self::InsufficientVram { .. } => "insufficient_vram",
            Self::ServiceBlocked { .. } => "service_blocked",
            Self::UpstreamUnavailable { .. } => "upstream_unavailable",
            Self::ProxyInternal { .. } => "proxy_internal",
            Self::NotImplemented { .. } => "not_implemented",
            Self::InvalidRequest { .. } => "invalid_request_error",
            Self::InvalidCursor => "invalid_cursor",
            Self::IfMatchRequired => "if_match_required",
            Self::HashMismatch { .. } => "hash_mismatch",
            Self::PersistFailed { .. } => "persist_failed",
        }
    }

    /// HTTP status code paired with this error class.
    pub fn status(&self) -> StatusCode {
        match self {
            Self::ModelNotFound { .. } | Self::ServiceNotFound { .. } => StatusCode::NOT_FOUND,
            Self::ServiceDisabled { .. }
            | Self::StartQueueFull { .. }
            | Self::StartFailed { .. }
            | Self::InsufficientVram { .. }
            | Self::ServiceBlocked { .. } => StatusCode::SERVICE_UNAVAILABLE,
            Self::UpstreamUnavailable { .. } => StatusCode::BAD_GATEWAY,
            Self::ProxyInternal { .. } | Self::PersistFailed { .. } => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
            Self::NotImplemented { .. } => StatusCode::NOT_IMPLEMENTED,
            Self::InvalidRequest { .. } | Self::InvalidCursor => StatusCode::BAD_REQUEST,
            Self::IfMatchRequired => StatusCode::PRECONDITION_REQUIRED,
            Self::HashMismatch { .. } => StatusCode::PRECONDITION_FAILED,
        }
    }

    /// OpenAI's error-type taxonomy. `invalid_request_error` for
    /// anything the client could have avoided, `server_error` for
    /// daemon-side problems. The management `ApiError` envelope used
    /// to fix this to `server_error` unconditionally; with the
    /// unified enum every surface now reports the accurate value.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::ModelNotFound { .. }
            | Self::ServiceNotFound { .. }
            | Self::NotImplemented { .. }
            | Self::InvalidRequest { .. }
            | Self::InvalidCursor
            | Self::IfMatchRequired
            | Self::HashMismatch { .. } => "invalid_request_error",
            Self::ServiceDisabled { .. }
            | Self::StartQueueFull { .. }
            | Self::StartFailed { .. }
            | Self::InsufficientVram { .. }
            | Self::ServiceBlocked { .. }
            | Self::UpstreamUnavailable { .. }
            | Self::ProxyInternal { .. }
            | Self::PersistFailed { .. } => "server_error",
        }
    }

    /// Human-readable message. Derived entirely from the variant's
    /// carry-data so the wording is consistent across every surface
    /// that renders the same error.
    pub fn message(&self) -> String {
        match self {
            Self::ModelNotFound { name } => format!("model `{name}` not found"),
            Self::ServiceNotFound { name } => format!("service `{name}` not found"),
            Self::ServiceDisabled { name, reason } => {
                format!("service `{name}` is disabled: {reason}")
            }
            Self::StartQueueFull { name } => format!("start queue full for service `{name}`"),
            Self::StartFailed { name, reason } => {
                format!("service `{name}` failed to start: {reason}")
            }
            Self::InsufficientVram { name, reason } => {
                format!("service `{name}` cannot fit: {reason}")
            }
            Self::ServiceBlocked { name, busy_peers } => {
                if busy_peers.is_empty() {
                    format!("service `{name}` is blocked by an unidentified busy peer")
                } else {
                    let list = busy_peers
                        .iter()
                        .map(|p| format!("`{p}`"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("service `{name}` is blocked by busy peer(s): {list}")
                }
            }
            Self::UpstreamUnavailable { reason } => reason.clone(),
            Self::ProxyInternal { reason } => reason.clone(),
            Self::NotImplemented { path } => format!("endpoint `{path}` is not implemented"),
            Self::InvalidRequest { reason } => reason.clone(),
            Self::InvalidCursor => "malformed `before` cursor".to_string(),
            Self::IfMatchRequired => {
                "PUT /api/config requires an If-Match header with the current config hash"
                    .to_string()
            }
            Self::HashMismatch { server_hash } => {
                format!("config was modified since last GET; current hash is {server_hash}")
            }
            Self::PersistFailed { reason } => format!("writing config to disk failed: {reason}"),
        }
    }
}

impl From<ApiErrorCode> for ApiError {
    fn from(code: ApiErrorCode) -> Self {
        // Build the body directly rather than going through
        // `ApiError::new`, which hardcodes `kind = "server_error"`.
        // The unified code reports its accurate kind on every surface.
        ApiError::with_kind(code.slug(), code.message(), code.kind())
    }
}

impl IntoResponse for ApiErrorCode {
    /// Render the standard `(status, JSON body)` shape used by every
    /// axum surface (management + OpenAI compat). The hyper proxy
    /// data plane has its own builder (see `proxy::error_response`)
    /// that takes the same `ApiErrorCode` and produces a byte-
    /// identical body off a different `Response<Body>` type.
    fn into_response(self) -> Response {
        let status = self.status();
        let body: ApiError = self.into();
        (status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_blocked_renders_named_peers() {
        let code = ApiErrorCode::ServiceBlocked {
            name: SmolStr::new("qwen"),
            busy_peers: vec![SmolStr::new("alpha"), SmolStr::new("beta")],
        };
        assert_eq!(code.slug(), "service_blocked");
        assert_eq!(code.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(code.kind(), "server_error");
        assert_eq!(
            code.message(),
            "service `qwen` is blocked by busy peer(s): `alpha`, `beta`"
        );
    }

    #[test]
    fn service_blocked_with_no_peers_falls_back_gracefully() {
        let code = ApiErrorCode::ServiceBlocked {
            name: SmolStr::new("qwen"),
            busy_peers: Vec::new(),
        };
        assert!(code.message().contains("unidentified"));
    }

    #[test]
    fn hash_mismatch_carries_server_hash_into_message() {
        let code = ApiErrorCode::HashMismatch {
            server_hash: "abc123".to_string(),
        };
        assert_eq!(code.status(), StatusCode::PRECONDITION_FAILED);
        assert!(code.message().contains("abc123"));
    }

    #[test]
    fn projection_to_api_error_uses_accurate_kind() {
        let code = ApiErrorCode::InvalidCursor;
        let body: ApiError = code.into();
        assert_eq!(body.error.code, "invalid_cursor");
        assert_eq!(body.error.kind, "invalid_request_error");
    }
}
