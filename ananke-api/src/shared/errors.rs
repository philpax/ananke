//! OpenAI-shaped error envelope used by every `/api/*` error response.
//!
//! The wire slug and taxonomy kind are typed enums ([`ApiErrorCodeSlug`] /
//! [`ApiErrorKind`]) so the OpenAPI spec surfaces them as enumerations
//! rather than free-form strings, and the daemon-side [`ApiErrorCode`]
//! (in the `ananke` crate) can project through them without string
//! literals.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// `{"error": {"code", "message", "type"}}`. The shared wire DTO; the
/// daemon-side `ApiErrorCode` enum is the source of truth for which
/// `code`, `message`, and `type` go together.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct ApiError {
    /// The nested body with the actual error metadata.
    pub error: ApiErrorBody,
}

/// Inner body of [`ApiError`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct ApiErrorBody {
    /// Short slug identifying the error class. Clients may switch on
    /// these strings; the [`ApiErrorCodeSlug`] enum enumerates every
    /// value the daemon emits.
    pub code: ApiErrorCodeSlug,
    /// Human-readable error message.
    pub message: String,
    /// OpenAI's taxonomy — [`ApiErrorKind::InvalidRequestError`] for
    /// things the client could have avoided, [`ApiErrorKind::ServerError`]
    /// for daemon-side problems. Derived from the daemon's `ApiErrorCode`
    /// variant.
    #[serde(rename = "type")]
    pub kind: ApiErrorKind,
}

/// Stable wire slug identifying an error class. Clients may switch on
/// these strings. The `Other` variant is a deserialization fallback so
/// clients don't break when the daemon adds a new code before they're
/// updated.
///
/// Variant names are serialised as `snake_case` strings (e.g.
/// `ModelNotFound` → `"model_not_found"`) via `#[serde(rename_all)]`.
/// The one exception is [`ApiErrorCodeSlug::InvalidRequest`], which
/// serialises as `"invalid_request_error"` to match OpenAI's
/// error-type taxonomy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ApiErrorCodeSlug {
    /// Client referenced a model name that isn't configured.
    ModelNotFound,
    /// Management caller referenced a service name that isn't in the
    /// live config.
    ServiceNotFound,
    /// Service is administratively disabled.
    ServiceDisabled,
    /// Supervisor's start queue saturated.
    StartQueueFull,
    /// Spawn / health-probe / queue-bus failure during ensure.
    StartFailed,
    /// Packer couldn't lay out the model.
    InsufficientVram,
    /// Queued behind a busy non-elastic peer.
    ServiceBlocked,
    /// Upstream child rejected the wire or never replied.
    UpstreamUnavailable,
    /// Bug inside the proxy itself.
    ProxyInternal,
    /// OpenAI endpoint the daemon hasn't implemented.
    NotImplemented,
    /// Client request was malformed (bad JSON, missing field, …).
    /// Renamed to `"invalid_request_error"` on the wire to match
    /// OpenAI's error-type taxonomy.
    #[serde(rename = "invalid_request_error")]
    InvalidRequest,
    /// Log-paging cursor failed to decode.
    InvalidCursor,
    /// Config PUT arrived without an `If-Match` precondition header.
    IfMatchRequired,
    /// Config PUT's `If-Match` didn't match the current on-disk hash.
    HashMismatch,
    /// Config write failed at the IO layer.
    PersistFailed,
    /// Deserialization fallback for forward compatibility.
    #[serde(other)]
    Other,
}

impl std::fmt::Display for ApiErrorCodeSlug {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Serialize produces a quoted JSON string like `"model_not_found"`;
        // strip the quotes so Display yields the bare slug.
        let s = serde_json::to_string(self).map_err(|_| std::fmt::Error)?;
        f.write_str(s.trim_matches('"'))
    }
}

/// OpenAI's error-type taxonomy. `InvalidRequestError` for anything the
/// client could have avoided, `ServerError` for daemon-side problems.
/// `Other` is a deserialization fallback.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ApiErrorKind {
    /// `"invalid_request_error"` — the client could have avoided this.
    InvalidRequestError,
    /// `"server_error"` — a daemon-side problem.
    ServerError,
    /// Deserialization fallback for forward compatibility.
    #[serde(other)]
    Other,
}

impl std::fmt::Display for ApiErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = serde_json::to_string(self).map_err(|_| std::fmt::Error)?;
        f.write_str(s.trim_matches('"'))
    }
}

impl ApiError {
    /// Build an error with `type: "server_error"`. Kept as the
    /// shorthand for the common case; new daemon-side code should
    /// go through `ApiErrorCode` and rely on `From<ApiErrorCode>
    /// for ApiError` so the kind is variant-accurate.
    pub fn new(code: ApiErrorCodeSlug, message: impl Into<String>) -> Self {
        Self {
            error: ApiErrorBody {
                code,
                message: message.into(),
                kind: ApiErrorKind::ServerError,
            },
        }
    }

    /// Build an error with an explicit `type`. Used by the
    /// daemon-side `From<ApiErrorCode> for ApiError` impl so the
    /// projection honours each variant's actual taxonomy bucket.
    pub fn with_kind(
        code: ApiErrorCodeSlug,
        message: impl Into<String>,
        kind: ApiErrorKind,
    ) -> Self {
        Self {
            error: ApiErrorBody {
                code,
                message: message.into(),
                kind,
            },
        }
    }
}
