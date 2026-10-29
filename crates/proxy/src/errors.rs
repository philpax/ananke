//! Every error class the daemon's HTTP/API surfaces can return.
//!
//! Variant data is exactly what's needed to render `message()`; no
//! freeform `String` payloads unless the underlying failure mode is
//! itself freeform (estimator errors, IO errors, etc.). The enum lives
//! in the proxy crate because the hyper proxy data-plane and the
//! management/OpenAI axum surfaces both project it; the axum-side
//! `IntoResponse` mapping stays in the daemon.

use ananke_api::shared::errors::{ApiError, ApiErrorCodeSlug, ApiErrorKind};
use http::StatusCode;
use smol_str::SmolStr;

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
    InsufficientCapacity { name: SmolStr, reason: String },
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
    /// A durable write failed: the config file or a local-store row.
    /// `what` names the thing being written, for the message.
    PersistFailed { what: &'static str, reason: String },
    /// A read against the daemon's local store failed.
    QueryFailed { reason: String },
    /// A launch-command preview could not be built. Distinct from
    /// `InsufficientCapacity`: not fitting is only one of the reasons a
    /// preview fails, and the others — no model path, an unreadable
    /// GGUF, an unsubstitutable placeholder — are service config the
    /// operator can correct rather than transient pressure to retry
    /// through.
    PreviewFailed { name: SmolStr, reason: String },
    /// A `/api/live` frame failed to parse, or broke a protocol limit
    /// such as the subscription id length.
    InvalidLiveRequest { reason: String },
    /// A `/api/live` subscribe reused an id already active on the
    /// connection.
    DuplicateSubscriptionId { id: String },
    /// A `/api/live` connection already holds `limit` subscriptions.
    SubscriptionLimit { limit: usize },
}

impl ApiErrorCode {
    /// Stable wire slug. Clients may switch on these strings, so each
    /// value is treated as part of the public API surface. The wire
    /// strings come from `#[serde(rename_all = "snake_case")]` on
    /// [`ApiErrorCodeSlug`], so there are no string literals here.
    pub fn slug(&self) -> ApiErrorCodeSlug {
        match self {
            Self::ModelNotFound { .. } => ApiErrorCodeSlug::ModelNotFound,
            Self::ServiceNotFound { .. } => ApiErrorCodeSlug::ServiceNotFound,
            Self::ServiceDisabled { .. } => ApiErrorCodeSlug::ServiceDisabled,
            Self::StartQueueFull { .. } => ApiErrorCodeSlug::StartQueueFull,
            Self::StartFailed { .. } => ApiErrorCodeSlug::StartFailed,
            Self::InsufficientCapacity { .. } => ApiErrorCodeSlug::InsufficientCapacity,
            Self::ServiceBlocked { .. } => ApiErrorCodeSlug::ServiceBlocked,
            Self::UpstreamUnavailable { .. } => ApiErrorCodeSlug::UpstreamUnavailable,
            Self::ProxyInternal { .. } => ApiErrorCodeSlug::ProxyInternal,
            Self::NotImplemented { .. } => ApiErrorCodeSlug::NotImplemented,
            Self::InvalidRequest { .. } => ApiErrorCodeSlug::InvalidRequest,
            Self::InvalidCursor => ApiErrorCodeSlug::InvalidCursor,
            Self::IfMatchRequired => ApiErrorCodeSlug::IfMatchRequired,
            Self::HashMismatch { .. } => ApiErrorCodeSlug::HashMismatch,
            Self::PersistFailed { .. } => ApiErrorCodeSlug::PersistFailed,
            Self::QueryFailed { .. } => ApiErrorCodeSlug::QueryFailed,
            Self::PreviewFailed { .. } => ApiErrorCodeSlug::PreviewFailed,
            Self::InvalidLiveRequest { .. } => ApiErrorCodeSlug::InvalidLiveRequest,
            Self::DuplicateSubscriptionId { .. } => ApiErrorCodeSlug::DuplicateSubscriptionId,
            Self::SubscriptionLimit { .. } => ApiErrorCodeSlug::SubscriptionLimit,
        }
    }

    /// HTTP status code paired with this error class.
    pub fn status(&self) -> StatusCode {
        match self {
            Self::ModelNotFound { .. } | Self::ServiceNotFound { .. } => StatusCode::NOT_FOUND,
            Self::ServiceDisabled { .. }
            | Self::StartQueueFull { .. }
            | Self::StartFailed { .. }
            | Self::InsufficientCapacity { .. }
            | Self::ServiceBlocked { .. } => StatusCode::SERVICE_UNAVAILABLE,
            Self::UpstreamUnavailable { .. } => StatusCode::BAD_GATEWAY,
            Self::ProxyInternal { .. } | Self::PersistFailed { .. } | Self::QueryFailed { .. } => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
            Self::NotImplemented { .. } => StatusCode::NOT_IMPLEMENTED,
            Self::InvalidRequest { .. } | Self::InvalidCursor | Self::InvalidLiveRequest { .. } => {
                StatusCode::BAD_REQUEST
            }
            Self::DuplicateSubscriptionId { .. } => StatusCode::CONFLICT,
            Self::SubscriptionLimit { .. } => StatusCode::TOO_MANY_REQUESTS,
            Self::PreviewFailed { .. } => StatusCode::UNPROCESSABLE_ENTITY,
            Self::IfMatchRequired => StatusCode::PRECONDITION_REQUIRED,
            Self::HashMismatch { .. } => StatusCode::PRECONDITION_FAILED,
        }
    }

    /// OpenAI's error-type taxonomy. `invalid_request_error` for
    /// anything the client could have avoided, `server_error` for
    /// daemon-side problems. One enum across every surface, so the
    /// management envelope reports the same accurate value the OpenAI
    /// one does rather than a blanket `server_error`.
    pub fn kind(&self) -> ApiErrorKind {
        match self {
            Self::ModelNotFound { .. }
            | Self::ServiceNotFound { .. }
            | Self::NotImplemented { .. }
            | Self::InvalidRequest { .. }
            | Self::InvalidCursor
            | Self::IfMatchRequired
            | Self::HashMismatch { .. }
            | Self::PreviewFailed { .. }
            | Self::InvalidLiveRequest { .. }
            | Self::DuplicateSubscriptionId { .. }
            | Self::SubscriptionLimit { .. } => ApiErrorKind::InvalidRequestError,
            Self::ServiceDisabled { .. }
            | Self::StartQueueFull { .. }
            | Self::StartFailed { .. }
            | Self::InsufficientCapacity { .. }
            | Self::ServiceBlocked { .. }
            | Self::UpstreamUnavailable { .. }
            | Self::ProxyInternal { .. }
            | Self::PersistFailed { .. }
            | Self::QueryFailed { .. } => ApiErrorKind::ServerError,
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
            Self::InsufficientCapacity { name, reason } => {
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
            Self::PersistFailed { what, reason } => format!("writing {what} failed: {reason}"),
            Self::QueryFailed { reason } => {
                format!("reading from the local store failed: {reason}")
            }
            Self::PreviewFailed { name, reason } => {
                format!("cannot build launch-command preview for `{name}`: {reason}")
            }
            Self::InvalidLiveRequest { reason } => format!("invalid live request: {reason}"),
            Self::DuplicateSubscriptionId { id } => {
                format!("subscription id `{id}` is already in use on this connection")
            }
            Self::SubscriptionLimit { limit } => {
                format!("this connection already holds the maximum of {limit} subscriptions")
            }
        }
    }
}

impl From<ApiErrorCode> for ApiError {
    fn from(code: ApiErrorCode) -> Self {
        ApiError::with_kind(code.slug(), code.message(), code.kind())
    }
}

impl axum::response::IntoResponse for ApiErrorCode {
    /// Render the standard `(status, JSON body)` shape used by every
    /// axum surface (management + OpenAI compat). The hyper proxy
    /// data plane has its own builder (see `proxy::error_response`)
    /// that takes the same `ApiErrorCode` and produces a byte-
    /// identical body off a different `Response<Body>` type.
    fn into_response(self) -> axum::response::Response {
        let status = self.status();
        let body: ApiError = self.into();
        (status, axum::Json(body)).into_response()
    }
}
