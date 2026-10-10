//! Multiplexed dashboard WebSocket requests, subscriptions, and responses.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{events::Event, openai::ChatCompletionEnvelope};

/// A historical window, optionally following the server clock.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DashboardRange {
    /// Inclusive lower bound for a fixed window, in Unix milliseconds.
    pub since: i64,
    /// Inclusive upper bound for a fixed window, in Unix milliseconds.
    pub until: i64,
    /// A rolling window's duration; fixed windows omit this field.
    pub live_ms: Option<u64>,
}

/// A dashboard read; subscriptions receive an initial snapshot and subsequent updates.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum DashboardQuery {
    /// The service inventory.
    Services,
    /// One service's current state and placement.
    Service {
        /// The configured service name.
        name: String,
    },
    /// The latest hardware measurements.
    Devices,
    /// Daemon addresses, configuration path, and uptime.
    Info,
    /// OpenAI-compatible model discovery.
    Models,
    /// Retained lifecycle events from this daemon instance.
    Events,
    /// Persisted request metrics.
    Metrics {
        /// The requested historical window.
        range: DashboardRange,
        /// An optional service filter.
        service: Option<String>,
        /// The aggregation bucket, e.g. `5m`.
        bucket: String,
    },
    /// Persisted automatic restart records.
    Restarts {
        /// The requested historical window.
        range: DashboardRange,
        /// An optional service filter.
        service: Option<String>,
    },
    /// Persisted device measurements.
    Samples {
        /// The requested historical window.
        range: DashboardRange,
    },
    /// A page of captured child-process logs.
    Logs {
        /// The configured service name.
        name: String,
        /// The requested historical window.
        range: DashboardRange,
        /// An opaque cursor for older lines.
        before: Option<String>,
    },
    /// The configuration file and its concurrency hash.
    Config,
}

/// A service lifecycle command.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum DashboardAction {
    /// Start an idle service.
    Start,
    /// Stop a service.
    Stop,
    /// Drain and restart a service.
    Restart,
    /// Enable a disabled service.
    Enable,
    /// Disable a service.
    Disable,
}

/// One client operation, correlated by its request or subscription id.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum DashboardRequest {
    /// Read once or subscribe until cancelled.
    Read {
        /// The connection-local correlation id.
        id: String,
        /// The requested resource.
        query: DashboardQuery,
        /// Whether to push subsequent changes.
        subscribe: bool,
    },
    /// Cancel a subscription or an in-progress request.
    Cancel {
        /// The id to cancel.
        id: String,
    },
    /// Execute a service lifecycle command.
    Action {
        /// The correlation id.
        id: String,
        /// The configured service name.
        name: String,
        /// The lifecycle command.
        action: DashboardAction,
    },
    /// Validate a configuration draft.
    Validate {
        /// The correlation id.
        id: String,
        /// The TOML draft.
        content: String,
    },
    /// Save a draft using the same optimistic concurrency contract as HTTP.
    Save {
        /// The correlation id.
        id: String,
        /// The TOML draft.
        content: String,
        /// The hash returned by the last configuration read.
        hash: String,
    },
    /// Stream a chat completion until completion or cancellation.
    Chat {
        /// The correlation id.
        id: String,
        /// The OpenAI-compatible request body.
        body: ChatCompletionEnvelope,
    },
}

/// A response or pushed snapshot; JSON bodies use the existing endpoint schemas.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DashboardMessage {
    /// A read result, pushed snapshot, or mutation acknowledgement.
    Response {
        /// The request or subscription id.
        id: String,
        /// The equivalent HTTP status code.
        status: u16,
        /// The endpoint's generated response DTO, or null for an empty response.
        body: serde_json::Value,
    },
    /// A fragment of the existing OpenAI event stream.
    Stream {
        /// The chat request id.
        id: String,
        /// UTF-8 SSE data, preserving upstream frame boundaries across fragments.
        text: String,
    },
    /// The stream has ended; absence of an OpenAI DONE marker remains an error.
    End {
        /// The chat request id.
        id: String,
    },
}

/// Retained events and the actual beginning of this daemon's capture window.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DashboardEvents {
    /// Oldest retained event time, or the daemon's capture start when empty.
    pub since_ms: i64,
    /// Events in publication order; the bounded history survives browser reconnects.
    pub events: Vec<Event>,
}
