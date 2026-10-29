//! `GET /api/live` WebSocket — multiplexed, read-only subscriptions.
//!
//! A client sends [`LiveRequest`]s naming a [`LiveResource`] under an id of
//! its choosing. The daemon answers each subscription with a full snapshot
//! straight away and another whenever the resource's body changes, as a
//! [`LiveMessage`]. Every snapshot body is the same DTO the resource's HTTP
//! `GET` returns.
//!
//! A client can also follow an append-only [`LiveFeed`], such as a service's
//! log, from a position it already holds. A feed has no snapshot: the daemon
//! sends what came after the position, then each new entry as it lands.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    devices::list::DeviceSummary,
    events::EventHistory,
    info::get::DaemonInfoResponse,
    internal::log_line::{LogLine, LogPosition, LogStream},
    reservations::ReservationsResponse,
    services::{detail::ServiceDetail, list::ServicesResponse},
    shared::errors::ApiError,
};

/// A frame the client sends on `/api/live`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum LiveRequest {
    /// Start a subscription. The daemon replies with a snapshot, then with
    /// another each time the resource's body changes.
    Subscribe {
        /// A connection-local id, 1 to 128 bytes, that every reply for this
        /// subscription carries. It must not be in use on this connection.
        id: String,
        /// What to subscribe to.
        resource: LiveResource,
    },
    /// Follow a feed from a position. The daemon sends what the feed holds
    /// after it, then each new entry as it lands. Follows share the id space
    /// and the limit with subscriptions, and end with `cancel` the same way.
    Follow {
        /// A connection-local id, as for `subscribe`.
        id: String,
        /// What to follow, and from where.
        feed: LiveFeed,
    },
    /// End a subscription or a follow. Unknown ids are ignored.
    Cancel {
        /// The id the subscription or follow was started with.
        id: String,
    },
}

/// An append-only feed a client can follow on `/api/live`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LiveFeed {
    /// A service's captured log lines, as `GET /api/services/{name}/logs`
    /// returns them.
    Logs(LogFeed),
}

/// Which log lines to follow, and from where.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct LogFeed {
    /// The service's name.
    pub service: String,
    /// Send only lines after this position, such as a logs page's
    /// `follow_after`. `null` starts from the oldest retained line, subject
    /// to the backfill cap.
    #[serde(default)]
    pub after: Option<LogPosition>,
    /// Restrict to one run.
    #[serde(default)]
    pub run: Option<i64>,
    /// Restrict to one stream.
    #[serde(default)]
    pub stream: Option<LogStream>,
}

/// A resource a client can subscribe to on `/api/live`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LiveResource {
    /// The body of `GET /api/services`.
    Services,
    /// The body of `GET /api/devices`.
    Devices,
    /// The body of `GET /api/events/history`.
    Events,
    /// The body of `GET /api/info`.
    Info,
    /// The body of `GET /api/services/{name}`.
    Service {
        /// The service's name.
        name: String,
    },
    /// The body of `GET /api/reservations`.
    Reservations,
}

/// A frame the daemon sends on `/api/live`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LiveMessage {
    /// The first frame on every connection. A client that sees `build`
    /// change across reconnects is talking to an upgraded daemon and
    /// should offer to reload.
    Hello {
        /// Identifies the daemon's version and embedded frontend build.
        build: String,
    },
    /// The full current body of a subscribed resource.
    Snapshot {
        /// The subscription's id.
        id: String,
        /// The resource body, tagged by resource kind.
        data: LiveData,
    },
    /// Log lines for a `logs` follow, oldest first. Every line is after the
    /// follow's position and after every line sent before it.
    LogLines {
        /// The follow's id.
        id: String,
        /// The lines, in position order.
        lines: Vec<LogLine>,
    },
    /// Lines between `after` and `before` that the follow will not send,
    /// because more matched than one backfill carries. Fetch them over HTTP
    /// if they are wanted.
    LogGap {
        /// The follow's id.
        id: String,
        /// The position the follow resumed from, or `null` from the start.
        after: Option<LogPosition>,
        /// The first line the follow sends after the gap.
        before: LogPosition,
        /// How many matching lines were skipped.
        skipped: u64,
    },
    /// The follow has sent everything stored when it last read the log, and
    /// is now sending new lines as they land.
    LogCaughtUp {
        /// The follow's id.
        id: String,
    },
    /// A subscription or protocol error. A resource error, such as an
    /// unknown service, leaves the subscription active; a later snapshot
    /// supersedes it.
    Error {
        /// The subscription the error belongs to, or `null` when the frame
        /// it answers could not be parsed.
        id: Option<String>,
        /// The HTTP status the same error would carry.
        status: u16,
        /// The standard error envelope.
        error: ApiError,
    },
}

/// A snapshot body, tagged by the resource it belongs to.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
#[serde(tag = "resource", content = "body", rename_all = "snake_case")]
pub enum LiveData {
    /// A `services` snapshot.
    Services(ServicesResponse),
    /// A `devices` snapshot.
    Devices(Vec<DeviceSummary>),
    /// An `events` snapshot.
    Events(EventHistory),
    /// An `info` snapshot.
    Info(DaemonInfoResponse),
    /// A `service` snapshot.
    Service(Box<ServiceDetail>),
    /// A `reservations` snapshot.
    Reservations(ReservationsResponse),
}
