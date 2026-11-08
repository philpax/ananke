//! The daemon's lifecycle event, as served by `/api/events/history` and the
//! `/api/live` `events` resource.
//!
//! `Event` is also used as a broadcast bus message in `supervise/`,
//! `allocator/`, and `config/`, so it lives in the internal module rather
//! than under a single endpoint.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use smol_str::SmolStr;
use utoipa::ToSchema;

use crate::shared::service_state::{DisableReason, ServiceStateKind};

/// One daemon lifecycle event. The `type` tag discriminates the variant, and
/// every variant carries `at_ms`. The same
/// `Event` is used as a broadcast bus message internally, so it must stay
/// serializable and stable — anything published here is part of the wire
/// contract the frontend and other clients read.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    /// A service's state machine moved to a new state.
    StateChanged {
        /// The service that changed state.
        #[schema(value_type = String)]
        service: SmolStr,
        /// The state it left.
        from: ServiceStateKind,
        /// The state it entered.
        to: ServiceStateKind,
        /// Why the service is disabled; present exactly when `to` is
        /// `disabled`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        disabled_reason: Option<DisableReason>,
        /// When the transition happened, in unix milliseconds.
        at_ms: i64,
    },
    /// A service's reservation table changed (a pledge was added, moved, or
    /// released on one or more devices).
    AllocationChanged {
        /// The service whose allocation changed.
        #[schema(value_type = String)]
        service: SmolStr,
        /// The per-device reserved byte counts after the change. Keys are
        /// device ids (`"cpu"` or `"gpu:N"`).
        reservations: BTreeMap<String, u64>,
        /// When the allocation changed, in unix milliseconds.
        at_ms: i64,
    },
    /// The config was reloaded or replaced, and a set of services changed.
    ConfigReloaded {
        /// When the reload happened, in unix milliseconds.
        at_ms: i64,
        /// The services whose effective config changed as a result.
        #[schema(value_type = Vec<String>)]
        changed_services: Vec<SmolStr>,
    },
    /// One of a service's rolling estimator corrections moved by more than 5%.
    /// `class` is the memory pool it belongs to — `"vram"` or `"host"` — since
    /// the two are learned independently from separate observations.
    EstimatorDrift {
        /// The service whose correction drifted.
        #[schema(value_type = String)]
        service: SmolStr,
        /// The memory pool: `"vram"` or `"host"`.
        class: String,
        /// The new rolling-mean correction factor.
        rolling_mean: f32,
        /// When the drift was recorded, in unix milliseconds.
        at_ms: i64,
    },
    /// A `Running` service was drained and respawned by its auto-restart
    /// policy. `trigger` is `"error_rate"`, `"periodic"`, `"ttft_stall"`,
    /// `"generation_stall"`, or `"spec_collapse"`; `detail` is a
    /// human-readable reason (e.g. the observed error rate and window).
    AutoRestarted {
        /// The service that was restarted.
        #[schema(value_type = String)]
        service: SmolStr,
        /// The auto-restart trigger that fired.
        trigger: String,
        /// A human-readable reason for the restart.
        detail: String,
        /// When the restart was recorded, in unix milliseconds.
        at_ms: i64,
    },
}

impl Event {
    /// When the event happened, in unix milliseconds.
    pub fn at_ms(&self) -> i64 {
        match self {
            Self::StateChanged { at_ms, .. }
            | Self::AllocationChanged { at_ms, .. }
            | Self::ConfigReloaded { at_ms, .. }
            | Self::EstimatorDrift { at_ms, .. }
            | Self::AutoRestarted { at_ms, .. } => *at_ms,
        }
    }
}
