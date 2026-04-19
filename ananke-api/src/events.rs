//! WebSocket event envelope published on `/api/events`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use smol_str::SmolStr;
use utoipa::ToSchema;

/// One event delivered over `/api/events`. The `type` tag discriminates the
/// variant; `at_ms` is present on every variant except `overflow`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
#[allow(missing_docs)]
pub enum Event {
    StateChanged {
        #[schema(value_type = String)]
        service: SmolStr,
        from: String,
        to: String,
        at_ms: i64,
    },
    AllocationChanged {
        #[schema(value_type = String)]
        service: SmolStr,
        reservations: BTreeMap<String, u64>,
        at_ms: i64,
    },
    ConfigReloaded {
        at_ms: i64,
        #[schema(value_type = Vec<String>)]
        changed_services: Vec<SmolStr>,
    },
    EstimatorDrift {
        #[schema(value_type = String)]
        service: SmolStr,
        rolling_mean: f32,
        at_ms: i64,
    },
    Overflow {
        dropped: u64,
    },
}
