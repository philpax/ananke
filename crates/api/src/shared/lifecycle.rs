//! When a service starts and stops. Spelled on the wire as it is in config.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// When a service starts and stops relative to the daemon's own lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum Lifecycle {
    /// Started at boot and restarted when it exits.
    Persistent,
    /// Started on first request and stopped after `idle_timeout`.
    OnDemand,
}

impl Lifecycle {
    /// The lifecycle name as it appears in config files and on the wire.
    pub fn as_str(self) -> &'static str {
        match self {
            Lifecycle::Persistent => "persistent",
            Lifecycle::OnDemand => "on_demand",
        }
    }
}

impl std::fmt::Display for Lifecycle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(self.as_str())
    }
}
