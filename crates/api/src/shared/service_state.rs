//! A service's lifecycle state as the management API reports it.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// The state a service is in. `docs/api.md` describes each one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ServiceStateKind {
    /// Not running; starts on demand.
    Idle,
    /// Spawned and waiting for its health check.
    Starting,
    /// Alive and healthy.
    Running,
    /// Shutting down while in-flight requests finish.
    Draining,
    /// Exited cleanly.
    Stopped,
    /// Stopped to free memory for a higher-priority service.
    Evicted,
    /// Failed to launch; retrying with backoff.
    Failed,
    /// Will not start until enabled; `disabled_reason` says why.
    Disabled,
    /// No supervisor exists for the service, which happens briefly while a
    /// config reload registers it.
    Unknown,
}

impl ServiceStateKind {
    /// The state's wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Starting => "starting",
            Self::Running => "running",
            Self::Draining => "draining",
            Self::Stopped => "stopped",
            Self::Evicted => "evicted",
            Self::Failed => "failed",
            Self::Disabled => "disabled",
            Self::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for ServiceStateKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(self.as_str())
    }
}

/// Why a `disabled` service is disabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum DisableReason {
    /// The process failed to launch three times.
    LaunchFailed,
    /// The health check did not pass within its timeout.
    HealthTimeout,
    /// The process was OOM-killed twice shortly after spawning.
    Oom,
    /// An operator disabled the service.
    UserDisabled,
    /// Auto-restart tripped its flap cap.
    AutoRestartLoop,
}

impl DisableReason {
    /// The reason's wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LaunchFailed => "launch_failed",
            Self::HealthTimeout => "health_timeout",
            Self::Oom => "oom",
            Self::UserDisabled => "user_disabled",
            Self::AutoRestartLoop => "auto_restart_loop",
        }
    }

    /// Whether the daemon disabled the service because something went wrong,
    /// as opposed to an operator choosing to.
    pub fn is_fault(self) -> bool {
        !matches!(self, Self::UserDisabled)
    }
}

impl std::fmt::Display for DisableReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(self.as_str())
    }
}
