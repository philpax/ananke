//! Service state machine.

// The wire enum is the internal one: every reason the supervisor records is
// one a client can see, so there is nothing to map and nothing to drift.
pub use ananke_api::shared::DisableReason;
use ananke_api::shared::ServiceStateKind;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ServiceState {
    Starting,
    Running,
    Draining,
    #[default]
    Idle,
    Stopped,
    Evicted,
    Failed {
        retry_count: u8,
    },
    Disabled {
        reason: DisableReason,
    },
}

impl ServiceState {
    pub fn name(&self) -> &'static str {
        self.kind().as_str()
    }

    /// The state as the management API reports it.
    pub fn kind(&self) -> ServiceStateKind {
        match self {
            ServiceState::Idle => ServiceStateKind::Idle,
            ServiceState::Starting => ServiceStateKind::Starting,
            ServiceState::Running => ServiceStateKind::Running,
            ServiceState::Draining => ServiceStateKind::Draining,
            ServiceState::Stopped => ServiceStateKind::Stopped,
            ServiceState::Evicted => ServiceStateKind::Evicted,
            ServiceState::Failed { .. } => ServiceStateKind::Failed,
            ServiceState::Disabled { .. } => ServiceStateKind::Disabled,
        }
    }

    /// Why the service is disabled, when it is.
    pub fn disabled_reason(&self) -> Option<DisableReason> {
        match self {
            ServiceState::Disabled { reason } => Some(*reason),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    SpawnRequested,
    HealthPassed,
    DrainRequested,
    DrainComplete,
    Stopped,
    LaunchFailed,
    HealthTimedOut,
    UserEnable,
    UserDisable,
    RetryAfterBackoff,
    /// Auto-restart tripped its flap cap; the running service is disabled
    /// instead of restarted again.
    AutoRestartLoop,
}

/// Returns the next state for this `(from, event)` pair.
///
/// Panics if the pair is not in the transition table. Callers must only pass
/// `(from, event)` combinations they know are legal given the supervisor's
/// current phase; passing an illegal pair is a programming bug that the
/// exhaustive panic surfaces loudly rather than silently swallowing.
pub fn transition(from: &ServiceState, event: Event) -> ServiceState {
    try_transition(from, event).unwrap_or_else(|| {
        panic!(
            "illegal state transition: from={from:?} event={event:?} (this is a programming bug)"
        )
    })
}

/// Returns the next state if the transition is defined, else `None`.
///
/// Prefer [`transition`] at call sites where the pair is known legal. Use this
/// only in genuinely fallible paths (validation, tests, or where an "illegal"
/// pair is recoverable with an explicit fallback).
pub fn try_transition(from: &ServiceState, event: Event) -> Option<ServiceState> {
    use ServiceState::*;
    match (from, event) {
        (Idle, Event::SpawnRequested) => Some(Starting),
        (Starting, Event::HealthPassed) => Some(Running),
        (Starting, Event::LaunchFailed) => Some(Failed { retry_count: 0 }),
        (Running, Event::DrainRequested) => Some(Draining),
        (Running, Event::Stopped) => Some(Stopped),
        (Draining, Event::DrainComplete) => Some(Idle),
        (Draining, Event::Stopped) => Some(Stopped),
        (Stopped, Event::SpawnRequested) => Some(Starting),
        (Failed { retry_count }, Event::RetryAfterBackoff) => {
            if *retry_count >= 2 {
                Some(Disabled {
                    reason: DisableReason::LaunchFailed,
                })
            } else {
                Some(Failed {
                    retry_count: retry_count + 1,
                })
            }
        }
        (Starting, Event::HealthTimedOut) => Some(Disabled {
            reason: DisableReason::HealthTimeout,
        }),
        (Running, Event::AutoRestartLoop) => Some(Disabled {
            reason: DisableReason::AutoRestartLoop,
        }),
        (Disabled { .. }, Event::UserEnable) => Some(Idle),
        (_, Event::UserDisable) => Some(Disabled {
            reason: DisableReason::UserDisabled,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_to_starting_on_spawn() {
        assert_eq!(
            transition(&ServiceState::Idle, Event::SpawnRequested),
            ServiceState::Starting,
        );
    }

    #[test]
    fn starting_to_running_on_health() {
        assert_eq!(
            transition(&ServiceState::Starting, Event::HealthPassed),
            ServiceState::Running,
        );
    }

    #[test]
    fn starting_health_timeout_disables_with_reason() {
        assert_eq!(
            transition(&ServiceState::Starting, Event::HealthTimedOut),
            ServiceState::Disabled {
                reason: DisableReason::HealthTimeout,
            },
        );
    }

    #[test]
    fn failed_retries_up_to_three_times_then_disables() {
        let s0 = ServiceState::Failed { retry_count: 0 };
        let s1 = transition(&s0, Event::RetryAfterBackoff);
        assert_eq!(s1, ServiceState::Failed { retry_count: 1 });
        let s2 = transition(&s1, Event::RetryAfterBackoff);
        assert_eq!(s2, ServiceState::Failed { retry_count: 2 });
        let s3 = transition(&s2, Event::RetryAfterBackoff);
        assert_eq!(
            s3,
            ServiceState::Disabled {
                reason: DisableReason::LaunchFailed,
            },
        );
    }

    #[test]
    fn invalid_transition_returns_none() {
        assert!(try_transition(&ServiceState::Idle, Event::DrainComplete).is_none());
    }

    #[test]
    #[should_panic(expected = "illegal state transition")]
    fn invalid_transition_panics_in_total_api() {
        let _ = transition(&ServiceState::Idle, Event::DrainComplete);
    }

    #[test]
    fn running_auto_restart_loop_disables_with_reason() {
        assert_eq!(
            transition(&ServiceState::Running, Event::AutoRestartLoop),
            ServiceState::Disabled {
                reason: DisableReason::AutoRestartLoop,
            },
        );
    }

    #[test]
    fn wire_state_carries_the_reason_only_when_disabled() {
        let cases = [
            (ServiceState::Idle, ServiceStateKind::Idle, None),
            (ServiceState::Starting, ServiceStateKind::Starting, None),
            (ServiceState::Running, ServiceStateKind::Running, None),
            (ServiceState::Draining, ServiceStateKind::Draining, None),
            (ServiceState::Stopped, ServiceStateKind::Stopped, None),
            (ServiceState::Evicted, ServiceStateKind::Evicted, None),
            (
                ServiceState::Failed { retry_count: 1 },
                ServiceStateKind::Failed,
                None,
            ),
            (
                ServiceState::Disabled {
                    reason: DisableReason::Oom,
                },
                ServiceStateKind::Disabled,
                Some(DisableReason::Oom),
            ),
        ];
        for (state, kind, reason) in cases {
            assert_eq!(state.kind(), kind, "{state:?}");
            assert_eq!(state.disabled_reason(), reason, "{state:?}");
            assert_eq!(state.name(), kind.as_str(), "{state:?}");
        }
    }

    #[test]
    fn disabled_can_be_re_enabled() {
        let s = ServiceState::Disabled {
            reason: DisableReason::HealthTimeout,
        };
        assert_eq!(transition(&s, Event::UserEnable), ServiceState::Idle);
    }
}
