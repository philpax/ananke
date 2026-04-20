//! Service state machine per spec §5.3.

use smol_str::SmolStr;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceState {
    Starting,
    Running,
    Draining,
    Idle,
    Stopped,
    Evicted,
    Failed { retry_count: u8 },
    Disabled { reason: DisableReason },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DisableReason {
    ConfigError(SmolStr),
    LaunchFailed,
    HealthTimeout,
    Oom,
    CrashLoop,
    NoFit,
    UserDisabled,
}

impl ServiceState {
    pub fn name(&self) -> &'static str {
        match self {
            ServiceState::Idle => "idle",
            ServiceState::Starting => "starting",
            ServiceState::Running => "running",
            ServiceState::Draining => "draining",
            ServiceState::Stopped => "stopped",
            ServiceState::Evicted => "evicted",
            ServiceState::Failed { .. } => "failed",
            ServiceState::Disabled { .. } => "disabled",
        }
    }
}

impl DisableReason {
    pub fn as_str(&self) -> &str {
        match self {
            DisableReason::ConfigError(_) => "config_error",
            DisableReason::LaunchFailed => "launch_failed",
            DisableReason::HealthTimeout => "health_timeout",
            DisableReason::Oom => "oom",
            DisableReason::CrashLoop => "crash_loop",
            DisableReason::NoFit => "no_fit",
            DisableReason::UserDisabled => "user_disabled",
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
    CrashLoop,
    UserEnable,
    UserDisable,
    RetryAfterBackoff,
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
        (Running, Event::CrashLoop) => Some(Disabled {
            reason: DisableReason::CrashLoop,
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
    fn disabled_can_be_re_enabled() {
        let s = ServiceState::Disabled {
            reason: DisableReason::HealthTimeout,
        };
        assert_eq!(transition(&s, Event::UserEnable), ServiceState::Idle);
    }
}
