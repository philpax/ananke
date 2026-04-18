//! Service state machine per spec §5.3.

use smol_str::SmolStr;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceState {
    Starting,
    Warming,
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
    WarmingComplete,
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

/// Returns the next state if the transition is valid, else `None`.
pub fn transition(from: &ServiceState, event: Event) -> Option<ServiceState> {
    use ServiceState::*;
    match (from, event) {
        (Idle, Event::SpawnRequested) => Some(Starting),
        (Starting, Event::HealthPassed) => Some(Warming),
        (Starting, Event::LaunchFailed) => Some(Failed { retry_count: 0 }),
        (Warming, Event::WarmingComplete) => Some(Running),
        (Warming, Event::HealthTimedOut) => Some(Disabled {
            reason: DisableReason::HealthTimeout,
        }),
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
        (Running | Warming, Event::CrashLoop) => Some(Disabled {
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
        assert!(matches!(
            transition(&ServiceState::Idle, Event::SpawnRequested),
            Some(ServiceState::Starting)
        ));
    }

    #[test]
    fn starting_to_warming_on_health() {
        assert!(matches!(
            transition(&ServiceState::Starting, Event::HealthPassed),
            Some(ServiceState::Warming)
        ));
    }

    #[test]
    fn warming_health_timeout_disables_with_reason() {
        let s = transition(&ServiceState::Warming, Event::HealthTimedOut).unwrap();
        assert!(matches!(
            s,
            ServiceState::Disabled {
                reason: DisableReason::HealthTimeout
            }
        ));
    }

    #[test]
    fn failed_retries_up_to_three_times_then_disables() {
        let s0 = ServiceState::Failed { retry_count: 0 };
        let s1 = transition(&s0, Event::RetryAfterBackoff).unwrap();
        assert_eq!(s1, ServiceState::Failed { retry_count: 1 });
        let s2 = transition(&s1, Event::RetryAfterBackoff).unwrap();
        assert_eq!(s2, ServiceState::Failed { retry_count: 2 });
        let s3 = transition(&s2, Event::RetryAfterBackoff).unwrap();
        assert!(matches!(
            s3,
            ServiceState::Disabled {
                reason: DisableReason::LaunchFailed
            }
        ));
    }

    #[test]
    fn invalid_transition_returns_none() {
        assert!(transition(&ServiceState::Idle, Event::DrainComplete).is_none());
    }

    #[test]
    fn disabled_can_be_re_enabled() {
        let s = ServiceState::Disabled {
            reason: DisableReason::HealthTimeout,
        };
        assert_eq!(transition(&s, Event::UserEnable), Some(ServiceState::Idle));
    }
}
