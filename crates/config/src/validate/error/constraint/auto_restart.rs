//! Constraint violations in the `auto_restart` watchdog settings.

use std::fmt;

/// Structured reason for a auto-restart constraint violation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum AutoRestartReason {
    PeriodicNeedsInterval,
    SpecCollapseRequiresSpecType,
    PeriodicModeInvalid { value: String },
    SpecCollapseWindowZero,
    SpecCollapseMinDraftTokensZero,
    SpecCollapsePollIntervalZero,
    GenerationStallTimeoutZero,
    GenerationStallPollIntervalZero,
    TtftStallTimeoutZero,
    ErrorRateOutOfRange { value: String },
    ErrorStatusClassInvalid { value: String },
    PeriodicMissingInterval,
}

impl fmt::Display for AutoRestartReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PeriodicNeedsInterval => {
                write!(
                    f,
                    "auto_restart.periodic = true needs an interval; write `periodic = {{ interval = \"6h\" }}`"
                )
            }
            Self::SpecCollapseRequiresSpecType => {
                write!(
                    f,
                    "auto_restart.spec_collapse requires spec_type to be set (without speculative decoding, responses carry no draft counts and the watchdog can never fire)"
                )
            }
            Self::PeriodicModeInvalid { value } => {
                write!(
                    f,
                    "auto_restart.periodic.mode must be `immediate`, `on-idle`, or `on-request`, got `{value}`"
                )
            }
            Self::SpecCollapseWindowZero => {
                write!(
                    f,
                    "auto_restart.spec_collapse.window must be greater than zero"
                )
            }
            Self::SpecCollapseMinDraftTokensZero => {
                write!(
                    f,
                    "auto_restart.spec_collapse.min_draft_tokens must be greater than zero"
                )
            }
            Self::SpecCollapsePollIntervalZero => {
                write!(
                    f,
                    "auto_restart.spec_collapse.poll_interval must be greater than zero"
                )
            }
            Self::GenerationStallTimeoutZero => {
                write!(
                    f,
                    "auto_restart.generation_stall.timeout must be greater than zero"
                )
            }
            Self::GenerationStallPollIntervalZero => {
                write!(
                    f,
                    "auto_restart.generation_stall.poll_interval must be greater than zero"
                )
            }
            Self::TtftStallTimeoutZero => {
                write!(
                    f,
                    "auto_restart.ttft_stall.timeout must be greater than zero"
                )
            }
            Self::ErrorRateOutOfRange { value } => {
                write!(
                    f,
                    "auto_restart.error_rate.max_error_rate must be in (0.0, 1.0], got {value}"
                )
            }
            Self::ErrorStatusClassInvalid { value } => {
                write!(
                    f,
                    "auto_restart.error_rate.error_statuses must be `5xx` or `4xx+5xx`, got `{value}`"
                )
            }
            Self::PeriodicMissingInterval => {
                write!(f, "auto_restart.periodic requires an `interval`")
            }
        }
    }
}
