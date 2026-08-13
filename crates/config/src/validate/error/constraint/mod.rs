//! The reason a fields-level constraint was violated.
//!
//! Split by validator family, mirroring the modules under `validate/` that
//! raise them. Reasons that belong to no family — one-off daemon, metadata,
//! filter, and port-pool rules — sit directly on [`ConstraintReason`].

mod allocation;
mod auto_restart;
mod command;
mod llama_cpp;
mod service;

use std::fmt;

pub use allocation::AllocationReason;
pub use auto_restart::AutoRestartReason;
pub use command::CommandReason;
pub use llama_cpp::LlamaCppReason;
pub use service::ServiceReason;

use crate::validate::error::DurationParseError;

/// Structured reason for a fields-level constraint violation.
///
/// Replaces the free-form `reason: String` so consumers can match on the
/// specific rule that failed rather than substring-searching a rendered message.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum ConstraintReason {
    LlamaCpp(LlamaCppReason),
    Command(CommandReason),
    Service(ServiceReason),
    AutoRestart(AutoRestartReason),
    Allocation(AllocationReason),
    DurationParse(DurationParseError),
    DaemonNonLoopbackWithoutFlag,
    MetadataInvalid {
        field: String,
        error: String,
    },
    FilterSetParamsInvalid {
        key: String,
        error: String,
    },
    PrivatePortExhausted {
        range_start: u16,
        range_end: u16,
        width: u32,
    },
}

impl fmt::Display for ConstraintReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LlamaCpp(reason) => write!(f, "{reason}"),
            Self::Command(reason) => write!(f, "{reason}"),
            Self::Service(reason) => write!(f, "{reason}"),
            Self::AutoRestart(reason) => write!(f, "{reason}"),
            Self::Allocation(reason) => write!(f, "{reason}"),
            Self::DurationParse(error) => write!(f, "{error}"),
            Self::DaemonNonLoopbackWithoutFlag => {
                write!(
                    f,
                    "daemon.management_listen is non-loopback but daemon.allow_external_management is false; the management API has no authentication"
                )
            }
            Self::MetadataInvalid { field, error } => write!(f, "{field}: {error}"),
            Self::FilterSetParamsInvalid { key, error } => {
                write!(f, "filters.set_params[{key}]: {error}")
            }
            Self::PrivatePortExhausted {
                range_start,
                range_end,
                width,
            } => {
                write!(
                    f,
                    "private_port_range [{range_start}, {range_end}] exhausted ({width} slots) — widen the range or reduce service count"
                )
            }
        }
    }
}

impl From<LlamaCppReason> for ConstraintReason {
    fn from(reason: LlamaCppReason) -> Self {
        Self::LlamaCpp(reason)
    }
}

impl From<CommandReason> for ConstraintReason {
    fn from(reason: CommandReason) -> Self {
        Self::Command(reason)
    }
}

impl From<ServiceReason> for ConstraintReason {
    fn from(reason: ServiceReason) -> Self {
        Self::Service(reason)
    }
}

impl From<AutoRestartReason> for ConstraintReason {
    fn from(reason: AutoRestartReason) -> Self {
        Self::AutoRestart(reason)
    }
}

impl From<AllocationReason> for ConstraintReason {
    fn from(reason: AllocationReason) -> Self {
        Self::Allocation(reason)
    }
}

impl From<DurationParseError> for ConstraintReason {
    fn from(error: DurationParseError) -> Self {
        Self::DurationParse(error)
    }
}
