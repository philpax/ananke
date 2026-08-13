//! Constraint violations in the `allocation` block.

use std::fmt;

/// Structured reason for a allocation constraint violation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum AllocationReason {
    StaticRequiresReserveGb,
    DynamicRequiresMinReserveGb,
    DynamicRequiresMaxReserveGb,
    MaxMustExceedMin,
    ModeUnknown { value: String },
    CommandRequiresMode,
}

impl fmt::Display for AllocationReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StaticRequiresReserveGb => {
                write!(f, "allocation.mode=static requires reserve_gb")
            }
            Self::DynamicRequiresMinReserveGb => {
                write!(f, "allocation.mode=dynamic requires min_reserve_gb")
            }
            Self::DynamicRequiresMaxReserveGb => {
                write!(f, "allocation.mode=dynamic requires max_reserve_gb")
            }
            Self::MaxMustExceedMin => {
                write!(f, "max_reserve_gb must be > min_reserve_gb")
            }
            Self::ModeUnknown { value } => {
                write!(f, "unknown allocation.mode `{value}`")
            }
            Self::CommandRequiresMode => {
                write!(
                    f,
                    "command template requires allocation.mode (static|dynamic)"
                )
            }
        }
    }
}
