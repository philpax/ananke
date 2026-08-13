//! Typed placeholder substitution failures, the one error vocabulary that
//! stays structured rather than being formatted into a diagnostic's message
//! at its construction site (its `Display` is reused directly there).

use std::fmt;

/// Typed placeholder substitution failures owned by the config domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlaceholderError {
    /// `{reserve_mb}` was used with a dynamic allocation.
    ReserveMbOnDynamic,
    /// `{reserve_mb}` was used with multiple static devices.
    ReserveMbMultiDevice,
    /// A placeholder name is not recognized.
    UnknownPlaceholder(String),
    /// `{args}` was embedded in a launcher argument.
    SplatInsideArg,
}

impl fmt::Display for PlaceholderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReserveMbOnDynamic => {
                write!(f, "{{reserve_mb}} is invalid with a dynamic allocation")
            }
            Self::ReserveMbMultiDevice => write!(
                f,
                "{{reserve_mb}} is valid only with a single-device static allocation"
            ),
            Self::UnknownPlaceholder(name) => write!(f, "unknown placeholder {{{name}}}"),
            Self::SplatInsideArg => write!(
                f,
                "splat placeholder {{args}} must be the entire launcher entry, not embedded"
            ),
        }
    }
}
