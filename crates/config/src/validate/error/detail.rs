//! Reason vocabularies for diagnostics that are not fields-level constraints:
//! duration parsing, merge and inheritance, placeholder substitution, and the
//! rendering detail a value diagnostic selects.

use std::fmt;

/// Parse error for duration strings.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum DurationParseError {
    /// The numeric portion failed to parse.
    InvalidNumber { input: String },
    /// The suffix is not recognised.
    UnrecognisedSuffix { input: String },
}

impl fmt::Display for DurationParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidNumber { input } => write!(f, "invalid number in `{input}`"),
            Self::UnrecognisedSuffix { input } => write!(f, "unrecognised duration: {input}"),
        }
    }
}

/// Structured reason for merge and inheritance failures.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum MergeReason {
    Cycle,
    ParentNotFound,
    ParentResolvedToNothing,
    ServiceNotFound,
    TemplateMismatch { child: String, parent: String },
    PortMustOverride,
    MigrationCycle,
    MissingNameDuringMigration,
}

impl fmt::Display for MergeReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cycle => write!(f, "extends cycle"),
            Self::ParentNotFound => write!(f, "parent does not exist"),
            Self::ParentResolvedToNothing => write!(f, "parent resolved to nothing"),
            Self::ServiceNotFound => {
                write!(f, "service not found during extends resolution")
            }
            Self::TemplateMismatch { child, parent } => {
                write!(
                    f,
                    "template `{child}` does not match parent's template `{parent}`; cross-template extends is not allowed"
                )
            }
            Self::PortMustOverride => write!(f, "must override port from parent"),
            Self::MigrationCycle => write!(f, "migrate_from cycle"),
            Self::MissingNameDuringMigration => {
                write!(f, "service without a name during migrate_from resolution")
            }
        }
    }
}

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

/// Typed detail for value diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueDiagnosticDetail {
    /// The value has no specialized rendering rule.
    Generic,
    /// The private-port range has inverted bounds.
    PrivatePortRangeInvalid,
    /// A tracking cgroup path is empty.
    TrackingEmpty,
    /// A tracking cgroup path is relative.
    TrackingNotAbsolute,
    /// A tracking cgroup path ends with a slash.
    TrackingTrailingSlash,
    /// A tracking cgroup path contains unsupported characters.
    TrackingInvalidCharacters,
}
