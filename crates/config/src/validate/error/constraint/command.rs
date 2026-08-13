//! Constraint violations specific to the command template.

use std::fmt;

/// Structured reason for a command template constraint violation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum CommandReason {
    MissingCommand,
    EmptyCommand,
    EmptyShutdownCommand,
    UpstreamModelEmpty,
}

impl fmt::Display for CommandReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingCommand => write!(f, "command template requires `command`"),
            Self::EmptyCommand => write!(f, "command is empty"),
            Self::EmptyShutdownCommand => {
                write!(f, "shutdown_command is present but empty")
            }
            Self::UpstreamModelEmpty => {
                write!(f, "openai_proxy.upstream_model must be a non-empty string")
            }
        }
    }
}
