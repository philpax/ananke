//! User-facing daemon errors with semantic exit codes.
//!
//! Internal programming errors use ad-hoc enums or panic.
//! `ExpectedError` is reserved for conditions where the daemon exits
//! non-zero and the user needs a clear message.

use std::fmt;
use std::path::PathBuf;

#[derive(Debug)]
pub struct ExpectedError {
    kind: ExpectedErrorKind,
}

#[derive(Debug)]
enum ExpectedErrorKind {
    BindFailed { addr: String, cause: String },
    ConfigUnparseable { path: PathBuf, cause: String },
    ConfigFileMissing { path: PathBuf },
    DatabaseOpenFailed { path: PathBuf, cause: String },
    NoDevices,
}

impl ExpectedError {
    pub fn bind_failed(addr: String, cause: String) -> Self {
        Self {
            kind: ExpectedErrorKind::BindFailed { addr, cause },
        }
    }

    pub fn config_unparseable(path: PathBuf, cause: String) -> Self {
        Self {
            kind: ExpectedErrorKind::ConfigUnparseable { path, cause },
        }
    }

    pub fn config_file_missing(path: PathBuf) -> Self {
        Self {
            kind: ExpectedErrorKind::ConfigFileMissing { path },
        }
    }

    pub fn database_open_failed(path: PathBuf, cause: String) -> Self {
        Self {
            kind: ExpectedErrorKind::DatabaseOpenFailed { path, cause },
        }
    }

    pub fn no_devices() -> Self {
        Self {
            kind: ExpectedErrorKind::NoDevices,
        }
    }

    pub fn exit_code(&self) -> u8 {
        match self.kind {
            ExpectedErrorKind::BindFailed { .. } => 2,
            ExpectedErrorKind::ConfigUnparseable { .. } => 3,
            ExpectedErrorKind::ConfigFileMissing { .. } => 3,
            ExpectedErrorKind::DatabaseOpenFailed { .. } => 5,
            ExpectedErrorKind::NoDevices => 4,
        }
    }
}

impl fmt::Display for ExpectedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ExpectedErrorKind::BindFailed { addr, cause } => {
                write!(f, "failed to bind {addr}: {cause}")
            }
            ExpectedErrorKind::ConfigUnparseable { path, cause } => {
                write!(f, "failed to parse config at {}: {cause}", path.display())
            }
            ExpectedErrorKind::ConfigFileMissing { path } => {
                write!(f, "config file not found at {}", path.display())
            }
            ExpectedErrorKind::DatabaseOpenFailed { path, cause } => {
                write!(f, "failed to open database at {}: {cause}", path.display())
            }
            ExpectedErrorKind::NoDevices => {
                write!(f, "no devices available: NVML and CPU probes both failed")
            }
        }
    }
}

impl std::error::Error for ExpectedError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_error_display_is_lowercase_fragment() {
        let err = ExpectedError::bind_failed("127.0.0.1:7777".into(), "permission denied".into());
        let msg = format!("{err}");
        assert_eq!(msg, "failed to bind 127.0.0.1:7777: permission denied");
        assert_eq!(err.exit_code(), 2);
    }

    #[test]
    fn config_error_kinds_distinguished() {
        let err =
            ExpectedError::config_unparseable("/tmp/x.toml".into(), "unexpected token".into());
        assert!(format!("{err}").contains("/tmp/x.toml"));
        assert_eq!(err.exit_code(), 3);
    }

    #[test]
    fn no_devices_exit_code_is_stable() {
        assert_eq!(ExpectedError::no_devices().exit_code(), 4);
    }
}
