//! Accumulation of diagnostics across the parse, merge, and validate stages,
//! and the error type the pipeline surfaces to callers.

use std::{fmt, path::PathBuf};

use crate::validate::error::ConfigDiagnostic;

/// Ordered accumulation of configuration diagnostics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConfigDiagnosticReport {
    diagnostics: Vec<ConfigDiagnostic>,
}

impl ConfigDiagnosticReport {
    /// Construct an empty report.
    pub const fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
        }
    }

    /// Append one diagnostic.
    pub fn push(&mut self, diagnostic: ConfigDiagnostic) {
        self.diagnostics.push(diagnostic);
    }

    /// Append all diagnostics from another report.
    pub fn extend(&mut self, other: Self) {
        self.diagnostics.extend(other.diagnostics);
    }

    /// Return whether no diagnostics were collected.
    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }

    /// Return the number of diagnostics.
    pub fn len(&self) -> usize {
        self.diagnostics.len()
    }

    /// Borrow the ordered diagnostics.
    pub fn as_slice(&self) -> &[ConfigDiagnostic] {
        &self.diagnostics
    }

    /// Consume the report into its ordered diagnostics.
    pub fn into_vec(self) -> Vec<ConfigDiagnostic> {
        self.diagnostics
    }

    /// Place service diagnostics after global diagnostics in original source order.
    pub fn sort_by_source_index(&mut self) {
        self.diagnostics.sort_by_key(|diagnostic| {
            (
                diagnostic.source_index.is_some(),
                diagnostic.source_index.unwrap_or(usize::MAX),
            )
        });
    }

    /// Convert the report to the compatibility startup error.
    pub fn into_expected_error(self, origin: PathBuf) -> ananke_errors::ExpectedError {
        let cause = self
            .diagnostics
            .iter()
            .map(|diagnostic| {
                if let Some(location) = &diagnostic.location {
                    format!("{}:{}: {}", location.line, location.column, diagnostic)
                } else {
                    diagnostic.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("; ");
        ananke_errors::ExpectedError::config_unparseable(origin, cause)
    }
}

impl IntoIterator for ConfigDiagnosticReport {
    type Item = ConfigDiagnostic;
    type IntoIter = std::vec::IntoIter<ConfigDiagnostic>;
    fn into_iter(self) -> Self::IntoIter {
        self.diagnostics.into_iter()
    }
}

impl fmt::Display for ConfigDiagnosticReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, diagnostic) in self.diagnostics.iter().enumerate() {
            if index != 0 {
                write!(f, "; ")?;
            }
            write!(f, "{diagnostic}")?;
        }
        Ok(())
    }
}

/// Failure category for the full config pipeline.
#[derive(Debug)]
pub enum ConfigPipelineError {
    /// TOML parser failure(s).
    Parse(ConfigDiagnosticReport),
    /// Inheritance or migration failure(s).
    Merge(ConfigDiagnosticReport),
    /// Semantic validation failure(s).
    Validation(ConfigDiagnosticReport),
    /// Operational GGUF preflight failure.
    Preflight(ananke_errors::ExpectedError),
}

impl ConfigPipelineError {
    /// Consume a validation-phase error as its ordered report.
    ///
    /// This adapter is only valid for parse, merge, and semantic validation
    /// failures. Operational preflight remains an `ExpectedError` and must be
    /// handled by [`Self::into_expected_error`].
    pub fn into_report(self) -> ConfigDiagnosticReport {
        match self {
            Self::Parse(report) | Self::Merge(report) | Self::Validation(report) => report,
            Self::Preflight(error) => panic!("preflight is not a validation report: {error}"),
        }
    }

    /// Convert the pipeline result to the startup-facing operational error.
    pub fn into_expected_error(self, origin: PathBuf) -> ananke_errors::ExpectedError {
        match self {
            Self::Parse(report) | Self::Merge(report) | Self::Validation(report) => {
                report.into_expected_error(origin)
            }
            Self::Preflight(error) => error,
        }
    }
}

impl fmt::Display for ConfigPipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(report) => write!(f, "{report}"),
            Self::Merge(report) => write!(f, "{report}"),
            Self::Validation(report) => write!(f, "{report}"),
            Self::Preflight(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for ConfigPipelineError {}

impl From<ConfigDiagnostic> for ananke_errors::ExpectedError {
    fn from(diagnostic: ConfigDiagnostic) -> Self {
        ConfigDiagnosticReport {
            diagnostics: vec![diagnostic],
        }
        .into_expected_error(PathBuf::from("<config>"))
    }
}

impl From<ConfigDiagnostic> for ConfigDiagnosticReport {
    fn from(diagnostic: ConfigDiagnostic) -> Self {
        Self {
            diagnostics: vec![diagnostic],
        }
    }
}

impl From<ConfigDiagnosticReport> for ananke_errors::ExpectedError {
    fn from(report: ConfigDiagnosticReport) -> Self {
        report.into_expected_error(PathBuf::from("<config>"))
    }
}

#[cfg(test)]
mod tests {
    use ananke_api::config::validate::ValidationErrorCode;

    use super::*;

    #[test]
    fn report_preserves_order_and_converts_all_messages() {
        let mut report = ConfigDiagnosticReport::new();
        report.push(ConfigDiagnostic::value(
            ValidationErrorCode::ValueInvalid,
            "port",
            "port: invalid value `x` (expected an integer)",
        ));
        report.push(ConfigDiagnostic::parse("bad TOML", None));
        assert_eq!(report.len(), 2);
        assert!(report.to_string().contains("port"));
        assert!(
            report
                .into_expected_error(PathBuf::from("/tmp/config.toml"))
                .to_string()
                .contains("bad TOML")
        );
    }
}
