//! Structured diagnostics produced by the configuration parse, merge, and
//! semantic-validation pipeline.
//!
//! [`ConfigDiagnostic`] is the single diagnostic type. Its payload is
//! [`ConfigDiagnosticKind`], which owns the stable [`ValidationErrorCode`] and
//! every fact a renderer or boundary adapter needs, so nothing downstream has
//! to parse a rendered message.

mod detail;
mod location;
mod report;

use std::fmt;

use ananke_api::config::validate::{ValidationError, ValidationErrorCode, ValidationLocation};
pub use detail::{DurationParseError, MergeReason, PlaceholderError, ValueDiagnosticDetail};
pub use location::{DiagnosticLocation, byte_offset_to_line_column};
pub use report::{ConfigDiagnosticReport, ConfigPipelineError};

/// The complete typed domain diagnostic. Its variants own the stable code and
/// every fact needed by renderers and boundary adapters.
// The associated-data variants are public for typed adapters, but their fields
// are fully described by the variant-level API and mirrored by `context()`.
#[expect(
    missing_docs,
    reason = "associated diagnostic payload fields are documented by context()"
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigDiagnosticKind {
    /// The parser rejected the TOML source.
    Parse { parser_message: String },
    /// Inheritance or migration resolution failed.
    Merge {
        service: Option<String>,
        index: Option<usize>,
        parent: Option<String>,
        reason: MergeReason,
    },
    /// A field-level constraint failed.
    Field {
        code: ValidationErrorCode,
        field: String,
        offending: Option<String>,
        expected: Option<String>,
    },
    /// A scalar value failed validation.
    Value {
        code: ValidationErrorCode,
        detail: ValueDiagnosticDetail,
        field: String,
        offending: String,
        expected: Option<String>,
    },
    /// A collection count failed validation.
    Count {
        code: ValidationErrorCode,
        field: String,
        got: usize,
        expected: usize,
    },
    /// An indexed collection value failed validation.
    Index {
        code: ValidationErrorCode,
        field: String,
        index: usize,
        value: String,
        expected: Option<String>,
    },
    /// Multiple fields participate in one constraint.
    Fields {
        code: ValidationErrorCode,
        fields: Vec<String>,
        service: Option<String>,
        message: String,
    },
    /// Placeholder substitution failed.
    Placeholder {
        code: ValidationErrorCode,
        service: Option<String>,
        field: String,
        argv_index: Option<usize>,
        argument: Option<String>,
        error: PlaceholderError,
    },
    /// A service identity or location was invalid.
    Service {
        code: ValidationErrorCode,
        service: Option<String>,
        index: Option<usize>,
        field: Option<String>,
    },
}

/// One typed configuration diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigDiagnostic {
    /// The complete typed diagnostic payload.
    pub kind: Box<ConfigDiagnosticKind>,
    /// Optional source location from the parser.
    pub location: Option<DiagnosticLocation>,
    /// Original zero-based service index, when the diagnostic belongs to a service.
    pub source_index: Option<usize>,
    /// Owning service name, when the diagnostic belongs to a named service.
    ///
    /// Attached by the orchestrator rather than by the individual validators, so
    /// that a diagnostic raised before the name is read — a missing `port`, say —
    /// still reaches the operator attributed to the block it came from.
    pub service: Option<String>,
}

impl ConfigDiagnostic {
    /// Construct a diagnostic from a typed payload and optional source location.
    pub fn new(kind: ConfigDiagnosticKind, location: Option<DiagnosticLocation>) -> Self {
        Self {
            kind: Box::new(kind),
            location,
            source_index: None,
            service: None,
        }
    }

    /// Attach the owning `[[service]]` block's source index and name.
    pub fn with_service_context(mut self, source_index: usize, service: Option<&str>) -> Self {
        self.source_index = Some(source_index);
        if self.service.is_none() {
            self.service = service.map(str::to_owned);
        }
        self
    }

    /// Construct a parser diagnostic, retaining the parser's original message.
    pub fn parse(message: impl Into<String>, location: Option<DiagnosticLocation>) -> Self {
        Self::new(
            ConfigDiagnosticKind::Parse {
                parser_message: message.into(),
            },
            location,
        )
    }

    /// Construct a merge diagnostic.
    pub fn merge(
        service: Option<String>,
        index: Option<usize>,
        parent: Option<String>,
        reason: MergeReason,
    ) -> Self {
        Self::new(
            ConfigDiagnosticKind::Merge {
                service,
                index,
                parent,
                reason,
            },
            None,
        )
    }

    /// Construct a field-value diagnostic.
    pub fn value(
        code: ValidationErrorCode,
        field: impl Into<String>,
        offending: impl Into<String>,
        expected: Option<String>,
    ) -> Self {
        Self::value_with_detail(
            code,
            ValueDiagnosticDetail::Generic,
            field,
            offending,
            expected,
        )
    }

    /// Construct a value diagnostic with typed rendering detail.
    pub fn value_with_detail(
        code: ValidationErrorCode,
        detail: ValueDiagnosticDetail,
        field: impl Into<String>,
        offending: impl Into<String>,
        expected: Option<String>,
    ) -> Self {
        Self::new(
            ConfigDiagnosticKind::Value {
                code,
                detail,
                field: field.into(),
                offending: offending.into(),
                expected,
            },
            None,
        )
    }

    /// Construct a collection-count diagnostic.
    pub fn count(
        code: ValidationErrorCode,
        field: impl Into<String>,
        got: usize,
        expected: usize,
    ) -> Self {
        Self::new(
            ConfigDiagnosticKind::Count {
                code,
                field: field.into(),
                got,
                expected,
            },
            None,
        )
    }

    /// Construct an indexed collection diagnostic.
    pub fn index(
        code: ValidationErrorCode,
        field: impl Into<String>,
        index: usize,
        value: impl Into<String>,
        expected: Option<String>,
    ) -> Self {
        Self::new(
            ConfigDiagnosticKind::Index {
                code,
                field: field.into(),
                index,
                value: value.into(),
                expected,
            },
            None,
        )
    }

    /// Construct a constraint diagnostic with explicit stable code and fields.
    pub fn constraint(
        code: ValidationErrorCode,
        service: Option<String>,
        fields: &[&str],
        message: String,
    ) -> Self {
        Self::new(
            ConfigDiagnosticKind::Fields {
                code,
                fields: fields.iter().map(|field| (*field).to_string()).collect(),
                service,
                message,
            },
            None,
        )
    }

    /// Construct a placeholder diagnostic.
    pub fn placeholder(
        service: Option<String>,
        field: impl Into<String>,
        argv_index: Option<usize>,
        argument: Option<String>,
        error: PlaceholderError,
    ) -> Self {
        Self::new(
            ConfigDiagnosticKind::Placeholder {
                code: ValidationErrorCode::PlaceholderInvalid,
                service,
                field: field.into(),
                argv_index,
                argument,
                error,
            },
            None,
        )
    }

    /// Return the stable code derived from the typed payload.
    pub const fn code(&self) -> ValidationErrorCode {
        match &*self.kind {
            ConfigDiagnosticKind::Parse { .. } => ValidationErrorCode::Parse,
            ConfigDiagnosticKind::Merge { .. } => ValidationErrorCode::MergeConstraint,
            ConfigDiagnosticKind::Field { code, .. }
            | ConfigDiagnosticKind::Value { code, .. }
            | ConfigDiagnosticKind::Count { code, .. }
            | ConfigDiagnosticKind::Index { code, .. }
            | ConfigDiagnosticKind::Fields { code, .. }
            | ConfigDiagnosticKind::Placeholder { code, .. }
            | ConfigDiagnosticKind::Service { code, .. } => *code,
        }
    }

    /// Return the field path, when the diagnostic has one.
    pub fn path(&self) -> Option<&str> {
        match &*self.kind {
            ConfigDiagnosticKind::Field { field, .. }
            | ConfigDiagnosticKind::Value { field, .. }
            | ConfigDiagnosticKind::Count { field, .. }
            | ConfigDiagnosticKind::Index { field, .. }
            | ConfigDiagnosticKind::Placeholder { field, .. } => Some(field),
            ConfigDiagnosticKind::Service { field, .. } => field.as_deref(),
            ConfigDiagnosticKind::Parse { .. }
            | ConfigDiagnosticKind::Merge { .. }
            | ConfigDiagnosticKind::Fields { .. } => None,
        }
    }

    /// The field paths a constraint diagnostic names, empty for other kinds.
    ///
    /// This is the structural discriminator: a test asserts on the path the
    /// rule rejected rather than on the sentence it rendered.
    pub fn fields(&self) -> &[String] {
        match &*self.kind {
            ConfigDiagnosticKind::Fields { fields, .. } => fields,
            _ => &[],
        }
    }

    /// Return the owning service name, preferring the one the payload carries.
    pub fn service_name(&self) -> Option<&str> {
        self.embedded_service().or(self.service.as_deref())
    }

    /// The service name the payload renders itself, if any.
    ///
    /// Kinds listed here already write `service {name}` in their own `Display`,
    /// so the shared prefix must not repeat it.
    fn embedded_service(&self) -> Option<&str> {
        match &*self.kind {
            ConfigDiagnosticKind::Merge { service, .. }
            | ConfigDiagnosticKind::Fields { service, .. }
            | ConfigDiagnosticKind::Placeholder { service, .. }
            | ConfigDiagnosticKind::Service { service, .. } => service.as_deref(),
            ConfigDiagnosticKind::Parse { .. }
            | ConfigDiagnosticKind::Field { .. }
            | ConfigDiagnosticKind::Value { .. }
            | ConfigDiagnosticKind::Count { .. }
            | ConfigDiagnosticKind::Index { .. } => None,
        }
    }

    /// Whether the payload writes its own service identity in `Display`.
    const fn renders_own_service(&self) -> bool {
        matches!(
            &*self.kind,
            ConfigDiagnosticKind::Merge { .. }
                | ConfigDiagnosticKind::Fields { .. }
                | ConfigDiagnosticKind::Placeholder { .. }
                | ConfigDiagnosticKind::Service { .. }
        )
    }
}

impl From<ConfigDiagnostic> for ValidationError {
    fn from(diagnostic: ConfigDiagnostic) -> Self {
        Self {
            code: diagnostic.code(),
            message: diagnostic.to_string(),
            path: diagnostic.path().map(str::to_owned),
            service: diagnostic.service_name().map(str::to_owned),
            service_index: diagnostic.source_index,
            location: diagnostic
                .location
                .as_ref()
                .map(|location| ValidationLocation {
                    start: location.start,
                    end: location.end,
                    line: location.line,
                    column: location.column,
                }),
        }
    }
}

impl fmt::Display for ConfigDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.renders_own_service() {
            match (self.service.as_deref(), self.source_index) {
                (Some(service), _) => write!(f, "service {service}: ")?,
                (None, Some(index)) => write!(f, "service[{index}]: ")?,
                (None, None) => {}
            }
        }
        match &*self.kind {
            ConfigDiagnosticKind::Parse { parser_message } => {
                write!(f, "parse error: {parser_message}")
            }
            ConfigDiagnosticKind::Merge {
                service,
                parent,
                reason,
                ..
            } => match (service, parent) {
                (Some(service), Some(parent)) => {
                    write!(f, "service {service} extends {parent}: {reason}")
                }
                (Some(service), None) => write!(f, "service {service}: {reason}"),
                _ => write!(f, "{reason}"),
            },
            ConfigDiagnosticKind::Field {
                field,
                offending,
                expected,
                ..
            } => {
                write!(f, "{field}: invalid field")?;
                if let Some(offending) = offending {
                    write!(f, " `{offending}`")?;
                }
                if let Some(expected) = expected {
                    write!(f, " (expected {expected})")?;
                }
                Ok(())
            }
            ConfigDiagnosticKind::Value {
                code,
                detail,
                field,
                offending,
                expected,
            } => match detail {
                ValueDiagnosticDetail::PrivatePortRangeInvalid => write!(
                    f,
                    "daemon.private_port_end must exceed daemon.private_port_start (got {offending})"
                ),
                ValueDiagnosticDetail::TrackingEmpty => write!(
                    f,
                    "tracking.cgroup_parent is empty — omit the field or supply a non-empty cgroup path"
                ),
                ValueDiagnosticDetail::TrackingNotAbsolute => write!(
                    f,
                    "tracking.cgroup_parent must be an absolute cgroup v2 path starting with `/` (got `{offending}`)"
                ),
                ValueDiagnosticDetail::TrackingTrailingSlash => write!(
                    f,
                    "tracking.cgroup_parent must not end with `/` (got `{offending}`)"
                ),
                ValueDiagnosticDetail::TrackingInvalidCharacters => write!(
                    f,
                    "tracking.cgroup_parent may only contain alphanumeric, `.`, `_`, `/`, and `-` characters (got `{offending}`)"
                ),
                ValueDiagnosticDetail::Generic => match code {
                    ValidationErrorCode::GpuAllowDuplicate => write!(
                        f,
                        "service: {field} must not contain duplicate GPU ids when tensor_split_weights is set"
                    ),
                    ValidationErrorCode::GpuAllowUnsorted => write!(
                        f,
                        "service: {field} must be in ascending GPU-id order when tensor_split_weights is set (got {offending})"
                    ),
                    ValidationErrorCode::ServiceNameDuplicate => {
                        write!(f, "duplicate service name `{offending}`")
                    }
                    ValidationErrorCode::ServicePortDuplicate => {
                        write!(f, "duplicate service port {offending}")
                    }
                    ValidationErrorCode::ServicePortManagementCollision => write!(
                        f,
                        "service port {offending} collides with daemon.management_listen"
                    ),
                    ValidationErrorCode::PrivatePortExhausted => write!(f, "{offending}"),
                    _ => {
                        write!(f, "{field}: invalid value `{offending}`")?;
                        if let Some(expected) = expected {
                            write!(f, " (expected {expected})")?;
                        }
                        Ok(())
                    }
                },
            },
            ConfigDiagnosticKind::Count {
                code,
                field,
                got,
                expected,
            } => {
                if *code == ValidationErrorCode::TensorSplitWeightsCount {
                    write!(
                        f,
                        "service: {field} has {got} entries but {expected} allowed GPU(s) (set via gpu_allow or [devices].gpu_ids)"
                    )
                } else {
                    write!(f, "{field}: got {got} entries, expected {expected}")
                }
            }
            ConfigDiagnosticKind::Index {
                code,
                field,
                index,
                value,
                expected,
            } => {
                if *code == ValidationErrorCode::TensorSplitWeightInvalid {
                    return write!(
                        f,
                        "service: {field}[{index}] must be a positive finite number, got {value}"
                    );
                }
                write!(f, "{field}[{index}]: invalid value `{value}`")?;
                if let Some(expected) = expected {
                    write!(f, " (expected {expected})")?;
                }
                Ok(())
            }
            ConfigDiagnosticKind::Fields {
                service, message, ..
            } => {
                if let Some(service) = service {
                    write!(f, "service {service}: ")?;
                }
                write!(f, "{message}")
            }
            ConfigDiagnosticKind::Placeholder {
                service,
                field,
                argv_index,
                argument,
                error,
                ..
            } => {
                if let Some(service) = service {
                    write!(f, "service {service} ")?;
                }
                write!(f, "{field}")?;
                if let Some(index) = argv_index {
                    write!(f, "[{index}]")?;
                }
                if let Some(argument) = argument {
                    write!(f, " {argument:?}")?;
                }
                write!(f, ": {error}")
            }
            ConfigDiagnosticKind::Service {
                service,
                index,
                field,
                ..
            } => {
                if let Some(service) = service {
                    write!(f, "service {service}")?;
                } else if let Some(index) = index {
                    write!(f, "service[{index}]")?;
                } else {
                    write!(f, "service")?;
                }
                if let Some(field) = field {
                    write!(f, ": {field}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for ConfigDiagnostic {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_kinds_retain_rule_data_and_render_without_string_dispatch() {
        let tracking = ConfigDiagnostic::value_with_detail(
            ValidationErrorCode::TrackingConstraint,
            ValueDiagnosticDetail::TrackingTrailingSlash,
            "tracking.cgroup_parent",
            "/system.slice/",
            Some("a path without a trailing slash".into()),
        );
        assert!(matches!(
            *tracking.kind,
            ConfigDiagnosticKind::Value {
                detail: ValueDiagnosticDetail::TrackingTrailingSlash,
                ..
            }
        ));
        assert!(tracking.to_string().contains("must not end with"));

        let placeholder = ConfigDiagnostic::placeholder(
            Some("demo".into()),
            "command",
            Some(1),
            Some("{bogus}".into()),
            PlaceholderError::UnknownPlaceholder("bogus".into()),
        );
        assert!(matches!(
            *placeholder.kind,
            ConfigDiagnosticKind::Placeholder {
                error: PlaceholderError::UnknownPlaceholder(ref name),
                ..
            } if name == "bogus"
        ));
        assert!(placeholder.to_string().contains("unknown placeholder"));
    }
}
