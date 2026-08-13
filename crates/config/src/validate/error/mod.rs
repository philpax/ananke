//! Structured diagnostics produced by the configuration parse, merge, and
//! semantic-validation pipeline.
//!
//! [`ConfigDiagnostic`] is the single diagnostic type. Every fact a renderer
//! or boundary adapter needs — the stable code, the owning service, the
//! field(s) it names, and its rendered message — is computed once at the
//! site that raises the diagnostic, so nothing downstream has to parse a
//! rendered message or dispatch on a payload variant.

mod detail;
mod location;
mod report;

use std::fmt;

use ananke_api::config::validate::{ValidationError, ValidationErrorCode, ValidationLocation};
pub use detail::PlaceholderError;
pub use location::{DiagnosticLocation, byte_offset_to_line_column};
pub use report::{ConfigDiagnosticReport, ConfigPipelineError};

/// One typed configuration diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigDiagnostic {
    /// The stable, machine-readable diagnostic code.
    pub code: ValidationErrorCode,
    /// Owning service name, when the diagnostic belongs to a named service.
    ///
    /// Attached either at construction (when the rule already knows the
    /// service) or by [`Self::with_service_context`], so that a diagnostic
    /// raised before the name is read — a missing `port`, say — still
    /// reaches the operator attributed to the block it came from.
    pub service: Option<String>,
    /// Original zero-based service index, when the diagnostic belongs to a service.
    pub source_index: Option<usize>,
    /// The field paths the diagnostic names, most significant first. A
    /// single-field rule names one; a cross-field constraint names each
    /// field that participates.
    pub fields: Vec<String>,
    /// The diagnostic's rendered message, formatted at the site that raised
    /// it (and, for constraint diagnostics, without a `service` prefix,
    /// which `Display` supplies uniformly).
    pub message: String,
    /// Optional source location from the parser.
    ///
    /// Boxed because only a parse diagnostic carries one, and it is the
    /// largest field on a type that is returned by `Result` throughout the
    /// pipeline.
    pub location: Option<Box<DiagnosticLocation>>,
}

impl ConfigDiagnostic {
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
        let mut diagnostic = Self::new(
            ValidationErrorCode::Parse,
            format!("parse error: {}", message.into()),
        );
        diagnostic.location = location.map(Box::new);
        diagnostic
    }

    /// Construct a merge-phase diagnostic from its fully rendered message.
    pub fn merge(message: impl Into<String>) -> Self {
        Self::new(ValidationErrorCode::MergeConstraint, message)
    }

    /// Construct a field-value diagnostic naming a single field.
    pub fn value(
        code: ValidationErrorCode,
        field: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        let mut diagnostic = Self::new(code, message);
        diagnostic.fields = vec![field.into()];
        diagnostic
    }

    /// Construct a constraint diagnostic with explicit stable code and fields.
    pub fn constraint(
        code: ValidationErrorCode,
        service: Option<String>,
        fields: &[&str],
        message: String,
    ) -> Self {
        let mut diagnostic = Self::new(code, message);
        diagnostic.service = service;
        diagnostic.fields = fields.iter().map(|field| (*field).to_string()).collect();
        diagnostic
    }

    /// Construct a template-constraint diagnostic for a named service.
    ///
    /// The overwhelming majority of service rules carry
    /// [`ValidationErrorCode::TemplateConstraint`]; spelling it at each of
    /// them buried the rule in its own bookkeeping. Reach for
    /// [`Self::constraint`] when the code differs.
    pub fn service(name: impl AsRef<str>, fields: &[&str], message: String) -> Self {
        Self::constraint(
            ValidationErrorCode::TemplateConstraint,
            Some(name.as_ref().to_string()),
            fields,
            message,
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
        let field = field.into();
        let mut message = field.clone();
        if let Some(index) = argv_index {
            message = format!("{message}[{index}]");
        }
        if let Some(argument) = &argument {
            message = format!("{message} {argument:?}");
        }
        message = format!("{message}: {error}");
        let mut diagnostic = Self::new(ValidationErrorCode::PlaceholderInvalid, message);
        diagnostic.service = service;
        diagnostic.fields = vec![field];
        diagnostic
    }

    /// Return the stable code.
    pub const fn code(&self) -> ValidationErrorCode {
        self.code
    }

    /// The most significant field the diagnostic names, for a consumer that
    /// can highlight only one.
    pub fn path(&self) -> Option<&str> {
        self.fields.first().map(String::as_str)
    }

    /// Every field path the diagnostic names, empty when it names none.
    ///
    /// This is the structural discriminator: a test asserts on the path the
    /// rule rejected rather than on the sentence it rendered.
    pub fn fields(&self) -> &[String] {
        &self.fields
    }

    /// Return the owning service name.
    pub fn service_name(&self) -> Option<&str> {
        self.service.as_deref()
    }

    /// Construct the shared shape every other constructor specializes.
    fn new(code: ValidationErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            service: None,
            source_index: None,
            fields: Vec::new(),
            message: message.into(),
            location: None,
        }
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
        match (self.service.as_deref(), self.source_index) {
            (Some(service), _) => write!(f, "service {service}: ")?,
            (None, Some(index)) => write!(f, "service[{index}]: ")?,
            (None, None) => {}
        }
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ConfigDiagnostic {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_diagnostics_render_the_message_formatted_at_the_construction_site() {
        let tracking = ConfigDiagnostic::value(
            ValidationErrorCode::TrackingConstraint,
            "tracking.cgroup_parent",
            "tracking.cgroup_parent must not end with `/` (got `/system.slice/`)",
        );
        assert_eq!(tracking.path(), Some("tracking.cgroup_parent"));
        assert!(tracking.to_string().contains("must not end with"));

        let placeholder = ConfigDiagnostic::placeholder(
            Some("demo".into()),
            "command",
            Some(1),
            Some("{bogus}".into()),
            PlaceholderError::UnknownPlaceholder("bogus".into()),
        );
        assert_eq!(placeholder.path(), Some("command"));
        assert!(placeholder.to_string().contains("unknown placeholder"));
        assert!(placeholder.to_string().starts_with("service demo: "));
    }
}
