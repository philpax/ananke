//! `POST /api/config/validate` — validate TOML without persisting.

use serde::{Deserialize, Deserializer, Serialize, de::Error as DeError};
use utoipa::ToSchema;

/// `POST /api/config/validate` request body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct ConfigValidateRequest {
    /// Raw TOML to validate without persisting.
    pub content: String,
}

/// `POST /api/config/validate` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct ConfigValidateResponse {
    /// `true` iff no errors were found.
    pub valid: bool,
    /// Ordered structured diagnostics.
    pub errors: Vec<ValidationError>,
}

/// Stable machine-readable validation code.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ValidationErrorCode {
    /// TOML parser or deserializer failure.
    Parse,
    /// Merge or inheritance failure.
    MergeConstraint,
    /// Duplicate GPU ids.
    GpuAllowDuplicate,
    /// Unsorted GPU ids.
    GpuAllowUnsorted,
    /// Tensor-split count mismatch.
    TensorSplitWeightsCount,
    /// Invalid tensor-split weight.
    TensorSplitWeightInvalid,
    /// Missing field.
    FieldMissing,
    /// Unknown field or value.
    FieldUnknown,
    /// Invalid value.
    ValueInvalid,
    /// Required-field dependency failure.
    FieldRequired,
    /// Incompatible fields.
    FieldsIncompatible,
    /// Duplicate service name.
    ServiceNameDuplicate,
    /// Duplicate service port.
    ServicePortDuplicate,
    /// Management-port collision.
    ServicePortManagementCollision,
    /// Invalid duration.
    DurationInvalid,
    /// Invalid placeholder.
    PlaceholderInvalid,
    /// Invalid allocation.
    AllocationInvalid,
    /// Invalid private-port range.
    PrivatePortRangeInvalid,
    /// Exhausted private-port range.
    PrivatePortExhausted,
    /// Invalid metadata.
    MetadataInvalid,
    /// Runtime constraint.
    RuntimeConstraint,
    /// Automatic-restart constraint.
    AutoRestartConstraint,
    /// Tracking constraint.
    TrackingConstraint,
    /// Placement constraint.
    PlacementConstraint,
    /// Command constraint.
    CommandConstraint,
    /// Template constraint.
    TemplateConstraint,
    /// Forward-compatible code.
    #[serde(other)]
    Other,
}

/// Source range and human position for a parser diagnostic.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct ValidationLocation {
    /// Zero-based byte offset at the start of the span.
    pub start: usize,
    /// Exclusive zero-based byte offset at the end of the span.
    pub end: usize,
    /// One-based source line.
    pub line: u32,
    /// One-based source column.
    pub column: u32,
}

/// Structured context for rendering and machine consumers.
#[derive(Debug, Clone, Serialize, PartialEq, Eq, ToSchema)]
#[serde(tag = "kind", content = "data")]
pub enum ValidationContext {
    /// Service identity and optional field context.
    Service {
        /// Service name.
        service: Option<String>,
        /// Original source index.
        index: Option<usize>,
        /// Field path.
        field: Option<String>,
    },
    /// Field context with optional values.
    Field {
        /// Field path.
        field: String,
        /// Offending value.
        offending: Option<String>,
        /// Expected value.
        expected: Option<String>,
    },
    /// A concrete invalid value.
    Value {
        /// Field path.
        field: String,
        /// Offending value.
        offending: String,
        /// Expected value.
        expected: Option<String>,
    },
    /// A count mismatch.
    Count {
        /// Field path.
        field: String,
        /// Actual count.
        got: usize,
        /// Expected count.
        expected: usize,
    },
    /// An invalid indexed value.
    Index {
        /// Field path.
        field: String,
        /// Index into the collection.
        index: usize,
        /// Offending value.
        value: String,
        /// Expected value.
        expected: Option<String>,
    },
    /// A multi-field constraint.
    Fields {
        /// Field paths.
        fields: Vec<String>,
        /// Constraint reason.
        reason: String,
    },
    /// Placeholder substitution context.
    Placeholder {
        /// Service name.
        service: Option<String>,
        /// Field path.
        field: String,
        /// Argument index.
        argv_index: Option<usize>,
        /// Argument text.
        argument: Option<String>,
        /// Substitution category.
        category: String,
    },
    /// Merge or inheritance context.
    Merge {
        /// Service name.
        service: Option<String>,
        /// Original source index.
        index: Option<usize>,
        /// Parent service.
        parent: Option<String>,
        /// Merge reason.
        reason: String,
    },
    /// Parser context.
    Parse {
        /// Original parser message.
        parser_message: String,
    },
    /// Forward-compatible context payload.
    Other {
        /// Opaque future payload.
        data: serde_json::Value,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", content = "data")]
enum KnownValidationContext {
    Service {
        service: Option<String>,
        index: Option<usize>,
        field: Option<String>,
    },
    Field {
        field: String,
        offending: Option<String>,
        expected: Option<String>,
    },
    Value {
        field: String,
        offending: String,
        expected: Option<String>,
    },
    Count {
        field: String,
        got: usize,
        expected: usize,
    },
    Index {
        field: String,
        index: usize,
        value: String,
        expected: Option<String>,
    },
    Fields {
        fields: Vec<String>,
        reason: String,
    },
    Placeholder {
        service: Option<String>,
        field: String,
        argv_index: Option<usize>,
        argument: Option<String>,
        category: String,
    },
    Merge {
        service: Option<String>,
        index: Option<usize>,
        parent: Option<String>,
        reason: String,
    },
    Parse {
        parser_message: String,
    },
    Other {
        data: serde_json::Value,
    },
}

impl<'de> Deserialize<'de> for ValidationContext {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let kind = value.get("kind").and_then(serde_json::Value::as_str);
        match kind {
            Some("Service") | Some("Field") | Some("Value") | Some("Count") | Some("Index")
            | Some("Fields") | Some("Placeholder") | Some("Merge") | Some("Parse")
            | Some("Other") => match serde_json::from_value(value).map_err(D::Error::custom)? {
                KnownValidationContext::Service {
                    service,
                    index,
                    field,
                } => Ok(Self::Service {
                    service,
                    index,
                    field,
                }),
                KnownValidationContext::Field {
                    field,
                    offending,
                    expected,
                } => Ok(Self::Field {
                    field,
                    offending,
                    expected,
                }),
                KnownValidationContext::Value {
                    field,
                    offending,
                    expected,
                } => Ok(Self::Value {
                    field,
                    offending,
                    expected,
                }),
                KnownValidationContext::Count {
                    field,
                    got,
                    expected,
                } => Ok(Self::Count {
                    field,
                    got,
                    expected,
                }),
                KnownValidationContext::Index {
                    field,
                    index,
                    value,
                    expected,
                } => Ok(Self::Index {
                    field,
                    index,
                    value,
                    expected,
                }),
                KnownValidationContext::Fields { fields, reason } => {
                    Ok(Self::Fields { fields, reason })
                }
                KnownValidationContext::Placeholder {
                    service,
                    field,
                    argv_index,
                    argument,
                    category,
                } => Ok(Self::Placeholder {
                    service,
                    field,
                    argv_index,
                    argument,
                    category,
                }),
                KnownValidationContext::Merge {
                    service,
                    index,
                    parent,
                    reason,
                } => Ok(Self::Merge {
                    service,
                    index,
                    parent,
                    reason,
                }),
                KnownValidationContext::Parse { parser_message } => {
                    Ok(Self::Parse { parser_message })
                }
                KnownValidationContext::Other { data } => Ok(Self::Other { data }),
            },
            Some(_) => Ok(Self::Other { data: value }),
            None => Err(D::Error::custom("validation context is missing kind")),
        }
    }
}

/// One structured config validation diagnostic.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct ValidationError {
    /// Stable machine-readable code.
    pub code: ValidationErrorCode,
    /// Centrally rendered human message.
    pub message: String,
    /// Field path, when available.
    pub path: Option<String>,
    /// Typed diagnostic context.
    pub context: ValidationContext,
    /// Authoritative parser source location, when available.
    pub location: Option<ValidationLocation>,
    /// Backwards-compatible one-based line, duplicated from `location`.
    pub line: Option<u32>,
    /// Backwards-compatible one-based column, duplicated from `location`.
    pub column: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_code_and_context_are_forward_compatible() {
        let value = serde_json::json!({
            "code": "future_code",
            "message": "future",
            "path": null,
            "context": { "kind": "Future", "data": { "x": 1 } },
            "location": null,
            "line": null,
            "column": null
        });
        let error: ValidationError = serde_json::from_value(value).unwrap();
        assert_eq!(error.code, ValidationErrorCode::Other);
        assert!(matches!(error.context, ValidationContext::Other { .. }));
    }
}
