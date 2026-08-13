//! `POST /api/config/validate` — validate TOML without persisting.

use std::fmt;

use serde::{Deserialize, Serialize};
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
    /// Invalid value.
    ValueInvalid,
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
    /// Invalid private-port range.
    PrivatePortRangeInvalid,
    /// Exhausted private-port range.
    PrivatePortExhausted,
    /// Tracking constraint.
    TrackingConstraint,
    /// Template constraint.
    TemplateConstraint,
    /// Forward-compatible code.
    #[serde(other)]
    Other,
}

impl ValidationErrorCode {
    /// The stable wire spelling, identical to the serialized form.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Parse => "parse",
            Self::MergeConstraint => "merge_constraint",
            Self::GpuAllowDuplicate => "gpu_allow_duplicate",
            Self::GpuAllowUnsorted => "gpu_allow_unsorted",
            Self::TensorSplitWeightsCount => "tensor_split_weights_count",
            Self::TensorSplitWeightInvalid => "tensor_split_weight_invalid",
            Self::FieldMissing => "field_missing",
            Self::ValueInvalid => "value_invalid",
            Self::ServiceNameDuplicate => "service_name_duplicate",
            Self::ServicePortDuplicate => "service_port_duplicate",
            Self::ServicePortManagementCollision => "service_port_management_collision",
            Self::DurationInvalid => "duration_invalid",
            Self::PlaceholderInvalid => "placeholder_invalid",
            Self::PrivatePortRangeInvalid => "private_port_range_invalid",
            Self::PrivatePortExhausted => "private_port_exhausted",
            Self::TrackingConstraint => "tracking_constraint",
            Self::TemplateConstraint => "template_constraint",
            Self::Other => "other",
        }
    }
}

impl fmt::Display for ValidationErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
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

/// One structured config validation diagnostic.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct ValidationError {
    /// Stable machine-readable code.
    pub code: ValidationErrorCode,
    /// Centrally rendered human message.
    pub message: String,
    /// Field path, when available.
    pub path: Option<String>,
    /// Owning service name, when the diagnostic belongs to one.
    pub service: Option<String>,
    /// Zero-based index of the owning `[[service]]` block in the original source.
    ///
    /// Present even when the name is missing or invalid, so a diagnostic can
    /// always be attributed back to the block that produced it.
    pub service_index: Option<usize>,
    /// Parser source location, when available.
    pub location: Option<ValidationLocation>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_code_is_forward_compatible() {
        let value = serde_json::json!({
            "code": "future_code",
            "message": "future",
            "path": null,
            "service": null,
            "service_index": null,
            "location": null
        });
        let error: ValidationError = serde_json::from_value(value).unwrap();
        assert_eq!(error.code, ValidationErrorCode::Other);
    }
}
