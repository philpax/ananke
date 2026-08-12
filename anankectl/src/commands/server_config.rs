//! Server-side (daemon) configuration management command handlers.

use std::{io::Read, path::Path};

use ananke_api::config::{
    get::ConfigResponse,
    validate::{ConfigValidateRequest, ConfigValidateResponse},
};

use crate::{
    client::{ApiClient, ApiClientError},
    output,
};

pub async fn show(client: &ApiClient, json: bool) -> Result<(), ApiClientError> {
    let resp: ConfigResponse = client.get_json("/api/config").await?;
    if json {
        output::print_json(&resp);
    } else {
        println!("{}", resp.content);
    }
    Ok(())
}

pub async fn validate(
    client: &ApiClient,
    json: bool,
    file: Option<&Path>,
) -> Result<(), ApiClientError> {
    let content = match file {
        Some(p) => std::fs::read_to_string(p)
            .map_err(|e| ApiClientError::Usage(format!("read {}: {e}", p.display())))?,
        None => {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s).ok();
            s
        }
    };
    let req = ConfigValidateRequest { content };
    let resp: ConfigValidateResponse = client.post_json("/api/config/validate", &req).await?;
    if json {
        output::print_json(&resp);
    } else if resp.valid {
        println!("ok: config is valid");
    } else {
        println!("error: config is invalid");
        print!("{}", format_validation_errors(&resp));
    }
    Ok(())
}

/// Format validation diagnostics for human CLI output.
pub fn format_validation_errors(response: &ConfigValidateResponse) -> String {
    response
        .errors
        .iter()
        .map(|error| {
            let location = error
                .location
                .as_ref()
                .map_or_else(String::new, |location| {
                    format!(" at {}:{}", location.line, location.column)
                });
            let path = error
                .path
                .as_deref()
                .map_or_else(String::new, |path| format!(" ({path})"));
            format!(
                "  [{}]{}{} context={:?} {}\n",
                error.code_label(),
                path,
                location,
                error.context,
                error.message
            )
        })
        .collect()
}

trait ValidationErrorLabel {
    fn code_label(&self) -> &'static str;
}

impl ValidationErrorLabel for ananke_api::config::validate::ValidationError {
    fn code_label(&self) -> &'static str {
        use ananke_api::config::validate::ValidationErrorCode::*;
        match self.code {
            Parse => "parse",
            MergeConstraint => "merge_constraint",
            GpuAllowDuplicate => "gpu_allow_duplicate",
            GpuAllowUnsorted => "gpu_allow_unsorted",
            TensorSplitWeightsCount => "tensor_split_weights_count",
            TensorSplitWeightInvalid => "tensor_split_weight_invalid",
            FieldMissing => "field_missing",
            FieldUnknown => "field_unknown",
            ValueInvalid => "value_invalid",
            FieldRequired => "field_required",
            FieldsIncompatible => "fields_incompatible",
            ServiceNameDuplicate => "service_name_duplicate",
            ServicePortDuplicate => "service_port_duplicate",
            ServicePortManagementCollision => "service_port_management_collision",
            DurationInvalid => "duration_invalid",
            PlaceholderInvalid => "placeholder_invalid",
            AllocationInvalid => "allocation_invalid",
            PrivatePortRangeInvalid => "private_port_range_invalid",
            PrivatePortExhausted => "private_port_exhausted",
            MetadataInvalid => "metadata_invalid",
            RuntimeConstraint => "runtime_constraint",
            AutoRestartConstraint => "auto_restart_constraint",
            TrackingConstraint => "tracking_constraint",
            PlacementConstraint => "placement_constraint",
            CommandConstraint => "command_constraint",
            TemplateConstraint => "template_constraint",
            Other => "other",
        }
    }
}

pub async fn reload(client: &ApiClient, _json: bool) -> Result<(), ApiClientError> {
    // Force-reload by PUTting the current file back to the server.
    // Read the file's current content via GET, then PUT it unchanged
    // with the matching If-Match hash.
    let resp: ConfigResponse = client.get_json("/api/config").await?;
    client
        .put_body("/api/config", resp.content, Some(&resp.hash))
        .await?;
    println!("ok: config reload requested");
    Ok(())
}

#[cfg(test)]
mod tests {
    use ananke_api::config::validate::{
        ValidationContext, ValidationError, ValidationErrorCode, ValidationLocation,
    };

    use super::*;

    fn error(code: ValidationErrorCode, location: Option<ValidationLocation>) -> ValidationError {
        ValidationError {
            code,
            message: "invalid config".into(),
            path: Some("service.port".into()),
            context: ValidationContext::Value {
                field: "service.port".into(),
                offending: "0".into(),
                expected: Some("a non-zero port".into()),
            },
            line: location.as_ref().map(|location| location.line),
            column: location.as_ref().map(|location| location.column),
            location,
        }
    }

    #[test]
    fn format_validation_errors_located() {
        let response = ConfigValidateResponse {
            valid: false,
            errors: vec![error(
                ValidationErrorCode::ValueInvalid,
                Some(ValidationLocation {
                    start: 4,
                    end: 8,
                    line: 2,
                    column: 5,
                }),
            )],
        };
        let formatted = format_validation_errors(&response);
        assert!(formatted.contains("[value_invalid] (service.port) at 2:5"));
        assert!(formatted.contains("invalid config"));
    }

    #[test]
    fn format_validation_errors_unlocated() {
        let response = ConfigValidateResponse {
            valid: false,
            errors: vec![error(ValidationErrorCode::FieldMissing, None)],
        };
        let formatted = format_validation_errors(&response);
        assert!(formatted.contains("[field_missing] (service.port)"));
        assert!(formatted.contains("context="));
        assert!(formatted.contains("invalid config"));
        assert!(!formatted.contains("0:0"));
    }

    #[test]
    fn format_validation_errors_multiple() {
        let response = ConfigValidateResponse {
            valid: false,
            errors: vec![
                error(ValidationErrorCode::Parse, None),
                error(ValidationErrorCode::MergeConstraint, None),
            ],
        };
        let formatted = format_validation_errors(&response);
        assert!(formatted.find("[parse]").unwrap() < formatted.find("[merge_constraint]").unwrap());
    }
}
