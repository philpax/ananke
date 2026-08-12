//! GET/PUT /api/config + POST /api/config/validate
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use ananke_api::{
    config::{
        get::ConfigResponse,
        validate::{
            ConfigValidateRequest, ConfigValidateResponse, ValidationContext, ValidationError,
            ValidationLocation,
        },
    },
    shared::errors::ApiError,
};
use ananke_config::validate::{ConfigDiagnostic, ConfigDiagnosticReport, DiagnosticContext};
use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode, header::IF_MATCH},
    response::{IntoResponse, Response},
};

use crate::{api::errors::ApiErrorCode, config::manager::ApplyError, daemon::app_state::AppState};

#[utoipa::path(
    summary = "Get or update the daemon config",
    get,
    path = "/api/config",
    responses((status = 200, body = ConfigResponse))
)]
pub async fn get_config(State(state): State<AppState>) -> Response {
    let (content, hash) = state.config.raw();
    let writable = state.config.writable();
    (
        StatusCode::OK,
        Json(ConfigResponse {
            content,
            hash,
            writable,
        }),
    )
        .into_response()
}

#[utoipa::path(
    summary = "Get or update the daemon config",
    put,
    path = "/api/config",
    request_body(content = String, description = "Raw TOML config"),
    responses(
        (status = 202),
        (status = 412, body = ApiError, description = "hash_mismatch"),
        (status = 422, body = ConfigValidateResponse),
        (status = 428, body = ApiError, description = "if_match_required"),
        (status = 500, body = ApiError, description = "persist_failed")
    )
)]
pub async fn put_config(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: String,
) -> Response {
    let Some(if_match) = headers
        .get(IF_MATCH)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim_matches('"').to_string())
    else {
        return ApiErrorCode::IfMatchRequired.into_response();
    };
    match state.config.apply(body, if_match).await {
        Ok(()) => StatusCode::ACCEPTED.into_response(),
        Err(ApplyError::HashMismatch { server_hash }) => {
            // ETag header has to be set on top of the standard
            // `ApiErrorCode` body, so build the response in pieces.
            // Invariant: `ConfigHash` is base64, and every base64 character is a
            // valid ETag header value, so the parse cannot fail.
            let etag = server_hash
                .parse()
                .unwrap_or_else(|_| unreachable!("base64 hash parses as an etag header value"));
            let mut resp = ApiErrorCode::HashMismatch { server_hash }.into_response();
            resp.headers_mut().insert(axum::http::header::ETAG, etag);
            resp
        }
        Err(ApplyError::Invalid(report)) => {
            let body = ConfigValidateResponse {
                valid: false,
                errors: project_report(report),
            };
            (StatusCode::UNPROCESSABLE_ENTITY, Json(body)).into_response()
        }
        Err(ApplyError::PersistFailed(io_err)) => ApiErrorCode::PersistFailed {
            reason: io_err.to_string(),
        }
        .into_response(),
    }
}

#[utoipa::path(
    summary = "Validate TOML config without persisting",
    post,
    path = "/api/config/validate",
    request_body = ConfigValidateRequest,
    responses((status = 200, body = ConfigValidateResponse))
)]
pub async fn post_validate(
    State(state): State<AppState>,
    Json(req): Json<ConfigValidateRequest>,
) -> Response {
    match state.config.validate(&req.content) {
        Ok(()) => Json(ConfigValidateResponse {
            valid: true,
            errors: vec![],
        })
        .into_response(),
        Err(report) => Json(ConfigValidateResponse {
            valid: false,
            errors: project_report(report),
        })
        .into_response(),
    }
}

fn project_report(report: ConfigDiagnosticReport) -> Vec<ValidationError> {
    report.into_iter().map(project_diagnostic).collect()
}

fn project_diagnostic(diagnostic: ConfigDiagnostic) -> ValidationError {
    let message = diagnostic.to_string();
    let location = diagnostic
        .location
        .as_ref()
        .map(|location| ValidationLocation {
            start: location.start,
            end: location.end,
            line: location.line,
            column: location.column,
        });
    let (line, column) = location.as_ref().map_or((None, None), |location| {
        (Some(location.line), Some(location.column))
    });
    let path = diagnostic.path().map(str::to_owned);
    let context = match diagnostic.context() {
        DiagnosticContext::Service {
            service,
            index,
            field,
        } => ValidationContext::Service {
            service,
            index,
            field,
        },
        DiagnosticContext::Field {
            field,
            offending,
            expected,
        } => ValidationContext::Field {
            field,
            offending,
            expected,
        },
        DiagnosticContext::Value {
            field,
            offending,
            expected,
        } => ValidationContext::Value {
            field,
            offending,
            expected,
        },
        DiagnosticContext::Count {
            field,
            got,
            expected,
        } => ValidationContext::Count {
            field,
            got,
            expected,
        },
        DiagnosticContext::Index {
            field,
            index,
            value,
            expected,
        } => ValidationContext::Index {
            field,
            index,
            value,
            expected,
        },
        DiagnosticContext::Fields { fields, reason, .. } => {
            ValidationContext::Fields { fields, reason }
        }
        DiagnosticContext::Placeholder {
            service,
            field,
            argv_index,
            argument,
            category,
        } => ValidationContext::Placeholder {
            service,
            field,
            argv_index,
            argument,
            category,
        },
        DiagnosticContext::Merge {
            service,
            index,
            parent,
            reason,
        } => ValidationContext::Merge {
            service,
            index,
            parent,
            reason,
        },
        DiagnosticContext::Parse { parser_message } => ValidationContext::Parse { parser_message },
    };
    ValidationError {
        code: diagnostic.code(),
        message,
        path,
        context,
        location,
        line,
        column,
    }
}

#[cfg(test)]
fn _force_link() {
    let _: Vec<ananke_api::config::validate::ValidationError> = vec![];
}
