//! GET/PUT /api/config + POST /api/config/validate

use ananke_api::{ApiError, ConfigResponse, ConfigValidateRequest, ConfigValidateResponse};
use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode, header::IF_MATCH},
    response::{IntoResponse, Response},
};

use crate::{api::errors::ApiErrorCode, config::manager::ApplyError, daemon::app_state::AppState};

#[utoipa::path(
    get,
    path = "/api/config",
    responses((status = 200, body = ConfigResponse))
)]
pub async fn get_config(State(state): State<AppState>) -> Response {
    let (content, hash) = state.config.raw();
    (StatusCode::OK, Json(ConfigResponse { content, hash })).into_response()
}

#[utoipa::path(
    put,
    path = "/api/config",
    request_body(content = String, description = "Raw TOML config"),
    responses(
        (status = 202),
        (status = 412, body = ApiError),
        (status = 422, body = ConfigValidateResponse),
        (status = 428, body = ApiError),
        (status = 500, body = ApiError)
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
            let etag = server_hash.parse().unwrap();
            let mut resp = ApiErrorCode::HashMismatch { server_hash }.into_response();
            resp.headers_mut().insert(axum::http::header::ETAG, etag);
            resp
        }
        Err(ApplyError::Invalid(errors)) => {
            let body = ConfigValidateResponse {
                valid: false,
                errors,
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
        Err(errors) => Json(ConfigValidateResponse {
            valid: false,
            errors,
        })
        .into_response(),
    }
}

#[cfg(test)]
#[allow(dead_code)]
fn _force_link() {
    let _: Vec<ananke_api::ValidationError> = vec![];
}
