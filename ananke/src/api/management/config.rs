//! GET/PUT /api/config + POST /api/config/validate

use ananke_api::{ApiError, ConfigResponse, ConfigValidateRequest, ConfigValidateResponse};
use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode, header::IF_MATCH},
    response::{IntoResponse, Response},
};

use crate::{config::manager::ApplyError, daemon::app_state::AppState};

pub async fn get_config(State(state): State<AppState>) -> Response {
    let (content, hash) = state.config.raw();
    (StatusCode::OK, Json(ConfigResponse { content, hash })).into_response()
}

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
        return (
            StatusCode::PRECONDITION_REQUIRED,
            Json(ApiError::new(
                "if_match_required",
                "PUT /api/config requires an If-Match header with the current config hash",
            )),
        )
            .into_response();
    };
    match state.config.apply(body, if_match).await {
        Ok(()) => StatusCode::ACCEPTED.into_response(),
        Err(ApplyError::HashMismatch { server_hash }) => {
            let mut resp = (
                StatusCode::PRECONDITION_FAILED,
                Json(ApiError::new(
                    "hash_mismatch",
                    format!("config was modified since last GET; current hash is {server_hash}"),
                )),
            )
                .into_response();
            resp.headers_mut()
                .insert(axum::http::header::ETAG, server_hash.parse().unwrap());
            resp
        }
        Err(ApplyError::Invalid(errors)) => {
            let body = ConfigValidateResponse {
                valid: false,
                errors,
            };
            (StatusCode::UNPROCESSABLE_ENTITY, Json(body)).into_response()
        }
        Err(ApplyError::PersistFailed(io_err)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiError::new(
                "persist_failed",
                format!("writing config to disk failed: {io_err}"),
            )),
        )
            .into_response(),
    }
}

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
