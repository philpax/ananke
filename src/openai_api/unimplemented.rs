//! 501 Not Implemented handlers for unsupported OpenAI endpoints.

use axum::Router;
use axum::extract::Path;
use axum::response::Response;
use axum::routing::any;

use crate::app_state::AppState;
use crate::openai_api::errors;

pub fn register(router: Router, _state: AppState) -> Router {
    router
        .route("/v1/audio/{*rest}", any(not_implemented))
        .route("/v1/images/{*rest}", any(not_implemented))
        .route("/v1/files/{*rest}", any(not_implemented))
        .route("/v1/fine_tuning/{*rest}", any(not_implemented))
        .route("/v1/batches", any(batches_not_implemented))
}

async fn not_implemented(Path(rest): Path<String>) -> Response {
    errors::not_implemented(&rest)
}

async fn batches_not_implemented() -> Response {
    errors::not_implemented("/v1/batches")
}
