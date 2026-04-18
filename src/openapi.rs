//! Aggregated OpenAPI document for the daemon.

use axum::Json;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::routing::{Router, get};
use utoipa::OpenApi;

use crate::app_state::AppState;
use crate::management_api::handlers as mgmt_handlers;
use crate::management_api::types as mgmt_types;
use crate::openai_api::errors as openai_errors;
use crate::openai_api::handlers as openai_handlers;
use crate::openai_api::schema as openai_schema;

#[derive(OpenApi)]
#[openapi(
    paths(
        openai_handlers::list_models,
        openai_handlers::chat_completions,
        openai_handlers::completions,
        openai_handlers::embeddings,
        mgmt_handlers::list_services,
        mgmt_handlers::service_detail,
        mgmt_handlers::list_devices,
    ),
    components(schemas(
        openai_schema::ModelListing,
        openai_schema::ModelsResponse,
        openai_schema::ChatCompletionEnvelope,
        openai_schema::CompletionEnvelope,
        openai_schema::EmbeddingEnvelope,
        openai_errors::ErrorBody,
        openai_errors::ErrorDetail,
        mgmt_types::ServiceSummary,
        mgmt_types::ServiceDetail,
        mgmt_types::LogLine,
        mgmt_types::DeviceSummary,
        mgmt_types::DeviceReservation,
    )),
    info(title = "Ananke API", version = "0.1.0"),
)]
pub struct AnankeApi;

pub fn register(router: Router, state: AppState) -> Router {
    let openapi: Router = Router::new()
        .route("/api/openapi.json", get(serve_openapi))
        .with_state(state);
    router.merge(openapi)
}

pub async fn serve_openapi(State(_state): State<AppState>) -> Response {
    (axum::http::StatusCode::OK, Json(AnankeApi::openapi())).into_response()
}
