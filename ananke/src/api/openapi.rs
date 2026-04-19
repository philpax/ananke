//! Aggregated OpenAPI document for the daemon.

use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response},
    routing::{Router, get},
};
use utoipa::OpenApi;

use crate::{
    api::{
        management::{handlers as mgmt_handlers, types as mgmt_types},
        openai::{errors as openai_errors, handlers as openai_handlers, schema as openai_schema},
    },
    daemon::app_state::AppState,
    oneshot::handlers as oneshot_handlers,
};

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
        oneshot_handlers::post_oneshot,
        oneshot_handlers::list_oneshots,
        oneshot_handlers::get_oneshot,
        oneshot_handlers::delete_oneshot,
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
        oneshot_handlers::OneshotRequest,
        oneshot_handlers::OneshotAllocation,
        oneshot_handlers::OneshotResponse,
        oneshot_handlers::OneshotStatus,
    )),
    info(title = "Ananke API", version = "0.1.0")
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
