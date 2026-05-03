//! Aggregated OpenAPI document for the daemon.

use ananke_api::{
    ApiError, ConfigResponse, ConfigValidateRequest, ConfigValidateResponse, DeviceReservation,
    DeviceSummary, DisableResponse, EnableResponse, LogLine, LogsResponse, ServiceDetail,
    ServiceSummary, ServicesResponse, StartResponse, StopResponse, ValidationError,
    oneshot::{OneshotAllocation, OneshotDevices, OneshotRequest, OneshotResponse, OneshotStatus},
};
use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response},
    routing::{Router, get},
};
use utoipa::OpenApi;

use crate::{
    api::{
        management::{
            config as mgmt_config, handlers as mgmt_handlers, lifecycle as mgmt_lifecycle,
            logs as mgmt_logs,
        },
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
        mgmt_lifecycle::post_start,
        mgmt_lifecycle::post_stop,
        mgmt_lifecycle::post_restart,
        mgmt_lifecycle::post_enable,
        mgmt_lifecycle::post_disable,
        mgmt_logs::get_logs,
        mgmt_config::get_config,
        mgmt_config::put_config,
        mgmt_config::post_validate,
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
        ServiceSummary,
        ServicesResponse,
        ServiceDetail,
        LogLine,
        LogsResponse,
        DeviceSummary,
        DeviceReservation,
        StartResponse,
        StopResponse,
        EnableResponse,
        DisableResponse,
        ConfigResponse,
        ConfigValidateRequest,
        ConfigValidateResponse,
        ValidationError,
        ApiError,
        OneshotRequest,
        OneshotAllocation,
        OneshotDevices,
        OneshotResponse,
        OneshotStatus,
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
