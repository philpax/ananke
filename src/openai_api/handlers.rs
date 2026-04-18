//! Handlers for /v1/models and the three POST body-rewriting endpoints.

use std::time::Duration;

use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{Router, get, post};
use bytes::Bytes;
use futures::TryStreamExt;
use http_body_util::{BodyExt, StreamBody};
use hyper::body::Frame;
use serde_json::Value;
use tokio::sync::broadcast;
use tracing::warn;

use crate::app_state::AppState;
use crate::openai_api::errors;
use crate::openai_api::filters;
use crate::openai_api::schema::{
    ChatCompletionEnvelope, CompletionEnvelope, EmbeddingEnvelope, ModelListing, ModelsResponse,
};
use crate::state::ServiceState;
use crate::supervise::{EnsureResponse, StartFailureKind, StartOutcome};

pub fn register(router: Router, state: AppState) -> Router {
    // Build the main routes against AppState, collapse to Router<()>, then
    // merge the unimplemented stubs (which already return Router<()>) and
    // the caller's router.
    let implemented: Router = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .with_state(state.clone());
    let stubs: Router = crate::openai_api::unimplemented::register(Router::new(), state);
    router.merge(implemented).merge(stubs)
}

#[utoipa::path(get, path = "/v1/models", responses((status = 200, body = ModelsResponse)))]
pub async fn list_models(State(state): State<AppState>) -> Response {
    let mut data = Vec::new();
    for (name, handle) in state.registry.all() {
        let Some(snap) = handle.snapshot().await else {
            continue;
        };
        match snap.state {
            ServiceState::Idle
            | ServiceState::Starting
            | ServiceState::Warming
            | ServiceState::Running => {
                data.push(ModelListing {
                    id: name.to_string(),
                    object: "model",
                    created: 0,
                    owned_by: "ananke",
                });
            }
            _ => {}
        }
    }
    let body = ModelsResponse {
        object: "list",
        data,
    };
    (StatusCode::OK, Json(body)).into_response()
}

#[utoipa::path(
    post,
    path = "/v1/chat/completions",
    request_body = ChatCompletionEnvelope,
    responses((status = 200, description = "Proxied from upstream"))
)]
pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    forward_json_post("/v1/chat/completions", state, headers, body).await
}

#[utoipa::path(
    post,
    path = "/v1/completions",
    request_body = CompletionEnvelope,
    responses((status = 200, description = "Proxied from upstream"))
)]
pub async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    forward_json_post("/v1/completions", state, headers, body).await
}

#[utoipa::path(
    post,
    path = "/v1/embeddings",
    request_body = EmbeddingEnvelope,
    responses((status = 200, description = "Proxied from upstream"))
)]
pub async fn embeddings(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    forward_json_post("/v1/embeddings", state, headers, body).await
}

async fn forward_json_post(
    path: &'static str,
    state: AppState,
    headers: HeaderMap,
    body_bytes: Bytes,
) -> Response {
    let mut parsed: Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => return errors::bad_request(format!("invalid JSON body: {e}")),
    };
    let model = match parsed.get("model").and_then(|v| v.as_str()) {
        Some(m) => m.to_string(),
        None => return errors::bad_request("request body missing `model` field"),
    };

    let handle = match state.registry.get(&model) {
        Some(h) => h,
        None => return errors::not_found_model(&model),
    };

    let svc = state.config.services.iter().find(|s| s.name == model);
    let Some(svc) = svc else {
        return errors::not_found_model(&model);
    };

    // Ensure the service is running (coalescing concurrent first-requests).
    let mut ensure_rx: Option<broadcast::Receiver<StartOutcome>> = None;
    match handle.ensure().await {
        Some(EnsureResponse::AlreadyRunning) => {}
        Some(EnsureResponse::Waiting { rx }) => ensure_rx = Some(rx),
        Some(EnsureResponse::QueueFull) => return errors::start_queue_full(&model),
        Some(EnsureResponse::Unavailable { reason }) => {
            if reason.starts_with("no fit") {
                return errors::insufficient_vram(&model, &reason);
            }
            return errors::service_disabled(&model, &reason);
        }
        None => return errors::start_failed(&model, "supervisor unreachable"),
    }

    if let Some(mut rx) = ensure_rx {
        let timeout = Duration::from_millis(svc.max_request_duration_ms);
        match tokio::time::timeout(timeout, rx.recv()).await {
            Ok(Ok(StartOutcome::Ok)) => {}
            Ok(Ok(StartOutcome::Err(f))) => {
                return match f.kind {
                    StartFailureKind::NoFit => errors::insufficient_vram(&model, &f.message),
                    StartFailureKind::HealthTimeout => {
                        errors::start_failed(&model, "health check timed out")
                    }
                    StartFailureKind::Disabled => errors::service_disabled(&model, &f.message),
                    StartFailureKind::LaunchFailed => errors::start_failed(&model, &f.message),
                    StartFailureKind::Oom => errors::insufficient_vram(&model, &f.message),
                };
            }
            Ok(Err(e)) => {
                return errors::start_failed(&model, &format!("start broadcast closed: {e}"));
            }
            Err(_) => return errors::start_failed(&model, "start timed out"),
        }
    }

    // Apply filters.
    filters::apply(&mut parsed, &svc.filters);
    let new_body = match serde_json::to_vec(&parsed) {
        Ok(b) => b,
        Err(e) => return errors::bad_request(format!("re-serialise failed: {e}")),
    };

    // Bump activity.
    state.activity.ping(&svc.name);

    // Build hyper client and forward to the upstream service.
    let client = hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
        .build_http::<http_body_util::combinators::BoxBody<
        Bytes,
        Box<dyn std::error::Error + Send + Sync>,
    >>();

    let uri = format!("http://127.0.0.1:{}{}", svc.private_port, path)
        .parse::<hyper::Uri>()
        .unwrap();
    let mut req = hyper::Request::builder().method("POST").uri(uri);
    for (k, v) in headers.iter() {
        if k == hyper::header::HOST || k == hyper::header::CONTENT_LENGTH {
            continue;
        }
        req = req.header(k, v);
    }
    req = req.header(hyper::header::CONTENT_TYPE, "application/json");
    req = req.header(hyper::header::CONTENT_LENGTH, new_body.len());
    let upstream_body = http_body_util::Full::new(Bytes::from(new_body))
        .map_err(|never| match never {})
        .boxed();
    let req = match req.body(upstream_body) {
        Ok(r) => r,
        Err(e) => return errors::bad_request(format!("build request: {e}")),
    };

    let resp = match client.request(req).await {
        Ok(r) => r,
        Err(e) => {
            warn!(error = %e, model = %model, "upstream request failed");
            return errors::start_failed(&model, "upstream unavailable");
        }
    };

    let (parts, upstream_body) = resp.into_parts();
    // Convert the upstream body into a stream of data frames for axum to proxy.
    // We convert to a BoxBody so the opaque Body::new bound (HttpBody<Data=Bytes>)
    // is satisfied without threading through the full generic chain.
    let stream = upstream_body.into_data_stream().map_ok(Frame::data);
    let boxed: http_body_util::combinators::BoxBody<
        Bytes,
        Box<dyn std::error::Error + Send + Sync>,
    > = BodyExt::map_err(
        StreamBody::new(stream),
        |e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) },
    )
    .boxed();
    let axum_body = Body::new(boxed);
    let mut out = Response::from_parts(parts, axum_body);
    out.headers_mut().remove(hyper::header::CONNECTION);
    out.headers_mut().remove("transfer-encoding");
    out
}
