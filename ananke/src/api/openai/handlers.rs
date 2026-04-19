//! Handlers for /v1/models and the three POST body-rewriting endpoints.

use std::{task::Poll, time::Duration};

use axum::{
    Json,
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{Router, get, post},
};
use bytes::Bytes;
use futures::TryStreamExt;
use http_body_util::{BodyExt, StreamBody};
use hyper::body::{Frame, SizeHint};
use serde_json::Value;
use tracing::warn;

use crate::{
    api::openai::{
        errors, filters,
        schema::{
            ChatCompletionEnvelope, CompletionEnvelope, EmbeddingEnvelope, ModelListing,
            ModelsResponse,
        },
    },
    daemon::app_state::AppState,
    supervise::{EnsureFailure, EnsureOutcome, await_ensure, state::ServiceState},
    tracking::inflight::InflightGuard,
};

pin_project_lite::pin_project! {
    /// Wraps a body and holds an [`InflightGuard`] so the counter stays elevated
    /// until the full response body (including SSE streams) has been consumed.
    struct GuardedBody<B> {
        #[pin]
        body: B,
        _guard: InflightGuard,
    }
}

impl<B: hyper::body::Body> hyper::body::Body for GuardedBody<B> {
    type Data = B::Data;
    type Error = B::Error;

    fn poll_frame(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        self.project().body.poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.body.size_hint()
    }
}

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
    let stubs: Router = crate::api::openai::unimplemented::register(Router::new(), state);
    router.merge(implemented).merge(stubs)
}

#[utoipa::path(get, path = "/v1/models", responses((status = 200, body = ModelsResponse)))]
pub async fn list_models(State(state): State<AppState>) -> Response {
    let mut data = Vec::new();
    for (name, handle) in state.registry.all() {
        // Hide services whose template doesn't produce an OpenAI-compatible
        // server. llama-cpp defaults to compat; `command` services opt in
        // via `metadata.openai_compat = true`.
        let compat = state
            .config
            .effective()
            .services
            .iter()
            .find(|s| s.name == name)
            .map(|s| s.openai_compat)
            .unwrap_or(false);
        if !compat {
            continue;
        }
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

    let eff = state.config.effective();
    let svc = eff.services.iter().find(|s| s.name == model);
    let Some(svc) = svc else {
        return errors::not_found_model(&model);
    };

    // Ensure the service is running (coalescing concurrent first-requests).
    let max_request_duration = Duration::from_millis(svc.max_request_duration_ms);
    match await_ensure(&handle, max_request_duration).await {
        EnsureOutcome::Ready => {}
        EnsureOutcome::Failed(EnsureFailure::InsufficientVram(msg)) => {
            return errors::insufficient_vram(&model, &msg);
        }
        EnsureOutcome::Failed(EnsureFailure::ServiceDisabled(msg)) => {
            return errors::service_disabled(&model, &msg);
        }
        EnsureOutcome::Failed(EnsureFailure::StartQueueFull) => {
            return errors::start_queue_full(&model);
        }
        EnsureOutcome::Failed(EnsureFailure::StartFailed(msg)) => {
            return errors::start_failed(&model, &msg);
        }
    }

    // Apply filters.
    filters::apply(&mut parsed, &svc.filters);
    let new_body = match serde_json::to_vec(&parsed) {
        Ok(b) => b,
        Err(e) => return errors::bad_request(format!("re-serialise failed: {e}")),
    };

    // Bump activity and acquire an in-flight guard before forwarding.
    state.activity.ping(&svc.name);
    let counter = state.inflight.counter(&svc.name);
    let guard = InflightGuard::new(counter);

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
    // Wrap in GuardedBody so the in-flight counter stays elevated for the full
    // duration of the response, including SSE streams.
    let stream = upstream_body.into_data_stream().map_ok(Frame::data);
    let boxed: http_body_util::combinators::BoxBody<
        Bytes,
        Box<dyn std::error::Error + Send + Sync>,
    > = BodyExt::map_err(
        StreamBody::new(stream),
        |e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) },
    )
    .boxed();
    let guarded = GuardedBody {
        body: boxed,
        _guard: guard,
    };
    let axum_body = Body::new(guarded);
    let mut out = Response::from_parts(parts, axum_body);
    out.headers_mut().remove(hyper::header::CONNECTION);
    out.headers_mut().remove("transfer-encoding");
    out
}
