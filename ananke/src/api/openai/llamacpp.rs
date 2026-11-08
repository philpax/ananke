//! Llama.cpp-native endpoint passthrough.
//!
//! llama-server exposes its own native surface alongside the OpenAI ones:
//! `/tokenize`, `/apply-template`, `/infill`, `/props`, `/slots`, and the
//! other routes in [`NATIVE_PATHS`]. Tools that speak llama.cpp's native
//! protocol (SillyTavern, the llama.cpp web UI, Kobold-style clients)
//! hit these paths directly. They used to fall through to an axum 404 on
//! the main inference listener; this module forwards them verbatim to the
//! upstream of a `llama-cpp` template service, so the main ananke URL
//! serves the same surface a bare llama-server would.
//!
//! The OpenAI- and Anthropic-shaped aliases (`/v1/*`, plus the non-`/v1`
//! `/models`, `/completions`, `/chat/completions`, `/responses`,
//! `/audio/transcriptions`, `/messages`, …) are deliberately **not**
//! proxied here: the `/v1/*` ones are ananke's own surface (implemented
//! or 501), and the non-`/v1` aliases would create unmonitored parallel
//! copies of it.
//!
//! Routing is by model name, the same selector the OpenAI surface uses:
//! a `model` field in the request body chooses the service, and a
//! request without one falls back to the sole llama-cpp service when
//! exactly one is configured (the reference single-model setup). A body
//! that names a non-llama-cpp service, or a multi-llama-cpp daemon
//! without a `model` field, gets a structured error naming the fix.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::{error::Error, sync::Arc, task::Poll, time::Duration};

use ananke_proxy::{InflightGuard, ProxyBody};
use axum::{
    RequestExt, Router,
    body::Body,
    extract::{Request, State},
    http::{HeaderMap, HeaderName, StatusCode},
    response::{IntoResponse, Response},
    routing::any,
};
use futures::TryStreamExt;
use http_body_util::{BodyExt, Full, LengthLimitError, StreamBody};
use hyper::{
    body::{Frame, SizeHint},
    header,
};
use smol_str::SmolStr;
use tracing::{info, warn};

use crate::{
    api::{errors::ApiErrorCode, openai::errors},
    config::Template,
    daemon::app_state::AppState,
    supervise::{EnsureFailure, EnsureOutcome, SupervisorHandle, await_ensure},
};

/// llama-server's native (non-OpenAI) surface, kept in sync with the
/// route table in llama.cpp's `tools/server/server.cpp`. Every path is
/// forwarded verbatim; the target service's template must be `llama-cpp`.
const NATIVE_PATHS: &[&str] = &[
    "/health",
    "/v1/health",
    "/metrics",
    "/props",
    "/slots",
    "/slots/:id_slot",
    "/lora-adapters",
    "/tokenize",
    "/detokenize",
    "/apply-template",
    "/completion",
    "/infill",
    "/embedding",
    "/embeddings",
    "/rerank",
    "/reranking",
];

/// Everything the forwarder needs from the resolved service, copied out
/// of the config so no guard or borrow survives an `.await`.
#[derive(Clone)]
struct Target {
    name: SmolStr,
    handle: Arc<SupervisorHandle>,
    private_port: u16,
    max_request_duration_ms: u64,
}

pin_project_lite::pin_project! {
    /// Keeps the service pinned until the downstream finishes or drops the body.
    struct InflightBody<B> {
        #[pin]
        body: B,
        _guard: InflightGuard,
    }
}

impl<B: hyper::body::Body> hyper::body::Body for InflightBody<B> {
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
    let native: Router = NATIVE_PATHS
        .iter()
        .fold(Router::new(), |r, path| r.route(path, any(passthrough)))
        .with_state(state);
    router.merge(native)
}

/// Forward one llama.cpp-native request to the resolved service's
/// upstream, verbatim: the original method, path, query, headers (minus
/// hop-by-hop), and body bytes are passed through untouched. Filters do
/// not apply (they expect OpenAI-shaped bodies), and no request is
/// recorded for metrics — matching the per-service proxy, which only
/// records the token-generating `/v1/*` endpoints.
async fn passthrough(State(state): State<AppState>, req: Request<Body>) -> Response {
    let (parts, body) = req.with_limited_body().into_parts();
    let method = parts.method;
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or("/")
        .to_string();
    let mut headers = parts.headers;
    let body_bytes = match body.collect().await {
        Ok(c) => c.to_bytes(),
        Err(e) if error_chain_contains::<LengthLimitError>(&e) => {
            return (StatusCode::PAYLOAD_TOO_LARGE, "request body too large").into_response();
        }
        Err(e) => return errors::bad_request(format!("read request body: {e}")),
    };

    let target = match select_target(&state, &body_bytes) {
        Ok(t) => t,
        Err(code) => return code.into_response(),
    };
    info!(path = %path_and_query, service = %target.name, "llama.cpp-native request received");

    // Ensure the service is running (coalescing concurrent first-requests),
    // exactly like the OpenAI surface and the per-service proxy.
    let max_request_duration = Duration::from_millis(target.max_request_duration_ms);
    match await_ensure(&target.handle, max_request_duration).await {
        EnsureOutcome::Ready { .. } => {}
        EnsureOutcome::Failed(f) => return ensure_failure_response(&target.name, f),
    }

    // Bump activity and acquire an in-flight guard so the service stays
    // pinned against drain/eviction for the full response (including SSE).
    state.activity.ping(&target.name);
    let guard = InflightGuard::new(state.inflight.counter(&target.name));

    let client = hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
        .build_http::<ProxyBody>();

    // Invariant: the port is a validated `u16` and the path came from the
    // client's own URI, so the constructed http URL always parses.
    let uri = format!("http://127.0.0.1:{}{}", target.private_port, path_and_query)
        .parse::<hyper::Uri>()
        .unwrap_or_else(|_| {
            unreachable!("uri from a validated port and a client-supplied path parses")
        });
    remove_hop_by_hop_headers(&mut headers);
    headers.remove(header::HOST);
    headers.remove(header::CONTENT_LENGTH);
    let mut upstream_req = Request::builder().method(method).uri(uri);
    for (k, v) in &headers {
        upstream_req = upstream_req.header(k, v);
    }
    if !body_bytes.is_empty() {
        upstream_req = upstream_req.header(header::CONTENT_LENGTH, body_bytes.len());
    }
    let upstream_body: ProxyBody = Full::new(body_bytes)
        .map_err(|never| match never {})
        .boxed();
    let upstream_req = match upstream_req.body(upstream_body) {
        Ok(r) => r,
        Err(e) => return errors::bad_request(format!("build request: {e}")),
    };

    let resp = match client.request(upstream_req).await {
        Ok(r) => r,
        Err(e) => {
            warn!(service = %target.name, error = %e, "llama.cpp-native passthrough: upstream request failed");
            return errors::upstream_unavailable(format!("upstream unavailable: {e}"));
        }
    };

    let (mut parts, body) = resp.into_parts();
    // Stream the body back without buffering — critical for SSE from
    // `/completion` with `"stream": true`.
    let stream = body.into_data_stream().map_ok(Frame::data);
    let boxed: ProxyBody = BodyExt::map_err(
        StreamBody::new(stream),
        |e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) },
    )
    .boxed();

    let guarded = InflightBody {
        body: boxed,
        _guard: guard,
    };
    remove_hop_by_hop_headers(&mut parts.headers);
    Response::from_parts(parts, Body::new(guarded))
}

fn error_chain_contains<T: Error + 'static>(error: &(dyn Error + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(error) = current {
        if error.is::<T>() {
            return true;
        }
        current = error.source();
    }
    false
}

fn remove_hop_by_hop_headers(headers: &mut HeaderMap) {
    let nominated = headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .filter_map(|name| HeaderName::from_bytes(name.trim().as_bytes()).ok())
        .collect::<Vec<_>>();
    for name in nominated {
        headers.remove(name);
    }
    for name in [
        header::CONNECTION,
        header::TE,
        header::TRAILER,
        header::TRANSFER_ENCODING,
        header::UPGRADE,
        header::PROXY_AUTHENTICATE,
        header::PROXY_AUTHORIZATION,
        HeaderName::from_static("keep-alive"),
        HeaderName::from_static("proxy-connection"),
    ] {
        headers.remove(name);
    }
}

/// Resolve which llama-cpp service a native request reaches: a `model`
/// field in the JSON body wins, then the sole llama-cpp service when
/// exactly one is configured. Every other case is an error that names
/// the fix. Carries [`ApiErrorCode`] rather than a rendered `Response`
/// so the happy path stays cheap to return.
fn select_target(state: &AppState, body_bytes: &[u8]) -> Result<Target, ApiErrorCode> {
    if let Some(model) = model_from_body(body_bytes) {
        return named_target(state, &model);
    }
    match llama_cpp_targets(state).as_slice() {
        [only] => Ok(only.clone()),
        [] => Err(ApiErrorCode::InvalidRequest {
            reason: "no llama-cpp service is configured to serve this endpoint".into(),
        }),
        many => {
            let names = many
                .iter()
                .map(|t| format!("`{}`", t.name))
                .collect::<Vec<_>>()
                .join(", ");
            Err(ApiErrorCode::InvalidRequest {
                reason: format!(
                    "multiple llama-cpp services are configured ({names}); add a \
                     `model` field to the request body naming one"
                ),
            })
        }
    }
}

/// The llama-cpp template services currently provisioned, in config order.
fn llama_cpp_targets(state: &AppState) -> Vec<Target> {
    let eff = state.config.effective();
    eff.services
        .iter()
        .filter(|s| s.template() == Template::LlamaCpp)
        .filter_map(|s| {
            state.registry.get(&s.name).map(|handle| Target {
                name: s.name.clone(),
                handle,
                private_port: s.private_port,
                max_request_duration_ms: s.max_request_duration_ms,
            })
        })
        .collect()
}

/// Resolve a request-level `model` name, mirroring the OpenAI surface:
/// registry lookup first, then the effective config, gated to the
/// llama-cpp template.
fn named_target(state: &AppState, model: &str) -> Result<Target, ApiErrorCode> {
    let Some(handle) = state.registry.get(model) else {
        return Err(ApiErrorCode::ModelNotFound {
            name: SmolStr::new(model),
        });
    };
    let eff = state.config.effective();
    let Some(svc) = eff.services.iter().find(|s| s.name == model) else {
        return Err(ApiErrorCode::ModelNotFound {
            name: SmolStr::new(model),
        });
    };
    if svc.template() != Template::LlamaCpp {
        return Err(ApiErrorCode::InvalidRequest {
            reason: format!(
                "service `{model}` does not use the llama-cpp template; \
                 llama.cpp-native endpoints only reach llama-cpp services"
            ),
        });
    }
    Ok(Target {
        name: svc.name.clone(),
        handle,
        private_port: svc.private_port,
        max_request_duration_ms: svc.max_request_duration_ms,
    })
}

/// Extract a `model` from a JSON request body if one is present. Native
/// llama.cpp endpoints take JSON bodies and ignore unknown fields, so a
/// client pointed at a gateway can disambiguate between multiple
/// llama-cpp services the same way it would on the OpenAI surface.
fn model_from_body(body_bytes: &[u8]) -> Option<String> {
    let parsed: serde_json::Value = serde_json::from_slice(body_bytes).ok()?;
    parsed
        .get("model")
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

fn ensure_failure_response(name: &SmolStr, f: EnsureFailure) -> Response {
    match f {
        EnsureFailure::InsufficientCapacity(msg) => errors::insufficient_capacity(name, &msg),
        EnsureFailure::ServiceDisabled(msg) => errors::service_disabled(name, &msg),
        EnsureFailure::StartQueueFull => errors::start_queue_full(name),
        EnsureFailure::StartFailed(msg) => errors::start_failed(name, &msg),
        EnsureFailure::Blocked { busy_peers } => errors::service_blocked(name, &busy_peers),
    }
}
