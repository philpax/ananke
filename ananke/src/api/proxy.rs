//! Per-service reverse HTTP proxy.

use std::{
    convert::Infallible,
    error::Error,
    net::SocketAddr,
    sync::{Arc, atomic::AtomicU64},
    time::{Duration, Instant},
};

use bytes::Bytes;
use futures::{TryStreamExt, future::BoxFuture};
use http_body_util::{BodyExt, Empty, Full, StreamBody};
use hyper::{
    Request, Response, StatusCode,
    body::{Frame, Incoming},
    header,
    service::service_fn,
};
use hyper_util::{
    client::legacy::Client,
    rt::{TokioExecutor, TokioIo},
    server::conn::auto,
};
use tokio::{net::TcpListener, sync::watch};
use tracing::{debug, info, warn};

use crate::{
    api::{
        errors::ApiErrorCode,
        openai::metrics::{MetricsBody, MetricsRecorder},
    },
    db::Database,
    errors::ExpectedError,
    tracking::inflight::InflightGuard,
};

/// How often to bump the per-service activity stamp while a WebSocket
/// session is open. Without this, the supervisor's idle-eviction loop
/// reads a stale stamp and SIGTERMs the service mid-session — there is
/// no traffic-based renewal because `copy_bidirectional` never touches
/// the activity table itself. 5 s is fast enough to keep any reasonable
/// `idle_timeout_ms` value (the smallest configurable is in the tens of
/// seconds) from racing the ticker, and the ping is just a single
/// `Mutex<Instant>` write so the overhead is negligible.
const WS_ACTIVITY_PING_INTERVAL: Duration = Duration::from_secs(5);

/// Boxed body type used for both upstream requests and downstream responses.
pub type ProxyBody =
    http_body_util::combinators::BoxBody<Bytes, Box<dyn std::error::Error + Send + Sync>>;

/// Short-circuit error response returned by a `before_request` hook. Alias of
/// `Response<ProxyBody>`; kept as a separate name so callers that build an
/// error reply need not reach into `ProxyBody` themselves.
pub type ProxyError = Response<ProxyBody>;

/// Per-upgrade-session bookkeeping handed down from `serve_with_activity`
/// to `handle_upgrade`. A WebSocket session lives well beyond the HTTP
/// handler that birthed it, so the proxy needs hooks that survive past
/// the `handle()` return:
///
/// * `inflight` mints a long-lived [`InflightGuard`] that pins the
///   service open against `drain_pipeline` (which polls until the
///   counter reaches zero, bounded by `max_request_duration`).
/// * `activity_ping` is invoked periodically by the splice task so the
///   running-state idle timeout never trips on a quietly-active session.
///
/// Cloneable so the same lifecycle pack can be threaded into every
/// per-connection task without recomputing the captures each time.
#[derive(Clone)]
pub struct WebSocketLifecycle {
    inflight: Arc<AtomicU64>,
    activity_ping: Arc<dyn Fn() + Send + Sync>,
}

impl WebSocketLifecycle {
    pub fn new(inflight: Arc<AtomicU64>, activity_ping: Arc<dyn Fn() + Send + Sync>) -> Self {
        Self {
            inflight,
            activity_ping,
        }
    }
}

/// Per-service metrics context threaded from `provision_service` into the
/// proxy request path. Present only for the per-service proxy (the OpenAI
/// multiplexer records its own metrics); when `None` the proxy is a pure
/// byte-forwarder as before.
///
/// Cloned per connection, so every field is cheap to clone: `Database` is an
/// `Arc`-backed handle, `model` is a `SmolStr`, and `run_id` is a closure
/// that reads the supervisor's mirror cell at request time (the run_id
/// changes on every reload, so it cannot be captured eagerly).
#[derive(Clone)]
pub struct ProxyMetrics {
    db: Database,
    /// Stable service row id, resolved once at provision time.
    service_id: i64,
    /// The service/model name recorded on each `RequestMetric`.
    model: smol_str::SmolStr,
    /// Reads the current run_id at request time — it is reassigned on each
    /// (re)load, so capturing it once would tag metrics with a stale run.
    run_id: Arc<dyn Fn() -> Option<i64> + Send + Sync>,
}

impl ProxyMetrics {
    pub fn new(
        db: Database,
        service_id: i64,
        model: smol_str::SmolStr,
        run_id: Arc<dyn Fn() -> Option<i64> + Send + Sync>,
    ) -> Self {
        Self {
            db,
            service_id,
            model,
            run_id,
        }
    }
}

/// Map a request path to the token-generating endpoint whose responses carry
/// `usage`/`timings`. Returns the matched `&'static str` for the metric's
/// `endpoint` column, or `None` for every other path (`/health`, `/metrics`,
/// `/v1/models`, upgrades, …) so those are forwarded without recording.
fn metrics_endpoint(path: &str) -> Option<&'static str> {
    match path {
        "/v1/chat/completions" => Some("/v1/chat/completions"),
        "/v1/completions" => Some("/v1/completions"),
        _ => None,
    }
}

pub async fn serve(
    listen: SocketAddr,
    upstream_port: u16,
    mut shutdown: watch::Receiver<bool>,
) -> Result<(), ExpectedError> {
    let listener = TcpListener::bind(listen)
        .await
        .map_err(|e| ExpectedError::bind_failed(listen.to_string(), e.to_string()))?;
    info!(%listen, upstream_port, "proxy listening");

    let client = Client::builder(TokioExecutor::new()).build_http::<ProxyBody>();

    loop {
        tokio::select! {
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!(%listen, "proxy shutting down");
                    return Ok(());
                }
            }
            accept = listener.accept() => {
                let (stream, peer) = match accept {
                    Ok(x) => x,
                    Err(e) => { warn!(error = %e, "accept failed"); continue; }
                };
                let io = TokioIo::new(stream);
                let client = client.clone();
                tokio::spawn(async move {
                    let svc = service_fn(move |req: Request<Incoming>| {
                        let client = client.clone();
                        async move { handle(req, client, upstream_port, peer, None, None).await }
                    });
                    if let Err(e) = auto::Builder::new(TokioExecutor::new())
                        .serve_connection_with_upgrades(io, svc)
                        .await
                    {
                        warn!(error = %e, "conn error");
                    }
                });
            }
        }
    }
}

/// Like [`serve`] but runs `before_request()` before forwarding each request
/// and threads a per-service [`WebSocketLifecycle`] through to the upgrade
/// path so long-lived sessions hold the service open and keep its activity
/// stamp fresh.
///
/// The closure returns a future that resolves to `None` (proceed with proxying)
/// or `Some(response)` (short-circuit with that response — used to trigger
/// on-demand service start and return 503s when the supervisor cannot start).
/// `inflight_counter` is incremented for each in-flight request and
/// decremented when the response completes; the same counter is reused
/// inside [`WebSocketLifecycle`] for the long-running splice guard.
pub async fn serve_with_activity(
    listen: SocketAddr,
    upstream_port: u16,
    mut shutdown: watch::Receiver<bool>,
    before_request: Arc<dyn Fn() -> BoxFuture<'static, Option<ProxyError>> + Send + Sync>,
    inflight_counter: Arc<AtomicU64>,
    activity_ping: Arc<dyn Fn() + Send + Sync>,
    metrics: Option<ProxyMetrics>,
) -> Result<(), ExpectedError> {
    let listener = TcpListener::bind(listen)
        .await
        .map_err(|e| ExpectedError::bind_failed(listen.to_string(), e.to_string()))?;
    info!(%listen, upstream_port, "proxy listening");

    let client = Client::builder(TokioExecutor::new()).build_http::<ProxyBody>();
    let ws_lifecycle = WebSocketLifecycle::new(inflight_counter.clone(), activity_ping);

    loop {
        tokio::select! {
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!(%listen, "proxy shutting down");
                    return Ok(());
                }
            }
            accept = listener.accept() => {
                let (stream, peer) = match accept {
                    Ok(x) => x,
                    Err(e) => { warn!(error = %e, "accept failed"); continue; }
                };
                let io = TokioIo::new(stream);
                let client = client.clone();
                let before_request = before_request.clone();
                let counter = inflight_counter.clone();
                let ws_lifecycle = ws_lifecycle.clone();
                let metrics = metrics.clone();
                tokio::spawn(async move {
                    let svc = service_fn(move |req: Request<Incoming>| {
                        let fut = (before_request)();
                        let counter = counter.clone();
                        let client = client.clone();
                        let ws_lifecycle = ws_lifecycle.clone();
                        let metrics = metrics.clone();
                        async move {
                            if let Some(short) = fut.await {
                                return Ok(short);
                            }
                            // The closure-scoped guard covers the HTTP
                            // handler's lifetime. For an upgrade request
                            // `handle_upgrade` mints a second, longer-lived
                            // guard out of `ws_lifecycle.inflight` and
                            // hands it to the splice task; the brief
                            // overlap is harmless because the drain
                            // pipeline only cares that the counter reaches
                            // zero, not what its peak was.
                            let _guard = InflightGuard::new(counter);
                            handle(req, client, upstream_port, peer, Some(ws_lifecycle), metrics)
                                .await
                        }
                    });
                    if let Err(e) = auto::Builder::new(TokioExecutor::new())
                        .serve_connection_with_upgrades(io, svc)
                        .await
                    {
                        warn!(error = %e, "conn error");
                    }
                });
            }
        }
    }
}

/// Build the standard JSON error response on the hyper proxy data
/// plane. Pairs with `ApiErrorCode::into_response` on the axum side —
/// both serialise the same `ApiError` body the typed code projects to,
/// so a client gets a byte-identical body regardless of which surface
/// emitted the failure.
pub fn error_response(code: ApiErrorCode) -> ProxyError {
    let status = code.status();
    let body: ananke_api::shared::errors::ApiError = code.into();
    let body_bytes = serde_json::to_vec(&body).unwrap_or_default();
    let full: ProxyBody = Full::new(Bytes::from(body_bytes))
        .map_err(|never| -> Box<dyn Error + Send + Sync> { match never {} })
        .boxed();
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(full)
        .unwrap()
}

/// Reverse-proxy a single request to the upstream, returning an infallible result.
///
/// All upstream and protocol errors are translated into HTTP error responses so that
/// `serve_connection` never sees a service error (avoiding the lifetime trouble with
/// `Box<dyn Error + Send + Sync>` and the `From` blanket impl).
async fn handle(
    req: Request<Incoming>,
    client: Client<hyper_util::client::legacy::connect::HttpConnector, ProxyBody>,
    upstream_port: u16,
    peer: SocketAddr,
    ws_lifecycle: Option<WebSocketLifecycle>,
    metrics: Option<ProxyMetrics>,
) -> Result<Response<ProxyBody>, Infallible> {
    match try_handle(req, client, upstream_port, peer, ws_lifecycle, metrics).await {
        Ok(resp) => Ok(resp),
        Err(e) => {
            warn!(error = %e, peer = %peer, "proxy error");
            Ok(error_response(ApiErrorCode::ProxyInternal {
                reason: e.to_string(),
            }))
        }
    }
}

async fn try_handle(
    req: Request<Incoming>,
    client: Client<hyper_util::client::legacy::connect::HttpConnector, ProxyBody>,
    upstream_port: u16,
    peer: SocketAddr,
    ws_lifecycle: Option<WebSocketLifecycle>,
    metrics: Option<ProxyMetrics>,
) -> Result<Response<ProxyBody>, Box<dyn std::error::Error + Send + Sync>> {
    // Upgrade requests (WebSocket and friends) need a raw byte splice between
    // the client and the upstream after the 101 — the pooled HTTP client
    // can't model that, and stripping the response's `Connection` header
    // would make aiohttp's WebSocket client reject the handshake. Upgrades
    // never carry token usage, so they bypass the metrics path entirely.
    if is_upgrade_request(&req) {
        return handle_upgrade(req, upstream_port, peer, ws_lifecycle).await;
    }

    // Only the token-generating endpoints get recorded; every other path is a
    // pure passthrough. Match on the path (sans query) before it is consumed.
    let metric_endpoint = metrics
        .as_ref()
        .and_then(|_| metrics_endpoint(req.uri().path()));
    // Wall-clock start for the request, captured before the upstream round-trip
    // so TTFT and total duration cover the full server-visible latency.
    let request_start = Instant::now();

    let (parts, body) = req.into_parts();
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or("/");
    let uri = format!("http://127.0.0.1:{upstream_port}{path_and_query}").parse::<hyper::Uri>()?;

    let mut upstream_req = Request::builder().method(parts.method.clone()).uri(uri);
    for (k, v) in parts.headers.iter() {
        if k == header::HOST {
            continue;
        }
        upstream_req = upstream_req.header(k, v);
    }
    let body_bytes = body.collect().await?.to_bytes();
    let upstream_body: ProxyBody = http_body_util::Full::new(body_bytes)
        .map_err(|never| match never {})
        .boxed();
    let upstream_req = upstream_req.body(upstream_body)?;

    let resp = match client.request(upstream_req).await {
        Ok(r) => r,
        Err(e) => {
            warn!(error = %e, peer = %peer, "upstream request failed");
            return Ok(error_response(ApiErrorCode::UpstreamUnavailable {
                reason: e.to_string(),
            }));
        }
    };

    let (parts, body) = resp.into_parts();
    // Stream the body back without buffering — critical for SSE.
    let stream = body.into_data_stream().map_ok(Frame::data);
    let boxed: ProxyBody = BodyExt::map_err(
        StreamBody::new(stream),
        |e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) },
    )
    .boxed();

    // Record per-request token metrics for the token-generating endpoints,
    // mirroring the OpenAI multiplexer's `MetricsBody` wrap. `is_streaming` is
    // inferred from the upstream `Content-Type` (`text/event-stream`); the
    // recorder handles both SSE and plain JSON, so this only affects TTFT
    // bookkeeping. Any other endpoint (or an absent context) forwards the body
    // verbatim.
    let body: ProxyBody = match (metric_endpoint, metrics) {
        (Some(endpoint), Some(metrics)) => {
            let status_code = parts.status.as_u16();
            let is_streaming = parts
                .headers
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .map(|v| v.starts_with("text/event-stream"))
                .unwrap_or(false);
            let recorder = MetricsRecorder::new(
                request_start,
                metrics.service_id,
                (metrics.run_id)(),
                metrics.model.to_string(),
                endpoint,
                is_streaming,
            );
            MetricsBody::new(boxed, recorder, metrics.db.clone(), status_code).boxed()
        }
        _ => boxed,
    };

    let mut out = Response::from_parts(parts, body);
    out.headers_mut().remove(header::CONNECTION);
    out.headers_mut().remove("transfer-encoding");
    Ok(out)
}

/// True if `req` carries an HTTP/1.1 `Upgrade` handshake (WebSocket, h2c, …).
///
/// Per RFC 7230 §6.7 the request must list the `upgrade` token in
/// `Connection` and name the target protocol in `Upgrade`. Both checks are
/// case-insensitive; `Connection` may contain other tokens alongside
/// `upgrade` (e.g. `keep-alive, Upgrade`).
fn is_upgrade_request(req: &Request<Incoming>) -> bool {
    let has_upgrade_token = req
        .headers()
        .get(header::CONNECTION)
        .and_then(|v| v.to_str().ok())
        .map(|s| {
            s.split(',')
                .any(|tok| tok.trim().eq_ignore_ascii_case("upgrade"))
        })
        .unwrap_or(false);
    let has_upgrade_target = req
        .headers()
        .get(header::UPGRADE)
        .and_then(|v| v.to_str().ok())
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    has_upgrade_token && has_upgrade_target
}

/// Proxy a single HTTP/1.1 upgrade request (typically WebSocket).
///
/// Opens a dedicated upstream connection (the pooled legacy client cannot
/// retain a half-upgraded socket), forwards the handshake, and — on a 101 —
/// splices both sides' upgraded I/O bidirectionally. The 101 response is
/// returned to the caller with the upstream's `Connection` and `Upgrade`
/// headers verbatim; this is the constraint aiohttp's WebSocket client
/// enforces (it rejects any `Connection` value other than literally
/// `upgrade`).
async fn handle_upgrade(
    mut req: Request<Incoming>,
    upstream_port: u16,
    peer: SocketAddr,
    ws_lifecycle: Option<WebSocketLifecycle>,
) -> Result<Response<ProxyBody>, Box<dyn std::error::Error + Send + Sync>> {
    // Detach the client-side upgrade handle before consuming the request.
    // It will resolve once hyper writes our 101 back and releases the TCP
    // socket for splicing.
    let client_upgrade = hyper::upgrade::on(&mut req);

    let (parts, _body) = req.into_parts();
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or("/");
    let uri = format!("http://127.0.0.1:{upstream_port}{path_and_query}").parse::<hyper::Uri>()?;

    let upstream_stream = match tokio::net::TcpStream::connect(("127.0.0.1", upstream_port)).await {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, peer = %peer, "upstream upgrade dial failed");
            return Ok(error_response(ApiErrorCode::UpstreamUnavailable {
                reason: e.to_string(),
            }));
        }
    };
    let upstream_io = TokioIo::new(upstream_stream);
    let (mut sender, conn) = hyper::client::conn::http1::handshake(upstream_io).await?;
    // The connection driver future must be polled for `send_request` to
    // make progress AND for the upstream's upgrade to fire — without
    // `with_upgrades()` the conn returns the 101 but never hands over
    // the socket. We let the spawned task run unobserved; it terminates
    // naturally once the upgrade fires (or the connection errors out).
    tokio::spawn(async move {
        if let Err(e) = conn.with_upgrades().await {
            debug!(error = %e, "upstream upgrade connection driver ended");
        }
    });

    let mut upstream_builder = Request::builder().method(parts.method.clone()).uri(uri);
    upstream_builder = upstream_builder.header(header::HOST, format!("127.0.0.1:{upstream_port}"));
    for (k, v) in parts.headers.iter() {
        if k == header::HOST {
            continue;
        }
        upstream_builder = upstream_builder.header(k, v);
    }
    let empty_body: ProxyBody = Empty::<Bytes>::new()
        .map_err(|never| -> Box<dyn std::error::Error + Send + Sync> { match never {} })
        .boxed();
    let upstream_req = upstream_builder.body(empty_body)?;

    let mut upstream_resp = match sender.send_request(upstream_req).await {
        Ok(r) => r,
        Err(e) => {
            warn!(error = %e, peer = %peer, "upstream upgrade request failed");
            return Ok(error_response(ApiErrorCode::UpstreamUnavailable {
                reason: e.to_string(),
            }));
        }
    };

    if upstream_resp.status() != StatusCode::SWITCHING_PROTOCOLS {
        // Upstream refused the upgrade. Pass the response back as a normal
        // streamed body and abandon the upgrade. Strip the connection
        // controls so hyper's encoder can pick the right framing for the
        // downstream socket.
        let (mut resp_parts, body) = upstream_resp.into_parts();
        resp_parts.headers.remove(header::CONNECTION);
        resp_parts.headers.remove("transfer-encoding");
        let stream = body.into_data_stream().map_ok(Frame::data);
        let boxed: ProxyBody = BodyExt::map_err(
            StreamBody::new(stream),
            |e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) },
        )
        .boxed();
        return Ok(Response::from_parts(resp_parts, boxed));
    }

    let upstream_upgrade = hyper::upgrade::on(&mut upstream_resp);
    let upstream_version = upstream_resp.version();
    let upstream_headers = upstream_resp.headers().clone();

    // Mint the session-scoped lifecycle bookkeeping *before* spawning the
    // splice task so the guard is owned by the spawned future from the
    // very first poll — the drain pipeline races the counter and we
    // don't want a window where the count has dipped to zero. The
    // closure-scoped guard the caller installed is still alive at this
    // point, so the count is at least 2 until `handle_upgrade` returns
    // and the caller's guard drops back to 1.
    let session_guard = ws_lifecycle
        .as_ref()
        .map(|l| InflightGuard::new(l.inflight.clone()));
    let activity_ping = ws_lifecycle.map(|l| l.activity_ping);

    // Once hyper writes our 101 the client_upgrade future resolves; the
    // upstream_upgrade future resolves the moment the connection driver
    // sees the 101 on the wire. Joining both gives us paired I/O halves
    // we can splice byte-for-byte.
    tokio::spawn(async move {
        // Pin the guard into the task's stack so an early-return path
        // (e.g. handshake failure on `try_join!`) still drops it when
        // the task ends rather than at any earlier scope boundary.
        let _session_guard = session_guard;
        let (client_upg, upstream_upg) = match tokio::try_join!(client_upgrade, upstream_upgrade) {
            Ok(pair) => pair,
            Err(e) => {
                warn!(error = %e, peer = %peer, "websocket upgrade negotiation failed");
                return;
            }
        };
        let client_io = TokioIo::new(client_upg);
        let upstream_io = TokioIo::new(upstream_upg);
        let (mut client_r, mut client_w) = tokio::io::split(client_io);
        let (mut upstream_r, mut upstream_w) = tokio::io::split(upstream_io);

        // First-close-wins splice. `copy_bidirectional` waits for *both*
        // halves to EOF, which leaks the task if either peer half-
        // closes — the standard failure mode when a client process
        // exits without sending a WebSocket Close frame. select! drops
        // the loser the moment one direction returns, and dropping it
        // closes the underlying socket halves on the next poll.
        let splice = async {
            tokio::select! {
                r = tokio::io::copy(&mut client_r, &mut upstream_w) => r.map(|_| ()),
                r = tokio::io::copy(&mut upstream_r, &mut client_w) => r.map(|_| ()),
            }
        };
        if let Some(ping) = activity_ping {
            tokio::pin!(splice);
            let mut ticker = tokio::time::interval(WS_ACTIVITY_PING_INTERVAL);
            // `Interval` fires immediately on first tick; skip that so
            // the first real ping happens one interval in, by which
            // point the closure-scoped `_guard` from the caller has
            // dropped and the session-scoped guard is the sole holder.
            ticker.tick().await;
            loop {
                tokio::select! {
                    result = &mut splice => {
                        if let Err(e) = result {
                            debug!(error = %e, peer = %peer, "websocket splice ended");
                        }
                        break;
                    }
                    _ = ticker.tick() => {
                        ping();
                    }
                }
            }
        } else if let Err(e) = splice.await {
            debug!(error = %e, peer = %peer, "websocket splice ended");
        }
    });

    // Forward the upstream's response headers verbatim — most importantly,
    // `Connection: Upgrade` and `Upgrade: websocket`. aiohttp's WebSocket
    // client validates `Connection` with exact equality after lowercasing,
    // so any rewrite (adding `keep-alive`, replacing with the encoder's
    // default, reordering) breaks the handshake.
    let mut builder = Response::builder()
        .status(StatusCode::SWITCHING_PROTOCOLS)
        .version(upstream_version);
    for (k, v) in upstream_headers.iter() {
        builder = builder.header(k, v);
    }
    let empty_body: ProxyBody = Empty::<Bytes>::new()
        .map_err(|never| -> Box<dyn std::error::Error + Send + Sync> { match never {} })
        .boxed();
    Ok(builder.body(empty_body)?)
}
