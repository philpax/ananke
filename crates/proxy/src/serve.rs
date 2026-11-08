//! Accept loops and per-connection lifecycle bookkeeping for the reverse
//! proxy: the plain [`serve`] loop, the activity-aware
//! [`serve_with_activity`] loop, and the [`WebSocketLifecycle`] /
//! [`ProxyMetrics`] contexts threaded through each connection.

use std::{net::SocketAddr, sync::Arc};

use ananke_errors::ExpectedError;
use futures::future::BoxFuture;
use hyper::{Request, body::Incoming, service::service_fn};
use hyper_util::{
    client::legacy::Client,
    rt::{TokioExecutor, TokioIo},
    server::conn::auto,
};
use tokio::{net::TcpListener, sync::watch};
use tracing::{info, warn};

use crate::{
    InflightGuard, ProxyBody, ProxyError, handle, inflight::InflightCounter,
    metrics::RecorderFactory,
};

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
    pub(crate) inflight: Arc<InflightCounter>,
    pub(crate) activity_ping: Arc<dyn Fn() + Send + Sync>,
}

impl WebSocketLifecycle {
    pub fn new(inflight: Arc<InflightCounter>, activity_ping: Arc<dyn Fn() + Send + Sync>) -> Self {
        Self {
            inflight,
            activity_ping,
        }
    }
}

/// Per-service metrics context threaded from `provision_service` into the
/// proxy request path. Present only for the per-service proxy (the OpenAI
/// multiplexer records its own metrics); when `None` the proxy is a pure
/// byte-forwarder.
///
/// Cloned per connection, so every field is cheap to clone: the factory
/// is an `Arc<dyn Fn>`, `model` is a `SmolStr`, and `run_id` is a closure
/// that reads the supervisor's mirror cell at request time (the run_id
/// changes on every reload, so it cannot be captured eagerly).
#[derive(Clone)]
pub struct ProxyMetrics {
    /// Mints one recorder per token-generating request, capturing the db
    /// handle (and anything else the recorder needs) at construction.
    pub(crate) recorder_factory: RecorderFactory,
    /// Stable service row id, resolved once at provision time.
    pub(crate) service_id: i64,
    /// The service/model name recorded on each `RequestMetric`.
    pub(crate) model: smol_str::SmolStr,
    /// Reads the current run_id at request time — it is reassigned on each
    /// (re)load, so capturing it once would tag metrics with a stale run.
    pub(crate) run_id: Arc<dyn Fn() -> Option<i64> + Send + Sync>,
}

impl ProxyMetrics {
    pub fn new(
        recorder_factory: RecorderFactory,
        service_id: i64,
        model: smol_str::SmolStr,
        run_id: Arc<dyn Fn() -> Option<i64> + Send + Sync>,
    ) -> Self {
        Self {
            recorder_factory,
            service_id,
            model,
            run_id,
        }
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
    inflight_counter: Arc<InflightCounter>,
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
