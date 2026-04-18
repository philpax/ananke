//! Per-service reverse HTTP proxy.

use std::convert::Infallible;
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use bytes::Bytes;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto;
use tokio::net::TcpListener;
use tokio::sync::watch;
use tracing::{info, warn};

use crate::errors::ExpectedError;
use crate::inflight::InflightGuard;

/// Boxed body type used for both upstream requests and downstream responses.
type ProxyBody =
    http_body_util::combinators::BoxBody<Bytes, Box<dyn std::error::Error + Send + Sync>>;

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
                        async move { handle(req, client, upstream_port, peer).await }
                    });
                    if let Err(e) = auto::Builder::new(TokioExecutor::new())
                        .serve_connection(io, svc)
                        .await
                    {
                        warn!(error = %e, "conn error");
                    }
                });
            }
        }
    }
}

/// Like [`serve`] but runs `before_request()` before forwarding each request.
///
/// The closure returns a future that resolves to `None` (proceed with proxying)
/// or `Some(response)` (short-circuit with that response — used to trigger
/// on-demand service start and return 503s when the supervisor cannot start).
/// Also accepts an `inflight_counter` that is incremented for each in-flight
/// request and decremented when the response (including streaming body) completes.
pub async fn serve_with_activity(
    listen: SocketAddr,
    upstream_port: u16,
    mut shutdown: watch::Receiver<bool>,
    before_request: Arc<dyn Fn() -> BoxFuture<'static, Option<Response<ProxyBody>>> + Send + Sync>,
    inflight_counter: Arc<AtomicU64>,
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
                let before_request = before_request.clone();
                let counter = inflight_counter.clone();
                tokio::spawn(async move {
                    let svc = service_fn(move |req: Request<Incoming>| {
                        let fut = (before_request)();
                        let counter = counter.clone();
                        let client = client.clone();
                        async move {
                            if let Some(short) = fut.await {
                                return Ok(short);
                            }
                            let _guard = InflightGuard::new(counter);
                            handle(req, client, upstream_port, peer).await
                        }
                    });
                    if let Err(e) = auto::Builder::new(TokioExecutor::new())
                        .serve_connection(io, svc)
                        .await
                    {
                        warn!(error = %e, "conn error");
                    }
                });
            }
        }
    }
}

/// Build a 503 Service Unavailable response with an OpenAI-shaped JSON error body.
pub fn error_response(code: &str, message: &str) -> Response<ProxyBody> {
    let body_json = serde_json::json!({
        "error": {"code": code, "message": message, "type": "server_error"}
    });
    let body_bytes = serde_json::to_vec(&body_json).unwrap_or_default();
    let full: ProxyBody = Full::new(Bytes::from(body_bytes))
        .map_err(|never| -> Box<dyn Error + Send + Sync> { match never {} })
        .boxed();
    Response::builder()
        .status(StatusCode::SERVICE_UNAVAILABLE)
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
) -> Result<Response<ProxyBody>, Infallible> {
    match try_handle(req, client, upstream_port, peer).await {
        Ok(resp) => Ok(resp),
        Err(e) => {
            warn!(error = %e, peer = %peer, "proxy error");
            let body = http_body_util::Full::new(Bytes::from("proxy error"))
                .map_err(|never| match never {})
                .boxed();
            Ok(Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(body)
                .unwrap_or_else(|_| {
                    Response::new(
                        http_body_util::Full::new(Bytes::new())
                            .map_err(|never| match never {})
                            .boxed(),
                    )
                }))
        }
    }
}

async fn try_handle(
    req: Request<Incoming>,
    client: Client<hyper_util::client::legacy::connect::HttpConnector, ProxyBody>,
    upstream_port: u16,
    peer: SocketAddr,
) -> Result<Response<ProxyBody>, Box<dyn std::error::Error + Send + Sync>> {
    let (parts, body) = req.into_parts();
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or("/");
    let uri = format!("http://127.0.0.1:{upstream_port}{path_and_query}").parse::<hyper::Uri>()?;

    let mut upstream_req = Request::builder().method(parts.method.clone()).uri(uri);
    for (k, v) in parts.headers.iter() {
        if k == hyper::header::HOST {
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
            let body = http_body_util::Full::new(Bytes::from("upstream unavailable"))
                .map_err(|never| match never {})
                .boxed();
            return Ok(Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(body)?);
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
    let mut out = Response::from_parts(parts, boxed);
    out.headers_mut().remove(hyper::header::CONNECTION);
    out.headers_mut().remove("transfer-encoding");
    Ok(out)
}
