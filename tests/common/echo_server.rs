//! Toy HTTP server used by integration tests in place of a real model backend.
//!
//! Routes:
//! - `GET /health`, `GET /v1/models` → 200 `{}`
//! - `GET /sse` → `text/event-stream` with 5 chunks `data: N\n\n`, 50 ms apart
//! - Anything else → 200 `hello`

// Not every integration test binary uses every symbol in this module.
#![allow(dead_code)]

use std::convert::Infallible;
use std::net::SocketAddr;

use bytes::Bytes;
use http_body_util::{BodyExt, StreamBody, combinators::BoxBody};
use hyper::body::{Frame, Incoming};
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto;
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio_stream::StreamExt as _;

type Body = BoxBody<Bytes, Box<dyn std::error::Error + Send + Sync>>;

fn full(s: &'static str) -> Body {
    http_body_util::Full::new(Bytes::from_static(s.as_bytes()))
        .map_err(|never| match never {})
        .boxed()
}

async fn handle(req: Request<Incoming>) -> Result<Response<Body>, Infallible> {
    let path = req.uri().path().to_owned();

    let resp = match path.as_str() {
        "/health" | "/v1/models" => Response::builder()
            .status(StatusCode::OK)
            .body(full("{}"))
            .unwrap(),

        "/sse" => {
            // Produce 5 SSE events, one every 50 ms.
            let stream = tokio_stream::iter(0u8..5).then(|n| async move {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                let chunk = format!("data: {n}\n\n");
                Ok::<Frame<Bytes>, Box<dyn std::error::Error + Send + Sync>>(Frame::data(
                    Bytes::from(chunk),
                ))
            });
            let body: Body = BodyExt::map_err(StreamBody::new(stream), |e| e).boxed();
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "text/event-stream")
                .header("cache-control", "no-cache")
                .body(body)
                .unwrap()
        }

        _ => Response::builder()
            .status(StatusCode::OK)
            .body(full("hello"))
            .unwrap(),
    };

    Ok(resp)
}

/// Spawns the echo server on `addr` and returns a shutdown sender.
///
/// Drop the returned `watch::Sender` or send `true` to initiate graceful
/// shutdown.  The server task exits within one accept timeout.
pub async fn spawn(addr: SocketAddr) -> watch::Sender<bool> {
    let listener = TcpListener::bind(addr).await.expect("bind echo server");
    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);

    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        return;
                    }
                }
                accept = listener.accept() => {
                    let (stream, _peer) = match accept {
                        Ok(x) => x,
                        Err(_) => continue,
                    };
                    let io = TokioIo::new(stream);
                    tokio::spawn(async move {
                        let svc = service_fn(handle);
                        let _ = auto::Builder::new(TokioExecutor::new())
                            .serve_connection(io, svc)
                            .await;
                    });
                }
            }
        }
    });

    shutdown_tx
}
