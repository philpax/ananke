//! Test-harness echo server: spawn counter, /sink, and configurable /v1/* bodies.
#![cfg(feature = "test-fakes")]
// Not every integration test binary uses every symbol in this module.
#![allow(dead_code)]

use std::{
    convert::Infallible,
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
    time::Duration,
};

use bytes::Bytes;
use http_body_util::{BodyExt, Full, StreamBody, combinators::BoxBody};
use hyper::{Request, Response, StatusCode, body::Frame, service::service_fn};
use hyper_util::{
    rt::{TokioExecutor, TokioIo},
    server::conn::auto,
};
use parking_lot::Mutex;
use tokio::{net::TcpListener, sync::watch};
use tokio_stream::wrappers::ReceiverStream;

/// State shared across all connections to track spawns and collect request bodies.
#[derive(Clone, Default)]
pub struct EchoState {
    /// Counter incremented each time `serve()` is called.
    pub spawn_counter: Arc<AtomicU32>,
    /// Sink for recording request bodies from /v1/* endpoints.
    pub sink: Arc<Mutex<Vec<serde_json::Value>>>,
}

type EchoBody = BoxBody<Bytes, Box<dyn std::error::Error + Send + Sync>>;

/// Serves HTTP requests on `addr` with the given `state`.
///
/// Increments `state.spawn_counter` on entry. On `/v1/chat/completions`,
/// `/v1/completions`, and `/v1/embeddings`, records the request body into
/// `state.sink`. `/health` and `/v1/models` responses include the
/// `x-echo-spawn-count` header. `/sse` streams 5 events 50ms apart.
/// Anything else returns "hello".
pub async fn serve(addr: SocketAddr, state: EchoState, mut shutdown: watch::Receiver<bool>) {
    state.spawn_counter.fetch_add(1, Ordering::Relaxed);
    let listener = TcpListener::bind(addr).await.expect("echo bind");
    loop {
        tokio::select! {
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    return;
                }
            }
            accept = listener.accept() => {
                let Ok((stream, _)) = accept else { continue; };
                let io = TokioIo::new(stream);
                let state = state.clone();
                tokio::spawn(async move {
                    let svc = service_fn(move |req| {
                        let state = state.clone();
                        handle(req, state)
                    });
                    let _ = auto::Builder::new(TokioExecutor::new())
                        .serve_connection(io, svc)
                        .await;
                });
            }
        }
    }
}

async fn handle(
    req: Request<hyper::body::Incoming>,
    state: EchoState,
) -> Result<Response<EchoBody>, Infallible> {
    let path = req.uri().path();

    match path {
        "/health" | "/v1/models" => {
            let body = Full::new(Bytes::from("{}")).map_err(|n| match n {}).boxed();
            Ok(Response::builder()
                .status(StatusCode::OK)
                .header(
                    "x-echo-spawn-count",
                    state.spawn_counter.load(Ordering::Relaxed).to_string(),
                )
                .body(body)
                .unwrap())
        }

        "/sse" => {
            let (tx, rx) = tokio::sync::mpsc::channel::<
                Result<Frame<Bytes>, Box<dyn std::error::Error + Send + Sync>>,
            >(8);
            tokio::spawn(async move {
                for i in 0..5 {
                    let chunk = format!("data: {i}\n\n");
                    if tx.send(Ok(Frame::data(Bytes::from(chunk)))).await.is_err() {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            });
            let stream = ReceiverStream::new(rx);
            let body = StreamBody::new(stream).boxed();
            Ok(Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "text/event-stream")
                .body(body)
                .unwrap())
        }

        "/v1/chat/completions" | "/v1/completions" | "/v1/embeddings" => {
            let body_bytes = req
                .into_body()
                .collect()
                .await
                .map(|c| c.to_bytes())
                .unwrap_or_default();
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&body_bytes) {
                state.sink.lock().push(v);
            }
            let body = Full::new(Bytes::from(
                r#"{"id":"cmpl-echo","choices":[{"message":{"role":"assistant","content":"ok"}}]}"#,
            ))
            .map_err(|n| match n {})
            .boxed();
            Ok(Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/json")
                .body(body)
                .unwrap())
        }

        _ => {
            let body = Full::new(Bytes::from("hello"))
                .map_err(|n| match n {})
                .boxed();
            Ok(Response::builder()
                .status(StatusCode::OK)
                .body(body)
                .unwrap())
        }
    }
}
