//! The `combined` log stream, end to end.
//!
//! Neither `docker logs` nor `podman logs` documents a framing that survives
//! the CLI boundary, so container output is tagged `combined` rather than
//! being passed off as stdout. These tests hold that label through the
//! batcher, the broadcast, the REST filter, and the WebSocket — and check
//! that native stdout/stderr capture is untouched by its arrival.
#![cfg(feature = "test-fakes")]

mod common;

use ananke_api::{
    internal::log_line::LogStream,
    live::{LiveFeed, LiveMessage, LiveRequest, LogFeed},
    services::LogsResponse,
};
use ananke_db::logs::{LogLine, Stream};
use ananke_supervise::logs::{spawn_pump_combined, spawn_pump_stderr, spawn_pump_stdout};
use axum::{body::to_bytes, http::StatusCode};
use common::live::{DEADLINE, connect, next_message, send};
use tower::util::ServiceExt;

/// Fetch `/api/services/demo/logs` with an optional `stream` filter.
async fn fetch_logs(state: &ananke::daemon::app_state::AppState, query: &str) -> LogsResponse {
    let app = ananke::api::management::router(state.clone());
    let req = axum::http::Request::builder()
        .method("GET")
        .uri(format!("/api/services/demo/logs{query}"))
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// A reader over `text`, standing in for a `logs --follow` child's stdout.
fn reader(text: &str) -> ananke_system::DynAsyncRead {
    Box::pin(std::io::Cursor::new(text.as_bytes().to_vec()))
}

#[tokio::test(flavor = "current_thread")]
async fn combined_stream_batches_and_broadcasts() {
    let h = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let service_id = h.state.db.upsert_service("demo", 0).await.unwrap();
    let mut live = h.state.batcher.subscribe();

    spawn_pump_combined(
        reader("loading model\nlisten 0.0.0.0:8000\n"),
        service_id,
        1,
        h.state.batcher.clone(),
    );

    // The live broadcast carries the label, so a websocket subscriber can
    // render it without waiting for the row to land.
    let mut received = Vec::new();
    while received.len() < 2 {
        let batch = live.recv().await.unwrap();
        received.extend(batch.for_service(service_id).cloned());
    }
    assert!(received.iter().all(|l| l.stream == "combined"));
    let texts: Vec<&str> = received.iter().map(|l| l.line.as_str()).collect();
    assert_eq!(texts, ["loading model", "listen 0.0.0.0:8000"]);

    h.state.batcher.flush().await;
    let rows = h.state.db.fetch_service_logs(service_id).await.unwrap();
    assert_eq!(rows.len(), 2);
    assert!(
        rows.iter().all(|r| r.stream == "combined"),
        "persisted rows must keep the combined label: {rows:?}"
    );
    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn combined_stream_rest_filter_roundtrip() {
    let h = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let service_id = h.state.db.upsert_service("demo", 0).await.unwrap();

    for (stream, line) in [
        (Stream::Stdout, "from stdout"),
        (Stream::Stderr, "from stderr"),
        (Stream::Combined, "from a container"),
    ] {
        h.state.batcher.push(LogLine {
            service_id,
            run_id: 1,
            timestamp_ms: 1_000,
            stream,
            line: line.to_string(),
        });
    }
    h.state.batcher.flush().await;

    // Unfiltered includes every stream.
    let all = fetch_logs(&h.state, "").await;
    assert_eq!(all.logs.len(), 3);

    // Asking for combined returns only the container's line — and asking for
    // stdout must not sweep it up, or a container's output would masquerade
    // as the child's own stdout.
    let combined = fetch_logs(&h.state, "?stream=combined").await;
    assert_eq!(combined.logs.len(), 1);
    assert_eq!(combined.logs[0].line, "from a container");
    assert_eq!(combined.logs[0].stream, "combined");

    let stdout = fetch_logs(&h.state, "?stream=stdout").await;
    assert_eq!(stdout.logs.len(), 1);
    assert_eq!(stdout.logs[0].line, "from stdout");

    let stderr = fetch_logs(&h.state, "?stream=stderr").await;
    assert_eq!(stderr.logs.len(), 1);
    assert_eq!(stderr.logs[0].line, "from stderr");

    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn native_workload_preserves_split_stdout_stderr() {
    // Adding a third stream must not blur the two a native process has.
    let h = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let service_id = h.state.db.upsert_service("demo", 0).await.unwrap();

    spawn_pump_stdout(reader("out one\n"), service_id, 1, h.state.batcher.clone());
    spawn_pump_stderr(reader("err one\n"), service_id, 1, h.state.batcher.clone());
    h.state.batcher.flush().await;
    // The pumps are spawned tasks; a second flush covers the interleaving.
    tokio::task::yield_now().await;
    h.state.batcher.flush().await;

    let rows = h.state.db.fetch_service_logs(service_id).await.unwrap();
    assert_eq!(rows.len(), 2, "got {rows:?}");
    let stdout = rows.iter().find(|r| r.stream == "stdout").unwrap();
    let stderr = rows.iter().find(|r| r.stream == "stderr").unwrap();
    assert_eq!(stdout.line, "out one");
    assert_eq!(stderr.line, "err one");
    assert!(!rows.iter().any(|r| r.stream == "combined"));

    h.cleanup().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn combined_stream_live_follow_delivery() {
    let h = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let service_id = h.state.db.upsert_service("demo", 0).await.unwrap();
    let mut socket = connect(&h).await;
    send(
        &mut socket,
        LiveRequest::Follow {
            id: "l".into(),
            feed: LiveFeed::Logs(LogFeed {
                service: "demo".into(),
                after: None,
                run: None,
                stream: Some(LogStream::Combined),
            }),
        },
    )
    .await;
    assert!(matches!(
        next_message(&mut socket, DEADLINE).await,
        Some(LiveMessage::LogCaughtUp { .. })
    ));

    h.state.batcher.push(LogLine {
        service_id,
        run_id: 1,
        timestamp_ms: 1_000,
        stream: Stream::Combined,
        line: "container says hello".to_string(),
    });
    h.state.batcher.flush().await;

    let Some(LiveMessage::LogLines { lines, .. }) = next_message(&mut socket, DEADLINE).await
    else {
        panic!("the follow did not deliver the line");
    };
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0].stream, "combined");
    assert_eq!(lines[0].line, "container says hello");

    h.cleanup().await;
}
