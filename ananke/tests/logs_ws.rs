//! Integration tests for the `GET /api/services/{name}/logs/stream` WebSocket endpoint.

mod common;

use ananke::db::logs::{LogLine, Stream};
use ananke_api::LogLine as ApiLogLine;
use futures::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

#[tokio::test(flavor = "current_thread")]
async fn log_line_arrives_over_websocket() {
    let harness = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let service_id = harness.state.db.upsert_service("demo", 0).await.unwrap();

    let addr = harness.spawn_management_server().await;
    let ws_url =
        common::management_url(addr, "/api/services/demo/logs/stream").replace("http://", "ws://");
    let (mut ws, _) = connect_async(ws_url).await.unwrap();

    // Push a log line and flush it so the broadcast fires.
    harness.state.batcher.push(LogLine {
        service_id,
        run_id: 1,
        timestamp_ms: 12345,
        stream: Stream::Stdout,
        line: "hello from test".into(),
    });
    harness.state.batcher.flush().await;

    // Expect the log line to arrive on the WebSocket within a reasonable window.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut received: Option<ApiLogLine> = None;
    while std::time::Instant::now() < deadline {
        let recv = tokio::time::timeout(std::time::Duration::from_millis(500), ws.next()).await;
        if let Ok(Some(Ok(Message::Text(s)))) = recv
            && let Ok(line) = serde_json::from_str::<ApiLogLine>(&s)
        {
            received = Some(line);
            break;
        }
    }

    let line = received.expect("expected a LogLine on the WebSocket");
    assert_eq!(line.timestamp_ms, 12345);
    assert_eq!(line.stream, "stdout");
    assert_eq!(line.line, "hello from test");
    assert_eq!(line.run_id, 1);
    assert_eq!(line.seq, 1);

    let _ = ws.send(Message::Close(None)).await;
    harness.shutdown().await;
}

#[tokio::test(flavor = "current_thread")]
async fn unknown_service_returns_not_found() {
    let harness = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let addr = harness.spawn_management_server().await;

    // The WS upgrade for an unknown service should be rejected at the HTTP
    // level with 404 before the WS handshake completes.
    let ws_url =
        common::management_url(addr, "/api/services/ghost/logs/stream").replace("http://", "ws://");
    let result = connect_async(ws_url).await;

    // tungstenite surfaces non-101 responses as Http errors; the server should
    // have returned 404.
    match result {
        Err(tokio_tungstenite::tungstenite::Error::Http(resp)) => {
            assert_eq!(resp.status(), 404);
        }
        other => panic!("expected an Http(404) error, got: {other:?}"),
    }

    harness.shutdown().await;
}

#[tokio::test(flavor = "current_thread")]
async fn lines_for_other_services_are_filtered() {
    let harness = common::build_harness(vec![
        common::minimal_llama_service("alpha", 0),
        common::minimal_llama_service("beta", 0),
    ])
    .await;

    let alpha_id = harness.state.db.upsert_service("alpha", 0).await.unwrap();
    let beta_id = harness.state.db.upsert_service("beta", 0).await.unwrap();

    let addr = harness.spawn_management_server().await;
    // Subscribe to "alpha" only.
    let ws_url =
        common::management_url(addr, "/api/services/alpha/logs/stream").replace("http://", "ws://");
    let (mut ws, _) = connect_async(ws_url).await.unwrap();

    // Push a line for "beta" then one for "alpha".
    harness.state.batcher.push(LogLine {
        service_id: beta_id,
        run_id: 1,
        timestamp_ms: 1,
        stream: Stream::Stderr,
        line: "beta line".into(),
    });
    harness.state.batcher.push(LogLine {
        service_id: alpha_id,
        run_id: 1,
        timestamp_ms: 2,
        stream: Stream::Stdout,
        line: "alpha line".into(),
    });
    harness.state.batcher.flush().await;

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut received_lines: Vec<ApiLogLine> = Vec::new();
    while std::time::Instant::now() < deadline {
        let recv = tokio::time::timeout(std::time::Duration::from_millis(500), ws.next()).await;
        if let Ok(Some(Ok(Message::Text(s)))) = recv
            && let Ok(line) = serde_json::from_str::<ApiLogLine>(&s)
        {
            received_lines.push(line);
            // We only expect one line (alpha's); stop once we have it.
            break;
        }
    }

    assert_eq!(
        received_lines.len(),
        1,
        "should receive exactly one line (alpha's)"
    );
    assert_eq!(received_lines[0].line, "alpha line");

    let _ = ws.send(Message::Close(None)).await;
    harness.shutdown().await;
}
