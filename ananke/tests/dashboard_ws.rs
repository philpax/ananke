//! Dashboard integration tests exercise the real handlers and producer notifications over one socket.
#![cfg(feature = "test-fakes")]

mod common;

use std::{sync::atomic::Ordering, time::Duration};

use ananke_api::{
    dashboard::{DashboardMessage, DashboardRequest},
    events::Event,
};
use ananke_db::{
    logs::{LogLine, Stream},
    models::RequestMetric,
};
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use smol_str::SmolStr;
use tokio::{net::TcpStream, time::timeout};
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async,
    tungstenite::{Message, client::IntoClientRequest},
};

type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;

async fn send(socket: &mut Socket, request: Value) {
    let _: DashboardRequest = serde_json::from_value(request.clone()).unwrap();
    socket
        .send(Message::Text(request.to_string().into()))
        .await
        .unwrap();
}

async fn receive(socket: &mut Socket, id: &str) -> (u16, Value) {
    timeout(Duration::from_secs(5), async {
        loop {
            if let Some(Ok(Message::Text(text))) = socket.next().await {
                let frame: DashboardMessage = serde_json::from_str(&text).unwrap();
                if let DashboardMessage::Response {
                    id: received,
                    status,
                    body,
                } = frame
                    && received == id
                {
                    return (status, body);
                }
            }
        }
    })
    .await
    .unwrap()
}

async fn read(socket: &mut Socket, id: &str, query: Value, subscribe: bool) -> (u16, Value) {
    send(
        socket,
        json!({"type":"read", "id":id, "query":query, "subscribe":subscribe}),
    )
    .await;
    receive(socket, id).await
}

fn range() -> Value {
    json!({"since":0, "until":ananke_time::now_unix_ms()+1000, "live_ms":null})
}

#[tokio::test]
async fn initial_backfill_push_and_cancel_use_the_same_connection() {
    let harness = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let db = &harness.state.db;
    let service_id = db.resolve_service_id("demo").await.unwrap().unwrap();
    harness.state.events.publish(Event::ConfigReloaded {
        at_ms: 123,
        changed_services: vec![],
    });
    db.insert_device_sample("cpu", 123, 100, 90).await.unwrap();
    harness.state.batcher.push(LogLine {
        service_id,
        run_id: 1,
        timestamp_ms: 123,
        stream: Stream::Stdout,
        line: "before connection".into(),
    });
    harness.state.batcher.flush().await;
    let addr = harness.spawn_management_server().await;
    let (mut socket, _) = connect_async(format!("ws://{addr}/api/dashboard"))
        .await
        .unwrap();
    let (status, inventory) =
        read(&mut socket, "inventory", json!({"type":"services"}), true).await;
    assert_eq!(status, 200);
    assert_eq!(inventory["services"][0]["name"], "demo");
    let (_, events) = read(&mut socket, "events", json!({"type":"events"}), true).await;
    assert!(
        events["events"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["at_ms"] == 123)
    );
    let (_, samples) = read(
        &mut socket,
        "samples",
        json!({"type":"samples", "range":range()}),
        true,
    )
    .await;
    assert_eq!(samples["samples"].as_array().unwrap().len(), 1);
    let (_, logs) = read(&mut socket, "logs", json!({"type":"logs", "name":"demo", "range":{"since":0,"until":0,"live_ms":60000}, "before":null}), true).await;
    // The fixed timestamp is outside this live window; backfill respects the range.
    assert_eq!(logs["logs"].as_array().unwrap().len(), 0);
    let (_, logs) = read(
        &mut socket,
        "old-logs",
        json!({"type":"logs", "name":"demo", "range":range(), "before":null}),
        false,
    )
    .await;
    assert_eq!(logs["logs"][0]["line"], "before connection");
    let (_, devices) = read(&mut socket, "devices", json!({"type":"devices"}), true).await;
    assert_eq!(devices[0]["id"], "cpu");
    assert!(
        timeout(Duration::from_millis(100), socket.next())
            .await
            .is_err(),
        "an idle dashboard does not poll or push unchanged data"
    );
    harness
        .state
        .snapshot
        .write()
        .cpu
        .as_mut()
        .unwrap()
        .available_bytes = 1234;
    let (_, devices) = receive(&mut socket, "devices").await;
    assert_eq!(devices[0]["free_bytes"], 1234);
    db.insert_device_sample("cpu", 124, 100, 80).await.unwrap();
    let (_, samples) = receive(&mut socket, "samples").await;
    assert_eq!(samples["samples"].as_array().unwrap().len(), 2);
    harness.state.batcher.push(LogLine {
        service_id,
        run_id: 1,
        timestamp_ms: ananke_time::now_unix_ms(),
        stream: Stream::Stderr,
        line: "after connection".into(),
    });
    harness.state.batcher.flush().await;
    let (_, logs) = receive(&mut socket, "logs").await;
    assert_eq!(logs["logs"][0]["line"], "after connection");
    send(&mut socket, json!({"type":"cancel", "id":"samples"})).await;
    read(&mut socket, "barrier", json!({"type":"config"}), false).await;
    db.insert_device_sample("cpu", 125, 100, 70).await.unwrap();
    assert!(
        timeout(Duration::from_millis(100), socket.next())
            .await
            .is_err()
    );
    socket.close(None).await.unwrap();
    let (mut socket, _) = connect_async(format!("ws://{addr}/api/dashboard"))
        .await
        .unwrap();
    let (_, samples) = read(
        &mut socket,
        "reconnect",
        json!({"type":"samples", "range":range()}),
        false,
    )
    .await;
    assert_eq!(samples["samples"].as_array().unwrap().len(), 3);
    let (_, logs) = read(
        &mut socket,
        "reconnect-logs",
        json!({"type":"logs", "name":"demo", "range":range(), "before":null}),
        false,
    )
    .await;
    assert_eq!(logs["logs"].as_array().unwrap().len(), 2);
    socket.close(None).await.unwrap();
    harness.shutdown().await;
}

#[tokio::test]
async fn mutations_preserve_handler_errors_config_conflicts_and_lifecycle_pushes() {
    let harness = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let addr = harness.spawn_management_server().await;
    let (mut socket, _) = connect_async(format!("ws://{addr}/api/dashboard"))
        .await
        .unwrap();
    read(&mut socket, "inventory", json!({"type":"services"}), true).await;
    send(
        &mut socket,
        json!({"type":"action", "id":"missing", "name":"missing", "action":"start"}),
    )
    .await;
    assert_eq!(receive(&mut socket, "missing").await.0, 404);
    send(
        &mut socket,
        json!({"type":"save", "id":"conflict", "content":"", "hash":"stale"}),
    )
    .await;
    assert_eq!(receive(&mut socket, "conflict").await.0, 412);
    send(
        &mut socket,
        json!({"type":"validate", "id":"validation", "content":"this is not TOML"}),
    )
    .await;
    let (status, body) = receive(&mut socket, "validation").await;
    assert_eq!(status, 200);
    assert_eq!(body["valid"], false);
    send(
        &mut socket,
        json!({"type":"action", "id":"start", "name":"demo", "action":"start"}),
    )
    .await;
    timeout(Duration::from_secs(5), async {
        let mut acknowledged = false;
        let mut pushed = false;
        while !acknowledged || !pushed {
            if let Some(Ok(Message::Text(text))) = socket.next().await {
                let frame: DashboardMessage = serde_json::from_str(&text).unwrap();
                if let DashboardMessage::Response { id, status, body } = frame {
                    if id == "start" {
                        assert_eq!(status, 202);
                        acknowledged = true;
                    }
                    if id == "inventory" && body["services"][0]["state"] == "running" {
                        pushed = true;
                    }
                }
            }
        }
    })
    .await
    .unwrap();
    socket.close(None).await.unwrap();
    harness.shutdown().await;
}

#[tokio::test]
async fn committed_metrics_push_to_a_subscriber_without_another_read() {
    let harness = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let addr = harness.spawn_management_server().await;
    let (mut socket, _) = connect_async(format!("ws://{addr}/api/dashboard"))
        .await
        .unwrap();
    let (_, body) = read(
        &mut socket,
        "metrics",
        json!({"type":"metrics", "range":range(), "bucket":"1m", "service":"demo"}),
        true,
    )
    .await;
    assert!(body["buckets"].as_array().unwrap().is_empty());
    harness
        .state
        .db
        .insert_request_metric(&RequestMetric {
            metric_id: 0,
            service_id: harness
                .state
                .db
                .resolve_service_id("demo")
                .await
                .unwrap()
                .unwrap(),
            run_id: None,
            timestamp_ms: ananke_time::now_unix_ms(),
            endpoint: "/v1/chat/completions".into(),
            model: "demo".into(),
            prompt_tokens: Some(4),
            completion_tokens: Some(2),
            prompt_eval_tokens: None,
            duration_ms: Some(100),
            ttft_ms: Some(5),
            prompt_ms: None,
            predicted_ms: None,
            draft_tokens: None,
            draft_tokens_accepted: None,
            status_code: 200,
        })
        .await
        .unwrap();
    let (status, body) = receive(&mut socket, "metrics").await;
    assert_eq!(status, 200);
    assert_eq!(body["buckets"][0]["request_count"], 1);
    socket.close(None).await.unwrap();
    harness.shutdown().await;
}

#[tokio::test]
async fn cancelling_a_chat_releases_the_proxy_and_does_not_block_other_reads() {
    let harness = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    harness.echo_state.hang.store(true, Ordering::Relaxed);
    let addr = harness.spawn_management_server().await;
    let (mut socket, _) = connect_async(format!("ws://{addr}/api/dashboard"))
        .await
        .unwrap();
    send(
        &mut socket,
        json!({"type":"chat", "id":"chat", "body":{"model":"demo", "messages":[], "stream":true}}),
    )
    .await;
    timeout(Duration::from_secs(5), async {
        while harness.echo_state.sink.lock().is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(harness.state.inflight.current(&SmolStr::new("demo")), 1);
    assert_eq!(
        read(&mut socket, "info", json!({"type":"info"}), false)
            .await
            .0,
        200
    );
    send(&mut socket, json!({"type":"cancel", "id":"chat"})).await;
    timeout(Duration::from_secs(5), async {
        while harness.state.inflight.current(&SmolStr::new("demo")) != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    socket.close(None).await.unwrap();
    harness.shutdown().await;
}

#[tokio::test]
async fn cookie_authenticated_socket_rejects_a_cross_site_origin() {
    let harness = common::build_harness(vec![]).await;
    let addr = harness.spawn_management_server().await;
    let mut request = format!("ws://{addr}/api/dashboard")
        .into_client_request()
        .unwrap();
    request
        .headers_mut()
        .insert("origin", "https://another.example".parse().unwrap());
    let error = connect_async(request).await.unwrap_err();
    assert!(
        matches!(error, tokio_tungstenite::tungstenite::Error::Http(response) if response.status() == 403)
    );
    let mut request = format!("ws://{addr}/api/dashboard")
        .into_client_request()
        .unwrap();
    request
        .headers_mut()
        .insert("origin", format!("http://{addr}").parse().unwrap());
    let (mut socket, _) = connect_async(request).await.unwrap();
    assert_eq!(
        read(&mut socket, "info", json!({"type":"info"}), false)
            .await
            .0,
        200
    );
    socket.close(None).await.unwrap();
    harness.shutdown().await;
}

#[tokio::test]
async fn streamed_chat_preserves_utf8_fragments_and_completion_without_blocking_data() {
    let harness = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let text = "data: {\"choices\":[{\"delta\":{\"content\":\"café 🪿\"}}]}\n\ndata: [DONE]\n\n";
    let split = text.find('é').unwrap() + 1;
    *harness.echo_state.chat_stream.lock() = Some(vec![
        Bytes::copy_from_slice(&text.as_bytes()[..split]),
        Bytes::copy_from_slice(&text.as_bytes()[split..]),
    ]);
    let addr = harness.spawn_management_server().await;
    let (mut socket, _) = connect_async(format!("ws://{addr}/api/dashboard"))
        .await
        .unwrap();
    send(&mut socket, json!({"type":"chat", "id":"chat", "body":{"model":"demo", "messages":[{"role":"user","content":"hello"}]}})).await;
    let mut output = String::new();
    timeout(Duration::from_secs(5), async {
        loop {
            if let Some(Ok(Message::Text(text))) = socket.next().await {
                match serde_json::from_str::<DashboardMessage>(&text).unwrap() {
                    DashboardMessage::Stream { id, text } if id == "chat" => output.push_str(&text),
                    DashboardMessage::End { id } if id == "chat" => break,
                    other => panic!("unexpected chat frame: {other:?}"),
                }
            }
        }
    })
    .await
    .unwrap();
    assert_eq!(output, text);
    assert_eq!(harness.echo_state.sink.lock()[0]["stream"], true);
    socket.close(None).await.unwrap();
    harness.shutdown().await;
}
