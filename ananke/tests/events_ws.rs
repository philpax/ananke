//! Integration tests for the `/api/events` WebSocket endpoint.
#![cfg(feature = "test-fakes")]

mod common;

use ananke_api::Event;
use futures::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

#[tokio::test(flavor = "current_thread")]
async fn state_change_fires_event() {
    let harness = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let addr = harness.spawn_management_server().await;
    let ws_url = common::management_url(addr, "/api/events").replace("http://", "ws://");
    let (mut ws, _) = connect_async(ws_url).await.unwrap();

    // Trigger a state change via the start endpoint.
    let start_url = common::management_url(addr, "/api/services/demo/start");
    tokio::spawn(async move {
        let _ = reqwest::Client::new().post(start_url).send().await;
    });

    // Expect at least one StateChanged event within a short window.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut saw_state_change = false;
    while std::time::Instant::now() < deadline {
        let recv = tokio::time::timeout(std::time::Duration::from_millis(500), ws.next()).await;
        if let Ok(Some(Ok(Message::Text(s)))) = recv {
            let event: Event = serde_json::from_str(&s).unwrap();
            if matches!(event, Event::StateChanged { .. }) {
                saw_state_change = true;
                break;
            }
        }
    }
    assert!(saw_state_change, "expected a StateChanged event");
    let _ = ws.send(Message::Close(None)).await;
    harness.shutdown().await;
}

#[tokio::test(flavor = "current_thread")]
async fn service_filter_excludes_other_services() {
    let harness = common::build_harness(vec![
        common::minimal_llama_service("alpha", 0),
        common::minimal_llama_service("beta", 0),
    ])
    .await;
    let addr = harness.spawn_management_server().await;
    // Subscribe filtered to "alpha" only.
    let ws_url =
        common::management_url(addr, "/api/events?service=alpha").replace("http://", "ws://");
    let (mut ws, _) = connect_async(ws_url).await.unwrap();

    // Trigger a start on "beta"; any StateChanged for "beta" should not arrive.
    // Trigger a start on "alpha"; its StateChanged should arrive.
    let beta_url = common::management_url(addr, "/api/services/beta/start");
    let alpha_url = common::management_url(addr, "/api/services/alpha/start");
    tokio::spawn(async move {
        let client = reqwest::Client::new();
        let _ = client.post(beta_url).send().await;
        let _ = client.post(alpha_url).send().await;
    });

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut saw_beta = false;
    let mut saw_alpha = false;
    while std::time::Instant::now() < deadline {
        let recv = tokio::time::timeout(std::time::Duration::from_millis(500), ws.next()).await;
        if let Ok(Some(Ok(Message::Text(s)))) = recv {
            let event: Event = serde_json::from_str(&s).unwrap();
            if let Event::StateChanged { ref service, .. } = event {
                if service.as_str() == "beta" {
                    saw_beta = true;
                }
                if service.as_str() == "alpha" {
                    saw_alpha = true;
                    break;
                }
            }
        }
    }
    assert!(!saw_beta, "filter should have excluded beta events");
    assert!(saw_alpha, "expected a StateChanged event for alpha");
    let _ = ws.send(Message::Close(None)).await;
    harness.shutdown().await;
}
