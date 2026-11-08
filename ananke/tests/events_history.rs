//! Integration tests for `GET /api/events/history`.
#![cfg(feature = "test-fakes")]

mod common;

use ananke_api::events::{Event, EventHistory};
use axum::{body::to_bytes, http::StatusCode};
use common::{build_harness, minimal_llama_service};
use smol_str::SmolStr;
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn history_lists_events_published_before_the_request() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0)]).await;
    let before = h.state.events.history();
    let published = Event::ConfigReloaded {
        at_ms: before.since_ms + 1,
        changed_services: vec![SmolStr::new("alpha")],
    };
    h.state.events.publish(published.clone());

    let app = ananke::api::management::router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("GET")
        .uri("/api/events/history")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let history: EventHistory = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(history.since_ms, before.since_ms);
    assert_eq!(history.events.last(), Some(&published));
    h.cleanup().await;
}
