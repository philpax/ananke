//! Integration tests for the management API's info endpoint.
#![cfg(feature = "test-fakes")]

mod common;

use std::time::Duration;

use ananke::daemon::app_state::AppState;
use ananke_api::info::get::DaemonInfoResponse;
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn api_info_uptime_follows_the_daemon_clock() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0)]).await;
    let before = get_info(&h.state).await;
    tokio::time::advance(Duration::from_secs(90)).await;
    let after = get_info(&h.state).await;
    assert!(
        after.uptime_ms - before.uptime_ms >= 90_000,
        "uptime went from {} to {}",
        before.uptime_ms,
        after.uptime_ms
    );
    h.cleanup().await;
}

async fn get_info(state: &AppState) -> DaemonInfoResponse {
    let app = ananke::api::management::router(state.clone());
    let req = Request::builder()
        .method("GET")
        .uri("/api/info")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}
