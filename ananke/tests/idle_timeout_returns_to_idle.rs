//! Integration test: after idle timeout the service returns to idle and
//! a subsequent request triggers a fresh spawn.
//!
//! Runs under `start_paused = true` with `current_thread` flavor so the
//! supervisor's tokio-based idle deadline advances virtually rather than
//! waiting wall-clock. The unification of activity tracking on
//! `tokio::time::Instant` (see `tracking::activity`) is what makes this
//! possible — previously the supervisor mixed wall-clock activity pings
//! with tokio time arithmetic and the test had to sleep real seconds.

mod common;

use std::time::Duration;

use ananke::api::openai;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn service_returns_to_idle_after_timeout_and_restarts() {
    let mut svc = minimal_llama_service("alpha", 0);
    svc.idle_timeout_ms = 500;

    let h = build_harness(vec![svc]).await;

    // First request: the service is idle, so this triggers a spawn and returns 200.
    let app = openai::router(h.state.clone());
    let body = r#"{"model":"alpha","messages":[]}"#;
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "first request must succeed");

    // Advance virtual time past the idle deadline. The supervisor's
    // `sleep_until(idle_deadline)` now fires, drains the fake child, and
    // transitions back to Idle. We advance past the drain grace too so the
    // SIGTERM → exit path has fully unwound.
    tokio::time::advance(Duration::from_secs(30)).await;
    // Yield so the supervisor task gets a chance to process the fired timer.
    tokio::task::yield_now().await;

    // Second request: the service should be idle again, triggering a fresh spawn.
    let req2 = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp2 = app.oneshot(req2).await.unwrap();
    assert_eq!(
        resp2.status(),
        StatusCode::OK,
        "second request must succeed after fresh spawn"
    );

    h.cleanup().await;
}
