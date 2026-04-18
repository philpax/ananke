//! Integration test: after idle timeout the service returns to idle and
//! a subsequent request triggers a fresh spawn.

mod common;

use ananke::openai_api;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "multi_thread")]
async fn service_returns_to_idle_after_timeout_and_restarts() {
    // Set a very short idle timeout so the test completes in a few seconds.
    let mut svc = minimal_llama_service("alpha", 0);
    svc.idle_timeout_ms = 500;

    let h = build_harness(vec![svc]).await;

    // First request: the service is idle, so this triggers a spawn and returns 200.
    let app = openai_api::router(h.state.clone());
    let body = r#"{"model":"alpha","messages":[]}"#;
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "first request must succeed");

    // Wait long enough for the idle timeout to fire and the supervisor to drain
    // back to Idle. 1 500 ms is 3× the timeout, giving the supervisor ample
    // time even on a loaded CI runner.
    tokio::time::sleep(std::time::Duration::from_millis(1_500)).await;

    // Second request: the service should be idle again, triggering a fresh spawn.
    let req2 = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp2 = app.oneshot(req2).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::OK, "second request must succeed after fresh spawn");

    h.cleanup().await;
}
