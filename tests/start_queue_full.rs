//! Integration test: when the start queue is full, excess requests get 503.
//!
//! A service is configured with `start_queue_depth = 2`. Four concurrent
//! requests are fired while the service is idle. At most 2 callers can wait on
//! the in-flight start; the rest must receive 503 start_queue_full. We assert
//! that at least 2 responses are 200 and at least 1 is 503.

mod common;

use ananke::api::openai;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use common::{build_harness, service_with_queue_depth};
use tokio::task::JoinSet;
use tower::util::ServiceExt;

#[tokio::test(flavor = "multi_thread")]
async fn excess_concurrent_requests_get_503_when_queue_full() {
    // Queue depth of 2: the first caller triggers the start (no queue slot
    // consumed yet) and then at most 2 additional callers may subscribe. The
    // fourth concurrent request should overflow and receive 503.
    let svc = service_with_queue_depth("alpha", 0, 2);
    let h = build_harness(vec![svc]).await;
    let app = openai::router(h.state.clone());

    let body = r#"{"model":"alpha","messages":[]}"#;
    let mut join_set = JoinSet::new();
    for _ in 0..4 {
        let app = app.clone();
        let body = body.to_string();
        join_set.spawn(async move {
            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap();
            app.oneshot(req).await.unwrap().status()
        });
    }

    let mut statuses = Vec::new();
    while let Some(result) = join_set.join_next().await {
        statuses.push(result.expect("task panicked"));
    }

    let ok_count = statuses.iter().filter(|s| **s == StatusCode::OK).count();
    let err_count = statuses
        .iter()
        .filter(|s| **s == StatusCode::SERVICE_UNAVAILABLE)
        .count();

    assert!(
        ok_count >= 2,
        "expected at least 2 successful responses, got {ok_count}; all: {statuses:?}"
    );
    assert!(
        err_count >= 1,
        "expected at least 1 queue-full 503, got {err_count}; all: {statuses:?}"
    );

    h.cleanup().await;
}
