//! Integration test: concurrent first-requests coalesce into one start.
//!
//! Five requests arrive while the service is idle. All five should receive 200
//! because the supervisor fans them out onto a single broadcast channel rather
//! than starting five separate processes or returning 503.

mod common;

use ananke::openai_api;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use common::{build_harness, minimal_llama_service};
use tokio::task::JoinSet;
use tower::util::ServiceExt;

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_first_requests_collapse_into_one_start() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0)]).await;
    let app = openai_api::router(h.state.clone());

    let body = r#"{"model":"alpha","messages":[]}"#;
    let mut join_set = JoinSet::new();
    for _ in 0..5 {
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

    // All concurrent callers must receive a successful response — coalescing
    // ensures they all wait on the same start outcome rather than racing.
    assert!(
        statuses.iter().all(|s| *s == StatusCode::OK),
        "expected all 200, got: {statuses:?}"
    );

    h.cleanup().await;
}
