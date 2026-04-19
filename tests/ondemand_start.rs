//! Integration test: an idle on-demand service starts on the first request.

mod common;

use ananke::openai_api;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn first_request_triggers_spawn_and_serves() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0)]).await;
    let app = openai_api::router(h.state.clone());

    let body = r#"{"model":"alpha","messages":[]}"#;
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    h.cleanup().await;
}
