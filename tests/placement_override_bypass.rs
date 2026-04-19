//! Integration test: a service with `placement_override` set bypasses the
//! estimator entirely. The request still succeeds (the echo server stands in
//! for llama-server).

mod common;

use ananke::api::openai;
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn placement_override_service_responds_200() {
    // `minimal_llama_service` sets a Cpu placement_override, so the supervisor
    // should skip the estimator and use the declared bytes directly.
    let h = build_harness(vec![minimal_llama_service("override-svc", 0)]).await;
    let app = openai::router(h.state.clone());

    let body = r#"{"model":"override-svc","messages":[{"role":"user","content":"hi"}]}"#;
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["id"], "cmpl-echo");

    h.cleanup().await;
}
