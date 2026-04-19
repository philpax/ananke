mod common;

use ananke::api::openai;
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn chat_completions_unknown_model_404() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0)]).await;
    let app = openai::router(h.state.clone());
    let body = r#"{"model":"nope","messages":[]}"#;
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn chat_completions_routes_through_echo() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0)]).await;
    let app = openai::router(h.state.clone());
    let body = r#"{"model":"alpha","messages":[{"role":"user","content":"hi"}]}"#;
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

    // The echo server's sink should have captured the forwarded request body.
    let sunk = h.echo_state.sink.lock().clone();
    assert_eq!(sunk.len(), 1);
    assert_eq!(sunk[0]["model"], "alpha");

    h.cleanup().await;
}
