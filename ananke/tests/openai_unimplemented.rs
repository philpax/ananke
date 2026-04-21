#![cfg(feature = "test-fakes")]
mod common;

use ananke::api::openai;
use axum::http::StatusCode;
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn audio_speech_returns_501() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0)]).await;
    let app = openai::router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/audio/speech")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn images_generations_returns_501() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0)]).await;
    let app = openai::router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/images/generations")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
    h.cleanup().await;
}
