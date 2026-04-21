//! Integration tests for the GET/PUT /api/config and POST /api/config/validate endpoints.
#![cfg(feature = "test-fakes")]

mod common;

use ananke_api::ConfigResponse;
use axum::{
    body::to_bytes,
    http::{StatusCode, header::IF_MATCH},
};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn get_config_returns_content_and_hash() {
    let h = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let app = ananke::api::management::router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("GET")
        .uri("/api/config")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: ConfigResponse = serde_json::from_slice(&bytes).unwrap();
    // The in-memory config manager starts with an empty raw string, so the
    // hash must still be present (it is the hash of the empty string).
    assert!(!parsed.hash.is_empty());
    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn put_without_if_match_is_428() {
    let h = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let app = ananke::api::management::router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("PUT")
        .uri("/api/config")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::PRECONDITION_REQUIRED);
    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn put_with_wrong_hash_is_412() {
    let h = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let app = ananke::api::management::router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("PUT")
        .uri("/api/config")
        .header(IF_MATCH, "\"wrong\"")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::PRECONDITION_FAILED);
    h.cleanup().await;
}
