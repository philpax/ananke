//! Integration tests for the GET /api/services/{name}/logs endpoint.
#![cfg(feature = "test-fakes")]

mod common;

use ananke_api::LogsResponse;
use axum::{body::to_bytes, http::StatusCode};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn returns_empty_for_idle_service() {
    let h = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let app = ananke::api::management::router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("GET")
        .uri("/api/services/demo/logs")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: LogsResponse = serde_json::from_slice(&bytes).unwrap();
    assert!(parsed.logs.is_empty());
    assert!(parsed.next_cursor.is_none());
    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn rejects_malformed_cursor() {
    let h = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let app = ananke::api::management::router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("GET")
        .uri("/api/services/demo/logs?before=notbase64!!!")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    h.cleanup().await;
}
