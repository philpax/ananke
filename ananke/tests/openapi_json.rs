#![cfg(feature = "test-fakes")]
mod common;

use ananke::api::management;
use axum::{body::to_bytes, http::StatusCode};
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn openapi_json_is_valid() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0)]).await;
    let app = management::router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("GET")
        .uri("/api/openapi.json")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = to_bytes(resp.into_body(), 10 * 1024 * 1024).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        parsed["openapi"].as_str().unwrap_or("").chars().next(),
        Some('3')
    );
    let paths = parsed["paths"].as_object().expect("paths object");
    assert!(paths.contains_key("/v1/models"));
    assert!(paths.contains_key("/api/services"));
    assert!(paths.contains_key("/api/devices"));

    h.cleanup().await;
}
