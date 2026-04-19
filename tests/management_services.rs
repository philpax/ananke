//! Integration tests for the management API's service endpoints.

mod common;

use axum::{body::to_bytes, http::StatusCode};
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn api_services_lists_registered() {
    let h = build_harness(vec![
        minimal_llama_service("alpha", 0),
        minimal_llama_service("beta", 0),
    ])
    .await;
    let app = ananke::api::management::router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("GET")
        .uri("/api/services")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let names: Vec<String> = parsed
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["name"].as_str().unwrap().to_string())
        .collect();
    assert!(names.contains(&"alpha".to_string()));
    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn api_service_detail_by_name() {
    let h = build_harness(vec![minimal_llama_service("alpha", 12345)]).await;
    let app = ananke::api::management::router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("GET")
        .uri("/api/services/alpha")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["name"], "alpha");
    assert_eq!(parsed["port"], 12345);
    h.cleanup().await;
}
