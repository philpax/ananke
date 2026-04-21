#![cfg(feature = "test-fakes")]
mod common;

use axum::{body::to_bytes, http::StatusCode};
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn api_services_includes_elastic_borrower_field() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0)]).await;
    let app = ananke::api::management::router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("GET")
        .uri("/api/services/alpha")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(
        parsed.get("elastic_borrower").is_some(),
        "elastic_borrower field missing: {}",
        parsed
    );
    h.cleanup().await;
}
