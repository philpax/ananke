mod common;

use ananke::openai_api;
use axum::{body::to_bytes, http::StatusCode};
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn get_v1_models_lists_registered_services() {
    let h = build_harness(vec![
        minimal_llama_service("alpha", 0),
        minimal_llama_service("beta", 0),
    ])
    .await;
    let app = openai_api::router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let names: Vec<String> = parsed["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();
    assert!(names.contains(&"alpha".to_string()));
    assert!(names.contains(&"beta".to_string()));
    h.cleanup().await;
}
