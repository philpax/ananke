//! Integration tests for the management API's device endpoint.
#![cfg(feature = "test-fakes")]

mod common;

use ananke::devices::{CpuSnapshot, DeviceSnapshot, GpuSnapshot};
use axum::{body::to_bytes, http::StatusCode};
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn api_devices_reflects_snapshot() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0)]).await;
    // Seed a snapshot with one GPU and CPU so the endpoint has data to return.
    *h.state.snapshot.write() = DeviceSnapshot {
        gpus: vec![GpuSnapshot {
            id: 0,
            name: "RTX 3090".into(),
            total_bytes: 24 * 1024 * 1024 * 1024,
            free_bytes: 20 * 1024 * 1024 * 1024,
        }],
        cpu: Some(CpuSnapshot {
            total_bytes: 64 * 1024 * 1024 * 1024,
            available_bytes: 40 * 1024 * 1024 * 1024,
        }),
        taken_at_ms: 0,
    };
    let app = ananke::api::management::router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("GET")
        .uri("/api/devices")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let ids: Vec<String> = parsed
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d["id"].as_str().unwrap().to_string())
        .collect();
    assert!(ids.contains(&"gpu:0".to_string()));
    assert!(ids.contains(&"cpu".to_string()));
    h.cleanup().await;
}
