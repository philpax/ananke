//! Integration test: the allocator returns 503 when there is insufficient memory.

mod common;

use ananke::config::DeviceSlot;
use ananke::devices::{CpuSnapshot, DeviceSnapshot};
use axum::body::to_bytes;
use axum::http::StatusCode;
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "multi_thread")]
async fn insufficient_vram_returns_503() {
    let mut svc = minimal_llama_service("big", 0);
    // Demand 10 GB of CPU so the allocator must check the snapshot.
    svc.placement_override.clear();
    svc.placement_override.insert(DeviceSlot::Cpu, 10 * 1024);

    let h = build_harness(vec![svc]).await;

    // Overwrite the harness's pre-seeded 64 GB with only 1 GB available.
    *h.state.snapshot.write() = DeviceSnapshot {
        gpus: Vec::new(),
        cpu: Some(CpuSnapshot {
            total_bytes: 16 * 1024 * 1024 * 1024,
            available_bytes: 1024 * 1024 * 1024,
        }),
        taken_at_ms: 0,
    };

    let app = ananke::openai_api::router(h.state.clone());
    let body = r#"{"model":"big","messages":[]}"#;
    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["error"]["code"], "insufficient_vram");
    h.cleanup().await;
}
