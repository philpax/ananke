//! Integration test: a user-driven request for a persistent service evicts a
//! running on-demand peer rather than yielding to it.
//!
//! Regression: when switching back to a persistent service after using a
//! non-persistent one (e.g. qwen3.6 → gemma → qwen3.6 again), the supervisor's
//! "persistent yields to active non-persistent" guard fired even on user-driven
//! requests, returning 503 instead of evicting and starting the persistent
//! service. The yield rule exists for background-watcher re-ensures only; a
//! user-driven ensure must be allowed to evict idle on-demand peers.
#![cfg(feature = "test-fakes")]

mod common;

use ananke::{
    api::openai,
    config::{DeviceSlot, Lifecycle},
    devices::{CpuSnapshot, DeviceSnapshot},
};
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use common::{build_harness, minimal_llama_service};
use smol_str::SmolStr;
use tower::util::ServiceExt;

async fn chat(app: axum::Router, model: &str) -> (StatusCode, String) {
    let body = format!(r#"{{"model":"{model}","messages":[]}}"#);
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap_or_default();
    (status, String::from_utf8_lossy(&bytes).to_string())
}

#[tokio::test(flavor = "multi_thread")]
async fn user_request_for_persistent_evicts_running_nonpersistent() {
    let mut on_demand = minimal_llama_service("on_demand", 0);
    let mut persistent = minimal_llama_service("persistent", 0);

    // 10 GB each on CPU, 15 GB available — only one fits at a time.
    for svc in [&mut on_demand, &mut persistent] {
        svc.placement_override.clear();
        svc.placement_override.insert(DeviceSlot::Cpu, 10 * 1024);
    }
    on_demand.priority = 50;
    persistent.priority = 50;
    persistent.lifecycle = Lifecycle::Persistent;

    let h = build_harness(vec![on_demand, persistent]).await;
    *h.state.snapshot.write() = DeviceSnapshot {
        gpus: Vec::new(),
        cpu: Some(CpuSnapshot {
            total_bytes: 15 * 1024 * 1024 * 1024,
            available_bytes: 15 * 1024 * 1024 * 1024,
        }),
        taken_at_ms: 0,
    };

    let on_demand_name = SmolStr::new("on_demand");
    let persistent_name = SmolStr::new("persistent");

    // Step 1: start the on-demand service via a user request and let it settle
    // into Running with inflight = 0.
    let (st, body) = chat(openai::router(h.state.clone()), "on_demand").await;
    assert_eq!(st, StatusCode::OK, "on_demand must load first: {body}");

    {
        let alloc = h.state.allocations.lock().clone();
        assert!(
            alloc.contains_key(&on_demand_name),
            "on_demand should hold the reservation; got {alloc:?}"
        );
        assert!(
            !alloc.contains_key(&persistent_name),
            "persistent should not hold any reservation yet; got {alloc:?}"
        );
    }

    // Step 2: user switches to the persistent service. on_demand is Running
    // but idle (inflight = 0), so it is evictable. A user-driven ensure must
    // evict it and start the persistent service rather than yielding with 503.
    let (st, body) = chat(openai::router(h.state.clone()), "persistent").await;
    assert_eq!(
        st,
        StatusCode::OK,
        "user-driven request for persistent service must succeed by evicting on_demand: {body}"
    );

    // Step 3: persistent holds the allocation; on_demand was evicted.
    let alloc = h.state.allocations.lock().clone();
    assert!(
        alloc.contains_key(&persistent_name),
        "persistent must hold allocation after evicting on_demand; got {alloc:?}"
    );
    assert!(
        !alloc.contains_key(&on_demand_name),
        "on_demand must have been evicted; got {alloc:?}"
    );

    h.cleanup().await;
}
