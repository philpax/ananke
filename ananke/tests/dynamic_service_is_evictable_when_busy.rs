//! Integration test: a busy dynamic-allocation service is evictable by a
//! tied-priority peer that can't otherwise pack.
//!
//! The hazard: ComfyUI runs with `allocation.mode = "dynamic"` and keeps a
//! `/ws` open from its web UI, which pegs its inflight counter at ≥1 even
//! when no image is generating. Read as "busy", it is never displaceable,
//! so an incoming tied-priority chat queues indefinitely against a peer
//! that is never going to idle and `anankectl chat` hangs silently.
//!
//! The rule that avoids it: a service whose operator-declared
//! `allocation_mode` is `Dynamic` is logically idle for eviction purposes.
//! The whole point of choosing dynamic mode is to opt the service into
//! "kill me when a tied-priority peer needs the VRAM."
#![cfg(feature = "test-fakes")]

mod common;

use std::time::Duration;

use ananke::{
    api::openai,
    config::{AllocationMode, DeviceSlot},
    supervise::state::ServiceState,
};
use ananke_placement::devices::{CpuSnapshot, DeviceSnapshot};
use ananke_proxy::InflightGuard;
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
async fn busy_dynamic_peer_is_evicted_by_tied_priority_request() {
    // `comfy` stands in for the ComfyUI-style elastic service: declared
    // `Dynamic`, pinned at default priority 50, and about to be held
    // perpetually busy by a stand-in for its WebSocket UI traffic.
    let mut comfy = minimal_llama_service("comfy", 0);
    let mut qwen = minimal_llama_service("qwen", 0);

    for svc in [&mut comfy, &mut qwen] {
        svc.placement_override.clear();
        svc.placement_override.insert(DeviceSlot::Cpu, 10 * 1024);
        svc.priority = 50;
    }
    comfy.allocation_mode = AllocationMode::Dynamic {
        min_mb: 2 * 1024,
        max_mb: 10 * 1024,
        min_borrower_runtime_ms: 60_000,
    };

    let h = build_harness(vec![comfy, qwen]).await;
    // 15 GB total / 15 GB available — fits one 10 GB pledge, not two.
    *h.state.snapshot.write() = DeviceSnapshot {
        gpus: Vec::new(),
        cpu: Some(CpuSnapshot {
            total_bytes: 15 * 1024 * 1024 * 1024,
            available_bytes: 15 * 1024 * 1024 * 1024,
        }),
        taken_at_ms: 0,
    };

    // Step 1: comfy loads and enters Running.
    let (st, body) = chat(openai::router(h.state.clone()), "comfy").await;
    assert_eq!(st, StatusCode::OK, "comfy must load first: {body}");

    // Step 2: simulate the open `/ws` from the ComfyUI web UI by pegging
    // comfy's inflight counter and holding it. No release task — the
    // counter stays at 1 for the rest of the test. That would make comfy
    // "busy" and unreachable to eviction, but the dynamic-mode declaration
    // overrides the busy bit and comfy stays evictable.
    let comfy_inflight = h.state.inflight.counter(&SmolStr::new("comfy"));
    let _comfy_busy = InflightGuard::new(comfy_inflight.clone());
    assert_eq!(comfy_inflight.current(), 1);

    // Step 3: qwen fires while comfy is "busy". The OpenAI handler's
    // `await_ensure` is bounded by the service's `max_request_duration_ms`
    // (5 s in `minimal_llama_service`). A comfy read as busy would either
    // 503 immediately with `insufficient_capacity` or hang the full 5 s and
    // 503 with "start timed out"; read as logically idle, it is picked as
    // the eviction victim and qwen proceeds.
    let (st, body) = chat(openai::router(h.state.clone()), "qwen").await;
    assert_eq!(
        st,
        StatusCode::OK,
        "qwen must evict busy dynamic comfy at tied priority: {body}"
    );

    // Settle for the drain to land in the registry mirror. The eviction
    // runs in a separate task, so the supervisor's state transition
    // (Running → Draining → Idle) doesn't necessarily happen before
    // `chat` returns 200 to the requester.
    for _ in 0..50 {
        let comfy_state = h
            .state
            .registry
            .get(&SmolStr::new("comfy"))
            .map(|s| s.peek_state());
        if matches!(
            comfy_state,
            Some(ServiceState::Idle | ServiceState::Disabled { .. })
        ) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // qwen now holds the CPU pledge; comfy has been drained out.
    let alloc = h.state.allocations.lock().clone();
    assert!(
        alloc.contains_key(&SmolStr::new("qwen")),
        "qwen should hold the reservation after the eviction; got {alloc:?}"
    );
    assert!(
        !alloc.contains_key(&SmolStr::new("comfy")),
        "comfy's reservation must have been released by the drain; got {alloc:?}"
    );

    h.cleanup().await;
}
