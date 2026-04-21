//! Integration test: an incoming request displaces an idle peer at the
//! same priority.
//!
//! Regression target: before the idle-eviction rule change, two services
//! at default priority (50) could deadlock the allocator — neither
//! could displace the other even while idle, so the second request
//! always got `insufficient_vram`. The allocator's eligibility rule is
//! now "idle is always evictable regardless of priority", and every
//! call site funnels through `EvictionCandidate::is_evictable_by` so
//! the two spots where the predicate lived can't drift again.
#![cfg(feature = "test-fakes")]

mod common;

use ananke::{
    api::openai,
    config::DeviceSlot,
    devices::{CpuSnapshot, DeviceSnapshot},
};
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use common::{build_harness, minimal_llama_service};
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
async fn second_request_displaces_idle_peer_at_equal_priority() {
    let mut a = minimal_llama_service("alpha", 0);
    let mut b = minimal_llama_service("beta", 0);

    // Both request 10 GB on CPU. Available snapshot below only fits one
    // at a time, which is the scenario the new rule is meant to cover.
    for svc in [&mut a, &mut b] {
        svc.placement_override.clear();
        svc.placement_override.insert(DeviceSlot::Cpu, 10 * 1024);
    }
    // Default priority (50) is already the same for both; being explicit
    // pins the scenario even if the default ever moves.
    a.priority = 50;
    b.priority = 50;

    let h = build_harness(vec![a, b]).await;

    // 15 GB total / 15 GB available → one fits, two don't.
    *h.state.snapshot.write() = DeviceSnapshot {
        gpus: Vec::new(),
        cpu: Some(CpuSnapshot {
            total_bytes: 15 * 1024 * 1024 * 1024,
            available_bytes: 15 * 1024 * 1024 * 1024,
        }),
        taken_at_ms: 0,
    };

    // Alpha loads.
    let (st, body) = chat(openai::router(h.state.clone()), "alpha").await;
    assert_eq!(st, StatusCode::OK, "alpha first request failed: {body}");

    // Beta arrives. Before the rule change this 503'd with
    // insufficient_vram because alpha was `Running` (not `Idle`) at the
    // same priority, and the allocator refused to displace it. Now,
    // alpha has no in-flight requests; it counts as idle; beta evicts.
    let (st, body) = chat(openai::router(h.state.clone()), "beta").await;
    assert_eq!(
        st,
        StatusCode::OK,
        "beta should displace idle alpha at the same priority: {body}"
    );

    // And back — alpha evicts beta on the rebound.
    let (st, body) = chat(openai::router(h.state.clone()), "alpha").await;
    assert_eq!(
        st,
        StatusCode::OK,
        "alpha should displace idle beta on the rebound: {body}"
    );

    h.cleanup().await;
}
