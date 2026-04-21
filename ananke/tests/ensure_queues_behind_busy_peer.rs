//! Integration test: an Ensure whose VRAM fit is blocked by a busy peer at
//! the same priority queues on a broadcast bus and resumes once the peer's
//! inflight counter drops to zero, rather than 503'ing immediately.
//!
//! Regression target: before the queue-on-busy-peer fix, any new request
//! that arrived while a same-priority peer had an in-flight generation
//! returned `503 insufficient_vram` even though the request would have fit
//! cleanly once the peer finished. That surfaced as "I asked for 35B while
//! 31B was still writing its response and got a 503".
#![cfg(feature = "test-fakes")]

mod common;

use std::{sync::atomic::Ordering, time::Duration};

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
async fn ensure_queues_behind_busy_same_priority_peer() {
    let mut alpha = minimal_llama_service("alpha", 0);
    let mut beta = minimal_llama_service("beta", 0);

    // Both pledge 10 GB on CPU. Available below is 15 GB — fits one, not two.
    for svc in [&mut alpha, &mut beta] {
        svc.placement_override.clear();
        svc.placement_override.insert(DeviceSlot::Cpu, 10 * 1024);
    }
    // Same default priority: neither can displace the other's *busy* work.
    alpha.priority = 50;
    beta.priority = 50;

    let h = build_harness(vec![alpha, beta]).await;
    *h.state.snapshot.write() = DeviceSnapshot {
        gpus: Vec::new(),
        cpu: Some(CpuSnapshot {
            total_bytes: 15 * 1024 * 1024 * 1024,
            available_bytes: 15 * 1024 * 1024 * 1024,
        }),
        taken_at_ms: 0,
    };

    // Step 1: alpha loads and enters Running.
    let (st, body) = chat(openai::router(h.state.clone()), "alpha").await;
    assert_eq!(st, StatusCode::OK, "alpha must load first: {body}");

    // Step 2: simulate alpha in the middle of a streaming request by bumping
    // its inflight counter. The real handler's `InflightGuard` does the same
    // thing around a proxied request; we stand in for it here so the test
    // isn't timing-sensitive on a real HTTP stream.
    let alpha_inflight = h.state.inflight.counter(&SmolStr::new("alpha"));
    alpha_inflight.fetch_add(1, Ordering::Relaxed);

    // Step 3: fire beta while alpha is "busy". The old code path rejected this
    // with 503. With the queue-on-busy-peer fix, the Ensure parks on a
    // broadcast bus and the Idle loop's poll-tick branch retries every ~250 ms
    // until alpha idles, at which point alpha becomes evictable and beta
    // proceeds normally.
    //
    // Run the request concurrently with a task that clears alpha's inflight
    // after a short delay; assert that beta eventually returns 200.
    let counter_for_release = alpha_inflight.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(400)).await;
        counter_for_release.fetch_sub(1, Ordering::Relaxed);
    });

    let (st, body) = chat(openai::router(h.state.clone()), "beta").await;
    assert_eq!(
        st,
        StatusCode::OK,
        "beta must queue behind busy alpha and succeed once alpha idles: {body}"
    );

    // At the end beta holds the allocation; alpha drained out to make room.
    let alloc = h.state.allocations.lock().clone();
    assert!(
        alloc.contains_key(&SmolStr::new("beta")),
        "beta should hold the reservation after the cascade; got {alloc:?}"
    );

    h.cleanup().await;
}
