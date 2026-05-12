//! Integration test: an Ensure that stays parked in the start queue
//! longer than `QUEUE_BLOCKED_GRACE` resolves with a structured
//! `service_blocked` error rather than hanging until the proxy's
//! `max_request_duration_ms` budget runs out.
//!
//! Regression target: before the queue-grace bound, a request queued
//! behind a tied-priority *non-elastic* busy peer (e.g. another model
//! mid-generation) would block silently on `await_start_bus` for the
//! full `max_request_duration_ms` (default 10 min in production) and
//! eventually 503 with the unhelpful `start timed out`. The CLI showed
//! nothing in the meantime. With the bound, the queue gives up after
//! ~30 s (overridden in the test via tokio's virtual clock) and the
//! daemon surfaces `service_blocked` with the busy peer name.
//!
//! Distinct from `ensure_queues_behind_busy_peer.rs`: that test exercises
//! the *successful* queue path where the peer idles quickly. This one
//! pins the timeout fallback for the case where the peer never releases.
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
async fn queued_ensure_fails_with_service_blocked_after_grace() {
    // Both static (default `AllocationMode::None`) so `collect_eviction_
    // candidates` keeps the busy bit semantics — a busy non-dynamic peer
    // is *not* evictable at tied priority, which is the case this test
    // pins. If either were dynamic, the Bug 1 fix would make eviction
    // succeed and we'd never reach the grace timeout.
    let mut alpha = minimal_llama_service("alpha", 0);
    let mut beta = minimal_llama_service("beta", 0);

    for svc in [&mut alpha, &mut beta] {
        svc.placement_override.clear();
        svc.placement_override.insert(DeviceSlot::Cpu, 10 * 1024);
        svc.priority = 50;
        // Default `max_request_duration_ms` is 5 s in `minimal_llama_service`
        // — too short to let the 30 s grace bound fire. Bump the request
        // budget so the failure path the test is observing is the grace
        // bound itself, not the outer request timeout.
        svc.max_request_duration_ms = 60_000;
    }

    let h = build_harness(vec![alpha, beta]).await;
    // 15 GB total / 15 GB available — fits one 10 GB pledge, not two.
    *h.state.snapshot.write() = DeviceSnapshot {
        gpus: Vec::new(),
        cpu: Some(CpuSnapshot {
            total_bytes: 15 * 1024 * 1024 * 1024,
            available_bytes: 15 * 1024 * 1024 * 1024,
        }),
        taken_at_ms: 0,
    };

    // Step 1: alpha loads.
    let (st, body) = chat(openai::router(h.state.clone()), "alpha").await;
    assert_eq!(st, StatusCode::OK, "alpha must load first: {body}");

    // Step 2: peg alpha's inflight counter so it's permanently "busy"
    // for the duration of the test. The release task in the
    // happy-path queue test is *intentionally absent* here — we want
    // the grace bound to fire.
    let alpha_inflight = h.state.inflight.counter(&SmolStr::new("alpha"));
    alpha_inflight.fetch_add(1, Ordering::Relaxed);

    // Step 3: beta fires. The supervisor enters the queue (alpha is busy
    // at tied priority and non-elastic), polls every 250 ms, and after
    // QUEUE_BLOCKED_GRACE (10 s) gives up and resolves the bus with
    // `service_blocked`. The test runs in real wall-time so it takes
    // ~10 s; the constant is sized for production and we just live
    // with the test duration.
    let (st, body) = chat(openai::router(h.state.clone()), "beta").await;
    assert_eq!(
        st,
        StatusCode::SERVICE_UNAVAILABLE,
        "beta must 503 after the grace bound: {body}"
    );
    assert!(
        body.contains("service_blocked"),
        "error body must carry the `service_blocked` code so the CLI can render a useful message: {body}"
    );
    assert!(
        body.contains("alpha"),
        "error body must name the busy peer so the operator knows what to kill or wait on: {body}"
    );

    // alpha still holds its allocation — the queue gave up cleanly
    // without evicting anything (the whole point of the structured
    // failure is "we're not going to displace this peer; the client
    // should decide what to do").
    let alloc = h.state.allocations.lock().clone();
    assert!(
        alloc.contains_key(&SmolStr::new("alpha")),
        "alpha should still hold its reservation after the failed queue: {alloc:?}"
    );
    assert!(
        !alloc.contains_key(&SmolStr::new("beta")),
        "beta should not have acquired any allocation: {alloc:?}"
    );

    // Defensive: if `QUEUE_BLOCKED_GRACE` is ever lengthened past
    // `max_request_duration_ms` (60 s above), beta will hit the outer
    // request timeout first and the assertions will fail with
    // `start_failed` instead of `service_blocked`, surfacing the mismatch.
    let _ = Duration::from_secs(10);

    h.cleanup().await;
}
