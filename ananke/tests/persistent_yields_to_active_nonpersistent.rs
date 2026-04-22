//! Integration test: a persistent service's Ensure stands down when a
//! non-persistent peer is currently `Starting` or `Running`, rather than
//! queueing up to evict the peer the moment it goes idle.
//!
//! Regression target: the persistent watcher's periodic `ensure()` would
//! race a user-driven on-demand service's startup. The watcher fired while
//! the pool was quiet, but by the time the supervisor ran the placement
//! retry loop a non-persistent peer had already transitioned into
//! `Starting` → `Running`. Because `EvictionCandidate::idle` flips to
//! `true` for a Running peer with no in-flight requests, the queued
//! persistent ensure immediately evicted the freshly-started peer — the
//! exact opposite of what the watcher is trying to achieve.
//!
//! With the yield rule in place, a persistent service's ensure returns
//! `Unavailable` when a non-persistent peer is loading or running, and
//! the watcher simply re-fires on its own cadence once the pool quiets.
#![cfg(feature = "test-fakes")]

mod common;

use ananke::{
    api::openai,
    config::{DeviceSlot, Lifecycle},
    devices::{CpuSnapshot, DeviceSnapshot},
    supervise::{EnsureFailure, EnsureResponse, EnsureSource},
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
async fn persistent_yields_when_nonpersistent_peer_is_running() {
    let mut on_demand = minimal_llama_service("on_demand", 0);
    let mut persistent = minimal_llama_service("persistent", 0);

    // 10 GB each on CPU, available 15 GB — fits one at a time.
    for svc in [&mut on_demand, &mut persistent] {
        svc.placement_override.clear();
        svc.placement_override.insert(DeviceSlot::Cpu, 10 * 1024);
    }
    // Same priority so the rule isn't about priority arbitration — the
    // persistent-yield rule is orthogonal and must fire regardless.
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

    // Step 1: on_demand loads and reaches Running. Chat returns 200 only
    // after the proxied request completes, at which point the service is
    // settled in Running with inflight back at zero.
    let (st, body) = chat(openai::router(h.state.clone()), "on_demand").await;
    assert_eq!(st, StatusCode::OK, "on_demand must load first: {body}");

    let on_demand_name = SmolStr::new("on_demand");
    let persistent_name = SmolStr::new("persistent");

    // Sanity check: on_demand holds the allocation; persistent doesn't.
    {
        let alloc = h.state.allocations.lock().clone();
        assert!(
            alloc.contains_key(&on_demand_name),
            "on_demand should hold the reservation at this point; got {alloc:?}"
        );
        assert!(
            !alloc.contains_key(&persistent_name),
            "persistent should not yet hold any reservation; got {alloc:?}"
        );
    }

    // Step 2: fire the persistent service's Ensure directly, simulating a
    // tick of the persistent_watcher. Because on_demand is Running (and
    // non-persistent), the persistent supervisor should yield rather than
    // evict it to make room.
    let persistent_handle = h
        .state
        .registry
        .get("persistent")
        .expect("persistent supervisor registered");
    let resp = persistent_handle
        .ensure(EnsureSource::BackgroundWatcher)
        .await
        .expect("ensure command delivered");

    match resp {
        EnsureResponse::Unavailable(EnsureFailure::InsufficientVram(msg)) => {
            assert!(
                msg.contains("yielding") || msg.contains("yield"),
                "yield message expected, got: {msg}"
            );
        }
        other => panic!(
            "expected Unavailable(InsufficientVram) from yielded persistent ensure, got {other:?}"
        ),
    }

    // Step 3: the allocation table must be unchanged — on_demand keeps
    // its reservation, persistent didn't get one, and nothing was drained.
    let alloc = h.state.allocations.lock().clone();
    assert!(
        alloc.contains_key(&on_demand_name),
        "on_demand must retain its allocation after persistent yielded; got {alloc:?}"
    );
    assert!(
        !alloc.contains_key(&persistent_name),
        "persistent must not hold an allocation after yielding; got {alloc:?}"
    );

    h.cleanup().await;
}
