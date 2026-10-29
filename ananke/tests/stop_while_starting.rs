//! Integration test: stopping a service while it is still Starting leaves its
//! supervisor able to start it again.
//!
//! The drain-while-starting path cancels the in-progress health probe. If that
//! cancellation outlives the aborted run, the next start's probe reports
//! `Cancelled` at once and the supervisor task exits, so every later request
//! fails with "supervisor unreachable" until the daemon restarts.
#![cfg(feature = "test-fakes")]

mod common;

use std::{sync::atomic::Ordering, time::Duration};

use ananke::{
    api::management::router,
    config::DeviceSlot,
    supervise::{EnsureSource, state::ServiceState},
};
use ananke_api::services::StartResponse;
use ananke_placement::devices::{CpuSnapshot, DeviceSnapshot};
use axum::{body::to_bytes, http::StatusCode};
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

async fn post(app: &axum::Router, uri: &str) -> (StatusCode, Vec<u8>) {
    let req = axum::http::Request::builder()
        .method("POST")
        .uri(uri)
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    (status, bytes.to_vec())
}

#[tokio::test(flavor = "multi_thread")]
async fn stop_while_starting_then_start_succeeds() {
    let h = build_harness(vec![minimal_llama_service("demo", 0)]).await;
    let app = router(h.state.clone());
    let handle = h.supervisors[0].clone();

    // Health never passes, so the first start parks in Starting.
    h.echo_state.unhealthy.store(true, Ordering::Relaxed);
    let first = handle
        .ensure(EnsureSource::UserRequest)
        .await
        .expect("supervisor answers the first ensure");
    drop(first);
    tokio::time::timeout(Duration::from_secs(5), async {
        while handle.peek_state() != ServiceState::Starting {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("service reaches Starting");

    let (status, _) = post(&app, "/api/services/demo/stop").await;
    assert_eq!(status, StatusCode::ACCEPTED);
    assert_eq!(handle.peek_state(), ServiceState::Idle);

    // The child is healthy now; a fresh start must reach Running.
    h.echo_state.unhealthy.store(false, Ordering::Relaxed);
    let (status, body) = post(&app, "/api/services/demo/start").await;
    assert_eq!(
        status,
        StatusCode::ACCEPTED,
        "start after stop-while-starting failed: {}",
        String::from_utf8_lossy(&body)
    );
    let parsed: StartResponse = serde_json::from_slice(&body).unwrap();
    assert!(
        matches!(parsed, StartResponse::Started { .. }),
        "unexpected start response: {parsed:?}"
    );
    assert_eq!(handle.peek_state(), ServiceState::Running);

    h.cleanup().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn disable_while_starting_then_enable_and_start_succeeds() {
    let h = build_harness(vec![minimal_llama_service("demo", 0)]).await;
    let app = router(h.state.clone());
    let handle = h.supervisors[0].clone();

    h.echo_state.unhealthy.store(true, Ordering::Relaxed);
    drop(handle.ensure(EnsureSource::UserRequest).await);
    tokio::time::timeout(Duration::from_secs(5), async {
        while handle.peek_state() != ServiceState::Starting {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("service reaches Starting");

    let (status, _) = post(&app, "/api/services/demo/disable").await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = post(&app, "/api/services/demo/enable").await;
    assert_eq!(status, StatusCode::OK);

    h.echo_state.unhealthy.store(false, Ordering::Relaxed);
    let (status, body) = post(&app, "/api/services/demo/start").await;
    assert_eq!(
        status,
        StatusCode::ACCEPTED,
        "start after disable-while-starting failed: {}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(handle.peek_state(), ServiceState::Running);

    h.cleanup().await;
}

async fn wait_for_state(
    handle: &ananke::supervise::SupervisorHandle,
    want: ServiceState,
    what: &str,
) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while handle.peek_state() != want {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("{what}: never reached {want:?}"));
}

/// The sequence seen live on 2026-10-29: a higher-priority peer's start
/// evicts the service while it is still Starting, and a later management
/// start of the service has to evict that peer in turn. The second start
/// spawned its child and then failed 20 ms later with "start broadcast
/// closed", leaving the service Starting with a dead pid.
#[tokio::test(flavor = "multi_thread")]
async fn evicted_while_starting_then_start_with_eviction_succeeds() {
    let mut explorer = minimal_llama_service("explorer", 0);
    let mut big = minimal_llama_service("big", 0);
    // 10 GiB each on a 15 GiB CPU: one fits at a time.
    for svc in [&mut explorer, &mut big] {
        svc.placement_override.clear();
        svc.placement_override.insert(DeviceSlot::Cpu, 10 * 1024);
    }
    explorer.priority = 40;
    big.priority = 60;
    let h = build_harness(vec![explorer, big]).await;
    *h.state.snapshot.write() = DeviceSnapshot {
        gpus: Vec::new(),
        cpu: Some(CpuSnapshot {
            total_bytes: 15 * 1024 * 1024 * 1024,
            available_bytes: 15 * 1024 * 1024 * 1024,
        }),
        taken_at_ms: 0,
    };
    let app = router(h.state.clone());
    let explorer = h.supervisors[0].clone();
    let big = h.supervisors[1].clone();

    // Explorer parks in Starting.
    h.echo_state.unhealthy.store(true, Ordering::Relaxed);
    drop(explorer.ensure(EnsureSource::UserRequest).await);
    wait_for_state(&explorer, ServiceState::Starting, "explorer").await;

    // Big needs explorer's room and evicts it mid-start.
    drop(big.ensure(EnsureSource::UserRequest).await);
    wait_for_state(&explorer, ServiceState::Idle, "evicted explorer").await;
    h.echo_state.unhealthy.store(false, Ordering::Relaxed);
    wait_for_state(&big, ServiceState::Running, "big").await;

    // Explorer's start now has to evict the idle big first.
    let (status, body) = post(&app, "/api/services/explorer/start").await;
    assert_eq!(
        status,
        StatusCode::ACCEPTED,
        "start after eviction-while-starting failed: {}",
        String::from_utf8_lossy(&body)
    );
    let parsed: StartResponse = serde_json::from_slice(&body).unwrap();
    assert!(
        matches!(parsed, StartResponse::Started { .. }),
        "unexpected start response: {parsed:?}"
    );
    assert_eq!(explorer.peek_state(), ServiceState::Running);
    assert_eq!(big.peek_state(), ServiceState::Idle);

    h.cleanup().await;
}
