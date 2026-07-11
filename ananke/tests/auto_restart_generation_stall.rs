//! Integration test: the generation-stall watchdog drains and respawns a
//! service whose `/metrics` progress counters stay flat under load.
//!
//! Mirrors a production incident the TTFT watchdog could not see:
//! `gemma-4-31b-it-qat` hit llama.cpp's SWA cache-invalidation wedge
//! (ggml-org/llama.cpp#22450) and produced zero tokens for ~74 minutes while
//! a *non-streaming* eval client retried every five minutes. Non-streaming
//! responses arrive fully buffered, so the proxy has no frame to watch and
//! the TTFT stall timer is never armed; the child's own Prometheus progress
//! counters are the only signal that separates "wedged" from "slow but
//! healthy". Here the echo server serves a controllable `/metrics` body and
//! the `hang` mode simulates the wedge. Runs under `start_paused` so the
//! stall timeout advances virtually.
#![cfg(feature = "test-fakes")]

mod common;

use std::{sync::atomic::Ordering, time::Duration};

use ananke::{
    api::openai,
    config::{AutoRestartSettings, GenerationStallTrigger},
    supervise::state::ServiceState,
    system::FakeProcessState,
};
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

/// A non-streaming chat request — the exact shape the TTFT watchdog is blind
/// to, and the reason this watchdog exists.
fn buffered_chat_request() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"alpha","messages":[]}"#))
        .unwrap()
}

/// Only the generation-stall watchdog is enabled, with short spans so the
/// virtual clock reaches them quickly.
fn genstall_only(timeout_ms: u64, max_restarts: u32) -> AutoRestartSettings {
    AutoRestartSettings {
        error_rate: None,
        periodic: None,
        ttft_stall: None,
        generation_stall: Some(GenerationStallTrigger {
            timeout_ms,
            poll_interval_ms: 5_000,
        }),
        min_uptime_ms: 0,
        max_restarts,
        flap_window_ms: 1_800_000,
    }
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn genstall_watchdog_restarts_wedged_service() {
    let mut svc = minimal_llama_service("alpha", 0);
    // Keep the idle timeout well out of the way so only the watchdog can drain.
    svc.idle_timeout_ms = 600_000;
    svc.auto_restart = genstall_only(60_000, 3);

    let h = build_harness(vec![svc]).await;
    h.echo_state.metrics_enabled.store(true, Ordering::Relaxed);
    h.echo_state.metrics_counter.store(100, Ordering::Relaxed);
    let app = openai::router(h.state.clone());

    // Cold-start to Running with a healthy request.
    let resp = app.clone().oneshot(buffered_chat_request()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    drop(resp);

    let sup = &h.supervisors[0];
    assert!(matches!(sup.peek_state(), ServiceState::Running));
    let run_id = sup.peek().run_id.expect("running has a run_id");

    // Wedge the upstream and HOLD the in-flight request: the counter stays
    // flat while the request never completes — the production signature.
    h.echo_state.hang.store(true, Ordering::Relaxed);
    let held = app.clone().oneshot(buffered_chat_request()).await.unwrap();
    assert_eq!(held.status(), StatusCode::OK);

    // Advance past the stall timeout; the watchdog fires and drains to Idle.
    let mut drained = false;
    for _ in 0..120 {
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
        if matches!(sup.peek_state(), ServiceState::Idle) {
            drained = true;
            break;
        }
    }
    assert!(
        drained,
        "generation-stall watchdog did not drain to Idle; state = {:?}",
        sup.peek_state()
    );

    // The wedged child was terminated, and nothing respawns without a request.
    let children = h.process_spawner.children();
    assert_eq!(
        children.len(),
        1,
        "no respawn should happen without traffic"
    );
    assert!(
        matches!(
            children[0].state,
            FakeProcessState::SigTerm | FakeProcessState::SigKill
        ),
        "wedged child was not terminated; state = {:?}",
        children[0].state
    );

    // Recovery: clear the wedge and issue a fresh request. It spawns a new run.
    drop(held);
    h.echo_state.hang.store(false, Ordering::Relaxed);
    let resp2 = app.oneshot(buffered_chat_request()).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::OK);
    tokio::task::yield_now().await;
    let new_run = sup.peek().run_id.expect("respawned run_id");
    assert_ne!(
        new_run, run_id,
        "expected a fresh run after the stall restart"
    );

    h.cleanup().await;
}

/// A slow-but-healthy generation advances the progress counters even though
/// its (non-streaming) response is still buffering — the watchdog must not
/// fire. This is the false-positive case that makes proxy-side detection of
/// non-streaming stalls impossible and forces the counter-based design.
#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn advancing_counters_never_fire() {
    let mut svc = minimal_llama_service("alpha", 0);
    svc.idle_timeout_ms = 600_000;
    svc.auto_restart = genstall_only(30_000, 3);

    let h = build_harness(vec![svc]).await;
    h.echo_state.metrics_enabled.store(true, Ordering::Relaxed);
    let app = openai::router(h.state.clone());

    let resp = app.clone().oneshot(buffered_chat_request()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    drop(resp);
    let sup = &h.supervisors[0];
    let run_id = sup.peek().run_id.expect("run id");

    // The request never completes (hang), but the counters keep advancing —
    // a healthy long generation from the child's point of view.
    h.echo_state.hang.store(true, Ordering::Relaxed);
    let held = app.oneshot(buffered_chat_request()).await.unwrap();
    assert_eq!(held.status(), StatusCode::OK);

    for _ in 0..120 {
        h.echo_state
            .metrics_counter
            .fetch_add(10, Ordering::Relaxed);
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
    }

    assert!(
        matches!(sup.peek_state(), ServiceState::Running),
        "advancing counters must not trip the watchdog; state = {:?}",
        sup.peek_state()
    );
    assert_eq!(sup.peek().run_id, Some(run_id), "the run must be unchanged");

    drop(held);
    h.cleanup().await;
}

/// Repeated generation stalls trip the shared flap cap and disable the
/// service — the watchdogs share the same restart budget.
#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn repeated_genstalls_trip_flap_cap_and_disable() {
    let mut svc = minimal_llama_service("alpha", 0);
    svc.idle_timeout_ms = 600_000;
    // One restart tolerated; the second stall disables instead.
    svc.auto_restart = genstall_only(60_000, 1);

    let h = build_harness(vec![svc]).await;
    h.echo_state.metrics_enabled.store(true, Ordering::Relaxed);
    let app = openai::router(h.state.clone());
    let sup = &h.supervisors[0];

    // Cold-start healthy, then wedge.
    let resp = app.clone().oneshot(buffered_chat_request()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    drop(resp);
    let run_a = sup.peek().run_id.expect("run A");
    h.echo_state.hang.store(true, Ordering::Relaxed);

    // First stall → restart to Idle.
    let held_a = app.clone().oneshot(buffered_chat_request()).await.unwrap();
    assert_eq!(held_a.status(), StatusCode::OK);
    let mut restarted = false;
    for _ in 0..120 {
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
        if matches!(sup.peek_state(), ServiceState::Idle) {
            restarted = true;
            break;
        }
    }
    assert!(restarted, "first stall should restart to Idle");
    drop(held_a);

    // Respawn (still wedged) and stall again → flap cap trips → Disabled.
    let held_b = app.oneshot(buffered_chat_request()).await.unwrap();
    assert_eq!(held_b.status(), StatusCode::OK);
    let run_b = sup.peek().run_id.expect("run B");
    assert_ne!(run_b, run_a);

    let mut disabled = false;
    for _ in 0..120 {
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
        if matches!(sup.peek_state(), ServiceState::Disabled { .. }) {
            disabled = true;
            break;
        }
    }
    assert!(
        disabled,
        "second stall should trip the flap cap and disable; state = {:?}",
        sup.peek_state()
    );
    assert!(
        matches!(
            sup.peek_state(),
            ServiceState::Disabled {
                reason: ananke::supervise::state::DisableReason::AutoRestartLoop
            }
        ),
        "expected AutoRestartLoop disable reason; state = {:?}",
        sup.peek_state()
    );

    drop(held_b);
    h.cleanup().await;
}

/// A child without a usable `/metrics` (endpoint absent or not llama.cpp)
/// must never be restarted by this watchdog: no signal means no verdict. The
/// echo server's default (non-metrics) mode returns a plain-text body with no
/// recognisable counters, standing in for a `command` service that opted in
/// without exposing the endpoint.
#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn missing_metrics_endpoint_never_fires() {
    let mut svc = minimal_llama_service("alpha", 0);
    svc.idle_timeout_ms = 600_000;
    svc.auto_restart = genstall_only(30_000, 3);

    let h = build_harness(vec![svc]).await;
    // metrics_enabled stays false: /metrics answers 200 "hello", which parses
    // to no counters at all.
    let app = openai::router(h.state.clone());

    let resp = app.clone().oneshot(buffered_chat_request()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    drop(resp);
    let sup = &h.supervisors[0];

    // Wedge with a held request — the situation where a *working* watchdog
    // would fire. Without counters it must stay silent.
    h.echo_state.hang.store(true, Ordering::Relaxed);
    let held = app.oneshot(buffered_chat_request()).await.unwrap();
    assert_eq!(held.status(), StatusCode::OK);

    for _ in 0..120 {
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
    }
    assert!(
        matches!(sup.peek_state(), ServiceState::Running),
        "no counters means no verdict — the watchdog must not fire; state = {:?}",
        sup.peek_state()
    );

    drop(held);
    h.cleanup().await;
}
