//! Integration test: the time-to-first-token stall watchdog drains and
//! respawns a service that accepts a request but never emits a token.
//!
//! Mirrors a production incident: `gemma-4-31b-it-qat` accepted a streaming
//! request — logging the chat format and launching the slot — then produced
//! no tokens and held the response open for over five hours. The process
//! stayed alive and every request ended in a client-side cancel, so neither
//! the crash path nor the 5xx error-rate watchdog (which only sees completed
//! requests) ever fired. Here the wedge is simulated by the echo server's
//! `hang` mode: 200 headers, then a body that never yields a frame. The
//! watchdog's per-request timer then observes the stall and self-heals.
//! Runs under `start_paused` so the stall timeout advances virtually.
#![cfg(feature = "test-fakes")]

mod common;

use std::{sync::atomic::Ordering, time::Duration};

use ananke::{
    api::openai,
    config::{AutoRestartSettings, TtftStallTrigger},
    supervise::state::ServiceState,
    system::FakeProcessState,
};
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use common::{build_harness, minimal_llama_service};
use smol_str::SmolStr;
use tower::util::ServiceExt;

/// A streaming chat request. Streaming matters: the stall watchdog waits for
/// the first *frame* on streaming requests (headers arrive before any token),
/// which is exactly the shape of the wedge.
fn streaming_chat_request() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"alpha","stream":true,"messages":[]}"#,
        ))
        .unwrap()
}

/// Only the stall watchdog is enabled, with a short timeout so the virtual
/// clock reaches it quickly. Error-rate is off so the test isolates the stall
/// path.
fn stall_only(timeout_ms: u64, max_restarts: u32) -> AutoRestartSettings {
    AutoRestartSettings {
        error_rate: None,
        periodic: None,
        ttft_stall: Some(TtftStallTrigger { timeout_ms }),
        min_uptime_ms: 0,
        max_restarts,
        flap_window_ms: 1_800_000,
    }
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn stall_watchdog_restarts_wedged_service() {
    let mut svc = minimal_llama_service("alpha", 0);
    // Keep the idle timeout well out of the way so only the watchdog can drain.
    svc.idle_timeout_ms = 600_000;
    svc.auto_restart = stall_only(60_000, 3);

    let h = build_harness(vec![svc]).await;
    let app = openai::router(h.state.clone());

    // Cold-start to Running with a healthy (non-hanging) request.
    let resp = app.clone().oneshot(streaming_chat_request()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    drop(resp);

    let sup = &h.supervisors[0];
    assert!(matches!(sup.peek_state(), ServiceState::Running));
    let run_id = sup.peek().run_id.expect("running has a run_id");

    // Wedge the upstream: subsequent requests get 200 headers then silence.
    h.echo_state.hang.store(true, Ordering::Relaxed);

    // Fire a request that will hang. The response headers arrive (200), but the
    // body never produces a frame, so the stall timer stays armed.
    let resp = app.clone().oneshot(streaming_chat_request()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    drop(resp);

    // Advance past the stall timeout; the watchdog fires and drains to Idle.
    let mut drained = false;
    for _ in 0..90 {
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
        if matches!(sup.peek_state(), ServiceState::Idle) {
            drained = true;
            break;
        }
    }
    assert!(
        drained,
        "stall watchdog did not drain to Idle; state = {:?}",
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
    h.echo_state.hang.store(false, Ordering::Relaxed);
    let resp2 = app.oneshot(streaming_chat_request()).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::OK);
    tokio::task::yield_now().await;
    let new_run = sup.peek().run_id.expect("respawned run_id");
    assert_ne!(
        new_run, run_id,
        "expected a fresh run after the stall restart"
    );

    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn healthy_stream_does_not_trigger_stall() {
    let mut svc = minimal_llama_service("alpha", 0);
    svc.idle_timeout_ms = 600_000;
    // A deliberately short timeout: if the watchdog were going to misfire on a
    // healthy request, this would give it every chance to.
    svc.auto_restart = stall_only(5_000, 3);

    let h = build_harness(vec![svc]).await;
    let app = openai::router(h.state.clone());

    // A normal streaming request. The echo server answers immediately, and the
    // proxied body carries a data frame, so the timer is disarmed.
    let resp = app.clone().oneshot(streaming_chat_request()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    // Drain the body so GuardedBody polls the frame and disarms the timer,
    // matching a real client that reads the stream.
    let _ = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();

    let sup = &h.supervisors[0];
    let run_id = sup.peek().run_id.expect("running has a run_id");

    // Advance well past the (short) stall timeout.
    for _ in 0..10 {
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
    }
    assert!(
        matches!(sup.peek_state(), ServiceState::Running),
        "a healthy request must not trip the stall watchdog; state = {:?}",
        sup.peek_state()
    );
    assert_eq!(
        sup.peek().run_id,
        Some(run_id),
        "the run must be unchanged after a healthy request"
    );

    h.cleanup().await;
}

/// Repeated stalls trip the shared flap cap and disable the service, exactly
/// as repeated error-rate storms do — the two watchdogs share the budget.
#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn repeated_stalls_trip_flap_cap_and_disable() {
    let mut svc = minimal_llama_service("alpha", 0);
    svc.idle_timeout_ms = 600_000;
    // One restart tolerated; the second stall disables instead.
    svc.auto_restart = stall_only(60_000, 1);

    let h = build_harness(vec![svc]).await;
    let app = openai::router(h.state.clone());
    let sup = &h.supervisors[0];

    // Cold-start healthy, then wedge.
    let resp = app.clone().oneshot(streaming_chat_request()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    drop(resp);
    let run_a = sup.peek().run_id.expect("run A");
    h.echo_state.hang.store(true, Ordering::Relaxed);

    // First stall → restart to Idle.
    let resp = app.clone().oneshot(streaming_chat_request()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    drop(resp);
    let mut restarted = false;
    for _ in 0..90 {
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
        if matches!(sup.peek_state(), ServiceState::Idle) {
            restarted = true;
            break;
        }
    }
    assert!(restarted, "first stall should restart to Idle");

    // Respawn (still wedged) and stall again → flap cap trips → Disabled.
    let resp = app.oneshot(streaming_chat_request()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    drop(resp);
    let run_b = sup.peek().run_id.expect("run B");
    assert_ne!(run_b, run_a);

    let mut disabled = false;
    for _ in 0..90 {
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

    h.cleanup().await;
}

/// A request that stalls while a *concurrent* request keeps the service
/// producing frames must NOT restart. This is the queued-behind-healthy-work
/// case (llama.cpp serialises at `--parallel 1`): the stalled request's own
/// timer expires, but the run-level progress check sees recent frames and
/// declines. Without that check this restarts a healthy service — and at the
/// flap cap, disables it. Concurrent healthy output is simulated by stamping
/// the service's shared progress cell as the virtual clock advances.
#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn concurrent_healthy_request_suppresses_stall_restart() {
    let mut svc = minimal_llama_service("alpha", 0);
    svc.idle_timeout_ms = 600_000;
    svc.auto_restart = stall_only(60_000, 3);

    let h = build_harness(vec![svc]).await;
    let app = openai::router(h.state.clone());

    // Cold-start (also creates the progress cell), then wedge the upstream.
    let resp = app.clone().oneshot(streaming_chat_request()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    drop(resp);
    let sup = &h.supervisors[0];
    let run_id = sup.peek().run_id.expect("run id");
    h.echo_state.hang.store(true, Ordering::Relaxed);

    // Fire the request that will stall (its body never produces a frame).
    let resp = app.oneshot(streaming_chat_request()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    drop(resp);

    // Advance past the stall timeout, but keep stamping progress each second to
    // stand in for a concurrent healthy generation streaming tokens.
    let cell = h.state.progress.stamp(&SmolStr::new("alpha"));
    for _ in 0..90 {
        cell.record();
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
    }

    assert!(
        matches!(sup.peek_state(), ServiceState::Running),
        "a stall alongside a healthy concurrent request must not restart; state = {:?}",
        sup.peek_state()
    );
    assert_eq!(sup.peek().run_id, Some(run_id), "the run must be unchanged");

    h.cleanup().await;
}

/// When the stall fires with the wedged request's client still connected (its
/// in-flight guard held), the restart must not sit for `max_request_duration`
/// waiting on a request that will never complete. With a 10-minute
/// `max_request_duration`, a correctly bounded drain still reaches Idle in
/// seconds of virtual time.
#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn stall_restart_does_not_wait_out_max_request_duration() {
    let mut svc = minimal_llama_service("alpha", 0);
    svc.idle_timeout_ms = 600_000;
    // Deliberately huge: an unbounded drain would wait this long for the
    // wedged request. The bounded stall drain must not.
    svc.max_request_duration_ms = 600_000;
    svc.auto_restart = stall_only(60_000, 3);

    let h = build_harness(vec![svc]).await;
    let app = openai::router(h.state.clone());

    let resp = app.clone().oneshot(streaming_chat_request()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    drop(resp);
    let sup = &h.supervisors[0];
    h.echo_state.hang.store(true, Ordering::Relaxed);

    // Fire the stalling request and HOLD its response: the in-flight guard
    // stays elevated, so the drain cannot rely on the client disconnecting.
    let held = app.oneshot(streaming_chat_request()).await.unwrap();
    assert_eq!(held.status(), StatusCode::OK);

    // Advance past the 60s stall timeout plus the bounded drain (5s inflight
    // grace + drain/sigterm), but far short of the 600s max_request_duration.
    let mut drained = false;
    for _ in 0..90 {
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
        if matches!(sup.peek_state(), ServiceState::Idle) {
            drained = true;
            break;
        }
    }
    assert!(
        drained,
        "bounded stall drain should reach Idle well under max_request_duration; state = {:?}",
        sup.peek_state()
    );

    drop(held);
    h.cleanup().await;
}
