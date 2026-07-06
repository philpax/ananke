//! Integration test: the periodic-restart trigger drains and respawns a
//! `Running` service when its interval elapses, in each firing mode.
//!
//! `immediate` restarts the moment the interval passes; `on-request` marks
//! the run stale and lets the next request drive the restart, blocking that
//! request on the fresh process. Runs under `start_paused` so the interval
//! advances virtually.
#![cfg(feature = "test-fakes")]

mod common;

use std::time::Duration;

use ananke::{
    api::openai,
    config::{AutoRestartSettings, PeriodicMode, PeriodicTrigger},
    supervise::state::ServiceState,
    system::FakeProcessState,
};
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

fn chat_request() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"alpha","messages":[]}"#))
        .unwrap()
}

fn periodic_only(mode: PeriodicMode, interval_ms: u64) -> AutoRestartSettings {
    AutoRestartSettings {
        error_rate: None,
        periodic: Some(PeriodicTrigger { interval_ms, mode }),
        ttft_stall: None,
        min_uptime_ms: 0,
        max_restarts: 3,
        flap_window_ms: 1_800_000,
    }
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn periodic_immediate_restarts_on_interval() {
    let mut svc = minimal_llama_service("alpha", 0);
    svc.idle_timeout_ms = 600_000;
    svc.auto_restart = periodic_only(PeriodicMode::Immediate, 30_000);

    let h = build_harness(vec![svc]).await;
    let app = openai::router(h.state.clone());

    let resp = app.clone().oneshot(chat_request()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let sup = &h.supervisors[0];
    assert!(matches!(sup.peek_state(), ServiceState::Running));
    let run_id = sup.peek().run_id.expect("running run_id");

    // Advance past the interval; immediate mode drains straight to Idle even
    // with no traffic.
    let mut drained = false;
    for _ in 0..60 {
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
        if matches!(sup.peek_state(), ServiceState::Idle) {
            drained = true;
            break;
        }
    }
    assert!(
        drained,
        "periodic immediate did not drain; state = {:?}",
        sup.peek_state()
    );

    let children = h.process_spawner.children();
    assert!(
        matches!(
            children[0].state,
            FakeProcessState::SigTerm | FakeProcessState::SigKill
        ),
        "child not terminated on periodic restart; state = {:?}",
        children[0].state
    );

    // Next request spawns a fresh run.
    let resp2 = app.oneshot(chat_request()).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::OK);
    tokio::task::yield_now().await;
    assert_ne!(sup.peek().run_id.unwrap(), run_id);

    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn periodic_on_request_restarts_on_next_request() {
    let mut svc = minimal_llama_service("alpha", 0);
    svc.idle_timeout_ms = 600_000;
    svc.auto_restart = periodic_only(PeriodicMode::OnRequest, 30_000);

    let h = build_harness(vec![svc]).await;
    let app = openai::router(h.state.clone());

    let resp = app.clone().oneshot(chat_request()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let sup = &h.supervisors[0];
    let run_id = sup.peek().run_id.expect("running run_id");

    // Advance past the interval. On-request mode only arms a flag — the service
    // must stay Running until a request arrives.
    for _ in 0..40 {
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
    }
    assert!(
        matches!(sup.peek_state(), ServiceState::Running),
        "on-request must not restart without traffic; state = {:?}",
        sup.peek_state()
    );
    assert_eq!(
        sup.peek().run_id.unwrap(),
        run_id,
        "run must be unchanged before the triggering request"
    );

    // The next request triggers the drain → respawn and must itself succeed
    // (it blocks on the fresh process rather than hitting the stale one).
    let resp2 = app.oneshot(chat_request()).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::OK);
    tokio::task::yield_now().await;

    // Two spawns total: the original plus the on-request respawn.
    let children = h.process_spawner.children();
    assert_eq!(
        children.len(),
        2,
        "expected the original plus one respawn; got {}",
        children.len()
    );
    assert!(
        matches!(
            children[0].state,
            FakeProcessState::SigTerm | FakeProcessState::SigKill
        ),
        "original child should have been drained; state = {:?}",
        children[0].state
    );
    assert_ne!(
        sup.peek().run_id.unwrap(),
        run_id,
        "the triggering request should have spawned a fresh run"
    );

    h.cleanup().await;
}
