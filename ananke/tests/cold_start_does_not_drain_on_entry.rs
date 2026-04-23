//! Regression: when a cold-start `Ensure` promotes the service from Idle to
//! Running, the idle-timeout arm in `run_running_loop` must not fire on the
//! very first tick. If it does, the child is SIGTERM'd before the caller's
//! proxied request can reach it.
//!
//! The hazard: `last_activity` is an `Arc<Mutex<Instant>>` shared between the
//! activity tracker and the supervisor. After a previous request drains the
//! service to Idle, the stamp is stale by construction (it is at most
//! `idle_timeout_ms` old relative to the drain, but the service can then sit
//! idle for arbitrarily long). The next `Ensure` runs through Starting →
//! Running and enters `run_running_loop`, whose `sleep_until` deadline is
//! based on the stale stamp and therefore already elapsed.
//!
//! The handler refreshes the stamp via `state.activity.ping(...)` only after
//! `await_ensure` returns, so the correctness of the first idle-timeout
//! evaluation depends on task-scheduling order — which is not a correctness
//! guarantee. Observed in production: Apr 23, supervisor logged
//! `state transition starting→running` and `idle timeout; draining to idle`
//! in the same millisecond, and the upstream proxy failed with `client error
//! (Connect)`.

#![cfg(feature = "test-fakes")]

mod common;

use std::time::Duration;

use ananke::{api::openai, supervise::state::ServiceState, system::FakeProcessState};
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn cold_start_does_not_drain_on_entry_to_running() {
    let mut svc = minimal_llama_service("alpha", 0);
    svc.idle_timeout_ms = 500;

    let h = build_harness(vec![svc]).await;
    let app = openai::router(h.state.clone());
    let body = r#"{"model":"alpha","messages":[]}"#;

    // First request: cold-start, spawn child, proxy, return 200. This also
    // primes `last_activity` via the handler's post-ensure ping.
    let req1 = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp1 = app.clone().oneshot(req1).await.unwrap();
    assert_eq!(resp1.status(), StatusCode::OK);

    // Advance past the idle deadline. The supervisor drains the first child
    // and returns to Idle. `last_activity` is now stale relative to virtual
    // `now()`.
    tokio::time::advance(Duration::from_secs(30)).await;
    tokio::task::yield_now().await;

    let sup = &h.supervisors[0];
    assert!(
        matches!(sup.peek_state(), ServiceState::Idle),
        "precondition: supervisor should be Idle after the advance; got {:?}",
        sup.peek_state()
    );

    // Second request: cold-start again with stale `last_activity`. The
    // supervisor must NOT idle-drain the newly-promoted child on the first
    // tick of `run_running_loop`.
    let req2 = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp2 = app.oneshot(req2).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::OK);
    tokio::task::yield_now().await;

    // With the race, the supervisor would already have transitioned back to
    // Idle (or to Draining) by this point, and the second fake child would be
    // in SigTerm. With the fix, both stay Running until the next idle window
    // legitimately elapses.
    assert!(
        matches!(sup.peek_state(), ServiceState::Running),
        "supervisor drained immediately after entering Running; state = {:?}",
        sup.peek_state()
    );

    let children = h.process_spawner.children();
    assert_eq!(
        children.len(),
        2,
        "expected exactly two spawns (cold-start per request); got {}",
        children.len()
    );
    assert!(
        matches!(children[1].state, FakeProcessState::Running),
        "second child was terminated during the cold-start race; state = {:?}",
        children[1].state
    );

    h.cleanup().await;
}
