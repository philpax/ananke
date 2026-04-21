//! Integration test: a service with `lifecycle = persistent` must not drain
//! to Idle via the idle-timeout path, even when `last_activity` is ancient
//! relative to `idle_timeout_ms`.
//!
//! Regression target: persistent services that came up without receiving
//! traffic would idle-time-out the instant they reached Running (their
//! `last_activity` stamp hadn't been pinged since before the spawn, so the
//! deadline was already in the past). The Idle state transition then
//! triggered `persistent_watcher` to re-ensure, producing an endless
//! ~15-second spawn/idle/respawn loop visible in the journal.

mod common;

use std::time::Duration;

use ananke::{api::openai, config::Lifecycle, supervise::state::ServiceState};
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use common::{build_harness, minimal_llama_service};
use smol_str::SmolStr;
use tower::util::ServiceExt;

#[tokio::test(flavor = "multi_thread")]
async fn persistent_service_never_idle_times_out() {
    let mut svc = minimal_llama_service("resident", 0);
    svc.lifecycle = Lifecycle::Persistent;
    // Deliberately short idle timeout: with the pre-fix code the service
    // would drain ~100 ms after entering Running. The test would still
    // expose the bug at a higher value; this just makes the assertion
    // cheap to wait on.
    svc.idle_timeout_ms = 100;

    let h = build_harness(vec![svc]).await;

    // Kick the service into Running via a chat request. (An alternative
    // path — persistent_watcher auto-ensure at boot — has the same shape
    // but is harder to synchronise against in tests.)
    let body = r#"{"model":"resident","messages":[]}"#;
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = openai::router(h.state.clone()).oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Wait for Running, then hold for several idle-timeout windows. A
    // non-persistent service at 100 ms timeout would have drained to Idle
    // within ~250 ms; a persistent one must stay Running the whole time.
    let handle = h
        .state
        .registry
        .get(&SmolStr::new("resident"))
        .expect("handle exists");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    while handle.peek_state() != ServiceState::Running {
        if tokio::time::Instant::now() >= deadline {
            panic!("never reached Running; state={:?}", handle.peek_state());
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(
        handle.peek_state(),
        ServiceState::Running,
        "persistent service must not drain via idle-timeout"
    );

    h.cleanup().await;
}
