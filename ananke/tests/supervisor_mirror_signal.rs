//! The supervisor mirror's change signal wakes a reader for every change it
//! must render: the pid appearing on start and clearing on stop.
#![cfg(feature = "test-fakes")]

mod common;

use std::time::Duration;

use ananke::api::management::router;
use axum::http::StatusCode;
use common::{build_harness, minimal_llama_service};
use tokio::sync::watch;
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn mirror_signal_wakes_on_pid_set_and_clear() {
    let h = build_harness(vec![minimal_llama_service("demo", 0)]).await;
    let handle = h.state.registry.get("demo").unwrap();
    let mut rx = h.state.mirror_changed.subscribe();

    post(&h.state, "/api/services/demo/start").await;
    wait_until(&mut rx, || handle.peek().pid.is_some()).await;

    post(&h.state, "/api/services/demo/stop").await;
    wait_until(&mut rx, || handle.peek().pid.is_none()).await;

    h.cleanup().await;
}

/// A child that dies on its own goes running → failed → idle without the
/// drain path; its pid and running row must still be cleared.
#[tokio::test(flavor = "current_thread")]
async fn a_crashed_child_clears_its_pid_and_running_row() {
    let h = build_harness(vec![minimal_llama_service("demo", 0)]).await;
    let handle = h.state.registry.get("demo").unwrap();
    let mut rx = h.state.mirror_changed.subscribe();

    post(&h.state, "/api/services/demo/start").await;
    wait_until(&mut rx, || handle.peek().state.name() == "running").await;
    let pid = handle.peek().pid.expect("a running child has a pid");
    assert_eq!(h.state.db.list_running().await.unwrap().len(), 1);

    assert!(h.process_spawner.exit(pid as u32, 1));
    wait_until(&mut rx, || {
        let peek = handle.peek();
        peek.state.name() == "idle" && peek.pid.is_none() && peek.run_id.is_none()
    })
    .await;
    assert!(h.state.db.list_running().await.unwrap().is_empty());

    h.cleanup().await;
}

async fn post(state: &ananke::daemon::app_state::AppState, uri: &str) {
    let req = axum::http::Request::builder()
        .method("POST")
        .uri(uri)
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = router(state.clone()).oneshot(req).await.unwrap();
    assert!(
        matches!(resp.status(), StatusCode::OK | StatusCode::ACCEPTED),
        "{uri}: {}",
        resp.status()
    );
}

/// Wait on the signal alone, never on a poll interval, so a missing
/// notification fails the test instead of being papered over.
async fn wait_until(rx: &mut watch::Receiver<u64>, condition: impl Fn() -> bool) {
    while !condition() {
        tokio::time::timeout(Duration::from_secs(5), rx.changed())
            .await
            .expect("the mirror signal must wake before the condition holds")
            .unwrap();
    }
}
