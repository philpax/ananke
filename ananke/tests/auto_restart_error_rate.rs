//! Integration test: the error-rate watchdog drains and respawns a service
//! that is alive but returning server errors.
//!
//! Mirrors the production incident (issue #17): a service ran healthy for
//! hours, then wedged into 100 % HTTP 500 while the process stayed alive, so
//! the crash path never fired. Here the wedge is simulated by recording 5xx
//! rows in `request_metrics` for the running run; the watchdog's poll then
//! observes the storm and self-heals. Runs under `start_paused` so the poll
//! interval and cooldown advance virtually.
#![cfg(feature = "test-fakes")]

mod common;

use std::time::Duration;

use ananke::{
    api::openai,
    config::{AutoRestartSettings, ErrorRateTrigger, ErrorStatusClass},
    db::models::RequestMetric,
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

/// Record `n` server-error requests against `(service_id, run_id)`, stamped
/// ~now so they land inside a default-length window.
async fn inject_server_errors(
    db: &ananke::db::Database,
    service_id: i64,
    run_id: i64,
    n: i64,
    status: i64,
) {
    let now = ananke::tracking::now_unix_ms();
    for i in 0..n {
        db.insert_request_metric(&RequestMetric {
            metric_id: 0,
            service_id,
            run_id: Some(run_id),
            timestamp_ms: now + i,
            endpoint: "/v1/chat/completions".into(),
            model: "alpha".into(),
            prompt_tokens: None,
            completion_tokens: None,
            duration_ms: Some(1000),
            ttft_ms: None,
            prompt_ms: None,
            predicted_ms: None,
            status_code: status,
        })
        .await
        .unwrap();
    }
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn error_rate_watchdog_restarts_wedged_service() {
    let mut svc = minimal_llama_service("alpha", 0);
    // Keep the idle timeout well out of the way so only the watchdog can drain.
    svc.idle_timeout_ms = 600_000;
    svc.auto_restart = AutoRestartSettings {
        error_rate: Some(ErrorRateTrigger {
            window_ms: 120_000,
            max_error_rate: 0.5,
            min_requests: 5,
            poll_interval_ms: 500,
            statuses: ErrorStatusClass::ServerOnly,
        }),
        periodic: None,
        min_uptime_ms: 1_000,
        max_restarts: 3,
        flap_window_ms: 1_800_000,
    };

    let h = build_harness(vec![svc]).await;
    let app = openai::router(h.state.clone());

    // Cold-start to Running.
    let resp = app.clone().oneshot(chat_request()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let sup = &h.supervisors[0];
    assert!(matches!(sup.peek_state(), ServiceState::Running));
    let run_id = sup.peek().run_id.expect("running has a run_id");
    let service_id = h
        .state
        .db
        .resolve_service_id("alpha")
        .await
        .unwrap()
        .unwrap();

    // Simulate the wedge: six server-error requests recorded against this run.
    inject_server_errors(&h.state.db, service_id, run_id, 6, 500).await;

    // Advance past the cooldown and a poll tick; the watchdog fires and drains.
    let mut drained = false;
    for _ in 0..40 {
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
        if matches!(sup.peek_state(), ServiceState::Idle) {
            drained = true;
            break;
        }
    }
    assert!(
        drained,
        "error-rate watchdog did not drain to Idle; state = {:?}",
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

    // A fresh request spawns a new run — the self-heal is complete, and the
    // fresh run's metrics start from zero so the watchdog does not re-fire.
    let resp2 = app.oneshot(chat_request()).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::OK);
    tokio::task::yield_now().await;
    let new_run = sup.peek().run_id.expect("respawned run_id");
    assert_ne!(new_run, run_id, "expected a fresh run after auto-restart");

    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn client_errors_do_not_trigger_under_default_5xx() {
    let mut svc = minimal_llama_service("alpha", 0);
    svc.idle_timeout_ms = 600_000;
    svc.auto_restart = AutoRestartSettings {
        error_rate: Some(ErrorRateTrigger {
            window_ms: 120_000,
            max_error_rate: 0.5,
            min_requests: 5,
            poll_interval_ms: 500,
            statuses: ErrorStatusClass::ServerOnly,
        }),
        periodic: None,
        min_uptime_ms: 1_000,
        max_restarts: 3,
        flap_window_ms: 1_800_000,
    };

    let h = build_harness(vec![svc]).await;
    let app = openai::router(h.state.clone());

    let resp = app.clone().oneshot(chat_request()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let sup = &h.supervisors[0];
    let run_id = sup.peek().run_id.expect("running has a run_id");
    let service_id = h
        .state
        .db
        .resolve_service_id("alpha")
        .await
        .unwrap()
        .unwrap();

    // A storm of 4xx client errors. Under the default server-only class these
    // must NOT trigger a restart — the client is at fault, not the service.
    inject_server_errors(&h.state.db, service_id, run_id, 20, 400).await;

    // Advance well past several poll intervals and the cooldown.
    for _ in 0..10 {
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
    }
    assert!(
        matches!(sup.peek_state(), ServiceState::Running),
        "4xx storm must not restart a server-only watchdog; state = {:?}",
        sup.peek_state()
    );

    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn repeated_storms_trip_flap_cap_and_disable() {
    let mut svc = minimal_llama_service("alpha", 0);
    svc.idle_timeout_ms = 600_000;
    svc.auto_restart = AutoRestartSettings {
        error_rate: Some(ErrorRateTrigger {
            window_ms: 120_000,
            max_error_rate: 0.5,
            min_requests: 5,
            poll_interval_ms: 500,
            statuses: ErrorStatusClass::ServerOnly,
        }),
        periodic: None,
        min_uptime_ms: 1_000,
        // One restart tolerated; the second storm disables instead.
        max_restarts: 1,
        flap_window_ms: 1_800_000,
    };

    let h = build_harness(vec![svc]).await;
    let app = openai::router(h.state.clone());
    let sup = &h.supervisors[0];
    let service_id = h
        .state
        .db
        .resolve_service_id("alpha")
        .await
        .unwrap()
        .unwrap();

    // First cycle: cold-start, storm, and confirm the watchdog restarts once.
    let resp = app.clone().oneshot(chat_request()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let run_a = sup.peek().run_id.expect("run A");
    inject_server_errors(&h.state.db, service_id, run_a, 6, 500).await;
    for _ in 0..40 {
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
        if matches!(sup.peek_state(), ServiceState::Idle) {
            break;
        }
    }
    assert!(
        matches!(sup.peek_state(), ServiceState::Idle),
        "first storm should restart to Idle; state = {:?}",
        sup.peek_state()
    );

    // Second cycle: respawn, storm again. This trips the flap cap and the
    // service is disabled rather than restarted a second time.
    let resp2 = app.oneshot(chat_request()).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::OK);
    let run_b = sup.peek().run_id.expect("run B");
    assert_ne!(run_b, run_a);
    inject_server_errors(&h.state.db, service_id, run_b, 6, 500).await;

    let mut disabled = false;
    for _ in 0..40 {
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
        if matches!(sup.peek_state(), ServiceState::Disabled { .. }) {
            disabled = true;
            break;
        }
    }
    assert!(
        disabled,
        "second storm should trip the flap cap and disable; state = {:?}",
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

/// Regression: a manual re-enable after an `AutoRestartLoop` disable must
/// grant a fresh restart budget. With `max_restarts = 1` the flap history is
/// non-empty at disable time; without clearing it on enable, the next storm
/// after re-enable would prune-but-still-see a full history and re-disable
/// immediately instead of restarting.
#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn reenable_after_flap_grants_fresh_restart_budget() {
    let mut svc = minimal_llama_service("alpha", 0);
    svc.idle_timeout_ms = 600_000;
    svc.auto_restart = AutoRestartSettings {
        error_rate: Some(ErrorRateTrigger {
            window_ms: 120_000,
            max_error_rate: 0.5,
            min_requests: 5,
            poll_interval_ms: 500,
            statuses: ErrorStatusClass::ServerOnly,
        }),
        periodic: None,
        min_uptime_ms: 1_000,
        // One restart tolerated, so the flap history holds one entry when the
        // second storm disables the service.
        max_restarts: 1,
        flap_window_ms: 1_800_000,
    };

    let h = build_harness(vec![svc]).await;
    let app = openai::router(h.state.clone());
    let sup = &h.supervisors[0];
    let service_id = h
        .state
        .db
        .resolve_service_id("alpha")
        .await
        .unwrap()
        .unwrap();

    let storm_and_wait = |app: axum::Router, want_disabled: bool| {
        let db = h.state.db.clone();
        async move {
            let resp = app.oneshot(chat_request()).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
            let run = sup.peek().run_id.expect("run id");
            inject_server_errors(&db, service_id, run, 6, 500).await;
            for _ in 0..40 {
                tokio::time::advance(Duration::from_secs(1)).await;
                tokio::task::yield_now().await;
                let done = if want_disabled {
                    matches!(sup.peek_state(), ServiceState::Disabled { .. })
                } else {
                    matches!(sup.peek_state(), ServiceState::Idle)
                };
                if done {
                    break;
                }
            }
            run
        }
    };

    // Cycle 1: restart to Idle. Cycle 2: flap cap trips → Disabled.
    let run_a = storm_and_wait(app.clone(), false).await;
    assert!(matches!(sup.peek_state(), ServiceState::Idle));
    let run_b = storm_and_wait(app.clone(), true).await;
    assert_ne!(run_b, run_a);
    assert!(
        matches!(
            sup.peek_state(),
            ServiceState::Disabled {
                reason: ananke::supervise::state::DisableReason::AutoRestartLoop
            }
        ),
        "second storm should disable with AutoRestartLoop; state = {:?}",
        sup.peek_state()
    );

    // Operator re-enables → Idle with a cleared flap history.
    sup.enable().await;
    for _ in 0..5 {
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
        if matches!(sup.peek_state(), ServiceState::Idle) {
            break;
        }
    }
    assert!(matches!(sup.peek_state(), ServiceState::Idle));

    // Cycle 3: with the budget reset, this storm must RESTART (reach Idle),
    // not immediately re-disable. With the bug the service would go straight
    // back to Disabled here.
    let run_c = storm_and_wait(app.clone(), false).await;
    assert_ne!(run_c, run_b, "re-enable should have produced a fresh run");
    assert!(
        matches!(sup.peek_state(), ServiceState::Idle),
        "re-enabled service must get a fresh restart, not an immediate re-disable; state = {:?}",
        sup.peek_state()
    );

    h.cleanup().await;
}
