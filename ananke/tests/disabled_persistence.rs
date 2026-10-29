//! Integration tests: an operator disable is stored in the database and
//! restored when the service is provisioned again (daemon restart or a
//! reload that adds it), before a persistent service can start; enabling
//! clears it.
#![cfg(feature = "test-fakes")]

mod common;

use std::time::Duration;

use ananke::{
    api::{management::router, openai},
    config::{EffectiveConfig, Lifecycle},
    supervise::{
        DisableResult, EnableResult,
        state::{DisableReason, ServiceState},
    },
};
use ananke_api::events::Event;
use axum::{body::to_bytes, http::StatusCode};
use common::{build_harness, minimal_llama_service};
use smol_str::SmolStr;
use tower::util::ServiceExt;

async fn post(app: &axum::Router, uri: &str) -> String {
    let req = axum::http::Request::builder()
        .method("POST")
        .uri(uri)
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

async fn disabled_at(h: &common::TestHarness, name: &str) -> Option<i64> {
    let id = h.state.db.resolve_service_id(name).await.unwrap().unwrap();
    h.state.db.service_disabled_at(id).await.unwrap()
}

#[tokio::test(flavor = "current_thread")]
async fn disable_and_enable_write_and_clear_the_stored_mark() {
    let h = build_harness(vec![minimal_llama_service("demo", 0)]).await;
    let app = router(h.state.clone());

    assert_eq!(disabled_at(&h, "demo").await, None);
    post(&app, "/api/services/demo/disable").await;
    assert!(disabled_at(&h, "demo").await.is_some());
    // A repeat disable keeps the mark and reports already-disabled.
    let body = post(&app, "/api/services/demo/disable").await;
    assert!(body.contains("already_disabled"), "{body}");
    assert!(disabled_at(&h, "demo").await.is_some());

    post(&app, "/api/services/demo/enable").await;
    assert_eq!(disabled_at(&h, "demo").await, None);

    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn enable_clears_a_stale_mark_on_a_service_that_is_not_disabled() {
    let h = build_harness(vec![minimal_llama_service("demo", 0)]).await;
    let app = router(h.state.clone());

    let id = h
        .state
        .db
        .resolve_service_id("demo")
        .await
        .unwrap()
        .unwrap();
    h.state.db.mark_service_disabled(id, 0).await.unwrap();
    let body = post(&app, "/api/services/demo/enable").await;
    assert!(body.contains("already_enabled"), "{body}");
    assert_eq!(disabled_at(&h, "demo").await, None);

    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn stored_disable_keeps_a_provisioned_persistent_service_from_starting() {
    let alpha = minimal_llama_service("alpha", 0);
    let h = build_harness(vec![alpha.clone()]).await;

    // An earlier run disabled `beta`; this daemon has not provisioned it.
    let beta_id = h.state.db.upsert_service("beta", 0).await.unwrap();
    h.state.db.mark_service_disabled(beta_id, 0).await.unwrap();

    let mut beta = minimal_llama_service("beta", 0);
    beta.lifecycle = Lifecycle::Persistent;
    beta.private_port = alpha.private_port;
    beta.port = common::free_port();
    let old = h.state.config.effective();
    let new = EffectiveConfig {
        daemon: old.daemon.clone(),
        services: vec![alpha.clone(), beta.clone()],
    };
    drop(old);
    h.state.config.swap_effective_for_test(new);
    h.state.events.publish(Event::ConfigReloaded {
        at_ms: 0,
        changed_services: vec![SmolStr::new("beta")],
    });

    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let handle = loop {
        if let Some(handle) = h.state.registry.get("beta") {
            break handle;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "reconciler never provisioned beta"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    };
    // Registered only once the restored disable has been acked.
    assert_eq!(
        handle.peek_state(),
        ServiceState::Disabled {
            reason: DisableReason::UserDisabled
        }
    );
    // Give a wrongly-sent persistent ensure time to spawn a child.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        h.process_spawner.children().is_empty(),
        "a disabled persistent service was started: {:?}",
        h.process_spawner.children()
    );

    let app = router(h.state.clone());
    let body = post(&app, "/api/services/beta/enable").await;
    assert!(body.contains("\"enabled\""), "{body}");
    assert_eq!(disabled_at(&h, "beta").await, None);

    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn disabling_a_running_service_stores_the_mark() {
    let h = build_harness(vec![minimal_llama_service("demo", 0)]).await;
    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(r#"{"model":"demo","messages":[]}"#))
        .unwrap();
    let resp = openai::router(h.state.clone()).oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(h.supervisors[0].peek_state(), ServiceState::Running);

    let body = post(&router(h.state.clone()), "/api/services/demo/disable").await;
    assert!(body.contains("\"disabled\""), "{body}");
    assert!(disabled_at(&h, "demo").await.is_some());
    assert_eq!(
        h.supervisors[0].peek_state(),
        ServiceState::Disabled {
            reason: DisableReason::UserDisabled
        }
    );

    h.cleanup().await;
}

/// Once the supervisor task is gone nothing can be stored, so enable and
/// disable must not report success.
#[tokio::test(flavor = "current_thread")]
async fn a_stopped_supervisor_does_not_report_success() {
    let h = build_harness(vec![minimal_llama_service("demo", 0)]).await;
    let handle = h.supervisors[0].clone();
    handle.shutdown().await;
    assert!(matches!(
        handle.disable().await,
        DisableResult::PersistFailed(_)
    ));
    assert!(matches!(
        handle.enable().await,
        EnableResult::PersistFailed(_)
    ));
    assert_eq!(disabled_at(&h, "demo").await, None);

    h.cleanup().await;
}
