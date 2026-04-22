//! Integration test: when a service disappears from the effective config,
//! the reload reconciler drains its supervisor, terminates its child, and
//! drops it from the registry. Without the reconciler, removing a service
//! from the TOML would leak its supervisor for the lifetime of the daemon.
//!
//! The test uses the in-memory [`FakeSpawner`] so assertions are made
//! against `FakeProcessState` transitions — no OS pids, no real signals,
//! no sleeps on real wall-clock. A regression that forgets to drain the
//! removed service leaves its child in `Running` rather than `SigTerm`.
#![cfg(feature = "test-fakes")]

mod common;

use std::{sync::Arc, time::Duration};

use ananke::{config::EffectiveConfig, supervise::SupervisorHandle, system::FakeProcessState};
use ananke_api::Event;
use common::{build_harness, minimal_llama_service};
use smol_str::SmolStr;

async fn wait_running_pid(handle: &SupervisorHandle, timeout_ms: u64) -> u32 {
    let deadline = tokio::time::Instant::now() + Duration::from_millis(timeout_ms);
    loop {
        if let Some(pid) = handle.peek().pid {
            return pid as u32;
        }
        if tokio::time::Instant::now() >= deadline {
            panic!("supervisor never reported a running pid");
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

#[tokio::test(flavor = "current_thread")]
async fn removing_service_from_config_drains_its_child_and_drops_registry() {
    let alpha = minimal_llama_service("alpha", 0);
    let beta = minimal_llama_service("beta", 0);
    let h = build_harness(vec![alpha.clone(), beta.clone()]).await;

    let alpha_handle = Arc::clone(
        h.supervisors
            .iter()
            .find(|s| s.name.as_str() == "alpha")
            .unwrap(),
    );
    let beta_handle = Arc::clone(
        h.supervisors
            .iter()
            .find(|s| s.name.as_str() == "beta")
            .unwrap(),
    );

    // Drive both supervisors to Running so the fake spawner records two
    // live children. `ensure()` awaits a successful start or unavailable
    // reply; the echo server satisfies the health probe in either case.
    let _ = alpha_handle.ensure(ananke::supervise::EnsureSource::UserRequest).await;
    let _ = beta_handle.ensure(ananke::supervise::EnsureSource::UserRequest).await;

    let alpha_pid = wait_running_pid(&alpha_handle, 5_000).await;
    let beta_pid = wait_running_pid(&beta_handle, 5_000).await;

    // Pre-condition: both fake children are Running.
    let snapshot = h.process_spawner.children();
    assert_eq!(snapshot.len(), 2, "expected exactly two spawned children");
    for s in &snapshot {
        assert_eq!(
            s.state,
            FakeProcessState::Running,
            "pid {} should be Running before reload",
            s.pid
        );
    }

    // Stage the post-reload state: beta is gone. Publish the reload event
    // so the reconciler can react.
    let old = h.state.config.effective();
    let new = EffectiveConfig {
        daemon: old.daemon.clone(),
        services: vec![alpha.clone()],
    };
    drop(old);
    h.state.config.swap_effective_for_test(new);
    h.state.events.publish(Event::ConfigReloaded {
        at_ms: 0,
        changed_services: vec![SmolStr::new("beta")],
    });

    // Wait for the reconciler to drive the drain to completion: beta's
    // registry slot is gone and its fake child is terminated. The drain
    // pipeline uses SIGTERM first; the fake responds instantly so we see
    // SigTerm rather than SigKill.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let beta_gone = h.state.registry.get("beta").is_none();
        let beta_terminated = h
            .process_spawner
            .child(beta_pid)
            .map(|c| !matches!(c.state, FakeProcessState::Running))
            .unwrap_or(false);
        if beta_gone && beta_terminated {
            break;
        }
        if tokio::time::Instant::now() >= deadline {
            panic!(
                "reconciler did not drain beta in time (registry gone? {beta_gone}; \
                 beta state: {:?})",
                h.process_spawner.child(beta_pid).map(|c| c.state)
            );
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    assert!(
        h.state.registry.get("alpha").is_some(),
        "alpha must remain registered"
    );
    assert!(
        h.state.registry.get("beta").is_none(),
        "beta must be dropped from the registry after reload"
    );

    let beta_final = h.process_spawner.child(beta_pid).expect("beta snapshot");
    assert_eq!(
        beta_final.state,
        FakeProcessState::SigTerm,
        "beta's child must be SIGTERM'd by the drain"
    );

    let alpha_final = h.process_spawner.child(alpha_pid).expect("alpha snapshot");
    assert_eq!(
        alpha_final.state,
        FakeProcessState::Running,
        "alpha's child must survive — it wasn't in changed_services"
    );

    h.cleanup().await;
}
