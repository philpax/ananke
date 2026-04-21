//! Integration test: `PUT /api/config` that changes a service's
//! `context` is reflected in the argv of the next child the supervisor
//! spawns. Pre-fix, `SupervisorInit.svc` was frozen at boot, so reload
//! edits silently bounced off already-spawned supervisors — the daemon
//! published `config_reloaded` but the next start still used the old
//! context. This test walks the full path: boot with context=4096,
//! reload to context=16384, stop+start, assert the spawned child's
//! argv contains the new `-c 16384`.
//!
//! Uses `FakeSpawner` so the assertion operates on recorded argv rather
//! than any real `llama-server` process.
#![cfg(feature = "test-fakes")]

mod common;

use std::{sync::Arc, time::Duration};

use ananke::{
    config::{EffectiveConfig, ServiceConfig, TemplateConfig},
    supervise::SupervisorHandle,
    system::FakeProcessState,
};
use ananke_api::Event;
use common::{build_harness, minimal_llama_service};
use smol_str::SmolStr;

fn with_context(mut svc: ServiceConfig, ctx: u32) -> ServiceConfig {
    let TemplateConfig::LlamaCpp(lc) = &mut svc.template_config else {
        unreachable!("test fixture is llama-cpp");
    };
    lc.context = Some(ctx);
    svc
}

async fn wait_running(handle: &SupervisorHandle, timeout_ms: u64) {
    let deadline = tokio::time::Instant::now() + Duration::from_millis(timeout_ms);
    loop {
        // Supervisor State enum's `name()` returns "running" once health
        // probes have passed. BeginDrain is a no-op during Starting, so the
        // test needs to wait for the full transition before issuing the drain.
        if format!("{:?}", handle.peek_state())
            .to_lowercase()
            .contains("running")
        {
            return;
        }
        if tokio::time::Instant::now() >= deadline {
            panic!("supervisor never reached Running");
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

#[tokio::test(flavor = "current_thread")]
async fn reload_context_change_takes_effect_on_next_spawn() {
    let initial = with_context(minimal_llama_service("ctx", 0), 4096);
    let h = build_harness(vec![initial.clone()]).await;

    let handle = Arc::clone(h.supervisors.first().unwrap());

    // First start: argv should carry the boot-time context (4096).
    let _ = handle.ensure().await;
    wait_running(&handle, 5_000).await;
    let first_argv = h
        .process_spawner
        .children()
        .into_iter()
        .next()
        .expect("first spawn");
    assert!(
        first_argv.args.iter().any(|a| a == "4096"),
        "first spawn should render context=4096, got args: {:?}",
        first_argv.args
    );

    // Drain back to Idle so the next Ensure forces a fresh spawn. No
    // sleeps or pid probes — begin_drain returns once the supervisor
    // transitions to Idle.
    handle
        .begin_drain(ananke::supervise::drain::DrainReason::UserKilled)
        .await;

    // Reload with context=16384. Publish ConfigReloaded so any subscribers
    // see the edit; the supervisor itself reads live from ConfigManager
    // on the next Ensure.
    let old = h.state.config.effective();
    let new = EffectiveConfig {
        daemon: old.daemon.clone(),
        services: vec![with_context(initial.clone(), 16384)],
    };
    drop(old);
    h.state.config.swap_effective_for_test(new);
    h.state.events.publish(Event::ConfigReloaded {
        at_ms: 0,
        changed_services: vec![SmolStr::new("ctx")],
    });

    // Second start: the supervisor must render argv from the live config,
    // so context=16384.
    let _ = handle.ensure().await;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let kids = h.process_spawner.children();
        if kids.len() >= 2 {
            let second = kids.last().unwrap().clone();
            assert!(
                second.args.iter().any(|a| a == "16384"),
                "second spawn must render the reloaded context=16384; args: {:?}",
                second.args
            );
            assert_eq!(
                second.state,
                FakeProcessState::Running,
                "second child should still be running"
            );
            break;
        }
        if tokio::time::Instant::now() >= deadline {
            panic!(
                "second spawn never recorded; fake children so far: {:?}",
                h.process_spawner.children()
            );
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    h.cleanup().await;
}
