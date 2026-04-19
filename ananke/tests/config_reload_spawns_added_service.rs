//! Integration test: `PUT /api/config` that adds a `[[service]]` block
//! provisions the supervisor and (for persistent services) kicks it
//! running — matching the daemon's boot loop behaviour. Pre-fix, reload
//! was a no-op for additions: the ConfigReloaded event fired but no
//! supervisor was spawned.
//!
//! Uses `FakeSpawner` so the assertion is on the fake child's recorded
//! state rather than a real process lifecycle.

mod common;

use std::time::Duration;

use ananke::{config::EffectiveConfig, system::FakeProcessState};
use ananke_api::Event;
use common::{build_harness, minimal_llama_service};
use smol_str::SmolStr;

#[tokio::test(flavor = "current_thread")]
async fn reload_added_persistent_service_is_provisioned_and_runs() {
    // Boot with a single on-demand service. That way we can unambiguously
    // attribute a newly-spawned fake child to the reload-added service.
    let alpha = minimal_llama_service("alpha", 0);
    let h = build_harness(vec![alpha.clone()]).await;

    // Pre-condition: exactly one registered supervisor, zero spawned
    // children (alpha is on-demand and we haven't poked it).
    assert!(h.state.registry.get("alpha").is_some());
    assert!(h.state.registry.get("beta").is_none());
    assert_eq!(
        h.process_spawner.children().len(),
        0,
        "expected zero spawned fake children before reload"
    );

    // Build the "after reload" config: add a Persistent "beta" that
    // points at the same echo-server port as alpha. The build_harness
    // rewrote alpha's private_port to the echo port, so we can reuse it
    // for beta's health probe.
    let mut beta = minimal_llama_service("beta", 0);
    beta.lifecycle = ananke::config::Lifecycle::Persistent;
    beta.private_port = alpha.private_port;
    // Distinct public port so the proxy bind doesn't collide with
    // alpha's listener (we're in an isolated harness — the OS just
    // needs unique ports).
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

    // Wait for the reconciler to provision beta. Persistent lifecycle
    // triggers an implicit ensure(), so the fake spawner should record
    // a child for it.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let provisioned = h.state.registry.get("beta").is_some();
        let spawned = h
            .process_spawner
            .children()
            .iter()
            .any(|c| matches!(c.state, FakeProcessState::Running));
        if provisioned && spawned {
            break;
        }
        if tokio::time::Instant::now() >= deadline {
            panic!(
                "reconciler never provisioned beta (registered={provisioned}, children={:?})",
                h.process_spawner.children()
            );
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    // beta's supervisor is in the registry and its fake child is Running.
    let beta_handle = h.state.registry.get("beta").expect("beta registered");
    assert_eq!(beta_handle.name.as_str(), "beta");

    h.cleanup().await;
}
