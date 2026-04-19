//! Reconcile live supervisors with the current `EffectiveConfig`.
//!
//! Subscribes to the daemon event bus. On each `ConfigReloaded`, every
//! service whose name appears in the change list but is no longer present
//! in `effective.services` has its supervisor drained and removed from the
//! registry. Without this reconciler the supervisor for a removed service
//! would live on for the lifetime of the daemon — its child would keep
//! consuming VRAM and its pledge would keep counting against placement —
//! because supervisors are spawned once at boot and never recycled.
//!
//! Scope limits: this reconciler handles *removal* only. Additions via
//! reload are still a no-op (no supervisor is spawned for a newly-added
//! service) because that path requires non-trivial rewiring of the
//! daemon's boot-time service/proxy setup; tracked separately.
//!
//! The per-service HTTP proxy task spawned in `daemon::run` is also *not*
//! torn down here. Requests to the removed service's public port will
//! return 502 after the drain; releasing the port cleanly requires a
//! per-proxy shutdown channel that doesn't exist yet. Tracked as a
//! follow-up.

use std::{collections::BTreeSet, sync::Arc};

use ananke_api::Event;
use smol_str::SmolStr;
use tokio::sync::{broadcast::error::RecvError, watch};
use tracing::{info, warn};

use crate::{
    config::manager::ConfigManager, daemon::events::EventBus,
    supervise::registry::ServiceRegistry,
};

/// Spawn the reconciler task. Returns a `JoinHandle` so the caller can
/// await it on daemon shutdown. The task exits when `shutdown_rx` fires
/// true or the event bus sender is dropped.
pub fn spawn(
    events: EventBus,
    config: Arc<ConfigManager>,
    registry: ServiceRegistry,
    mut shutdown_rx: watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
    let mut rx = events.subscribe();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        return;
                    }
                }
                ev = rx.recv() => match ev {
                    Ok(Event::ConfigReloaded { changed_services, .. }) => {
                        handle_reload(&config, &registry, &changed_services).await;
                    }
                    Ok(_) => {}
                    Err(RecvError::Lagged(n)) => {
                        warn!(dropped = n, "reload reconciler lagged on event bus");
                    }
                    Err(RecvError::Closed) => return,
                }
            }
        }
    })
}

async fn handle_reload(
    config: &ConfigManager,
    registry: &ServiceRegistry,
    changed: &[SmolStr],
) {
    // Snapshot the current service-name set and release the arc-swap guard
    // before awaiting anywhere — `Guard` is `!Send` in practice and we want
    // the live config to remain swappable while we drain.
    let live: BTreeSet<SmolStr> = {
        let current = config.effective();
        current.services.iter().map(|s| s.name.clone()).collect()
    };

    for name in changed {
        if live.contains(name) {
            continue;
        }
        let Some(handle) = registry.remove(name) else {
            continue;
        };
        info!(%name, "draining service removed by config reload");
        handle.shutdown().await;
    }
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use parking_lot::Mutex;
    use smol_str::SmolStr;
    use tokio::sync::watch;

    use super::*;
    use crate::{
        allocator::AllocationTable,
        config::{
            EffectiveConfig, Lifecycle,
            validate::test_fixtures::minimal_llama_cpp_service,
        },
        db::{Database, logs::spawn as spawn_batcher},
        devices::{Allocation, snapshotter},
        supervise::{SupervisorDeps, SupervisorInit, spawn_supervisor},
        tracking::{
            activity::ActivityTable, inflight::InflightTable, observation::ObservationTable,
            rolling::RollingTable,
        },
    };

    /// Spawn a supervisor pointing at the given `EffectiveConfig` and insert
    /// it into the registry. Returns the registry + event bus the reconciler
    /// needs.
    async fn fixture(
        services: Vec<crate::config::ServiceConfig>,
    ) -> (ServiceRegistry, EventBus, Arc<ConfigManager>) {
        let effective = EffectiveConfig {
            daemon: crate::config::DaemonSettings {
                management_listen: String::new(),
                openai_listen: String::new(),
                data_dir: PathBuf::new(),
                shutdown_timeout_ms: 5_000,
                allow_external_management: false,
            },
            services: services.clone(),
        };
        let events = EventBus::new();
        let config = ConfigManager::in_memory(effective.clone(), events.clone());
        let registry = ServiceRegistry::new();

        let db = Database::open_in_memory().await.unwrap();
        let batcher = spawn_batcher(db.clone());
        let activity = ActivityTable::new();
        let allocations = Arc::new(Mutex::new(AllocationTable::new()));
        let rolling = RollingTable::new();
        let observation = ObservationTable::new();
        let deps = SupervisorDeps {
            db: db.clone(),
            batcher,
            snapshot: snapshotter::new_shared(),
            allocations: allocations.clone(),
            rolling,
            observation,
            registry: registry.clone(),
            config: config.clone(),
            events: events.clone(),
            system: crate::system::SystemDeps::fake().0,
        };

        for svc in services {
            let service_id = db.upsert_service(&svc.name, 0).await.unwrap();
            let init = SupervisorInit {
                svc: svc.clone(),
                allocation: Allocation::from_override(&svc.placement_override),
                service_id,
                last_activity: activity.get_or_init(&svc.name),
                inflight: InflightTable::new().counter(&svc.name),
            };
            let handle = Arc::new(spawn_supervisor(init, deps.clone()));
            registry.insert(svc.name.clone(), handle);
        }

        (registry, events, config)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn removed_service_is_drained_and_dropped_from_registry() {
        let svc_a = {
            let mut s = minimal_llama_cpp_service("a");
            s.lifecycle = Lifecycle::OnDemand;
            s
        };
        let svc_b = {
            let mut s = minimal_llama_cpp_service("b");
            s.lifecycle = Lifecycle::OnDemand;
            s
        };
        let (registry, events, config) = fixture(vec![svc_a.clone(), svc_b.clone()]).await;

        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let join = spawn(events.clone(), config.clone(), registry.clone(), shutdown_rx);

        // Simulate the apply() path by publishing a ConfigReloaded whose
        // changed_services includes "b" — but first mutate the config so
        // the reconciler sees "b" as no longer live.
        let new_effective = EffectiveConfig {
            daemon: config.effective().daemon.clone(),
            services: vec![svc_a.clone()],
        };
        config.swap_effective_for_test(new_effective);

        events.publish(Event::ConfigReloaded {
            at_ms: 0,
            changed_services: vec![SmolStr::new("b")],
        });

        // Wait for the reconciler to observe + drain. shutdown() on a
        // minimal on-demand supervisor that never Ran completes fast, but
        // give the broadcast + actor loop a tick.
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(500);
        while registry.get("b").is_some() && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        assert!(registry.get("a").is_some(), "a should stay registered");
        assert!(registry.get("b").is_none(), "b should be drained and dropped");

        let _ = shutdown_tx.send(true);
        let _ = join.await;
    }
}
