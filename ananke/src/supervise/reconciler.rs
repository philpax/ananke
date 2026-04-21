//! Reconcile live supervisors with the current `EffectiveConfig`.
//!
//! Subscribes to the daemon event bus. On each `ConfigReloaded`, the
//! reconciler cross-references the change list with the new effective
//! config:
//!
//! - Services **removed** from the config: the reconciler drains the
//!   supervisor (via `shutdown()`, which SIGTERMs the child and cleans up
//!   the allocation table) and evicts the handle from the registry. The
//!   per-service HTTP proxy task is *not* torn down here — released
//!   cleanly it needs a per-proxy shutdown channel that doesn't exist
//!   yet (tracked as a follow-up). Requests to the removed service's
//!   public port will return 502 until the daemon restarts.
//! - Services **added** to the config: the reconciler calls
//!   [`provision::provision_service`] to spawn a supervisor, bind the
//!   proxy listener, spawn the balloon resolver if the service is
//!   dynamic, and (for persistent services) fire the implicit `ensure()`.
//!   This mirrors the daemon's boot loop exactly, so added services
//!   behave as if they had been present at boot.

use std::{collections::BTreeSet, sync::Arc};

use ananke_api::Event;
use smol_str::SmolStr;
use tokio::sync::{broadcast::error::RecvError, watch};
use tracing::{error, info, warn};

use crate::{
    config::manager::ConfigManager,
    daemon::events::EventBus,
    supervise::{provision::ProvisioningDeps, registry::ServiceRegistry},
};

/// Spawn the reconciler task. Returns a `JoinHandle` so the caller can
/// await it on daemon shutdown. The task exits when `shutdown_rx` fires
/// true or the event bus sender is dropped.
///
/// `provisioning` is optional so lib-level tests (which don't need the
/// add-service path) can pass `None`. Production always supplies it so
/// reload-add actually provisions.
pub fn spawn(
    events: EventBus,
    config: Arc<ConfigManager>,
    registry: ServiceRegistry,
    provisioning: Option<ProvisioningDeps>,
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
                        handle_reload(&config, &registry, provisioning.as_ref(), &changed_services).await;
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
    provisioning: Option<&ProvisioningDeps>,
    changed: &[SmolStr],
) {
    // Snapshot the live services as `(name, full ServiceConfig)` before
    // awaiting, so the arc-swap guard releases promptly. We need the full
    // ServiceConfig to provision adds; cloning is cheap at the ~tens-of-
    // services scale daemons run.
    let live: BTreeSet<SmolStr>;
    let live_configs: std::collections::BTreeMap<SmolStr, crate::config::ServiceConfig>;
    {
        let current = config.effective();
        live = current.services.iter().map(|s| s.name.clone()).collect();
        live_configs = current
            .services
            .iter()
            .map(|s| (s.name.clone(), s.clone()))
            .collect();
    }

    for name in changed {
        let in_live = live.contains(name);
        let in_registry = registry.get(name).is_some();

        match (in_live, in_registry) {
            // Removed: live has it? No. Registry has it? Yes. Drain it.
            (false, true) => {
                if let Some(handle) = registry.remove(name) {
                    info!(%name, "draining service removed by config reload");
                    handle.shutdown().await;
                }
            }
            // Added: live has it, registry doesn't — provision.
            (true, false) => {
                let Some(prov) = provisioning else {
                    warn!(%name, "reload added a service but reconciler has no provisioning deps; supervisor not spawned");
                    continue;
                };
                let Some(svc) = live_configs.get(name).cloned() else {
                    continue;
                };
                info!(%name, "provisioning service added by config reload");
                if let Err(e) = crate::supervise::provision::provision_service(svc, prov).await {
                    error!(%name, error = %e, "provisioning failed for reload-added service");
                }
            }
            // In both (edit to existing service) or in neither (stale
            // event): nothing to do here — edit-in-place is handled by
            // the supervisor's `current_svc()` live read at Ensure time.
            _ => {}
        }
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
        config::{EffectiveConfig, Lifecycle, validate::test_fixtures::minimal_llama_cpp_service},
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
                allow_external_services: false,
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
            inflight: crate::tracking::inflight::InflightTable::new(),
            activity: activity.clone(),
        };

        for svc in services {
            let service_id = db.upsert_service(&svc.name, 0).await.unwrap();
            let init = SupervisorInit {
                identity: crate::supervise::ServiceIdentity::from_service(&svc),
                allocation: Allocation::from_override(&svc.placement_override),
                service_id,
                last_activity: activity.get_or_init(&svc.name),
                inflight: InflightTable::new().counter(&svc.name),
            };
            let handle = Arc::new(spawn_supervisor(init, svc.clone(), deps.clone()));
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
        let join = spawn(
            events.clone(),
            config.clone(),
            registry.clone(),
            None,
            shutdown_rx,
        );

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
        assert!(
            registry.get("b").is_none(),
            "b should be drained and dropped"
        );

        let _ = shutdown_tx.send(true);
        let _ = join.await;
    }
}
