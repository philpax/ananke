//! Shared lookup from service name to `SupervisorHandle`.
//!
//! The generic `ServiceRegistry<T>` lives in `ananke-placement`; this
//! module instantiates it with the daemon's `SupervisorHandle` and
//! re-exports the concrete alias so `crate::registry::…`
//! keeps resolving.

pub use ananke_placement::{DrainReason, KillHandle, ServiceRegistry, slot_to_key};

use crate::SupervisorHandle;

/// The daemon's registry: service name → supervisor handle.
pub type SupervisorRegistry = ServiceRegistry<SupervisorHandle>;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ananke_config::validate::{Lifecycle, test_fixtures::minimal_service};
    use ananke_db::{Database, logs::spawn as spawn_batcher};
    use ananke_devices::Allocation;
    use smol_str::SmolStr;

    use super::*;
    use crate::spawn_supervisor;

    #[tokio::test(flavor = "current_thread")]
    async fn insert_and_get() {
        let db = Database::open_in_memory().await.unwrap();
        let mut svc = minimal_service("demo");
        svc.lifecycle = Lifecycle::Persistent;
        let effective = ananke_config::EffectiveConfig {
            daemon: ananke_config::DaemonSettings::default(),
            services: vec![svc.clone()],
        };
        let events = ananke_events::EventBus::new();
        let config = ananke_config::manager::ConfigManager::in_memory(effective, events.clone());
        let init = crate::SupervisorInit {
            identity: crate::ServiceIdentity::from_service(&svc),
            allocation: Allocation::from_override(&svc.placement_override),
            service_id: 1,
            last_activity: Arc::new(parking_lot::Mutex::new(tokio::time::Instant::now())),
            inflight: ananke_proxy::InflightCounter::detached(),
        };
        let deps = crate::SupervisorDeps {
            db: db.clone(),
            batcher: spawn_batcher(db),
            snapshot: ananke_observation::new_shared(),
            allocations: Arc::new(parking_lot::Mutex::new(
                ananke_allocator::AllocationTable::new(),
            )),
            rolling: ananke_tracking::rolling::RollingTable::new(),
            observation: ananke_observation::ObservationTable::new(),
            registry: SupervisorRegistry::new(),
            config,
            events,
            system: ananke_system::SystemDeps::fake().0,
            inflight: ananke_tracking::inflight::InflightTable::new(),
            activity: ananke_tracking::activity::ActivityTable::new(),
            estimate_cache: crate::estimate_cache::EstimateCacheHandle::new(),
            mirror_changed: ananke_events::ChangeSignal::new(),
        };
        let handle = Arc::new(spawn_supervisor(init, svc.clone(), deps));

        let registry = SupervisorRegistry::new();
        registry.insert(SmolStr::new("demo"), handle.clone());
        assert!(registry.get("demo").is_some());
        assert!(registry.get("missing").is_none());
        assert_eq!(registry.names(), vec![SmolStr::new("demo")]);

        // Remove returns the evicted handle; subsequent lookups miss.
        let taken = registry.remove("demo").expect("registry had demo");
        assert!(registry.get("demo").is_none());
        assert_eq!(registry.names(), Vec::<SmolStr>::new());
        // Shutdown the evicted handle so the supervisor actor exits cleanly.
        taken.shutdown().await;

        handle.shutdown().await;
    }
}
