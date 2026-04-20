//! Shared lookup from service name to `SupervisorHandle`.
//!
//! Read-heavy; wrapped in an `Arc<RwLock<...>>` so both HTTP routers and
//! the daemon lifecycle code can share visibility without cloning the
//! whole map per request.

use std::{collections::BTreeMap, sync::Arc};

use parking_lot::RwLock;
use smol_str::SmolStr;

use crate::supervise::SupervisorHandle;

#[derive(Clone, Default)]
pub struct ServiceRegistry {
    inner: Arc<RwLock<BTreeMap<SmolStr, Arc<SupervisorHandle>>>>,
}

impl ServiceRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&self, name: SmolStr, handle: Arc<SupervisorHandle>) {
        self.inner.write().insert(name, handle);
    }

    pub fn get(&self, name: &str) -> Option<Arc<SupervisorHandle>> {
        self.inner.read().get(name).cloned()
    }

    /// Evict `name` and return its handle if present. The caller typically
    /// awaits `shutdown()` on the returned handle to drain the underlying
    /// child before the `Arc` is dropped.
    pub fn remove(&self, name: &str) -> Option<Arc<SupervisorHandle>> {
        self.inner.write().remove(name)
    }

    pub fn names(&self) -> Vec<SmolStr> {
        self.inner.read().keys().cloned().collect()
    }

    pub fn all(&self) -> Vec<(SmolStr, Arc<SupervisorHandle>)> {
        self.inner
            .read()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use smol_str::SmolStr;

    use super::*;
    use crate::{
        config::validate::{Lifecycle, test_fixtures::minimal_service},
        db::{Database, logs::spawn as spawn_batcher},
        devices::Allocation,
        supervise::spawn_supervisor,
    };

    #[tokio::test(flavor = "current_thread")]
    async fn insert_and_get() {
        let db = Database::open_in_memory().await.unwrap();
        let mut svc = minimal_service("demo");
        svc.lifecycle = Lifecycle::Persistent;
        let effective = crate::config::EffectiveConfig {
            daemon: crate::config::DaemonSettings {
                management_listen: String::new(),
                openai_listen: String::new(),
                data_dir: std::path::PathBuf::new(),
                shutdown_timeout_ms: 5000,
                allow_external_management: false,
            },
            services: vec![svc.clone()],
        };
        let events = crate::daemon::events::EventBus::new();
        let config = crate::config::manager::ConfigManager::in_memory(effective, events.clone());
        let init = crate::supervise::SupervisorInit {
            identity: crate::supervise::ServiceIdentity::from_service(&svc),
            allocation: Allocation::from_override(&svc.placement_override),
            service_id: 1,
            last_activity: Arc::new(parking_lot::Mutex::new(tokio::time::Instant::now())),
            inflight: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        };
        let deps = crate::supervise::SupervisorDeps {
            db: db.clone(),
            batcher: spawn_batcher(db),
            snapshot: crate::devices::snapshotter::new_shared(),
            allocations: Arc::new(parking_lot::Mutex::new(
                crate::allocator::AllocationTable::new(),
            )),
            rolling: crate::tracking::rolling::RollingTable::new(),
            observation: crate::tracking::observation::ObservationTable::new(),
            registry: ServiceRegistry::new(),
            config,
            events,
            system: crate::system::SystemDeps::fake().0,
        };
        let handle = Arc::new(spawn_supervisor(init, svc.clone(), deps));

        let registry = ServiceRegistry::new();
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
