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
    use std::{
        collections::BTreeMap,
        path::PathBuf,
        sync::{Arc, atomic::AtomicU64},
    };

    use smol_str::SmolStr;
    use tempfile::tempdir;

    use super::*;
    use crate::{
        config::{
            parse::RawService,
            validate::{
                AllocationMode, DeviceSlot, HealthSettings, Lifecycle, PlacementPolicy,
                ServiceConfig, Template,
            },
        },
        db::{Database, logs::spawn as spawn_batcher},
        devices::Allocation,
        supervise::spawn_supervisor,
    };

    fn minimal_svc(name: &str) -> ServiceConfig {
        let mut override_map = BTreeMap::new();
        override_map.insert(DeviceSlot::Cpu, 100);
        ServiceConfig {
            name: SmolStr::new(name),
            template: Template::LlamaCpp,
            port: 0,
            private_port: 0,
            lifecycle: Lifecycle::Persistent,
            priority: 50,
            health: HealthSettings {
                http_path: "/".into(),
                timeout_ms: 1000,
                probe_interval_ms: 500,
            },
            placement_override: override_map,
            placement_policy: PlacementPolicy::CpuOnly,
            filters: Default::default(),
            idle_timeout_ms: 600_000,
            warming_grace_ms: 1000,
            drain_timeout_ms: 1000,
            extended_stream_drain_ms: 1000,
            max_request_duration_ms: 1000,
            allocation_mode: AllocationMode::None,
            command: None,
            workdir: None,
            openai_compat: true,
            raw: RawService {
                name: Some(SmolStr::new(name)),
                template: Some(SmolStr::new("llama-cpp")),
                model: Some(PathBuf::from("/fake/path")),
                port: Some(0),
                ..Default::default()
            },
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn insert_and_get() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("a.sqlite")).await.unwrap();
        let batcher = spawn_batcher(db.clone());
        let svc = minimal_svc("demo");
        let alloc = Allocation::from_override(&svc.placement_override);
        let last_activity = Arc::new(AtomicU64::new(0));
        let snapshot = crate::devices::snapshotter::new_shared();
        let allocations = Arc::new(parking_lot::Mutex::new(
            crate::allocator::AllocationTable::new(),
        ));
        let inflight = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let registry = ServiceRegistry::new();
        let effective = Arc::new(crate::config::EffectiveConfig {
            daemon: crate::config::DaemonSettings {
                management_listen: String::new(),
                openai_listen: String::new(),
                data_dir: std::path::PathBuf::new(),
                shutdown_timeout_ms: 5000,
            },
            services: vec![svc.clone()],
        });
        let handle = Arc::new(spawn_supervisor(
            svc.clone(),
            alloc,
            db.clone(),
            batcher.clone(),
            1,
            last_activity,
            snapshot,
            allocations,
            crate::tracking::rolling::RollingTable::new(),
            crate::tracking::observation::ObservationTable::new(),
            inflight,
            registry,
            effective,
        ));

        let registry = ServiceRegistry::new();
        registry.insert(SmolStr::new("demo"), handle.clone());
        assert!(registry.get("demo").is_some());
        assert!(registry.get("missing").is_none());
        assert_eq!(registry.names(), vec![SmolStr::new("demo")]);

        handle.shutdown().await;
    }
}
