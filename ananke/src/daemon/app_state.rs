//! Shared application state passed to every Axum handler via `State(...)`.

use std::sync::Arc;

use parking_lot::Mutex;

use crate::{
    allocator::AllocationTable,
    config::manager::ConfigManager,
    daemon::{estimate_cache::EstimateCache, events::EventBus},
    db::{Database, logs::BatcherHandle},
    devices::snapshotter::SharedSnapshot,
    oneshot::{OneshotRegistry, PortPool},
    supervise::registry::ServiceRegistry,
    tracking::{
        activity::ActivityTable, inflight::InflightTable, observation::ObservationTable,
        progress::ProgressTable, rolling::RollingTable,
    },
};

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<ConfigManager>,
    pub registry: ServiceRegistry,
    pub allocations: Arc<Mutex<AllocationTable>>,
    pub snapshot: SharedSnapshot,
    pub activity: ActivityTable,
    pub rolling: RollingTable,
    pub observation: ObservationTable,
    pub db: Database,
    pub inflight: InflightTable,
    /// Per-service timestamp of the last forwarded response frame, read by the
    /// time-to-first-token stall watchdog to tell a wedged child from a
    /// request queued behind healthy work.
    pub progress: ProgressTable,
    pub port_pool: Arc<Mutex<PortPool>>,
    pub oneshots: OneshotRegistry,
    pub batcher: BatcherHandle,
    pub events: EventBus,
    pub system: crate::system::SystemDeps,
    /// Memoised GGUF summary + estimator output, keyed by service
    /// name. Populated lazily by the management `ServiceDetail`
    /// handler so successive detail polls don't re-parse the GGUF.
    pub estimate_cache: EstimateCache,
}

impl AppState {
    /// Bundle the shared-daemon fields a `spawn_supervisor` call needs.
    /// The returned struct is trivially cloneable.
    pub fn supervisor_deps(&self) -> crate::supervise::SupervisorDeps {
        crate::supervise::SupervisorDeps {
            db: self.db.clone(),
            batcher: self.batcher.clone(),
            snapshot: self.snapshot.clone(),
            allocations: self.allocations.clone(),
            rolling: self.rolling.clone(),
            observation: self.observation.clone(),
            registry: self.registry.clone(),
            config: self.config.clone(),
            events: self.events.clone(),
            system: self.system.clone(),
            inflight: self.inflight.clone(),
            activity: self.activity.clone(),
            estimate_cache: self.estimate_cache.clone(),
        }
    }
}
