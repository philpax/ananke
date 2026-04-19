//! Shared application state passed to every Axum handler via `State(...)`.

use std::sync::Arc;

use parking_lot::Mutex;

use crate::{
    allocator::AllocationTable,
    config::EffectiveConfig,
    daemon::events::EventBus,
    db::{Database, logs::BatcherHandle},
    devices::snapshotter::SharedSnapshot,
    oneshot::{OneshotRegistry, PortPool},
    supervise::registry::ServiceRegistry,
    tracking::{
        activity::ActivityTable, inflight::InflightTable, observation::ObservationTable,
        rolling::RollingTable,
    },
};

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<EffectiveConfig>,
    pub registry: ServiceRegistry,
    pub allocations: Arc<Mutex<AllocationTable>>,
    pub snapshot: SharedSnapshot,
    pub activity: ActivityTable,
    pub rolling: RollingTable,
    pub observation: ObservationTable,
    pub db: Database,
    pub inflight: InflightTable,
    pub port_pool: Arc<Mutex<PortPool>>,
    pub oneshots: OneshotRegistry,
    pub batcher: BatcherHandle,
    pub events: EventBus,
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
            effective: self.config.clone(),
            events: self.events.clone(),
        }
    }
}
