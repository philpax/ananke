//! Shared application state passed to every Axum handler via `State(...)`.

use std::sync::Arc;

use parking_lot::Mutex;

use crate::activity::ActivityTable;
use crate::allocator::AllocationTable;
use crate::config::EffectiveConfig;
use crate::db::Database;
use crate::db::logs::BatcherHandle;
use crate::inflight::InflightTable;
use crate::observation::ObservationTable;
use crate::oneshot::{OneshotRegistry, PortPool};
use crate::rolling::RollingTable;
use crate::service_registry::ServiceRegistry;
use crate::snapshotter::SharedSnapshot;

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
}

impl AppState {
    /// Spawn a oneshot service from a validated request.
    ///
    /// Full implementation wired in Task 13; returns an error until then.
    pub async fn spawn_oneshot(
        &self,
        _id: crate::oneshot::OneshotId,
        _req: crate::oneshot::handlers::OneshotRequest,
        _port: u16,
        _ttl_ms: u64,
    ) -> Result<(), String> {
        Err("not wired yet".into())
    }
}
