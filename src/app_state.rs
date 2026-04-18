//! Shared application state passed to every Axum handler via `State(...)`.

use std::sync::Arc;

use parking_lot::Mutex;

use crate::activity::ActivityTable;
use crate::allocator::AllocationTable;
use crate::config::EffectiveConfig;
use crate::db::Database;
use crate::service_registry::ServiceRegistry;
use crate::snapshotter::SharedSnapshot;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<EffectiveConfig>,
    pub registry: ServiceRegistry,
    pub allocations: Arc<Mutex<AllocationTable>>,
    pub snapshot: SharedSnapshot,
    pub activity: ActivityTable,
    pub db: Database,
}
