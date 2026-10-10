//! Shared application state passed to every Axum handler via `State(...)`.

use std::sync::Arc;

use ananke_allocator::AllocationTable;
use ananke_db::{Database, logs::BatcherHandle};
use ananke_events::EventBus;
use ananke_observation::{ObservationTable, SharedSnapshot};
use ananke_tracking::{
    activity::ActivityTable, inflight::InflightTable, progress::ProgressTable,
    rolling::RollingTable,
};
use parking_lot::Mutex;

use crate::{
    config::manager::ConfigManager,
    oneshot::{OneshotRegistry, PortPool},
    supervise::{estimate_cache::EstimateCacheHandle, registry::SupervisorRegistry},
};

#[derive(Clone)]
pub struct AppState {
    pub started_at: tokio::time::Instant,
    pub config: Arc<ConfigManager>,
    pub registry: SupervisorRegistry,
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
    pub system: ananke_system::SystemDeps,
    /// Memoised GGUF summary + estimator output, keyed by service
    /// name. Populated lazily by the management `ServiceDetail`
    /// handler so successive detail polls don't re-parse the GGUF.
    pub estimate_cache: EstimateCacheHandle,
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

    /// Assemble a `ProvisioningDeps` from this state and the daemon-wide
    /// shutdown channel. Every field already lives on `AppState`; this
    /// constructor keeps the call sites in `daemon::run` and the test
    /// harness from enumerating them by hand.
    pub fn provisioning_deps(
        &self,
        shutdown_rx: tokio::sync::watch::Receiver<bool>,
    ) -> crate::supervise::provision::ProvisioningDeps {
        let metrics_db = self.db.clone();
        crate::supervise::provision::ProvisioningDeps {
            db: self.db.clone(),
            activity: self.activity.clone(),
            inflight: self.inflight.clone(),
            observation: self.observation.clone(),
            allocations: self.allocations.clone(),
            supervisor_deps: self.supervisor_deps(),
            shutdown_rx,
            metrics_factory: std::sync::Arc::new(
                move |start, service_id, run_id, model, endpoint, is_streaming| {
                    Box::new(crate::api::openai::metrics::RequestMetricsRecorder {
                        recorder: crate::api::openai::metrics::MetricsRecorder::new(
                            start,
                            service_id,
                            run_id,
                            model,
                            endpoint,
                            is_streaming,
                        ),
                        db: metrics_db.clone(),
                    })
                },
            ),
        }
    }
}
