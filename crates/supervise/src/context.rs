//! [`RunLoop`]'s basic constructor and small state-mirror accessors shared by
//! every phase: reading/writing the lifecycle state, the run/pid mirror, the
//! allocation-changed event, and the two teardown helpers (shutdown command,
//! drain-complete bookkeeping) every phase calls into.

use std::{sync::Arc, time::Duration};

use ananke_api::events::Event;
use ananke_config::validate::ServiceConfig;
use ananke_db::Database;
use parking_lot::Mutex as SyncMutex;
use tokio::sync::{mpsc, watch};
use tracing::{info, warn};

use crate::{
    RunLoop,
    handle::{EnsureSource, MirroredState, SupervisorCommand, SupervisorDeps, SupervisorInit},
    rolling::RollingBase,
    state::ServiceState,
};

/// Upper bound for a command service's optional `shutdown_command`.
/// Long enough for a `docker stop` (default 10s docker grace + slack)
/// but short enough that a hung shutdown doesn't block the drain
/// forever. A timeout escalates to SIGKILL of the shutdown child.
const SHUTDOWN_COMMAND_TIMEOUT: Duration = Duration::from_secs(30);

impl RunLoop {
    pub(crate) fn new(
        init: SupervisorInit,
        boot_svc: ServiceConfig,
        deps: SupervisorDeps,
        rx: mpsc::Receiver<SupervisorCommand>,
        mirror: Arc<SyncMutex<MirroredState>>,
    ) -> Self {
        // `MirroredState::default()` already seeds `Idle`; the explicit write
        // here is defensive in case the caller reused a handle's mirror.
        *mirror.lock() = MirroredState::default();
        let (cancel_tx, cancel_rx) = watch::channel(false);
        Self {
            init,
            deps,
            rx,
            mirror,
            cancel_tx,
            cancel_rx,
            start_bus_carry: None,
            pending_ensure_bus: None,
            pending_ensure_source: EnsureSource::default(),
            queued_watch: Vec::new(),
            queued_since: None,
            packed_for_spawn: None,
            oom_attempts: 0,
            rolling_base: RollingBase::default(),
            boot_svc,
            auto_restart_history: Vec::new(),
            restart_pending: false,
            deferred_ensure: None,
        }
    }

    /// Read the current lifecycle phase from the shared mirror.
    pub(crate) fn read_state(&self) -> ServiceState {
        self.mirror.lock().state.clone()
    }

    /// Resolve the latest `ServiceConfig` for this supervisor's service.
    ///
    /// Reads from the live [`ConfigManager`]'s arc-swapped effective config so
    /// that `PUT /api/config` edits (priority, context, override_tensor,
    /// idle_timeout, sampling, etc.) reach already-spawned supervisors the
    /// next time they hit the spawn or eviction path. Falls back to the
    /// boot-time snapshot if the service has been removed from the config —
    /// the reload reconciler will shut the supervisor down shortly, but we
    /// want any lookups in the interim to return a sensible value rather
    /// than panicking. The returned `ServiceConfig` is cloned so the
    /// arc-swap guard is released before any `.await`.
    pub(crate) fn current_svc(&self) -> ServiceConfig {
        let eff = self.deps.config.effective();
        eff.services
            .iter()
            .find(|s| s.name == self.init.identity.name)
            .cloned()
            .unwrap_or_else(|| self.boot_svc.clone())
    }

    pub(crate) fn set_state(&mut self, new_state: ServiceState) {
        let prior_state = {
            let mut m = self.mirror.lock();
            let prior = m.state.clone();
            m.state = new_state.clone();
            prior
        };
        self.deps.mirror_changed.notify();
        info!(
            service = %self.init.identity.name,
            from = %prior_state.name(),
            to = %new_state.name(),
            "state transition"
        );
        self.deps.events.publish(Event::StateChanged {
            service: self.init.identity.name.clone(),
            from: prior_state.kind(),
            to: new_state.kind(),
            disabled_reason: new_state.disabled_reason(),
            at_ms: ananke_tracking::now_unix_ms(),
        });
    }

    /// Stamp `run_id` + `pid` into the shared mirror. Called once per spawn,
    /// right after `insert_running_row` assigns the run_id, so every
    /// `SupervisorHandle::peek()` from that point sees the identifiers.
    pub(crate) fn set_running_ids(&mut self, run_id: i64, pid: i32) {
        {
            let mut m = self.mirror.lock();
            m.run_id = Some(run_id);
            m.pid = Some(pid);
        }
        self.deps.mirror_changed.notify();
    }

    /// Clear `run_id` + `pid` from the shared mirror. Called from every
    /// teardown path (drain complete, child exited, eviction, etc.) alongside
    /// the `delete_running_row` DB update, so peeks don't keep reporting a
    /// stale child. Prefer [`Self::end_run`] at call sites that need both.
    pub(crate) fn clear_running_ids(&mut self) {
        {
            let mut m = self.mirror.lock();
            m.run_id = None;
            m.pid = None;
        }
        self.deps.mirror_changed.notify();
    }

    /// Combined teardown: delete the DB `running_services` row and clear the
    /// mirror's `run_id` + `pid`. Every exit from the running/draining loops
    /// needs both, and keeping them in one helper means neither can be
    /// forgotten.
    pub(crate) async fn end_run(&mut self, run_id: i64) {
        delete_running_row(&self.deps.db, self.init.service_id, run_id).await;
        self.clear_running_ids();
    }

    /// Publish an `AllocationChanged` event reflecting the current state of
    /// this service's entry in the allocation table. Called after every
    /// reserve, drain, or eviction that touches our row.
    pub(crate) fn emit_allocation_changed(&self) {
        let reservations: std::collections::BTreeMap<String, u64> = self
            .deps
            .allocations
            .lock()
            .get(&self.init.identity.name)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|(slot, mb)| (slot_to_key(&slot), mb * 1024 * 1024))
            .collect();
        self.deps.events.publish(Event::AllocationChanged {
            service: self.init.identity.name.clone(),
            reservations,
            at_ms: ananke_tracking::now_unix_ms(),
        });
    }

    /// Run a command-template service's `shutdown_command`, if any, after
    /// the normal SIGTERM/SIGKILL pipeline completes. No-op for llama-cpp
    /// services and for command services without a shutdown command.
    ///
    /// Gives the shutdown child a bounded window to exit; logs (but does
    /// not propagate) failures — a drain is already terminal and the
    /// caller can't usefully recover from a shutdown-command error.
    pub(crate) async fn run_shutdown_command(&self) {
        let svc = self.current_svc();
        let Some(render_result) = crate::spawn::render_shutdown_argv(&svc, &self.init.allocation)
        else {
            return;
        };
        let cfg = match render_result {
            Ok(c) => c,
            Err(e) => {
                warn!(
                    service = %self.init.identity.name,
                    error = %e,
                    "shutdown_command placeholder substitution failed; skipping"
                );
                return;
            }
        };
        info!(service = %self.init.identity.name, binary = %cfg.binary, "drain: running shutdown_command");
        let mut child = match self.deps.system.process_spawner.spawn(&cfg).await {
            Ok(c) => c,
            Err(e) => {
                warn!(service = %self.init.identity.name, error = %e, "shutdown_command spawn failed");
                return;
            }
        };
        match tokio::time::timeout(SHUTDOWN_COMMAND_TIMEOUT, child.wait()).await {
            Ok(Ok(status)) => {
                if !status.success() {
                    warn!(
                        service = %self.init.identity.name,
                        ?status,
                        "shutdown_command exited non-zero"
                    );
                }
            }
            Ok(Err(e)) => {
                warn!(service = %self.init.identity.name, error = %e, "shutdown_command wait failed");
            }
            Err(_) => {
                warn!(
                    service = %self.init.identity.name,
                    timeout_s = SHUTDOWN_COMMAND_TIMEOUT.as_secs(),
                    "shutdown_command timed out; SIGKILLing it"
                );
                let _ = child.sigkill().await;
            }
        }
    }

    /// Remove this service's reservation, clear observation, and finalise the
    /// rolling correction. Used when the child exits or is drained back to
    /// Idle. Logs nothing itself; callers emit the tracing event that fits
    /// their context.
    pub(crate) fn record_drain_complete(&mut self) {
        self.record_rolling_observation();
        self.deps.observation.clear(&self.init.identity.name);
        self.deps
            .allocations
            .lock()
            .remove(&self.init.identity.name);
        self.emit_allocation_changed();
    }
}

pub use ananke_placement::slot_to_key;

/// Delete the `running_services` row for `(service_id, run_id)` if present.
async fn delete_running_row(db: &Database, service_id: i64, run_id: i64) {
    if let Err(e) = db.delete_running(service_id, run_id).await {
        warn!(error = %e, "running_services delete failed");
    }
}
