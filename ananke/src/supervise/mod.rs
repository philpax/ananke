//! Service supervision: per-service tokio tasks, child lifetimes, health loops.
//!
//! Linux-coupled via `os::unix::process::ExitStatusExt` (signal() on ExitStatus)
//! and the submodules it delegates to (`drain`, `orphans`, `spawn`).

pub mod drain;
pub mod health;
pub mod logs;
pub mod orphans;
pub mod provision;
pub mod reconciler;
pub mod registry;
pub mod spawn;
pub mod state;

use std::{
    os::unix::process::ExitStatusExt,
    sync::{Arc, atomic::AtomicU64},
    time::{Duration, Instant},
};

use ananke_api::Event;
pub use orphans::{OrphanDisposition, reconcile};
use parking_lot::Mutex as SyncMutex;
pub use spawn::{SpawnConfig, render_argv};
use tokio::{
    sync::{broadcast, mpsc, watch},
    task::JoinHandle,
};
use tracing::{error, info, warn};

use crate::{
    allocator::placement::Packed,
    config::validate::{DEFAULT_SERVICE_PRIORITY, ServiceConfig},
    daemon::events::EventBus,
    db::{Database, logs::BatcherHandle},
    devices::Allocation,
    supervise::{
        drain::{DrainConfig, drain_pipeline, fast_kill},
        health::{HealthConfig, HealthOutcome, wait_healthy},
        logs::{spawn_pump_stderr, spawn_pump_stdout},
        registry::ServiceRegistry,
        state::{DisableReason, Event as StateEvent, ServiceState, transition},
    },
    system::ManagedChild,
    tracking::{observation::ObservationTable, rolling::RollingTable},
};

#[derive(Debug)]
pub enum SupervisorCommand {
    Shutdown {
        ack: tokio::sync::oneshot::Sender<()>,
    },
    /// Request state snapshot for tests / management surface.
    Snapshot {
        ack: tokio::sync::oneshot::Sender<SupervisorSnapshot>,
    },
    /// Ensure the service is started (or starting). Returns a broadcast
    /// receiver the caller can await for the start outcome. If the
    /// start queue is full, returns `EnsureResponse::QueueFull` via the
    /// single-shot `ack`.
    Ensure {
        ack: tokio::sync::oneshot::Sender<EnsureResponse>,
    },
    /// Record that a request was served; resets the idle timer.
    ActivityPing,
    /// Enter the full drain pipeline for eviction / TTL / user-kill.
    BeginDrain {
        reason: crate::supervise::drain::DrainReason,
        ack: tokio::sync::oneshot::Sender<()>,
    },
    /// Balloon-resolver fast-path: 5 s SIGTERM grace then SIGKILL.
    FastKill {
        reason: crate::supervise::drain::DrainReason,
        ack: tokio::sync::oneshot::Sender<()>,
    },
    /// Re-enable a disabled service, returning it to Idle.
    Enable {
        ack: tokio::sync::oneshot::Sender<EnableResult>,
    },
    /// Administratively disable a running or idle service.
    Disable {
        ack: tokio::sync::oneshot::Sender<DisableResult>,
    },
}

/// Result of a `SupervisorCommand::Enable`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnableResult {
    /// Was `Disabled`; now `Idle`.
    Enabled,
    /// Already in a non-disabled state (Idle, Running, etc.).
    NotDisabled,
}

/// Result of a `SupervisorCommand::Disable`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisableResult {
    /// Transitioned to `Disabled`.
    Disabled,
    /// Was already `Disabled`; no change.
    AlreadyDisabled,
}

#[derive(Debug)]
pub enum EnsureResponse {
    /// Service is already running; proceed directly.
    AlreadyRunning,
    /// Service is idle/starting/warming; subscribe and wait.
    Waiting {
        rx: tokio::sync::broadcast::Receiver<StartOutcome>,
    },
    /// Start queue is full; reject with 503.
    QueueFull,
    /// Service cannot be started right now; the variant carries the
    /// semantic reason so callers don't have to sniff the message.
    Unavailable(EnsureFailure),
}

#[derive(Debug, Clone)]
pub enum StartOutcome {
    Ok,
    Err(StartFailure),
}

#[derive(Debug, Clone)]
pub struct StartFailure {
    pub kind: StartFailureKind,
    pub message: String,
}

#[derive(Debug, Clone)]
pub enum StartFailureKind {
    NoFit,
    LaunchFailed,
    HealthTimeout,
    Disabled,
    Oom,
}

#[derive(Debug, Clone)]
pub struct SupervisorSnapshot {
    pub name: smol_str::SmolStr,
    pub state: ServiceState,
    pub run_id: Option<i64>,
    pub pid: Option<i32>,
}

pub struct SupervisorHandle {
    pub name: smol_str::SmolStr,
    tx: mpsc::Sender<SupervisorCommand>,
    join: tokio::sync::Mutex<Option<JoinHandle<()>>>,
}

impl SupervisorHandle {
    pub async fn shutdown(&self) {
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        let _ = self
            .tx
            .send(SupervisorCommand::Shutdown { ack: ack_tx })
            .await;
        let _ = ack_rx.await;
        if let Some(handle) = self.join.lock().await.take() {
            let _ = handle.await;
        }
    }

    pub async fn snapshot(&self) -> Option<SupervisorSnapshot> {
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        if self
            .tx
            .send(SupervisorCommand::Snapshot { ack: ack_tx })
            .await
            .is_err()
        {
            return None;
        }
        ack_rx.await.ok()
    }

    pub async fn ensure(&self) -> Option<EnsureResponse> {
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(SupervisorCommand::Ensure { ack: ack_tx })
            .await
            .ok()?;
        ack_rx.await.ok()
    }

    /// Fetch the current `run_id` from the supervisor's snapshot. Returns
    /// `None` if the supervisor is unreachable or not in a running state.
    pub async fn run_id(&self) -> Option<i64> {
        self.snapshot().await?.run_id
    }

    pub fn ping(&self) {
        let _ = self.tx.try_send(SupervisorCommand::ActivityPing);
    }

    pub async fn begin_drain(&self, reason: crate::supervise::drain::DrainReason) {
        let (ack, rx) = tokio::sync::oneshot::channel();
        let _ = self
            .tx
            .send(SupervisorCommand::BeginDrain { reason, ack })
            .await;
        let _ = rx.await;
    }

    pub async fn fast_kill(&self, reason: crate::supervise::drain::DrainReason) {
        let (ack, rx) = tokio::sync::oneshot::channel();
        let _ = self
            .tx
            .send(SupervisorCommand::FastKill { reason, ack })
            .await;
        let _ = rx.await;
    }

    pub async fn enable(&self) -> EnableResult {
        let (ack, rx) = tokio::sync::oneshot::channel();
        let _ = self.tx.send(SupervisorCommand::Enable { ack }).await;
        rx.await.unwrap_or(EnableResult::NotDisabled)
    }

    pub async fn disable(&self) -> DisableResult {
        let (ack, rx) = tokio::sync::oneshot::channel();
        let _ = self.tx.send(SupervisorCommand::Disable { ack }).await;
        rx.await.unwrap_or(DisableResult::AlreadyDisabled)
    }
}

/// Daemon-wide shared state every supervisor borrows. Cloning it is cheap
/// (every field is `Arc`-backed). Outside-world capabilities live inside
/// `system` ([`crate::system::SystemDeps`]); everything else is
/// daemon-internal state.
///
/// `config` is the live [`ConfigManager`](crate::config::manager::ConfigManager),
/// not a frozen `Arc<EffectiveConfig>`. Every supervisor reads its current
/// `ServiceConfig` through it so that `PUT /api/config` edits (priority,
/// context, override_tensor, idle_timeout, etc.) take effect on the next
/// spawn or eviction check without requiring a daemon restart. Identity
/// fields the supervisor can't live-update — name, port, private_port —
/// stay on `SupervisorInit` which is boot-time.
#[derive(Clone)]
pub struct SupervisorDeps {
    pub db: Database,
    pub batcher: BatcherHandle,
    pub snapshot: crate::devices::snapshotter::SharedSnapshot,
    pub allocations: Arc<parking_lot::Mutex<crate::allocator::AllocationTable>>,
    pub rolling: RollingTable,
    pub observation: ObservationTable,
    pub registry: ServiceRegistry,
    pub config: Arc<crate::config::manager::ConfigManager>,
    pub events: EventBus,
    pub system: crate::system::SystemDeps,
}

/// Identity fields that don't change across a reload. Everything else a
/// supervisor needs about its service (priority, context, health paths,
/// etc.) is fetched live from [`SupervisorDeps::config`] so edits pushed
/// via `PUT /api/config` reach already-spawned supervisors.
#[derive(Debug, Clone)]
pub struct ServiceIdentity {
    /// Service name — the primary key for registry lookups, events, the
    /// allocation table, etc.
    pub name: smol_str::SmolStr,
    /// Upstream port bound at daemon start; used to build the health probe
    /// URL. The reconciler refuses to change `private_port` across a
    /// reload (a port change respawns the supervisor), so freezing it
    /// here is safe.
    pub private_port: u16,
}

impl ServiceIdentity {
    /// Derive an identity from a boot-time [`ServiceConfig`].
    pub fn from_service(svc: &ServiceConfig) -> Self {
        Self {
            name: svc.name.clone(),
            private_port: svc.private_port,
        }
    }
}

/// Per-service initialisation for a single supervisor task.
pub struct SupervisorInit {
    pub identity: ServiceIdentity,
    pub allocation: Allocation,
    pub service_id: i64,
    pub last_activity: crate::tracking::activity::ActivityStamp,
    pub inflight: Arc<AtomicU64>,
}

/// Spawn a supervisor task. `boot_svc` seeds the `current_svc()` fallback
/// used when the service is briefly removed from live config during a
/// reload — the reconciler will shut the supervisor down shortly after,
/// but any interim lookup still returns a sensible value.
pub fn spawn_supervisor(
    init: SupervisorInit,
    boot_svc: ServiceConfig,
    deps: SupervisorDeps,
) -> SupervisorHandle {
    let (tx, rx) = mpsc::channel(SUPERVISOR_COMMAND_MAILBOX);
    let name = init.identity.name.clone();
    let join = tokio::spawn(run(init, boot_svc, deps, rx));
    SupervisorHandle {
        name,
        tx,
        join: tokio::sync::Mutex::new(Some(join)),
    }
}

/// Mailbox depth for per-supervisor command channels.
const SUPERVISOR_COMMAND_MAILBOX: usize = 32;

/// Outcome of waiting for a supervisor to reach Running.
///
/// Both the per-service transparent proxy and the OpenAI-compat router need
/// the same "ensure + wait-on-bus + map to error kind" dance before
/// forwarding. [`await_ensure`] packages it once; each caller renders its
/// own error response from the [`EnsureFailure`] kinds.
pub enum EnsureOutcome {
    /// Service is Running (or was already).
    ///
    /// `was_already_running` is `true` when the supervisor was already in the
    /// Running state at the time of the call; `false` when the call triggered
    /// an Idle → Starting transition and the caller waited for the child to
    /// become healthy.
    Ready { was_already_running: bool },
    /// Service cannot serve the request.
    Failed(EnsureFailure),
}

/// Why `compute_reservation_map` couldn't produce a reservation for the
/// next spawn. Each variant carries the structured inner error it wraps
/// so callers can inspect the specific cause without parsing a message.
#[derive(Debug, Clone)]
enum ReservationFailure {
    /// Service config is missing something required to even try estimation
    /// (e.g. no model path on a llama-cpp service). Not recoverable by
    /// eviction — the service needs a config fix first.
    Misconfigured(MisconfiguredKind),
    /// The estimator refused or failed on this GGUF. Carries the concrete
    /// [`estimator::EstimatorError`] so callers can tell an unknown-arch
    /// from a GGUF-read failure.
    EstimatorError(crate::estimator::EstimatorError),
    /// The packer couldn't lay the model down given current reservations.
    /// Carries the structured [`placement::PackError`] — the supervisor
    /// branches on this specifically to retry with eviction.
    PackFailed(crate::allocator::placement::PackError),
}

/// Concrete ways a service's config can prevent the estimator from even
/// running. Expands over time as new check surfaces are added.
#[derive(Debug, Clone, PartialEq, Eq)]
enum MisconfiguredKind {
    /// Llama-cpp service without a `model` path. Should have been caught
    /// at config validation but the supervisor double-checks defensively.
    NoModelPath,
}

impl std::fmt::Display for MisconfiguredKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoModelPath => f.write_str("no model path configured"),
        }
    }
}

impl ReservationFailure {
    /// Flatten to the string the proxy / OpenAI layer shows operators.
    /// The shape matches the pre-enum behaviour so log scrapes keep
    /// working; inspectors should match on the enum instead.
    fn message(self) -> String {
        match self {
            Self::Misconfigured(k) => k.to_string(),
            Self::EstimatorError(e) => format!("estimator: {e}"),
            Self::PackFailed(p) => format!("placement: {p}"),
        }
    }
}

/// Semantic bucket of an [`EnsureOutcome::Failed`]. Callers map these onto
/// their own error-response surface (OpenAI error JSON, proxy error body).
#[derive(Debug, Clone)]
pub enum EnsureFailure {
    /// The start fit-check rejected or the child got OOM-killed.
    InsufficientVram(String),
    /// The service is Disabled (config or health) or otherwise unavailable.
    ServiceDisabled(String),
    /// The supervisor's start queue is saturated.
    StartQueueFull,
    /// The start itself failed (launch error, health timeout, bus closed,
    /// overall timeout, or the supervisor task is gone).
    StartFailed(String),
}

/// Ensure the service is Running, waiting up to `max_request_duration` for
/// an in-flight start to finish. Used by every HTTP path that forwards to
/// a supervised child.
pub async fn await_ensure(
    handle: &SupervisorHandle,
    max_request_duration: Duration,
) -> EnsureOutcome {
    let rx = match handle.ensure().await {
        Some(EnsureResponse::AlreadyRunning) => {
            return EnsureOutcome::Ready {
                was_already_running: true,
            };
        }
        Some(EnsureResponse::Waiting { rx }) => rx,
        Some(EnsureResponse::QueueFull) => {
            return EnsureOutcome::Failed(EnsureFailure::StartQueueFull);
        }
        Some(EnsureResponse::Unavailable(failure)) => {
            return EnsureOutcome::Failed(failure);
        }
        None => {
            return EnsureOutcome::Failed(EnsureFailure::StartFailed(
                "supervisor unreachable".into(),
            ));
        }
    };
    await_start_bus(rx, max_request_duration).await
}

async fn await_start_bus(
    mut rx: tokio::sync::broadcast::Receiver<StartOutcome>,
    max_request_duration: Duration,
) -> EnsureOutcome {
    match tokio::time::timeout(max_request_duration, rx.recv()).await {
        Ok(Ok(StartOutcome::Ok)) => EnsureOutcome::Ready {
            was_already_running: false,
        },
        Ok(Ok(StartOutcome::Err(f))) => EnsureOutcome::Failed(match f.kind {
            StartFailureKind::NoFit | StartFailureKind::Oom => {
                EnsureFailure::InsufficientVram(f.message)
            }
            StartFailureKind::Disabled => EnsureFailure::ServiceDisabled(f.message),
            StartFailureKind::HealthTimeout => {
                EnsureFailure::StartFailed("health check timed out".into())
            }
            StartFailureKind::LaunchFailed => EnsureFailure::StartFailed(f.message),
        }),
        Ok(Err(e)) => EnsureOutcome::Failed(EnsureFailure::StartFailed(format!(
            "start broadcast closed: {e}"
        ))),
        Err(_) => EnsureOutcome::Failed(EnsureFailure::StartFailed("start timed out".into())),
    }
}

/// If a child dies with SIGKILL within this window from spawn, we treat it
/// as an OOM kill and bump the rolling safety factor for the next attempt.
const OOM_KILL_WINDOW: Duration = Duration::from_secs(30);

/// SIGTERM grace during Starting/Warming where the child may not yet be ready
/// to drain gracefully. Short so we do not block shutdown on a half-loaded
/// child.
const STARTING_SIGTERM_GRACE: Duration = Duration::from_secs(5);

/// SIGTERM grace during Running or during command-initiated drain. Longer
/// because the child is healthy and may be mid-request.
const RUNNING_SIGTERM_GRACE: Duration = Duration::from_secs(10);

/// Upper bound for a command service's optional `shutdown_command`.
/// Long enough for a `docker stop` (default 10s docker grace + slack)
/// but short enough that a hung shutdown doesn't block the drain
/// forever. A timeout escalates to SIGKILL of the shutdown child.
const SHUTDOWN_COMMAND_TIMEOUT: Duration = Duration::from_secs(30);

/// Backoff schedule for consecutive start failures before transitioning to
/// Disabled. Indexed by `retry_count` (0-based).
const FAILED_RETRY_BACKOFFS: [Duration; 3] = [
    Duration::from_secs(2),
    Duration::from_secs(5),
    Duration::from_secs(15),
];

/// 31-bit mask for `run_id`. `run_id` is derived from wall-clock millis; we
/// clip to a positive `i64` so it round-trips through SQLite's `INTEGER`
/// without sign surprises.
const RUN_ID_MASK: i64 = 0x7FFF_FFFF;

/// Clock-skew tolerance on the idle-timeout re-check. Lets a ping that raced
/// our deadline extend the idle window rather than immediately draining.
const IDLE_DEADLINE_SKEW_MS: u64 = 100;

async fn run(
    init: SupervisorInit,
    boot_svc: ServiceConfig,
    deps: SupervisorDeps,
    rx: mpsc::Receiver<SupervisorCommand>,
) {
    let mut loop_state = RunLoop::new(init, boot_svc, deps, rx);
    loop {
        let step = match loop_state.state.clone() {
            ServiceState::Idle => loop_state.handle_idle().await,
            ServiceState::Starting => loop_state.handle_active_lifecycle().await,
            ServiceState::Failed { retry_count } => loop_state.handle_failed(retry_count).await,
            ServiceState::Disabled { .. } => loop_state.handle_disabled().await,
            other => {
                warn!(state = ?other, "unexpected state in supervisor loop");
                return;
            }
        };
        if matches!(step, Step::Exit) {
            return;
        }
    }
}

/// Result of a `handle_*` method: either continue the outer dispatcher loop
/// (consulting the updated `state`) or exit the supervisor task entirely.
enum Step {
    Continue,
    Exit,
}

/// Mutable context threaded through every `handle_*` method. Owns every
/// binding that outlives a single state's body and is read or mutated across
/// transitions.
struct RunLoop {
    init: SupervisorInit,
    deps: SupervisorDeps,
    rx: mpsc::Receiver<SupervisorCommand>,
    state: ServiceState,
    state_mirror: Arc<SyncMutex<ServiceState>>,
    cancel_tx: watch::Sender<bool>,
    cancel_rx: watch::Receiver<bool>,
    /// Carries a broadcast sender from Idle through to Starting so waiters can
    /// be notified of the outcome once the child finishes warming.
    start_bus_carry: Option<broadcast::Sender<StartOutcome>>,
    /// Carries the placement-derived `CommandArgs` from Idle (where they are
    /// computed) into Starting (where `render_argv` consumes them).
    packed_for_spawn: Option<Packed>,
    /// Counts consecutive OOM kills for the current service.
    oom_attempts: u32,
    /// Total reserved bytes captured at Ensure time, used as the base for the
    /// rolling update that fires when the service later drains back to Idle.
    base_total_bytes_for_rolling: u64,
    /// Boot-time snapshot of this service's config, consulted by
    /// [`Self::current_svc`] as a fallback only when the live config has
    /// dropped the service (mid-reload). Never mutated — the live
    /// [`ConfigManager`] is always preferred.
    boot_svc: ServiceConfig,
}

impl RunLoop {
    fn new(
        init: SupervisorInit,
        boot_svc: ServiceConfig,
        deps: SupervisorDeps,
        rx: mpsc::Receiver<SupervisorCommand>,
    ) -> Self {
        let state = ServiceState::Idle;
        let state_mirror = Arc::new(SyncMutex::new(state.clone()));
        let (cancel_tx, cancel_rx) = watch::channel(false);
        Self {
            init,
            deps,
            rx,
            state,
            state_mirror,
            cancel_tx,
            cancel_rx,
            start_bus_carry: None,
            packed_for_spawn: None,
            oom_attempts: 0,
            base_total_bytes_for_rolling: 0,
            boot_svc,
        }
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
    fn current_svc(&self) -> ServiceConfig {
        let eff = self.deps.config.effective();
        eff.services
            .iter()
            .find(|s| s.name == self.init.identity.name)
            .cloned()
            .unwrap_or_else(|| self.boot_svc.clone())
    }

    fn set_state(&mut self, new_state: ServiceState) {
        let prior_state = self.state.clone();
        self.state = new_state;
        *self.state_mirror.lock() = self.state.clone();
        self.deps.events.publish(Event::StateChanged {
            service: self.init.identity.name.clone(),
            from: prior_state.name().to_string(),
            to: self.state.name().to_string(),
            at_ms: crate::tracking::now_unix_ms(),
        });
    }

    /// Publish an `AllocationChanged` event reflecting the current state of
    /// this service's entry in the allocation table. Called after every
    /// reserve, drain, or eviction that touches our row.
    fn emit_allocation_changed(&self) {
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
            at_ms: crate::tracking::now_unix_ms(),
        });
    }

    /// Run a command-template service's `shutdown_command`, if any, after
    /// the normal SIGTERM/SIGKILL pipeline completes. No-op for llama-cpp
    /// services and for command services without a shutdown command.
    ///
    /// Gives the shutdown child a bounded window to exit; logs (but does
    /// not propagate) failures — a drain is already terminal and the
    /// caller can't usefully recover from a shutdown-command error.
    async fn run_shutdown_command(&self) {
        let svc = self.current_svc();
        let Some(render_result) =
            crate::supervise::spawn::render_shutdown_argv(&svc, &self.init.allocation)
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
    fn record_drain_complete(&mut self) {
        self.deps.rolling.update(
            &self.init.identity.name,
            self.deps.observation.read_peak(&self.init.identity.name),
            self.base_total_bytes_for_rolling,
        );
        self.deps.observation.clear(&self.init.identity.name);
        self.deps
            .allocations
            .lock()
            .remove(&self.init.identity.name);
        self.emit_allocation_changed();
    }

    /// For on_demand services we wait here for an Ensure. Persistent services
    /// have the daemon call ensure() synthetically at boot.
    async fn handle_idle(&mut self) -> Step {
        loop {
            match self.rx.recv().await {
                Some(SupervisorCommand::Shutdown { ack }) => {
                    let _ = ack.send(());
                    return Step::Exit;
                }
                Some(SupervisorCommand::Snapshot { ack }) => {
                    let _ = ack.send(SupervisorSnapshot {
                        name: self.init.identity.name.clone(),
                        state: self.state.clone(),
                        run_id: None,
                        pid: None,
                    });
                }
                Some(SupervisorCommand::Ensure { ack }) => {
                    if self.handle_idle_ensure(ack).await {
                        return Step::Continue;
                    }
                }
                Some(SupervisorCommand::ActivityPing) => {}
                // Service is not running; drain/kill commands are no-ops.
                Some(SupervisorCommand::BeginDrain { ack, .. }) => {
                    let _ = ack.send(());
                }
                Some(SupervisorCommand::FastKill { ack, .. }) => {
                    let _ = ack.send(());
                }
                Some(SupervisorCommand::Enable { ack }) => {
                    // Idle is already enabled.
                    let _ = ack.send(EnableResult::NotDisabled);
                }
                Some(SupervisorCommand::Disable { ack }) => {
                    // Transition idle service directly to Disabled.
                    self.set_state(ServiceState::Disabled {
                        reason: DisableReason::UserDisabled,
                    });
                    let _ = ack.send(DisableResult::Disabled);
                    return Step::Continue;
                }
                None => return Step::Exit,
            }
        }
    }

    /// Body of an `Ensure` received while Idle. Returns `true` if the
    /// reservation succeeded and the state has transitioned to Starting (so
    /// the Idle loop should break); `false` if the ensure was rejected and
    /// the loop should keep waiting for the next command.
    async fn handle_idle_ensure(
        &mut self,
        ack: tokio::sync::oneshot::Sender<EnsureResponse>,
    ) -> bool {
        let snap = self.deps.snapshot.read().clone();
        let table = self.deps.allocations.lock().clone();

        let (want, pre_evicted) = match self.compute_reservation_map(&snap, &table) {
            Ok(w) => (w, Vec::new()),
            Err(ReservationFailure::PackFailed(_)) => {
                // Pack couldn't lay the model down given current reservations
                // (e.g. an in-between layer didn't fit on any allowed GPU).
                // Retry with lower-priority services treated as evicted; if
                // pack succeeds, drain those victims and carry them through
                // to the feasibility check.
                match self.retry_pack_with_eviction(&snap, &table).await {
                    Ok((want, victims)) => (want, victims),
                    Err(retry_reason) => {
                        let _ = ack.send(EnsureResponse::Unavailable(
                            EnsureFailure::InsufficientVram(retry_reason),
                        ));
                        return false;
                    }
                }
            }
            Err(other) => {
                let _ = ack.send(EnsureResponse::Unavailable(EnsureFailure::ServiceDisabled(
                    other.message(),
                )));
                return false;
            }
        };

        // Feasibility check. When we came through `retry_pack_with_eviction`,
        // the victims it drained are still sitting in the allocation table
        // (drains are in-flight) and their supervisors are blocked executing
        // the drain — `try_eviction_to_fit` can't poll their state. Use
        // `can_fit_after_eviction` with the already-committed victim list so
        // we don't loop back through select_for_slot only to find zero
        // candidates.
        let fit_result = if pre_evicted.is_empty() {
            crate::allocator::can_fit(&want, &snap, &table, Some(&self.init.identity.name))
        } else {
            crate::allocator::can_fit_after_eviction(
                &want,
                &snap,
                &table,
                Some(&self.init.identity.name),
                &pre_evicted,
            )
        };
        if let Err(nofit) = fit_result
            && let Err(reason) = self.try_eviction_to_fit(&want, &nofit).await
        {
            let _ = ack.send(EnsureResponse::Unavailable(
                EnsureFailure::InsufficientVram(reason),
            ));
            return false;
        }

        // Reserve in the allocation table before spawning. Capture the total
        // reserved bytes (MB → bytes) for the rolling update that fires when
        // the service later drains back to Idle.
        self.base_total_bytes_for_rolling = want.values().sum::<u64>() * 1024 * 1024;
        self.deps
            .allocations
            .lock()
            .insert(self.init.identity.name.clone(), want);
        self.emit_allocation_changed();

        // Create broadcast channel and subscribe the caller.
        let sender = tokio::sync::broadcast::channel::<StartOutcome>(16).0;
        let bus_rx = sender.subscribe();
        let _ = ack.send(EnsureResponse::Waiting { rx: bus_rx });
        self.start_bus_carry = Some(sender);

        let next = transition(&self.state, StateEvent::SpawnRequested);
        self.set_state(next);
        true
    }

    /// Determine the reservation map for an Ensure. On the llama-cpp path this
    /// runs the estimator + packer and caches `Packed` on `self` for the
    /// eventual `render_argv` call. Typed `Err` so the caller can branch on
    /// pack failures (retry with eviction) vs config / estimator failures
    /// (surface verbatim).
    fn compute_reservation_map(
        &mut self,
        snap: &crate::devices::DeviceSnapshot,
        table: &crate::allocator::AllocationTable,
    ) -> Result<std::collections::BTreeMap<crate::config::DeviceSlot, u64>, ReservationFailure>
    {
        self.compute_reservation_map_inner(snap, table, false)
    }

    /// Variant of [`compute_reservation_map`] that uses the optimistic
    /// planner (`pack_optimistic`). Used by the retry-after-eviction path,
    /// where `table` has had victims filtered out and nvml still shows their
    /// realized usage.
    fn compute_reservation_map_optimistic(
        &mut self,
        snap: &crate::devices::DeviceSnapshot,
        table: &crate::allocator::AllocationTable,
    ) -> Result<std::collections::BTreeMap<crate::config::DeviceSlot, u64>, ReservationFailure>
    {
        self.compute_reservation_map_inner(snap, table, true)
    }

    fn compute_reservation_map_inner(
        &mut self,
        snap: &crate::devices::DeviceSnapshot,
        table: &crate::allocator::AllocationTable,
        optimistic: bool,
    ) -> Result<std::collections::BTreeMap<crate::config::DeviceSlot, u64>, ReservationFailure>
    {
        let current = self.current_svc();
        let svc = &current;
        if matches!(svc.template(), crate::config::Template::Command) {
            self.packed_for_spawn = None;
            let bytes_mb = match svc.allocation_mode {
                crate::config::AllocationMode::Static { vram_mb } => vram_mb,
                crate::config::AllocationMode::Dynamic { min_mb, .. } => min_mb,
                crate::config::AllocationMode::None => 0,
            };
            let target_gpu: Option<u32> = svc
                .gpu_allow
                .first()
                .copied()
                .or_else(|| snap.gpus.first().map(|g| g.id));
            let mut map = std::collections::BTreeMap::new();
            if bytes_mb > 0 {
                let slot = match svc.placement_policy {
                    crate::config::PlacementPolicy::CpuOnly => crate::config::DeviceSlot::Cpu,
                    _ => match target_gpu {
                        Some(id) => crate::config::DeviceSlot::Gpu(id),
                        None => crate::config::DeviceSlot::Cpu,
                    },
                };
                map.insert(slot, bytes_mb);
            }
            return Ok(map);
        }
        if !svc.placement_override.is_empty() {
            self.packed_for_spawn = None;
            return Ok(svc.placement_override.clone());
        }

        // Estimator + placement path.
        let inputs = crate::estimator::EstimatorInputs::from_service(svc).ok_or(
            ReservationFailure::Misconfigured(MisconfiguredKind::NoModelPath),
        )?;
        let mut est = crate::estimator::estimate_from_path(self.deps.system.fs.as_ref(), &inputs)
            .map_err(ReservationFailure::EstimatorError)?;
        // Apply rolling correction to weights_bytes.
        let rc = self.deps.rolling.get(&svc.name);
        est.weights_bytes = (est.weights_bytes as f64 * rc.rolling_mean) as u64;

        let packed = if optimistic {
            crate::allocator::placement::pack_optimistic(&est, svc, snap, table)
        } else {
            crate::allocator::placement::pack(&est, svc, snap, table)
        }
        .map_err(ReservationFailure::PackFailed)?;
        // Convert Allocation bytes (per-DeviceId, in bytes) to the
        // BTreeMap<DeviceSlot, u64> in MB that can_fit + insert expects.
        let want_mb: std::collections::BTreeMap<crate::config::DeviceSlot, u64> = packed
            .allocation
            .bytes
            .iter()
            .map(|(id, bytes)| {
                let slot = match id {
                    crate::devices::DeviceId::Cpu => crate::config::DeviceSlot::Cpu,
                    crate::devices::DeviceId::Gpu(n) => crate::config::DeviceSlot::Gpu(*n),
                };
                // Convert bytes → MB, rounding up so we never under-reserve.
                let mb = bytes.div_ceil(1024 * 1024);
                (slot, mb)
            })
            .collect();
        self.packed_for_spawn = Some(packed);
        Ok(want_mb)
    }

    /// Pack failed against the current allocation table. Try again pretending
    /// every lower-priority service has been evicted; if pack now succeeds,
    /// actually drain those victims and return the new reservation plus the
    /// committed victim list so the caller can use
    /// `can_fit_after_eviction` for the final feasibility check. (Going
    /// through the normal `try_eviction_to_fit` loop here would fail:
    /// `collect_eviction_candidates` polls each supervisor's snapshot, but
    /// the ones we just begin_drained are blocked inside their drain.)
    async fn retry_pack_with_eviction(
        &mut self,
        snap: &crate::devices::DeviceSnapshot,
        table: &crate::allocator::AllocationTable,
    ) -> Result<
        (
            std::collections::BTreeMap<crate::config::DeviceSlot, u64>,
            Vec<smol_str::SmolStr>,
        ),
        String,
    > {
        let candidates = self.collect_eviction_candidates().await;
        let my_priority = self.current_svc().priority;
        let victims: Vec<smol_str::SmolStr> = candidates
            .iter()
            .filter(|c| c.priority < my_priority)
            .map(|c| c.name.clone())
            .collect();
        if victims.is_empty() {
            let _ = self.packed_for_spawn.take();
            let reason = self
                .compute_reservation_map(snap, table)
                .err()
                .map(ReservationFailure::message)
                .unwrap_or_else(|| "placement: unknown failure".into());
            return Err(reason);
        }

        let mut filtered = table.clone();
        for v in &victims {
            filtered.remove(v);
        }
        let want = self
            .compute_reservation_map_optimistic(snap, &filtered)
            .map_err(ReservationFailure::message)?;

        warn!(
            service = %self.init.identity.name,
            evict_count = victims.len(),
            "pack succeeded after pretending lower-priority victims were evicted; draining them"
        );
        for victim in &victims {
            if let Some(handle) = self.deps.registry.get(victim) {
                handle
                    .begin_drain(crate::supervise::drain::DrainReason::Eviction)
                    .await;
            }
        }
        Ok((want, victims))
    }

    /// Try to make room by draining lower-priority services, then re-check
    /// `can_fit`. Returns `Ok(())` on success (caller proceeds to reserve),
    /// `Err(reason)` if eviction can't help (caller should unavailable-reply).
    async fn try_eviction_to_fit(
        &mut self,
        want: &std::collections::BTreeMap<crate::config::DeviceSlot, u64>,
        nofit: &crate::allocator::NoFit,
    ) -> Result<(), String> {
        let candidates = self.collect_eviction_candidates().await;

        let reservations_now = self.deps.allocations.lock().clone();
        let snap = self.deps.snapshot.read().clone();
        let free_on_slot = snap.free_bytes(&nofit.slot).unwrap_or(0);
        let to_evict = crate::allocator::eviction::select_for_slot(
            nofit.needed_bytes,
            &nofit.slot,
            self.current_svc().priority,
            &candidates,
            &reservations_now,
            free_on_slot,
        );

        if to_evict.is_empty() {
            // Clear any computed packed args so they are not used on the next
            // Ensure attempt after this failure.
            let _ = self.packed_for_spawn.take();
            return Err(format!("{nofit}"));
        }

        warn!(service = %self.init.identity.name, evict_count = to_evict.len(), "eviction planned to make room");
        for victim in &to_evict {
            if let Some(handle) = self.deps.registry.get(victim) {
                handle
                    .begin_drain(crate::supervise::drain::DrainReason::Eviction)
                    .await;
            }
        }

        // Re-attempt feasibility after evictions. `begin_drain` is
        // non-blocking — the victims' allocation rows aren't actually removed
        // until each drain finishes (see `record_drain_complete`), so the raw
        // allocation table still reflects them at this moment.
        // `can_fit_after_eviction` treats the planned evictees as already
        // freed; if the drains later fail or stall, the child spawn will see
        // the real state and OOM-retry will handle it.
        let snap2 = self.deps.snapshot.read().clone();
        let table2 = self.deps.allocations.lock().clone();
        if let Err(again) = crate::allocator::can_fit_after_eviction(
            want,
            &snap2,
            &table2,
            Some(&self.init.identity.name),
            &to_evict,
        ) {
            let _ = self.packed_for_spawn.take();
            return Err(format!("eviction insufficient: {again}"));
        }
        Ok(())
    }

    /// Enumerate every other service as an eviction candidate. Skips self so
    /// we don't deadlock snapshotting our own supervisor.
    async fn collect_eviction_candidates(
        &self,
    ) -> Vec<crate::allocator::eviction::EvictionCandidate> {
        let all_services = self.deps.registry.all();
        // Materialise the priority map from the live config before any
        // `.await` below, so the arc-swap guard is released promptly.
        let priority_by_name: std::collections::BTreeMap<_, _> = {
            let eff = self.deps.config.effective();
            eff.services
                .iter()
                .map(|s| (s.name.clone(), s.priority))
                .collect()
        };
        let mut out = Vec::new();
        for (_name, handle) in all_services {
            if handle.name.as_str() == self.init.identity.name.as_str() {
                continue;
            }
            let Some(service_snap) = handle.snapshot().await else {
                continue;
            };
            let idle = matches!(service_snap.state, ServiceState::Idle);
            let alloc_mb = self
                .deps
                .allocations
                .lock()
                .get(&handle.name)
                .cloned()
                .unwrap_or_default();
            let bytes = alloc_mb.values().sum::<u64>() * 1024 * 1024;
            let priority = priority_by_name
                .get(&handle.name)
                .copied()
                .unwrap_or(DEFAULT_SERVICE_PRIORITY);
            out.push(crate::allocator::eviction::EvictionCandidate {
                name: handle.name.clone(),
                priority,
                idle,
                allocation_bytes: bytes,
            });
        }
        out
    }

    /// The whole Starting → Warming → Running → (Draining|Idle|Failed|Disabled)
    /// pipeline. State transitions within this body never escape back to the
    /// outer dispatcher; we only return when the child has been cleaned up and
    /// the next outer-loop state is either a terminal variant or Idle.
    async fn handle_active_lifecycle(&mut self) -> Step {
        // Pull the latest ServiceConfig for this spawn. `render_argv`,
        // `HealthConfig`, and the per-command branches below all read
        // fields that a reload may have changed (context, override_tensor,
        // cache_type_k/v, sampling, health probe settings, etc.).
        let current = self.current_svc();
        // When the placement engine has run (`packed_for_spawn = Some`), use
        // its computed Allocation for CUDA_VISIBLE_DEVICES rendering —
        // `self.init.allocation` is built from `placement_override` at
        // registry-init time and is empty for any estimator-driven service,
        // which would otherwise leave the child with `CUDA_VISIBLE_DEVICES=`
        // and silently fall back to CPU.
        let spawn_alloc = self
            .packed_for_spawn
            .as_ref()
            .map(|p| &p.allocation)
            .unwrap_or(&self.init.allocation);
        let spawn_cfg = match render_argv(
            &current,
            spawn_alloc,
            self.packed_for_spawn.as_ref().map(|p| &p.args),
        ) {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "placeholder substitution failed; aborting spawn");
                if let Some(bus) = self.start_bus_carry.take() {
                    let _ = bus.send(StartOutcome::Err(StartFailure {
                        kind: StartFailureKind::LaunchFailed,
                        message: format!("placeholder substitution failed: {e}"),
                    }));
                }
                self.deps
                    .allocations
                    .lock()
                    .remove(&self.init.identity.name);
                self.emit_allocation_changed();
                self.set_state(ServiceState::Failed { retry_count: 0 });
                return Step::Continue;
            }
        };
        let cmdline = format!("{} {}", spawn_cfg.binary, spawn_cfg.args.join(" "));
        let mut child = match self.deps.system.process_spawner.spawn(&spawn_cfg).await {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "spawn failed");
                if let Some(bus) = self.start_bus_carry.take() {
                    let _ = bus.send(StartOutcome::Err(StartFailure {
                        kind: StartFailureKind::LaunchFailed,
                        message: format!("{e}"),
                    }));
                }
                self.deps
                    .allocations
                    .lock()
                    .remove(&self.init.identity.name);
                self.emit_allocation_changed();
                self.set_state(ServiceState::Failed { retry_count: 0 });
                return Step::Continue;
            }
        };

        let pid = child.id().unwrap_or(0) as i32;
        self.deps
            .observation
            .register(&self.init.identity.name, pid as u32);
        let spawn_time = Instant::now();
        let run_id = crate::tracking::now_unix_ms() & RUN_ID_MASK;
        let allocation_json = serde_json::to_string(
            &self
                .init
                .allocation
                .bytes
                .iter()
                .map(|(k, v)| (k.as_display(), *v))
                .collect::<std::collections::BTreeMap<_, _>>(),
        )
        .unwrap_or_default();
        insert_running_row(
            &self.deps.db,
            self.init.service_id,
            run_id,
            pid as i64,
            cmdline.clone(),
            allocation_json,
        )
        .await;

        if let Some(stdout) = child.take_stdout() {
            spawn_pump_stdout(
                stdout,
                self.init.service_id,
                run_id,
                self.deps.batcher.clone(),
            );
        }
        if let Some(stderr) = child.take_stderr() {
            spawn_pump_stderr(
                stderr,
                self.init.service_id,
                run_id,
                self.deps.batcher.clone(),
            );
        }

        let health_cfg = HealthConfig {
            // `private_port` is fixed at boot (proxy binding can't move live),
            // so read it from init; the rest of the health config is live.
            url: format!(
                "http://127.0.0.1:{}{}",
                self.init.identity.private_port, current.health.http_path
            ),
            probe_interval: Duration::from_millis(current.health.probe_interval_ms),
            timeout: Duration::from_millis(current.health.timeout_ms),
        };

        let cancel_rx_h = self.cancel_rx.clone();
        let health_task = tokio::spawn(wait_healthy(health_cfg, cancel_rx_h));
        tokio::pin!(health_task);

        loop {
            tokio::select! {
                exit = child.wait() => {
                    return self.on_child_exit_during_start(exit, spawn_time);
                }
                outcome = &mut health_task => {
                    match self.on_health_outcome(outcome, &mut *child, run_id).await {
                        StartingOutcome::Continue => {}
                        StartingOutcome::Break => break,
                        StartingOutcome::Exit => return Step::Exit,
                    }
                }
                cmd = self.rx.recv() => {
                    match self.on_starting_command(cmd, &mut *child, run_id).await {
                        StartingOutcome::Continue => {}
                        StartingOutcome::Break => break,
                        StartingOutcome::Exit => return Step::Exit,
                    }
                }
            }
        }
        Step::Continue
    }

    /// Child exited while we were still in Starting or Warming (before the
    /// health probe passed). Detects OOM, updates state, and notifies waiters.
    fn on_child_exit_during_start(
        &mut self,
        exit: std::io::Result<std::process::ExitStatus>,
        spawn_time: Instant,
    ) -> Step {
        warn!(?exit, "child exited during starting/warming");
        self.deps
            .allocations
            .lock()
            .remove(&self.init.identity.name);
        self.deps.observation.clear(&self.init.identity.name);
        self.emit_allocation_changed();

        // Detect OOM kill: process died within 30 s and was killed by
        // SIGKILL (kernel OOM killer or cgroup limit).
        let runtime = spawn_time.elapsed();
        let was_sigkill = exit
            .as_ref()
            .ok()
            .and_then(|s| s.signal())
            .map(|sig| sig == libc::SIGKILL)
            .unwrap_or(false);
        if runtime < OOM_KILL_WINDOW && was_sigkill {
            self.oom_attempts += 1;
            if self.oom_attempts >= 2 {
                warn!(service = %self.init.identity.name, attempts = self.oom_attempts, "OOM retry limit reached; disabling");
                if let Some(bus) = self.start_bus_carry.take() {
                    let _ = bus.send(StartOutcome::Err(StartFailure {
                        kind: StartFailureKind::Oom,
                        message: "disabled after repeated OOM kills".into(),
                    }));
                }
                self.set_state(ServiceState::Disabled {
                    reason: DisableReason::Oom,
                });
            } else {
                warn!(service = %self.init.identity.name, "OOM kill detected; bumping rolling factor for retry");
                self.deps
                    .rolling
                    .bump_for_oom_retry(&self.init.identity.name);
                // Return to Idle so the next Ensure triggers a re-estimated
                // start with the bumped safety factor.
                if let Some(bus) = self.start_bus_carry.take() {
                    let _ = bus.send(StartOutcome::Err(StartFailure {
                        kind: StartFailureKind::Oom,
                        message: "OOM kill; retrying with larger reservation".into(),
                    }));
                }
                self.set_state(ServiceState::Idle);
            }
            return Step::Continue;
        }

        if let Some(bus) = self.start_bus_carry.take() {
            let _ = bus.send(StartOutcome::Err(StartFailure {
                kind: StartFailureKind::LaunchFailed,
                message: "child exited during starting".into(),
            }));
        }
        self.set_state(ServiceState::Failed { retry_count: 0 });
        Step::Continue
    }

    /// Handle the result of the health probe task. On Healthy we run the
    /// warming grace and fall through to the Running loop; on other outcomes
    /// we tear the child down and update state.
    async fn on_health_outcome(
        &mut self,
        outcome: Result<HealthOutcome, tokio::task::JoinError>,
        child: &mut dyn ManagedChild,
        run_id: i64,
    ) -> StartingOutcome {
        match outcome {
            Ok(HealthOutcome::Healthy) => {
                let next = transition(&self.state, StateEvent::HealthPassed);
                self.set_state(next);
                match self.run_warming_grace(child, run_id).await {
                    WarmingOutcome::Complete => {}
                    WarmingOutcome::ChildExited => return StartingOutcome::Break,
                    WarmingOutcome::Shutdown => return StartingOutcome::Exit,
                }

                // Notify waiters that the service is now running.
                if let Some(bus) = self.start_bus_carry.take() {
                    let _ = bus.send(StartOutcome::Ok);
                }

                self.run_running_loop(child, run_id).await
            }
            Ok(HealthOutcome::TimedOut) => {
                warn!(service = %self.init.identity.name, "health timed out; disabling");
                if let Some(bus) = self.start_bus_carry.take() {
                    let _ = bus.send(StartOutcome::Err(StartFailure {
                        kind: StartFailureKind::HealthTimeout,
                        message: "health check timed out".into(),
                    }));
                }
                self.deps
                    .allocations
                    .lock()
                    .remove(&self.init.identity.name);
                self.emit_allocation_changed();
                self.set_state(ServiceState::Disabled {
                    reason: DisableReason::HealthTimeout,
                });
                drain::sigterm_then_sigkill(child, STARTING_SIGTERM_GRACE).await;
                self.run_shutdown_command().await;
                StartingOutcome::Break
            }
            Ok(HealthOutcome::Cancelled) | Err(_) => {
                self.deps
                    .allocations
                    .lock()
                    .remove(&self.init.identity.name);
                self.emit_allocation_changed();
                drain::sigterm_then_sigkill(child, STARTING_SIGTERM_GRACE).await;
                self.run_shutdown_command().await;
                StartingOutcome::Exit
            }
        }
    }

    /// Warming grace: sleep `warming_grace_ms` while also watching for child
    /// exit and Shutdown.
    async fn run_warming_grace(
        &mut self,
        child: &mut dyn ManagedChild,
        run_id: i64,
    ) -> WarmingOutcome {
        let grace = Duration::from_millis(self.current_svc().warming_grace_ms);
        tokio::select! {
            _ = tokio::time::sleep(grace) => {
                let next = transition(&self.state, StateEvent::WarmingComplete);
                self.set_state(next);
                WarmingOutcome::Complete
            }
            _ = child.wait() => {
                warn!("child exited during warming grace");
                if let Some(bus) = self.start_bus_carry.take() {
                    let _ = bus.send(StartOutcome::Err(StartFailure {
                        kind: StartFailureKind::LaunchFailed,
                        message: "child exited during warming".into(),
                    }));
                }
                self.deps.allocations.lock().remove(&self.init.identity.name);
                self.emit_allocation_changed();
                self.set_state(ServiceState::Failed { retry_count: 0 });
                WarmingOutcome::ChildExited
            }
            cmd = self.rx.recv() => {
                if let Some(SupervisorCommand::Shutdown { ack }) = cmd {
                    info!(service = %self.init.identity.name, "draining during warming");
                    let _ = self.cancel_tx.send(true);
                    drain::sigterm_then_sigkill(child, RUNNING_SIGTERM_GRACE).await;
                    self.run_shutdown_command().await;
                    delete_running_row(&self.deps.db, self.init.service_id, run_id).await;
                    self.deps.allocations.lock().remove(&self.init.identity.name);
                    self.emit_allocation_changed();
                    let _ = ack.send(());
                    return WarmingOutcome::Shutdown;
                }
                // Snapshot or channel-closed: fall through to warming complete.
                let next = transition(&self.state, StateEvent::WarmingComplete);
                self.set_state(next);
                WarmingOutcome::Complete
            }
        }
    }

    /// The Running inner loop: wait for child exit, idle timeout, or commands.
    async fn run_running_loop(
        &mut self,
        child: &mut dyn ManagedChild,
        run_id: i64,
    ) -> StartingOutcome {
        let pid = child.id().unwrap_or(0) as i32;
        loop {
            tokio::select! {
                exit = child.wait() => {
                    warn!(?exit, "child exited from running");
                    self.record_drain_complete();
                    self.set_state(ServiceState::Failed { retry_count: 0 });
                    return StartingOutcome::Break;
                }
                _ = tokio::time::sleep_until(idle_deadline_for(&self.init.last_activity, self.current_svc().idle_timeout_ms)) => {
                    // Re-check the stamp; a recent ping may have extended the deadline.
                    let now = tokio::time::Instant::now();
                    let last = *self.init.last_activity.lock();
                    let fresh_deadline =
                        last + Duration::from_millis(self.current_svc().idle_timeout_ms);
                    if now + Duration::from_millis(IDLE_DEADLINE_SKEW_MS) < fresh_deadline {
                        // A ping arrived; loop again with a fresh deadline.
                        continue;
                    }
                    info!(service = %self.init.identity.name, "idle timeout; draining to idle");
                    drain::sigterm_then_sigkill(child, RUNNING_SIGTERM_GRACE).await;
                    self.run_shutdown_command().await;
                    delete_running_row(&self.deps.db, self.init.service_id, run_id).await;
                    self.record_drain_complete();
                    self.set_state(ServiceState::Idle);
                    return StartingOutcome::Break;
                }
                cmd = self.rx.recv() => {
                    match self.on_running_command(cmd, child, run_id, pid).await {
                        RunningOutcome::Continue => {}
                        RunningOutcome::Break => return StartingOutcome::Break,
                        RunningOutcome::Exit => return StartingOutcome::Exit,
                    }
                }
            }
        }
    }

    /// Full drain pipeline for a running child: transitions to Draining, runs
    /// the drain pipeline, deletes the DB row, and clears the allocation.
    /// Caller is responsible for transitioning to the next state after this
    /// returns.
    async fn drain_now(
        &mut self,
        child: &mut dyn ManagedChild,
        run_id: i64,
        reason: crate::supervise::drain::DrainReason,
    ) {
        self.set_state(ServiceState::Draining);
        let current = self.current_svc();
        let cfg = DrainConfig {
            max_request_duration: Duration::from_millis(current.max_request_duration_ms),
            drain_timeout: Duration::from_millis(current.drain_timeout_ms),
            extended_stream_drain: Duration::from_millis(current.extended_stream_drain_ms),
            sigterm_grace: RUNNING_SIGTERM_GRACE,
        };
        drain_pipeline(child, &cfg, self.init.inflight.clone(), reason).await;
        self.run_shutdown_command().await;
        delete_running_row(&self.deps.db, self.init.service_id, run_id).await;
        self.record_drain_complete();
    }

    /// Dispatch a command received while the service is Running.
    async fn on_running_command(
        &mut self,
        cmd: Option<SupervisorCommand>,
        child: &mut dyn ManagedChild,
        run_id: i64,
        pid: i32,
    ) -> RunningOutcome {
        match cmd {
            Some(SupervisorCommand::Shutdown { ack }) => {
                info!(service = %self.init.identity.name, "draining");
                let next = transition(&self.state, StateEvent::DrainRequested);
                self.set_state(next);
                let _ = self.cancel_tx.send(true);
                drain::sigterm_then_sigkill(child, RUNNING_SIGTERM_GRACE).await;
                self.run_shutdown_command().await;
                delete_running_row(&self.deps.db, self.init.service_id, run_id).await;
                self.record_drain_complete();
                let _ = ack.send(());
                RunningOutcome::Exit
            }
            Some(SupervisorCommand::Snapshot { ack }) => {
                let _ = ack.send(SupervisorSnapshot {
                    name: self.init.identity.name.clone(),
                    state: self.state.clone(),
                    run_id: Some(run_id),
                    pid: Some(pid),
                });
                RunningOutcome::Continue
            }
            Some(SupervisorCommand::Ensure { ack }) => {
                let _ = ack.send(EnsureResponse::AlreadyRunning);
                RunningOutcome::Continue
            }
            Some(SupervisorCommand::ActivityPing) => {
                *self.init.last_activity.lock() = tokio::time::Instant::now();
                RunningOutcome::Continue
            }
            Some(SupervisorCommand::BeginDrain { reason, ack }) => {
                info!(service = %self.init.identity.name, ?reason, "BeginDrain received; draining");
                self.drain_now(child, run_id, reason).await;
                let _ = ack.send(());
                self.set_state(ServiceState::Idle);
                RunningOutcome::Break
            }
            Some(SupervisorCommand::FastKill { reason, ack }) => {
                info!(service = %self.init.identity.name, ?reason, "FastKill received");
                self.set_state(ServiceState::Draining);

                fast_kill(child, reason).await;
                self.run_shutdown_command().await;

                delete_running_row(&self.deps.db, self.init.service_id, run_id).await;
                self.deps
                    .allocations
                    .lock()
                    .remove(&self.init.identity.name);
                self.deps.observation.clear(&self.init.identity.name);
                self.emit_allocation_changed();
                let _ = ack.send(());
                self.set_state(ServiceState::Idle);
                RunningOutcome::Break
            }
            Some(SupervisorCommand::Enable { ack }) => {
                // Already running; enable is a no-op.
                let _ = ack.send(EnableResult::NotDisabled);
                RunningOutcome::Continue
            }
            Some(SupervisorCommand::Disable { ack }) => {
                info!(service = %self.init.identity.name, "Disable received; draining then disabling");
                self.drain_now(child, run_id, drain::DrainReason::UserKilled)
                    .await;
                self.set_state(ServiceState::Disabled {
                    reason: DisableReason::UserDisabled,
                });
                let _ = ack.send(DisableResult::Disabled);
                RunningOutcome::Break
            }
            None => RunningOutcome::Exit,
        }
    }

    /// Dispatch a command received while the service is Starting (before the
    /// health probe has resolved). Drain/kill commands are no-ops here.
    async fn on_starting_command(
        &mut self,
        cmd: Option<SupervisorCommand>,
        child: &mut dyn ManagedChild,
        run_id: i64,
    ) -> StartingOutcome {
        let pid = child.id().unwrap_or(0) as i32;
        match cmd {
            Some(SupervisorCommand::Shutdown { ack }) => {
                let _ = self.cancel_tx.send(true);
                drain::sigterm_then_sigkill(child, STARTING_SIGTERM_GRACE).await;
                self.run_shutdown_command().await;
                self.deps
                    .allocations
                    .lock()
                    .remove(&self.init.identity.name);
                self.emit_allocation_changed();
                let _ = ack.send(());
                StartingOutcome::Exit
            }
            Some(SupervisorCommand::Snapshot { ack }) => {
                let _ = ack.send(SupervisorSnapshot {
                    name: self.init.identity.name.clone(),
                    state: self.state.clone(),
                    run_id: None,
                    pid: Some(pid),
                });
                StartingOutcome::Continue
            }
            Some(SupervisorCommand::Ensure { ack }) => {
                // Already in Starting; subscribe to existing bus or report running.
                if let Some(sender) = self.start_bus_carry.as_ref() {
                    if sender.receiver_count() >= self.current_svc().start_queue_depth {
                        let _ = ack.send(EnsureResponse::QueueFull);
                    } else {
                        let bus_rx = sender.subscribe();
                        let _ = ack.send(EnsureResponse::Waiting { rx: bus_rx });
                    }
                } else {
                    // No bus; best-effort.
                    let _ = ack.send(EnsureResponse::AlreadyRunning);
                }
                StartingOutcome::Continue
            }
            Some(SupervisorCommand::ActivityPing) => StartingOutcome::Continue,
            // Service is not yet running; drain/kill are no-ops during starting.
            Some(SupervisorCommand::BeginDrain { ack, .. }) => {
                let _ = ack.send(());
                StartingOutcome::Continue
            }
            Some(SupervisorCommand::FastKill { ack, .. }) => {
                let _ = ack.send(());
                StartingOutcome::Continue
            }
            Some(SupervisorCommand::Enable { ack }) => {
                // Already starting; not disabled.
                let _ = ack.send(EnableResult::NotDisabled);
                StartingOutcome::Continue
            }
            Some(SupervisorCommand::Disable { ack }) => {
                // Disable during starting: drain the child, clean up, and
                // transition to Disabled.
                let _ = self.cancel_tx.send(true);
                drain::sigterm_then_sigkill(child, STARTING_SIGTERM_GRACE).await;
                self.run_shutdown_command().await;
                delete_running_row(&self.deps.db, self.init.service_id, run_id).await;
                self.deps
                    .allocations
                    .lock()
                    .remove(&self.init.identity.name);
                self.deps.observation.clear(&self.init.identity.name);
                self.emit_allocation_changed();
                if let Some(bus) = self.start_bus_carry.take() {
                    let _ = bus.send(StartOutcome::Err(StartFailure {
                        kind: StartFailureKind::Disabled,
                        message: "service disabled by operator".into(),
                    }));
                }
                self.set_state(ServiceState::Disabled {
                    reason: DisableReason::UserDisabled,
                });
                let _ = ack.send(DisableResult::Disabled);
                StartingOutcome::Break
            }
            None => StartingOutcome::Exit,
        }
    }

    async fn handle_failed(&mut self, retry_count: u8) -> Step {
        let idx = (retry_count as usize).min(FAILED_RETRY_BACKOFFS.len() - 1);
        let delay = FAILED_RETRY_BACKOFFS[idx];
        tokio::select! {
            _ = tokio::time::sleep(delay) => {
                // `handle_failed` is only reached in the `Failed` state, for which
                // `RetryAfterBackoff` is always defined — either bumping retry_count
                // or promoting to Disabled at the cap.
                let next = transition(&self.state, StateEvent::RetryAfterBackoff);
                let next = if !matches!(next, ServiceState::Disabled { .. }) {
                    // Move back to Idle so the next Ensure triggers a fresh start.
                    ServiceState::Idle
                } else {
                    next
                };
                self.set_state(next);
                Step::Continue
            }
            cmd = self.rx.recv() => {
                match cmd {
                    Some(SupervisorCommand::Shutdown { ack }) => {
                        let _ = ack.send(());
                        return Step::Exit;
                    }
                    Some(SupervisorCommand::Enable { ack }) => {
                        // Failed is not disabled; enable is a no-op.
                        let _ = ack.send(EnableResult::NotDisabled);
                    }
                    Some(SupervisorCommand::Disable { ack }) => {
                        // Disable a failed service: skip the retry and go to Disabled.
                        self.set_state(ServiceState::Disabled {
                            reason: DisableReason::UserDisabled,
                        });
                        let _ = ack.send(DisableResult::Disabled);
                        return Step::Continue;
                    }
                    _ => {}
                }
                Step::Continue
            }
        }
    }

    async fn handle_disabled(&mut self) -> Step {
        info!(service = %self.init.identity.name, "disabled; awaiting shutdown or enable");
        loop {
            match self.rx.recv().await {
                Some(SupervisorCommand::Shutdown { ack }) => {
                    let _ = ack.send(());
                    return Step::Exit;
                }
                Some(SupervisorCommand::Snapshot { ack }) => {
                    let _ = ack.send(SupervisorSnapshot {
                        name: self.init.identity.name.clone(),
                        state: self.state.clone(),
                        run_id: None,
                        pid: None,
                    });
                }
                Some(SupervisorCommand::Ensure { ack }) => {
                    let _ = ack.send(EnsureResponse::Unavailable(EnsureFailure::ServiceDisabled(
                        "service disabled".into(),
                    )));
                }
                Some(SupervisorCommand::ActivityPing) => {}
                // Service is disabled; drain/kill are no-ops.
                Some(SupervisorCommand::BeginDrain { ack, .. }) => {
                    let _ = ack.send(());
                }
                Some(SupervisorCommand::FastKill { ack, .. }) => {
                    let _ = ack.send(());
                }
                Some(SupervisorCommand::Enable { ack }) => {
                    // Transition back to Idle so the next Ensure can start it.
                    let next = transition(&self.state, StateEvent::UserEnable);
                    self.set_state(next);
                    let _ = ack.send(EnableResult::Enabled);
                    return Step::Continue;
                }
                Some(SupervisorCommand::Disable { ack }) => {
                    // Already disabled.
                    let _ = ack.send(DisableResult::AlreadyDisabled);
                }
                None => return Step::Exit,
            }
        }
    }
}

/// Outcome of a sub-step inside the Starting-through-Draining pipeline.
enum StartingOutcome {
    /// Keep spinning the current inner select.
    Continue,
    /// Fall out of the Starting outer select (back to the dispatcher).
    Break,
    /// Exit the supervisor task entirely.
    Exit,
}

/// Outcome of the warming-grace select.
enum WarmingOutcome {
    /// Warming finished normally; proceed into Running.
    Complete,
    /// Child exited during the grace window.
    ChildExited,
    /// Shutdown received during the grace window.
    Shutdown,
}

/// Outcome of a single command dispatch inside the Running inner loop.
enum RunningOutcome {
    Continue,
    Break,
    Exit,
}

/// Convert a `DeviceSlot` to the canonical string key used in
/// `AllocationChanged` reservations (`"cpu"` or `"gpu:N"`).
fn slot_to_key(slot: &crate::config::DeviceSlot) -> String {
    match slot {
        crate::config::DeviceSlot::Cpu => "cpu".to_string(),
        crate::config::DeviceSlot::Gpu(n) => format!("gpu:{n}"),
    }
}

/// Compute the tokio `Instant` at which the idle deadline fires, based on the
/// last recorded activity instant. Lives entirely on the tokio monotonic
/// clock so `tokio::time::pause()` can freeze and advance it virtually.
fn idle_deadline_for(
    last_activity: &crate::tracking::activity::ActivityStamp,
    timeout_ms: u64,
) -> tokio::time::Instant {
    let last = *last_activity.lock();
    last + Duration::from_millis(timeout_ms)
}

/// Insert a `running_services` row.
async fn insert_running_row(
    db: &Database,
    service_id: i64,
    run_id: i64,
    pid: i64,
    command_line: String,
    allocation: String,
) {
    use crate::db::models::RunningService;

    let row = RunningService {
        service_id,
        run_id,
        pid,
        spawned_at: crate::tracking::now_unix_ms(),
        command_line,
        allocation,
        state: "starting".to_string(),
    };
    if let Err(e) = db.insert_running(&row).await {
        warn!(error = %e, "running_services insert failed");
    }
}

/// Delete the `running_services` row for `(service_id, run_id)` if present.
async fn delete_running_row(db: &Database, service_id: i64, run_id: i64) {
    if let Err(e) = db.delete_running(service_id, run_id).await {
        warn!(error = %e, "running_services delete failed");
    }
}
