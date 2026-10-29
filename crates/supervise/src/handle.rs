//! The supervisor's client-facing entry point: the command/response types
//! exchanged over its mailbox, [`SupervisorHandle`] itself, and
//! [`spawn_supervisor`], which starts the actor task and returns a handle to it.

use std::sync::Arc;

use ananke_config::validate::ServiceConfig;
use ananke_db::{Database, logs::BatcherHandle};
use ananke_devices::Allocation;
use ananke_events::{ChangeSignal, EventBus};
use ananke_observation::ObservationTable;
use ananke_proxy::InflightCounter;
use ananke_tracking::rolling::RollingTable;
use parking_lot::Mutex as SyncMutex;
use tokio::{sync::mpsc, task::JoinHandle};

use crate::{
    KillHandle, ensure::EnsureFailure, estimate_cache::EstimateCacheHandle,
    registry::SupervisorRegistry, run, state::ServiceState,
};

/// Distinguishes who initiated an `Ensure` command. The yield rule that
/// prevents a persistent service from evicting a running on-demand peer
/// applies only to background-watcher re-ensures; user-driven requests must
/// be allowed to evict idle non-persistent peers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EnsureSource {
    /// Initiated by an incoming user request (e.g. an OpenAI API call or a
    /// boot-time provision that reflects explicit operator intent).
    UserRequest,
    /// Initiated by the background persistent-service watcher, which reclaims
    /// idle VRAM but must not fight active on-demand traffic for it.
    #[default]
    BackgroundWatcher,
}

#[derive(Debug)]
pub enum SupervisorCommand {
    Shutdown {
        ack: tokio::sync::oneshot::Sender<()>,
    },
    /// Ensure the service is started (or starting). Returns a broadcast
    /// receiver the caller can await for the start outcome. If the
    /// start queue is full, returns `EnsureResponse::QueueFull` via the
    /// single-shot `ack`.
    Ensure {
        ack: tokio::sync::oneshot::Sender<EnsureResponse>,
        source: EnsureSource,
    },
    /// Record that a request was served; resets the idle timer.
    ActivityPing,
    /// Enter the full drain pipeline for eviction / idle timeout / user-kill.
    BeginDrain {
        reason: crate::drain::DrainReason,
        ack: tokio::sync::oneshot::Sender<()>,
    },
    /// Balloon-resolver fast-path: 5 s SIGTERM grace then SIGKILL.
    FastKill {
        reason: crate::drain::DrainReason,
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
    /// The time-to-first-token stall watchdog observed a proxied request that
    /// stayed in-flight past its timeout without producing a token. `run_id`
    /// is the run the stalled request was forwarded to; the handler ignores
    /// the command unless it still matches the current run, so a stall from an
    /// already-replaced run can't restart a healthy fresh one. Fire-and-forget
    /// (no ack): the request path sends it and moves on.
    WatchdogStall { run_id: i64 },
}

/// Result of a `SupervisorCommand::Enable`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnableResult {
    /// Was `Disabled`; now `Idle`.
    Enabled,
    /// Already in a non-disabled state (Idle, Running, etc.).
    NotDisabled,
    /// Clearing the stored disable failed; the state is unchanged.
    PersistFailed(String),
}

/// Result of a `SupervisorCommand::Disable`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DisableResult {
    /// Transitioned to `Disabled`.
    Disabled,
    /// Was already `Disabled`; no change.
    AlreadyDisabled,
    /// Storing the disable failed; the state is unchanged.
    PersistFailed(String),
}

#[derive(Debug)]
pub enum EnsureResponse {
    /// Service is already running; proceed directly.
    AlreadyRunning,
    /// Service is idle/starting; subscribe and wait.
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
    /// The request stayed parked in the queue for [`QUEUE_BLOCKED_GRACE`]
    /// without a single peer idling. The structured `busy_peers` list
    /// (rather than a freeform message) lets the wire layer render each
    /// blocker on its own and lets clients programmatically detect
    /// "blocked by `X`" if they want to.
    Blocked {
        busy_peers: Vec<smol_str::SmolStr>,
    },
}

/// The full outward-visible state of a supervisor. Always synthesised from
/// the shared [`MirroredState`] plus the handle's name — there is no separate
/// "async snapshot" path, because the supervisor's phase / run_id / pid all
/// live in one lock-free cell that readers can inspect directly.
#[derive(Debug, Clone)]
pub struct SupervisorSnapshot {
    pub name: smol_str::SmolStr,
    pub state: ServiceState,
    pub run_id: Option<i64>,
    pub pid: Option<i32>,
}

/// Shared cell holding every piece of supervisor state that is readable from
/// outside the task. The supervisor task is the sole writer; `SupervisorHandle`
/// is a reader. One source of truth, rather than a `ServiceState` mirror and a
/// `Snapshot` mailbox command kept in sync with each other.
#[derive(Debug, Clone, Default)]
pub(crate) struct MirroredState {
    pub(crate) state: ServiceState,
    pub(crate) run_id: Option<i64>,
    pub(crate) pid: Option<i32>,
}

impl MirroredState {
    /// The state a new supervisor starts in: `Idle`, or `Disabled {
    /// UserDisabled }` when restoring an operator disable.
    pub(crate) fn initial(disabled: bool) -> Self {
        let mut m = Self::default();
        if disabled {
            m.state = crate::state::ServiceState::Disabled {
                reason: crate::state::DisableReason::UserDisabled,
            };
        }
        m
    }
}

pub struct SupervisorHandle {
    pub name: smol_str::SmolStr,
    tx: mpsc::Sender<SupervisorCommand>,
    join: tokio::sync::Mutex<Option<JoinHandle<()>>>,
    /// Sole source of truth for the supervisor's state, shared between the
    /// task and every handle. Locked reads are non-blocking
    /// (`parking_lot::Mutex`) and never go through the command channel, so
    /// it's safe to call these from inside another supervisor's
    /// `handle_idle_ensure` or the eviction planner.
    mirror: Arc<SyncMutex<MirroredState>>,
}

impl SupervisorHandle {
    /// Build a registry-presence-only handle for unit tests that exercise
    /// pure-data logic against a `ServiceRegistry` (e.g. the balloon
    /// resolver's contention pick) without standing up a real supervisor
    /// task. The returned handle's mailbox is closed — anything that
    /// actually sends a command to it will silently drop or error, which
    /// is precisely the no-op behaviour those tests want.
    #[cfg(any(test, feature = "test-fakes"))]
    pub fn stub_for_test() -> Self {
        let (tx, _rx) = mpsc::channel(1);
        Self {
            name: smol_str::SmolStr::new(""),
            tx,
            join: tokio::sync::Mutex::new(None),
            mirror: Arc::new(SyncMutex::new(MirroredState::default())),
        }
    }

    /// Like [`Self::stub_for_test`], but hands back the mailbox receiver so a
    /// test can assert on the commands the handle was sent. Use this when the
    /// behaviour under test *is* the command dispatch (e.g. "does the balloon
    /// watchdog actually fast-kill?"); `stub_for_test`'s closed mailbox
    /// silently swallows commands and would make such a test vacuous.
    #[cfg(any(test, feature = "test-fakes"))]
    pub fn stub_with_mailbox() -> (Self, mpsc::Receiver<SupervisorCommand>) {
        let (tx, rx) = mpsc::channel(8);
        (
            Self {
                name: smol_str::SmolStr::new(""),
                tx,
                join: tokio::sync::Mutex::new(None),
                mirror: Arc::new(SyncMutex::new(MirroredState::default())),
            },
            rx,
        )
    }

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

    /// Non-blocking full snapshot of the supervisor's state, pid, and run_id.
    /// Always succeeds — the data lives in an always-present mirror cell,
    /// not in the supervisor task's local variables.
    pub fn peek(&self) -> SupervisorSnapshot {
        let m = self.mirror.lock();
        SupervisorSnapshot {
            name: self.name.clone(),
            state: m.state.clone(),
            run_id: m.run_id,
            pid: m.pid,
        }
    }

    /// Shorthand for [`Self::peek`] when only the lifecycle phase is needed.
    pub fn peek_state(&self) -> ServiceState {
        self.mirror.lock().state.clone()
    }

    pub async fn ensure(&self, source: EnsureSource) -> Option<EnsureResponse> {
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(SupervisorCommand::Ensure {
                ack: ack_tx,
                source,
            })
            .await
            .ok()?;
        ack_rx.await.ok()
    }

    pub fn ping(&self) {
        let _ = self.tx.try_send(SupervisorCommand::ActivityPing);
    }

    /// Signal that a proxied request forwarded to `run_id` stalled without a
    /// first response frame. Non-blocking and best-effort: if the mailbox is
    /// full the command is dropped; a later stalled request (or the drain the
    /// first accepted command initiates) covers the gap. The Running handler
    /// ignores the command unless `run_id` still matches the current run.
    pub fn watchdog_stall(&self, run_id: i64) {
        let _ = self
            .tx
            .try_send(SupervisorCommand::WatchdogStall { run_id });
    }

    pub async fn begin_drain(&self, reason: crate::drain::DrainReason) {
        let (ack, rx) = tokio::sync::oneshot::channel();
        let _ = self
            .tx
            .send(SupervisorCommand::BeginDrain { reason, ack })
            .await;
        let _ = rx.await;
    }

    pub async fn fast_kill(&self, reason: crate::drain::DrainReason) {
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
        // A stopped supervisor wrote nothing; don't report success.
        rx.await
            .unwrap_or_else(|_| EnableResult::PersistFailed(SUPERVISOR_STOPPED.into()))
    }

    pub async fn disable(&self) -> DisableResult {
        let (ack, rx) = tokio::sync::oneshot::channel();
        let _ = self.tx.send(SupervisorCommand::Disable { ack }).await;
        rx.await
            .unwrap_or_else(|_| DisableResult::PersistFailed(SUPERVISOR_STOPPED.into()))
    }
}

#[async_trait::async_trait]
impl KillHandle for SupervisorHandle {
    async fn fast_kill(&self, reason: crate::drain::DrainReason) {
        self.fast_kill(reason).await;
    }
}

/// Daemon-wide shared state every supervisor borrows. Cloning it is cheap
/// (every field is `Arc`-backed). Outside-world capabilities live inside
/// `system` ([`ananke_system::SystemDeps`]); everything else is
/// daemon-internal state.
///
/// `config` is the live [`ConfigManager`](ananke_config::manager::ConfigManager),
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
    pub snapshot: ananke_observation::SharedSnapshot,
    pub allocations: Arc<parking_lot::Mutex<ananke_allocator::AllocationTable>>,
    pub rolling: RollingTable,
    pub observation: ObservationTable,
    pub registry: SupervisorRegistry,
    pub config: Arc<ananke_config::manager::ConfigManager>,
    pub events: EventBus,
    pub system: ananke_system::SystemDeps,
    pub inflight: ananke_tracking::inflight::InflightTable,
    /// Global activity table. Exposed on `SupervisorDeps` so the eviction
    /// planner can rank peers by LRU when picking a minimum victim set.
    pub activity: ananke_tracking::activity::ActivityTable,
    /// Shared GGUF + estimator cache. The supervisor's spawn-time
    /// estimator run writes into this cache so the management
    /// `ServiceDetail` handler sees the same numbers without doing a
    /// second GGUF read.
    pub estimate_cache: EstimateCacheHandle,
    /// Bumped whenever any supervisor's mirrored state, run id, or pid
    /// changes. The run id and pid move without an event of their own.
    pub mirror_changed: ChangeSignal,
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
    pub last_activity: ananke_tracking::activity::ActivityStamp,
    pub inflight: Arc<InflightCounter>,
    /// Start in `Disabled { UserDisabled }` rather than `Idle`: the operator
    /// disabled the service in an earlier run and the database still holds
    /// the mark.
    pub disabled: bool,
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
    // Shared with `RunLoop`: the supervisor task writes through this cell,
    // every `SupervisorHandle::peek*` reads from it. No separate in-task
    // copy of the state lives alongside it — there is exactly one
    // `{state, run_id, pid}` tuple per supervisor.
    let mirror = Arc::new(SyncMutex::new(MirroredState::initial(init.disabled)));
    let join = tokio::spawn(run(init, boot_svc, deps, rx, mirror.clone()));
    SupervisorHandle {
        name,
        tx,
        join: tokio::sync::Mutex::new(Some(join)),
        mirror,
    }
}

/// `PersistFailed` reason when the supervisor task is gone before it acks an
/// enable or disable.
const SUPERVISOR_STOPPED: &str = "the service's supervisor has stopped";

/// Mailbox depth for per-supervisor command channels.
const SUPERVISOR_COMMAND_MAILBOX: usize = 32;
