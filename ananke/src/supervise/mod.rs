//! Service supervision: per-service tokio tasks, child lifetimes, health loops.
//!
//! Linux-coupled via `os::unix::process::ExitStatusExt` (signal() on ExitStatus)
//! and the submodules it delegates to (`drain`, `orphans`, `spawn`).

pub mod drain;
pub mod health;
pub mod logs;
pub mod orphans;
pub mod persistent_watcher;
pub mod preview;
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

use ananke_api::events::Event;
pub use orphans::{OrphanDisposition, reconcile};
use parking_lot::Mutex as SyncMutex;
pub use preview::{
    PlacementOutcome, PreviewError, preview_command, preview_command_placement,
    preview_override_placement, preview_placement,
};
pub use spawn::{SpawnConfig, render_argv};
use tokio::{
    sync::{broadcast, mpsc, watch},
    task::JoinHandle,
};
use tracing::{error, info, warn};

use crate::{
    allocator::placement::Packed,
    config::validate::{
        DEFAULT_SERVICE_PRIORITY, ErrorRateTrigger, Lifecycle, PeriodicMode, ServiceConfig,
    },
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
    /// The time-to-first-token stall watchdog observed a proxied request that
    /// stayed in-flight past its timeout without producing a token. `run_id`
    /// is the run the stalled request was forwarded to; the handler ignores
    /// the command unless it still matches the current run, so a stall from an
    /// already-replaced run can't restart a healthy fresh one. Fire-and-forget
    /// (no ack): the request path sends it and moves on.
    WatchdogStall { run_id: i64 },
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
/// "async snapshot" path anymore, because the supervisor's phase / run_id /
/// pid all live in one lock-free cell that readers can inspect directly.
#[derive(Debug, Clone)]
pub struct SupervisorSnapshot {
    pub name: smol_str::SmolStr,
    pub state: ServiceState,
    pub run_id: Option<i64>,
    pub pid: Option<i32>,
}

/// Shared cell holding every piece of supervisor state that is readable from
/// outside the task. The supervisor task is the sole writer; `SupervisorHandle`
/// is a reader. Replaced the old `ServiceState` mirror + dedicated `Snapshot`
/// mailbox command: one source of truth instead of two locations kept in sync
/// and one slow async path kept in parallel with one lock-free fast path.
#[derive(Debug, Clone, Default)]
struct MirroredState {
    state: ServiceState,
    run_id: Option<i64>,
    pid: Option<i32>,
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
    pub inflight: crate::tracking::inflight::InflightTable,
    /// Global activity table. Exposed on `SupervisorDeps` so the eviction
    /// planner can rank peers by LRU when picking a minimum victim set.
    pub activity: crate::tracking::activity::ActivityTable,
    /// Shared GGUF + estimator cache. The supervisor's spawn-time
    /// estimator run writes into this cache so the management
    /// `ServiceDetail` handler sees the same numbers without doing a
    /// second GGUF read.
    pub estimate_cache: crate::daemon::estimate_cache::EstimateCache,
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
    // Shared with `RunLoop`: the supervisor task writes through this cell,
    // every `SupervisorHandle::peek*` reads from it. No separate in-task
    // copy of the state lives alongside it — there is exactly one
    // `{state, run_id, pid}` tuple per supervisor.
    let mirror = Arc::new(SyncMutex::new(MirroredState::default()));
    let join = tokio::spawn(run(init, boot_svc, deps, rx, mirror.clone()));
    SupervisorHandle {
        name,
        tx,
        join: tokio::sync::Mutex::new(Some(join)),
        mirror,
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

/// Outcome of the retry-with-eviction pack attempt. The supervisor branches
/// on the `WaitForBusy` variant to queue the request (poll until a busy peer
/// idles out) rather than 503'ing immediately.
#[derive(Debug, Clone)]
enum RetryPackFailure {
    /// Every current candidate is busy at our priority or below; none is
    /// evictable right now, but each will be once its in-flight request
    /// finishes. `busy_peers` is the set the caller should watch: when any
    /// of their inflight counters drops to zero, the queue is woken up to
    /// retry the pack.
    WaitForBusy { busy_peers: Vec<smol_str::SmolStr> },
    /// Pack is infeasible no matter what: either there is no peer to evict,
    /// all peers are higher priority, or the optimistic pack still fails
    /// after treating every evictable peer as gone. Reject outright.
    NotPossible(String),
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
    /// The request parked in the start queue for [`QUEUE_BLOCKED_GRACE`]
    /// without a single watched peer idling. Carries the structured
    /// list of blocking peer names — wire-layer renderers turn this
    /// into a 503 + `service_blocked` body that names each blocker.
    /// Distinct from `InsufficientVram` because the *fit* is fine — the
    /// planner just can't displace the current occupant on its own.
    Blocked { busy_peers: Vec<smol_str::SmolStr> },
}

/// Ensure the service is Running, waiting up to `max_request_duration` for
/// an in-flight start to finish. Used by every HTTP path that forwards to
/// a supervised child.
pub async fn await_ensure(
    handle: &SupervisorHandle,
    max_request_duration: Duration,
) -> EnsureOutcome {
    let rx = match handle.ensure(EnsureSource::UserRequest).await {
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
            StartFailureKind::Blocked { busy_peers } => EnsureFailure::Blocked { busy_peers },
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

/// SIGTERM grace during Starting where the child may not yet be ready to
/// drain gracefully. Short so we do not block shutdown on a half-loaded
/// child.
const STARTING_SIGTERM_GRACE: Duration = Duration::from_secs(5);

/// SIGTERM grace during Running or during command-initiated drain. Longer
/// because the child is healthy and may be mid-request.
const RUNNING_SIGTERM_GRACE: Duration = Duration::from_secs(10);

/// In-flight drain grace for a stall-triggered restart. Short by design: the
/// stall watchdog only fires once the run has produced nothing for the whole
/// timeout window, so its wedged in-flight request will never complete and
/// there is no healthy traffic to preserve — a brief grace for a tailing
/// packet, then SIGTERM.
const STALL_DRAIN_INFLIGHT_WAIT: Duration = Duration::from_secs(5);

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

/// Poll interval while a supervisor is queued behind a busy peer waiting
/// for it to idle. Each tick is a cheap atomic-load precheck against the
/// watched peers' inflight counters (see `retry_queued_ensure`); only when
/// a peer has actually gone idle do we run the full estimator + packer.
/// 250 ms is fast enough that a queued request wakes up within a quarter
/// second of the peer finishing its response, while keeping the tick
/// noise low enough that the logs aren't drowned when a queued ensure
/// is waiting for a peer that's loading.
const QUEUE_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Hard upper bound on how long an Ensure may sit in the start queue
/// waiting for a busy peer to idle. Past this point the queue gives up
/// and resolves with [`StartFailureKind::Blocked`] so the client sees a
/// 503 with a clear "blocked by peer X" message instead of waiting the
/// full `max_request_duration_ms` (default 10 min) and silently 503'ing
/// with "start timed out".
///
/// Sized to absorb the brief "peer just finishing its response" window
/// (the happy-path queue test releases at 400 ms; a multi-token
/// streaming completion that's about to end fits comfortably) without
/// silently absorbing the genuinely-stuck case. Past 10 s the
/// expectation is the client should see a structured error and decide
/// whether to wait, kill the blocker, or try a different model. The
/// blocker is, by definition, a non-elastic peer at our priority or
/// higher — dynamic-allocation services are always evictable (see
/// `collect_eviction_candidates`), so this timeout exists for the
/// genuinely-stuck case where waiting longer would not help.
const QUEUE_BLOCKED_GRACE: Duration = Duration::from_secs(10);

async fn run(
    init: SupervisorInit,
    boot_svc: ServiceConfig,
    deps: SupervisorDeps,
    rx: mpsc::Receiver<SupervisorCommand>,
    mirror: Arc<SyncMutex<MirroredState>>,
) {
    let mut loop_state = RunLoop::new(init, boot_svc, deps, rx, mirror);
    loop {
        let step = match loop_state.read_state() {
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
/// transitions. The lifecycle phase, run_id, and pid all live in the shared
/// `mirror` cell; there is no local `state` field — readers call
/// [`Self::read_state`] or [`Self::read_full`].
struct RunLoop {
    init: SupervisorInit,
    deps: SupervisorDeps,
    rx: mpsc::Receiver<SupervisorCommand>,
    mirror: Arc<SyncMutex<MirroredState>>,
    cancel_tx: watch::Sender<bool>,
    cancel_rx: watch::Receiver<bool>,
    /// Carries a broadcast sender from Idle through to Starting so waiters can
    /// be notified of the outcome once the child passes the health probe.
    start_bus_carry: Option<broadcast::Sender<StartOutcome>>,
    /// When Some, the supervisor is parked in Idle waiting for a busy peer
    /// to idle out so we can evict it. The bus is shared with the original
    /// caller (via `EnsureResponse::Waiting`) and resolves to `Ok` once the
    /// fit succeeds or to `Err` on shutdown / disable / hard-reject.
    pending_ensure_bus: Option<broadcast::Sender<StartOutcome>>,
    /// The source of the currently parked Ensure (matches the original
    /// `EnsureSource` that entered the queue). Consulted by
    /// `retry_queued_ensure` for the yield-to-nonpersistent check.
    pending_ensure_source: EnsureSource,
    /// Busy peers we're waiting on for the queued Ensure. The poll-tick
    /// branch of the Idle loop does a cheap atomic-load check against
    /// these peers' inflight counters and skips the expensive estimator-
    /// plus-packer path entirely while they're all still above zero.
    /// Updated every time the retry returns `WaitForBusy` with a
    /// potentially different set of peers.
    queued_watch: Vec<smol_str::SmolStr>,
    /// Wall-monotonic stamp captured the first time the current Ensure
    /// entered the queue. [`QUEUE_BLOCKED_GRACE`] is measured against
    /// this; the value is preserved across retry ticks (only cleared
    /// when the queue resolves, succeeds, or hard-fails) so a flapping
    /// `busy_peers` set doesn't reset the wait clock and let the client
    /// hang forever.
    queued_since: Option<tokio::time::Instant>,
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
    /// Monotonic timestamps of recent error-rate auto-restarts, pruned to the
    /// flap window. Lives on the supervisor (not the run) so it survives the
    /// drain → respawn cycle and can trip the flap cap. Only the error-rate
    /// trigger touches this — periodic restarts are intentional maintenance
    /// and must not count toward the cap.
    auto_restart_history: Vec<tokio::time::Instant>,
    /// Set by the periodic `on-request` trigger when the interval elapses: the
    /// current run is stale and the next incoming request drives a drain →
    /// respawn (carried through [`Self::deferred_ensure`]) instead of hitting
    /// the wedged child.
    restart_pending: bool,
    /// An `Ensure` whose reply is deferred across an on-request restart drain.
    /// Held from the Running `Ensure` handler through the drain, then replayed
    /// at the top of [`Self::handle_idle`] so the triggering request blocks on
    /// the fresh process via the normal idle-ensure spawn path.
    deferred_ensure: Option<(tokio::sync::oneshot::Sender<EnsureResponse>, EnsureSource)>,
}

impl RunLoop {
    fn new(
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
            base_total_bytes_for_rolling: 0,
            boot_svc,
            auto_restart_history: Vec::new(),
            restart_pending: false,
            deferred_ensure: None,
        }
    }

    /// Read the current lifecycle phase from the shared mirror.
    fn read_state(&self) -> ServiceState {
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
    fn current_svc(&self) -> ServiceConfig {
        let eff = self.deps.config.effective();
        eff.services
            .iter()
            .find(|s| s.name == self.init.identity.name)
            .cloned()
            .unwrap_or_else(|| self.boot_svc.clone())
    }

    fn set_state(&mut self, new_state: ServiceState) {
        let prior_state = {
            let mut m = self.mirror.lock();
            let prior = m.state.clone();
            m.state = new_state.clone();
            prior
        };
        info!(
            service = %self.init.identity.name,
            from = %prior_state.name(),
            to = %new_state.name(),
            "state transition"
        );
        self.deps.events.publish(Event::StateChanged {
            service: self.init.identity.name.clone(),
            from: prior_state.name().to_string(),
            to: new_state.name().to_string(),
            at_ms: crate::tracking::now_unix_ms(),
        });
    }

    /// Stamp `run_id` + `pid` into the shared mirror. Called once per spawn,
    /// right after `insert_running_row` assigns the run_id, so every
    /// `SupervisorHandle::peek()` from that point sees the identifiers.
    fn set_running_ids(&mut self, run_id: i64, pid: i32) {
        let mut m = self.mirror.lock();
        m.run_id = Some(run_id);
        m.pid = Some(pid);
    }

    /// Clear `run_id` + `pid` from the shared mirror. Called from every
    /// teardown path (drain complete, child exited, eviction, etc.) alongside
    /// the `delete_running_row` DB update, so peeks don't keep reporting a
    /// stale child. Prefer [`Self::end_run`] at call sites that need both.
    fn clear_running_ids(&mut self) {
        let mut m = self.mirror.lock();
        m.run_id = None;
        m.pid = None;
    }

    /// Combined teardown: delete the DB `running_services` row and clear the
    /// mirror's `run_id` + `pid`. Replaces the pattern of calling both in
    /// sequence at every exit from the running/draining loops — keeping the
    /// two in one helper means we can't forget one.
    async fn end_run(&mut self, run_id: i64) {
        delete_running_row(&self.deps.db, self.init.service_id, run_id).await;
        self.clear_running_ids();
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
            // The rolling base is a VRAM-only pledge, so the observed peak must
            // be VRAM-only too. Using the combined VRAM+RSS peak here inflates
            // the ratio by the process's host-memory footprint and over-pledges
            // VRAM on the next placement — which has pushed a shard past a GPU's
            // capacity and blocked re-placement.
            self.deps
                .observation
                .read_peak_vram(&self.init.identity.name),
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
    ///
    /// The poll-tick branch of the select is only armed while a queued Ensure
    /// is parked on `pending_ensure_bus` (busy peer waiting to idle); when no
    /// queue is pending, the loop is a pure command-recv.
    async fn handle_idle(&mut self) -> Step {
        // An on-request periodic restart drained to Idle while holding the
        // triggering request's Ensure. Replay it through the normal
        // idle-ensure path now so the caller blocks on the fresh spawn (a
        // Waiting response on a real start bus) rather than getting
        // AlreadyRunning against a child that no longer exists.
        if let Some((ack, source)) = self.deferred_ensure.take()
            && self.handle_idle_ensure(ack, source).await
        {
            return Step::Continue;
        }
        loop {
            tokio::select! {
                _ = tokio::time::sleep(QUEUE_POLL_INTERVAL), if self.pending_ensure_bus.is_some() => {
                    if self.retry_queued_ensure().await {
                        return Step::Continue;
                    }
                }
                cmd = self.rx.recv() => {
                    match cmd {
                        Some(SupervisorCommand::Shutdown { ack }) => {
                            self.fail_queue(
                                StartFailureKind::Disabled,
                                "supervisor shutting down".into(),
                            );
                            let _ = ack.send(());
                            return Step::Exit;
                        }
                        Some(SupervisorCommand::Ensure { ack, source }) => {
                            // If there's already a queued Ensure, subscribe
                            // the new caller to the same bus (coalesce) up to
                            // `start_queue_depth`. Otherwise go through the
                            // normal fit path.
                            if let Some(sender) = self.pending_ensure_bus.as_ref() {
                                if sender.receiver_count() >= self.current_svc().start_queue_depth {
                                    let _ = ack.send(EnsureResponse::QueueFull);
                                } else {
                                    let rx = sender.subscribe();
                                    let _ = ack.send(EnsureResponse::Waiting { rx });
                                }
                            } else if self.handle_idle_ensure(ack, source).await {
                                return Step::Continue;
                            }
                        }
                        Some(SupervisorCommand::ActivityPing) => {}
                        // Not running: a stall can only be reported against a
                        // Running run, so this is always stale here.
                        Some(SupervisorCommand::WatchdogStall { .. }) => {}
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
                            self.fail_queue(
                                StartFailureKind::Disabled,
                                "service disabled by operator".into(),
                            );
                            self.set_state(ServiceState::Disabled {
                                reason: DisableReason::UserDisabled,
                            });
                            let _ = ack.send(DisableResult::Disabled);
                            return Step::Continue;
                        }
                        None => {
                            self.fail_queue(
                                StartFailureKind::Disabled,
                                "supervisor channel closed".into(),
                            );
                            return Step::Exit;
                        }
                    }
                }
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
        source: EnsureSource,
    ) -> bool {
        let snap = self.deps.snapshot.read().clone();
        let table = self.deps.allocations.lock().clone();

        let (want, pre_evicted) = match self.compute_reservation_map(&snap, &table) {
            Ok(w) => (w, Vec::new()),
            Err(ReservationFailure::PackFailed(msg)) => {
                // Pack couldn't lay the model down given current reservations
                // (e.g. an in-between layer didn't fit on any allowed GPU).
                // Retry with lower-priority services treated as evicted; if
                // pack succeeds, drain those victims and carry them through
                // to the feasibility check.
                //
                // Before attempting eviction, check the "persistent yields to
                // active non-persistent" rule: a persistent service about to
                // evict a peer stands down if any non-persistent peer is
                // Starting or Running — but only when the ensure originated
                // from the background watcher, not from a user request. A
                // user explicitly asking for a persistent service should be
                // allowed to evict idle on-demand peers.
                if self.should_yield_to_active_nonpersistent(source) {
                    info!(
                        service = %self.init.identity.name,
                        "persistent ensure yielding to active non-persistent peer"
                    );
                    let _ = ack.send(EnsureResponse::Unavailable(
                        EnsureFailure::InsufficientVram(
                            "persistent service yielding to active non-persistent peer".into(),
                        ),
                    ));
                    return false;
                }
                info!(
                    service = %self.init.identity.name,
                    reason = %msg,
                    "initial pack failed; retrying with eviction"
                );
                match self.retry_pack_with_eviction(&snap, &table).await {
                    Ok((want, victims)) => (want, victims),
                    Err(RetryPackFailure::NotPossible(retry_reason)) => {
                        // Reason is already logged by `retry_pack_with_eviction`
                        // (either "optimistic pack failed" or "no evictable
                        // candidates"), so no second log here — the consuming
                        // handler emits the client-facing line.
                        let _ = ack.send(EnsureResponse::Unavailable(
                            EnsureFailure::InsufficientVram(retry_reason),
                        ));
                        return false;
                    }
                    Err(RetryPackFailure::WaitForBusy { busy_peers }) => {
                        // Park the caller on a broadcast bus and stay in
                        // Idle. The Idle loop's poll-tick branch retries
                        // the ensure periodically via `retry_queued_ensure`;
                        // Shutdown/Disable/another Ensure that arrive
                        // while we're queued flow through the normal
                        // command-recv arm and drain the bus appropriately.
                        return self.enter_queue(ack, busy_peers, source);
                    }
                }
            }
            Err(other) => {
                let msg = other.message();
                warn!(
                    service = %self.init.identity.name,
                    reason = %msg,
                    "ensure failed: reservation computation error"
                );
                let _ = ack.send(EnsureResponse::Unavailable(EnsureFailure::ServiceDisabled(
                    msg,
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
        info!(
            service = %self.init.identity.name,
            fit_ok = fit_result.is_ok(),
            pre_evicted = ?pre_evicted,
            "fit_result computed"
        );
        if let Err(nofit) = fit_result {
            match self.try_eviction_to_fit(&want, &nofit, source).await {
                Ok(()) => {}
                Err(RetryPackFailure::NotPossible(reason)) => {
                    let _ = ack.send(EnsureResponse::Unavailable(
                        EnsureFailure::InsufficientVram(reason),
                    ));
                    return false;
                }
                Err(RetryPackFailure::WaitForBusy { busy_peers }) => {
                    return self.enter_queue(ack, busy_peers, source);
                }
            }
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

        let next = transition(&self.read_state(), StateEvent::SpawnRequested);
        self.set_state(next);
        true
    }

    /// Park an Ensure whose fit is blocked by a busy peer. Creates the shared
    /// broadcast bus, replies `Waiting` to the caller, records the busy peers
    /// for the cheap poll-tick precheck, and stashes the sender on
    /// `self.pending_ensure_bus` so the Idle loop's poll-tick branch can
    /// later retry the fit. Always returns `false` — the supervisor stays in
    /// Idle until either a retry succeeds or a command drains the bus.
    fn enter_queue(
        &mut self,
        ack: tokio::sync::oneshot::Sender<EnsureResponse>,
        busy_peers: Vec<smol_str::SmolStr>,
        source: EnsureSource,
    ) -> bool {
        let sender = tokio::sync::broadcast::channel::<StartOutcome>(16).0;
        let bus_rx = sender.subscribe();
        let _ = ack.send(EnsureResponse::Waiting { rx: bus_rx });
        self.pending_ensure_bus = Some(sender);
        self.pending_ensure_source = source;
        self.queued_watch = busy_peers;
        // Stamp the first time this Ensure parks in the queue so
        // QUEUE_BLOCKED_GRACE can fire even if `queued_watch` keeps
        // shifting across retry ticks.
        self.queued_since = Some(tokio::time::Instant::now());
        false
    }

    /// Retry a queued Ensure. Called from the Idle loop's poll-tick when
    /// `pending_ensure_bus` is Some. On success, promotes the queued bus to
    /// `start_bus_carry` and transitions to Starting. On hard-fail, drains
    /// the bus with `Err` and clears the queue. On continued soft-wait,
    /// leaves the queue in place for the next tick.
    ///
    /// Cheap precheck up front: if every peer in `queued_watch` still has
    /// inflight > 0, nothing actionable has changed since last tick and we
    /// skip the expensive estimator + packer path entirely. Only when at
    /// least one watched peer has gone idle do we run the full retry. This
    /// is what keeps a 30-second wait from producing 60+ GGUF reads and
    /// 180+ info log lines.
    async fn retry_queued_ensure(&mut self) -> bool {
        // Bail out of the queue entirely once we've been parked here for
        // longer than `QUEUE_BLOCKED_GRACE`. Without this, a request
        // blocked by a tied-priority non-elastic peer (e.g. another
        // model mid-generation that's about to exceed the user's
        // patience) hangs all the way to `max_request_duration_ms` and
        // returns a generic "start timed out". With it, the client sees
        // a 503 + structured "blocked by peer X" within ~30 s. Dynamic
        // peers don't reach this branch because they are always
        // evictable (see `collect_eviction_candidates`).
        if let Some(since) = self.queued_since
            && since.elapsed() > QUEUE_BLOCKED_GRACE
        {
            let busy_peers = self.queued_watch.clone();
            let log_summary = if busy_peers.is_empty() {
                "unknown".to_string()
            } else {
                busy_peers
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            info!(
                service = %self.init.identity.name,
                busy_peers = %log_summary,
                grace_secs = QUEUE_BLOCKED_GRACE.as_secs(),
                "queue grace exceeded; failing queued ensure"
            );
            // The `message` field on `StartFailure` is now just a
            // human-readable log breadcrumb; the wire layer renders
            // off the structured `busy_peers` list inside the kind
            // and ignores this string.
            let log_message = format!(
                "blocked by busy peer(s) for {:?}: {log_summary}",
                QUEUE_BLOCKED_GRACE
            );
            self.fail_queue(StartFailureKind::Blocked { busy_peers }, log_message);
            return false;
        }

        // Same yield rule as `handle_idle_ensure`: a background-watcher-driven
        // persistent ensure queued behind a busy peer stands down the moment
        // any non-persistent peer enters Starting/Running. Without this check,
        // the watcher would wait for the peer to finish loading and then
        // immediately evict it — the exact opposite of what it should do.
        // User-driven ensures are exempt: they may proceed to eviction.
        let source = self.pending_ensure_source;
        if self.should_yield_to_active_nonpersistent(source) {
            info!(
                service = %self.init.identity.name,
                "queued persistent ensure yielding to active non-persistent peer"
            );
            self.fail_queue(
                StartFailureKind::NoFit,
                "persistent service yielding to active non-persistent peer".into(),
            );
            return false;
        }

        if !self.queued_watch.is_empty()
            && self
                .queued_watch
                .iter()
                .all(|name| self.peer_still_busy_for_precheck(name))
        {
            return false;
        }

        let snap = self.deps.snapshot.read().clone();
        let table = self.deps.allocations.lock().clone();

        let (want, pre_evicted) = match self.compute_reservation_map(&snap, &table) {
            Ok(w) => (w, Vec::new()),
            Err(ReservationFailure::PackFailed(_)) => {
                match self.retry_pack_with_eviction(&snap, &table).await {
                    Ok((want, victims)) => (want, victims),
                    Err(RetryPackFailure::WaitForBusy { busy_peers }) => {
                        self.queued_watch = busy_peers;
                        return false;
                    }
                    Err(RetryPackFailure::NotPossible(reason)) => {
                        self.fail_queue(StartFailureKind::LaunchFailed, reason);
                        return false;
                    }
                }
            }
            Err(other) => {
                self.fail_queue(StartFailureKind::Disabled, other.message());
                return false;
            }
        };

        // Feasibility re-check on the post-eviction table, mirroring the
        // shape of `handle_idle_ensure`'s fit branch.
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
        if let Err(nofit) = fit_result {
            match self.try_eviction_to_fit(&want, &nofit, source).await {
                Ok(()) => {}
                Err(RetryPackFailure::NotPossible(reason)) => {
                    self.fail_queue(StartFailureKind::LaunchFailed, reason);
                    return false;
                }
                // Still waiting for the busy peer; refresh the watch set
                // so next tick skips until something changes.
                Err(RetryPackFailure::WaitForBusy { busy_peers }) => {
                    self.queued_watch = busy_peers;
                    return false;
                }
            }
        }

        // Commit the reservation + promote the queued bus to the start
        // carry, then transition to Starting. `handle_active_lifecycle`
        // will pick it up from here.
        self.base_total_bytes_for_rolling = want.values().sum::<u64>() * 1024 * 1024;
        self.deps
            .allocations
            .lock()
            .insert(self.init.identity.name.clone(), want);
        self.emit_allocation_changed();
        if let Some(sender) = self.pending_ensure_bus.take() {
            self.start_bus_carry = Some(sender);
        }
        self.queued_watch.clear();
        self.queued_since = None;
        let next = transition(&self.read_state(), StateEvent::SpawnRequested);
        self.set_state(next);
        true
    }

    /// Drain the queued Ensure bus with an error outcome and clear the
    /// pending state. Called on hard-reject, shutdown, disable, or any
    /// other terminal interrupt while queued.
    fn fail_queue(&mut self, kind: StartFailureKind, message: String) {
        self.queued_watch.clear();
        self.queued_since = None;
        if let Some(sender) = self.pending_ensure_bus.take() {
            let _ = sender.send(StartOutcome::Err(StartFailure { kind, message }));
        }
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
            return self.compute_command_reservation(svc, snap, table, optimistic);
        }
        if !svc.placement_override.is_empty() {
            self.packed_for_spawn = None;
            return Ok(svc.placement_override.clone());
        }

        // Estimator + placement path.
        let inputs = crate::estimator::EstimatorInputs::from_service(svc).ok_or(
            ReservationFailure::Misconfigured(MisconfiguredKind::NoModelPath),
        )?;
        let fingerprint = inputs.config_fingerprint();
        let (summary, mut est) =
            crate::estimator::estimate_with_summary(self.deps.system.fs.as_ref(), &inputs)
                .map_err(ReservationFailure::EstimatorError)?;
        // Warm the daemon-wide estimate cache with the *base* estimate
        // (pre-rolling-correction) and the GGUF summary we just
        // parsed. The management `ServiceDetail` handler reads this
        // cache instead of re-parsing the file on every detail poll;
        // populating it here turns the first detail-page view of a
        // running service into a cache hit. The cache stores the base
        // numbers because that's what the wire `EstimateSummary`
        // documents — the rolling correction applied below is a
        // supervisor-internal placement tweak, not a user-facing
        // estimate.
        let lc = svc.llama_cpp();
        if let Some(lc) = lc {
            self.deps.estimate_cache.insert(
                svc.name.clone(),
                crate::daemon::estimate_cache::CacheEntry::build(
                    &summary,
                    &est,
                    lc.model.clone(),
                    lc.mmproj.clone(),
                    fingerprint,
                ),
            );
        }
        // Apply rolling correction to weights_bytes. `effective_mean()` gates
        // the factor to a neutral 1.0 until enough samples accumulate, so a
        // single noisy early observation can't over-pledge a shard past a GPU's
        // capacity.
        let rc = self.deps.rolling.get(&svc.name);
        est.weights_bytes = (est.weights_bytes as f64 * rc.effective_mean()) as u64;

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

    /// Reservation-map computation for command-template services.
    ///
    /// Picks the GPU with the most available headroom (subject to
    /// `gpu_allow`) via `placement::pick_command_gpu`, falling back to
    /// `PackError::WeightsDoNotFit` when nothing fits — that variant is the
    /// trigger for the supervisor's eviction-retry path. The chosen
    /// allocation is stashed on `packed_for_spawn` so `handle_active_lifecycle`
    /// renders `CUDA_VISIBLE_DEVICES` against the actual pick instead of the
    /// always-empty `init.allocation` (which is built from
    /// `placement_override` only).
    fn compute_command_reservation(
        &mut self,
        svc: &crate::config::ServiceConfig,
        snap: &crate::devices::DeviceSnapshot,
        table: &crate::allocator::AllocationTable,
        optimistic: bool,
    ) -> Result<std::collections::BTreeMap<crate::config::DeviceSlot, u64>, ReservationFailure>
    {
        self.packed_for_spawn = None;
        let (min_mb, prefer_mb) = match svc.allocation_mode {
            crate::config::AllocationMode::Static { vram_mb } => (vram_mb, Some(vram_mb)),
            crate::config::AllocationMode::Dynamic { min_mb, max_mb, .. } => (min_mb, Some(max_mb)),
            crate::config::AllocationMode::None => (0, None),
        };
        let mut map = std::collections::BTreeMap::new();
        if min_mb == 0 {
            // No reservation requested. Still publish an empty Packed so the
            // spawn path renders a deterministic (empty) CUDA_VISIBLE_DEVICES.
            let alloc = crate::devices::Allocation::from_override(&map);
            self.packed_for_spawn = Some(crate::allocator::placement::Packed {
                allocation: alloc,
                args: crate::allocator::placement::CommandArgs::default(),
                expert_offload_bytes: 0,
                expert_offload_layers: 0,
            });
            return Ok(map);
        }
        // Operator pinned the layout explicitly (e.g. multi-GPU vLLM with
        // TP=2). Honour the per-device pledge verbatim instead of trying to
        // land the whole `min_mb` on a single GPU.
        if !svc.placement_override.is_empty() {
            crate::allocator::placement::check_command_placement_override(
                svc, snap, table, optimistic,
            )
            .map_err(ReservationFailure::PackFailed)?;
            map = svc.placement_override.clone();
            let alloc = crate::devices::Allocation::from_override(&map);
            self.packed_for_spawn = Some(crate::allocator::placement::Packed {
                allocation: alloc,
                args: crate::allocator::placement::CommandArgs::default(),
                expert_offload_bytes: 0,
                expert_offload_layers: 0,
            });
            return Ok(map);
        }
        let slot = if matches!(
            svc.placement_policy,
            crate::config::PlacementPolicy::CpuOnly
        ) {
            crate::config::DeviceSlot::Cpu
        } else {
            match crate::allocator::placement::pick_command_gpu(
                svc, snap, table, min_mb, prefer_mb, optimistic,
            ) {
                Some(id) => crate::config::DeviceSlot::Gpu(id),
                None if snap.gpus.is_empty() => {
                    // No GPUs visible at all (typical in tests with a CPU-only
                    // snapshot). Fall back to CPU so the reservation lands
                    // somewhere — matches the pre-fix behaviour and keeps
                    // the test harness working.
                    crate::config::DeviceSlot::Cpu
                }
                None => {
                    return Err(ReservationFailure::PackFailed(
                        crate::allocator::placement::PackError::WeightsDoNotFit,
                    ));
                }
            }
        };
        map.insert(slot.clone(), min_mb);
        let alloc = crate::devices::Allocation::from_override(&map);
        self.packed_for_spawn = Some(crate::allocator::placement::Packed {
            allocation: alloc,
            args: crate::allocator::placement::CommandArgs::default(),
            expert_offload_bytes: 0,
            expert_offload_layers: 0,
        });
        Ok(map)
    }

    /// Pack failed against the current allocation table. Walk evictable peers
    /// in least-recently-used order (oldest activity first), adding one at
    /// a time to the victim set until the optimistic pack succeeds. The
    /// minimum number of LRU-first peers we need to drain, no more — a busy
    /// service whose cold peers already cover the demand never gets touched.
    ///
    /// Choosing LRU over "single biggest victim" is deliberate: among idle
    /// peers the disruption cost is near-zero (no one is using them), so
    /// evicting two cold services to preserve one warm one is the right
    /// trade. If the warm service happens to be the one with the most
    /// VRAM, we leave it alone and evict multiple LRU peers to cover the
    /// deficit.
    ///
    /// (Going through `try_eviction_to_fit` here instead wouldn't work:
    /// `collect_eviction_candidates` reads peer state through the mirror,
    /// which is fine, but that helper is sized for the
    /// "fit-but-missing-headroom" case and doesn't re-run the packer after
    /// each eviction; this one does.)
    async fn retry_pack_with_eviction(
        &mut self,
        snap: &crate::devices::DeviceSnapshot,
        table: &crate::allocator::AllocationTable,
    ) -> Result<
        (
            std::collections::BTreeMap<crate::config::DeviceSlot, u64>,
            Vec<smol_str::SmolStr>,
        ),
        RetryPackFailure,
    > {
        let candidates = self.collect_eviction_candidates().await;
        let my_priority = self.current_svc().priority;
        let my_lifecycle = self.current_svc().lifecycle;

        // LRU-first ordering over evictable peers. `collect_eviction_candidates`
        // already excludes zero-allocation peers, so everything in `candidates`
        // holds real VRAM. Services that have never been pinged sort as "most
        // LRU" (effectively infinite age), so an idle peer is a first-class
        // victim candidate — except for tied-priority persistent peers when
        // the requester is on-demand, which `is_evictable_by` filters out.
        let now = tokio::time::Instant::now();
        let mut ranked: Vec<(&crate::allocator::eviction::EvictionCandidate, u128)> = candidates
            .iter()
            .filter(|c| c.is_evictable_by(my_priority, my_lifecycle))
            .map(|c| {
                let age_ms = self
                    .deps
                    .activity
                    .last(&c.name)
                    .map(|t| now.saturating_duration_since(t).as_millis())
                    .unwrap_or(u128::MAX);
                (c, age_ms)
            })
            .collect();
        ranked.sort_by_key(|(_, age)| std::cmp::Reverse(*age));

        if ranked.is_empty() {
            // No peer is currently evictable. Distinguish two cases so the
            // caller can queue vs reject appropriately:
            //   - there IS a busy peer at our priority or lower, so waiting
            //     for it to idle would make it evictable → `WaitForBusy`
            //     (returning the set of peers to watch so the queue loop
            //     can skip expensive retries while they're all still busy).
            //   - all peers are higher priority, or there are none → hard
            //     reject with the pack reason.
            let busy_peers: Vec<smol_str::SmolStr> = candidates
                .iter()
                .filter(|c| !c.idle && c.priority <= my_priority)
                .map(|c| c.name.clone())
                .collect();
            let _ = self.packed_for_spawn.take();
            if !busy_peers.is_empty() {
                info!(
                    service = %self.init.identity.name,
                    busy_peers = ?busy_peers,
                    my_priority,
                    "no evictable candidates yet; waiting for busy peer to idle"
                );
                return Err(RetryPackFailure::WaitForBusy { busy_peers });
            }
            info!(
                service = %self.init.identity.name,
                candidates = candidates.len(),
                my_priority,
                "eviction selection failed: no evictable candidates for placement"
            );
            let reason = self
                .compute_reservation_map(snap, table)
                .err()
                .map(ReservationFailure::message)
                .unwrap_or_else(|| "placement: unknown failure".into());
            return Err(RetryPackFailure::NotPossible(reason));
        }

        // Greedy LRU-first fill: walk peers in LRU order, extending the
        // victim set one at a time, and re-run the optimistic packer after
        // every addition. Stop as soon as the pack succeeds — that's the
        // minimum LRU-first victim set that covers the layout.
        let mut victims: Vec<smol_str::SmolStr> = Vec::new();
        let mut winning_want: Option<std::collections::BTreeMap<crate::config::DeviceSlot, u64>> =
            None;
        for (cand, _) in &ranked {
            victims.push(cand.name.clone());
            let mut filtered = table.clone();
            for v in &victims {
                filtered.remove(v);
            }
            if let Ok(w) = self.compute_reservation_map_optimistic(snap, &filtered) {
                winning_want = Some(w);
                break;
            }
        }

        let Some(want) = winning_want else {
            // Even with every evictable peer treated as gone the packer
            // still can't lay out the model. Report the last pack error
            // so the operator sees the actual deficit (which GPU, what
            // bytes), not a generic "no fit".
            let mut filtered = table.clone();
            for v in &victims {
                filtered.remove(v);
            }
            let reason = self
                .compute_reservation_map_optimistic(snap, &filtered)
                .err()
                .map(ReservationFailure::message)
                .unwrap_or_else(|| "placement: unknown failure".into());
            warn!(
                service = %self.init.identity.name,
                reason = %reason,
                evictable_count = victims.len(),
                "optimistic pack failed even with all evictable peers treated as gone"
            );
            return Err(RetryPackFailure::NotPossible(reason));
        };

        info!(
            service = %self.init.identity.name,
            evict_count = victims.len(),
            victims = ?victims,
            evictable_considered = ranked.len(),
            "eviction planned (LRU-first minimum victim set)"
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
    /// `can_fit`. `Ok(())` means the caller can proceed to reserve;
    /// `Err(RetryPackFailure::WaitForBusy)` means eviction couldn't pick a
    /// victim right now but a busy same-or-lower-priority peer exists, so
    /// the caller should queue and retry; `Err(RetryPackFailure::NotPossible)`
    /// means no amount of waiting will help.
    async fn try_eviction_to_fit(
        &mut self,
        want: &std::collections::BTreeMap<crate::config::DeviceSlot, u64>,
        nofit: &crate::allocator::NoFit,
        source: EnsureSource,
    ) -> Result<(), RetryPackFailure> {
        // Matching the check in `handle_idle_ensure` and
        // `retry_queued_ensure`: a background-watcher-driven persistent
        // ensure yields when a non-persistent peer is actively loading or
        // running. User-driven ensures may proceed to eviction.
        if self.should_yield_to_active_nonpersistent(source) {
            info!(
                service = %self.init.identity.name,
                "persistent ensure yielding to active non-persistent peer (post-pack fit)"
            );
            return Err(RetryPackFailure::NotPossible(
                "persistent service yielding to active non-persistent peer".into(),
            ));
        }
        let candidates = self.collect_eviction_candidates().await;

        let reservations_now = self.deps.allocations.lock().clone();
        // `nofit.available_bytes` is already reservation-adjusted
        // (`min(snap.free, snap.total - sum_of_reservations)`), so pass
        // it through verbatim. Using `snap.free_bytes` here would let
        // running services' pledges hide behind the raw snapshot free
        // and short-circuit eviction — exactly what happens in the
        // fake-spawner harness where the snapshot doesn't move as
        // services start up.
        let to_evict = crate::allocator::eviction::select_for_slot(
            nofit.needed_bytes,
            &nofit.slot,
            self.current_svc().priority,
            self.current_svc().lifecycle,
            &candidates,
            &reservations_now,
            nofit.available_bytes,
        );

        if to_evict.is_empty() {
            // Same distinction as `retry_pack_with_eviction`: is there a busy
            // peer we could evict once it idles, or are we genuinely stuck?
            let my_priority = self.current_svc().priority;
            let busy_peers: Vec<smol_str::SmolStr> = candidates
                .iter()
                .filter(|c| !c.idle && c.priority <= my_priority)
                .map(|c| c.name.clone())
                .collect();
            let _ = self.packed_for_spawn.take();
            if !busy_peers.is_empty() {
                info!(
                    service = %self.init.identity.name,
                    busy_peers = ?busy_peers,
                    needed_bytes = nofit.needed_bytes,
                    available_bytes = nofit.available_bytes,
                    slot = ?nofit.slot,
                    "no evictable candidates yet; waiting for busy peer to idle"
                );
                return Err(RetryPackFailure::WaitForBusy { busy_peers });
            }
            info!(
                service = %self.init.identity.name,
                candidates = candidates.len(),
                needed_bytes = nofit.needed_bytes,
                available_bytes = nofit.available_bytes,
                slot = ?nofit.slot,
                "eviction selection failed: no evictable candidates cover the deficit"
            );
            return Err(RetryPackFailure::NotPossible(format!("{nofit}")));
        }

        info!(
            service = %self.init.identity.name,
            evict_count = to_evict.len(),
            victims = ?to_evict,
            "eviction planned (fit feasible after drain)"
        );
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
            return Err(RetryPackFailure::NotPossible(format!(
                "eviction insufficient: {again}"
            )));
        }
        Ok(())
    }

    /// Enumerate every other service that currently holds a VRAM reservation
    /// and could plausibly be displaced. Skips self so we don't deadlock
    /// snapshotting our own supervisor, and skips peers with no allocation
    /// — an empty slot in the registry has nothing to give up, so it isn't
    /// an eviction candidate, it's just a registered service that hasn't
    /// started yet.
    async fn collect_eviction_candidates(
        &self,
    ) -> Vec<crate::allocator::eviction::EvictionCandidate> {
        let all_services = self.deps.registry.all();
        // Materialise the priority + lifecycle + dynamic-mode flag from the
        // live config before any `.await` below, so the arc-swap guard is
        // released promptly. The dynamic-mode bit is what lets the planner
        // treat balloon services as elastic — see the idle-computation
        // comment below.
        let svc_meta_by_name: std::collections::BTreeMap<_, _> = {
            let eff = self.deps.config.effective();
            eff.services
                .iter()
                .map(|s| {
                    let is_dynamic = matches!(
                        s.allocation_mode,
                        crate::config::AllocationMode::Dynamic { .. }
                    );
                    (s.name.clone(), (s.priority, s.lifecycle, is_dynamic))
                })
                .collect()
        };
        let mut out = Vec::new();
        for (_name, handle) in all_services {
            if handle.name.as_str() == self.init.identity.name.as_str() {
                continue;
            }
            let alloc_mb = self
                .deps
                .allocations
                .lock()
                .get(&handle.name)
                .cloned()
                .unwrap_or_default();
            let bytes = alloc_mb.values().sum::<u64>() * 1024 * 1024;
            if bytes == 0 {
                continue;
            }
            // `peek_state` reads the shared mirror under a parking_lot
            // mutex — no mailbox hop, no circular wait. When this is
            // called from inside `handle_idle_ensure` the peer supervisor
            // may be mid-drain and unable to service commands; reading the
            // mirror directly is the only safe path.
            let state = handle.peek_state();
            let (priority, lifecycle, is_dynamic) = svc_meta_by_name
                .get(&handle.name)
                .copied()
                .unwrap_or((DEFAULT_SERVICE_PRIORITY, Lifecycle::OnDemand, false));
            // "Idle" for eviction purposes means "no user-facing work
            // in flight on a settled supervisor" — either literally
            // Idle (not running) or Running with no in-flight requests.
            // Starting is excluded: the child is spawned but not yet
            // healthy and its start-bus still holds queued callers
            // who'd all fail if we tore it down.
            //
            // Dynamic-allocation services are the explicit exception. By
            // choosing `allocation.mode = "dynamic"` the operator has
            // declared the service elastic — happy to be torn down when
            // a peer needs VRAM. The most common case is ComfyUI: its
            // web UI keeps a long-lived `/ws` open for live updates, so
            // its inflight counter is rarely zero even when no image is
            // actually generating. Treating that idle-with-open-UI as
            // "busy" deadlocks tied-priority on-demand peers (chat hits
            // 503 / silent queue) while ComfyUI's 7 GiB pledge sits
            // unused. Funnelling dynamic services into the idle bucket
            // restores the eviction path the operator opted into.
            let in_flight = self.deps.inflight.current(&handle.name);
            let settled = matches!(state, ServiceState::Idle | ServiceState::Running);
            let idle = settled && (is_dynamic || in_flight == 0);
            out.push(crate::allocator::eviction::EvictionCandidate {
                name: handle.name.clone(),
                priority,
                lifecycle,
                idle,
                allocation_bytes: bytes,
            });
        }
        out
    }

    /// The "persistent yields to active non-persistent" predicate: this
    /// service is `Persistent` and at least one peer service with
    /// `Lifecycle::OnDemand` is currently in `Starting` or `Running`.
    ///
    /// Callers use this to stand down from an eviction-requiring start
    /// rather than snipe a peer that's in the middle of loading or
    /// running. The persistent watcher will retry on its own cadence
    /// when the pool quiets, so there's no reclamation-deadline pressure.
    fn should_yield_to_active_nonpersistent(&self, source: EnsureSource) -> bool {
        // User-driven requests are allowed to evict idle on-demand peers
        // regardless of their running state; only background watcher re-ensures
        // should stand down when a non-persistent peer is active.
        if source == EnsureSource::UserRequest {
            return false;
        }
        if self.current_svc().lifecycle != crate::config::Lifecycle::Persistent {
            return false;
        }
        let lifecycle_by_name: std::collections::BTreeMap<_, _> = {
            let eff = self.deps.config.effective();
            eff.services
                .iter()
                .map(|s| (s.name.clone(), s.lifecycle))
                .collect()
        };
        for (_, handle) in self.deps.registry.all() {
            if handle.name.as_str() == self.init.identity.name.as_str() {
                continue;
            }
            let lifecycle = lifecycle_by_name
                .get(&handle.name)
                .copied()
                .unwrap_or(crate::config::Lifecycle::OnDemand);
            if lifecycle == crate::config::Lifecycle::Persistent {
                continue;
            }
            if matches!(
                handle.peek_state(),
                ServiceState::Starting | ServiceState::Running,
            ) {
                return true;
            }
        }
        false
    }

    /// True if a named peer is currently "still busy" for the purpose of
    /// the queued-ensure retry precheck. A peer counts as busy when it
    /// has in-flight work OR is in `Starting` (loading, not yet eligible
    /// for eviction). Without the `Starting`-aware branch the 250 ms tick
    /// would re-run the full estimator + packer every tick while a peer
    /// loads, because `inflight == 0` during startup.
    fn peer_still_busy_for_precheck(&self, name: &smol_str::SmolStr) -> bool {
        if self.deps.inflight.current(name) > 0 {
            return true;
        }
        self.deps
            .registry
            .get(name.as_str())
            .map(|h| matches!(h.peek_state(), ServiceState::Starting))
            .unwrap_or(false)
    }

    /// The whole Starting → Running → (Draining|Idle|Failed|Disabled)
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
        info!(service = %self.init.identity.name, binary = %spawn_cfg.binary, "spawning child");
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
        self.deps.observation.set_cgroup_parent(
            &self.init.identity.name,
            current.tracking.cgroup_parent.clone(),
        );
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
        self.set_running_ids(run_id, pid);

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

        // When no health check path is configured, transition to Running
        // immediately after spawn. The service is assumed ready as soon as
        // the child process exists. Used by oneshots that don't expose an
        // HTTP health endpoint.
        if let Some(http_path) = &current.health.http_path {
            let health_cfg = HealthConfig {
                url: format!(
                    "http://127.0.0.1:{}{}",
                    self.init.identity.private_port, http_path
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
        } else {
            // No health check: transition to Running immediately.
            match self
                .on_health_outcome(Ok(HealthOutcome::Healthy), &mut *child, run_id)
                .await
            {
                StartingOutcome::Continue => {}
                StartingOutcome::Break => {}
                StartingOutcome::Exit => return Step::Exit,
            }
        }
        Step::Continue
    }

    /// Child exited while we were still in Starting (before the health
    /// probe passed). Detects OOM, updates state, and notifies waiters.
    fn on_child_exit_during_start(
        &mut self,
        exit: std::io::Result<std::process::ExitStatus>,
        spawn_time: Instant,
    ) -> Step {
        warn!(?exit, "child exited during starting");
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

    /// Handle the result of the health probe task. On Healthy we transition
    /// to Running and notify waiters; on other outcomes we tear the child
    /// down and update state.
    async fn on_health_outcome(
        &mut self,
        outcome: Result<HealthOutcome, tokio::task::JoinError>,
        child: &mut dyn ManagedChild,
        run_id: i64,
    ) -> StartingOutcome {
        match outcome {
            Ok(HealthOutcome::Healthy) => {
                let next = transition(&self.read_state(), StateEvent::HealthPassed);
                self.set_state(next);

                // Reset the idle window at the moment the service becomes
                // ready. Without this, a stale `last_activity` (left over
                // from the request that preceded the most recent drain) can
                // make the idle deadline already elapsed on the first poll
                // of `run_running_loop`'s select, racing the waiter's
                // post-`await_ensure` ping and draining the freshly-spawned
                // child before it can serve the request that started it.
                *self.init.last_activity.lock() = tokio::time::Instant::now();

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

    /// The Running inner loop: wait for child exit, idle timeout, or commands.
    async fn run_running_loop(
        &mut self,
        child: &mut dyn ManagedChild,
        run_id: i64,
    ) -> StartingOutcome {
        // Timers are seeded once from the config at Running entry; threshold
        // values (rate, min_requests, cooldown, flap cap) are re-read live in
        // the handlers so a `PUT /api/config` edit takes effect without a
        // respawn. The `running_since` stamp anchors both the periodic
        // deadline and the error-rate cooldown.
        let running_since = tokio::time::Instant::now();
        self.restart_pending = false;
        let auto_restart = self.current_svc().auto_restart;

        // Error-rate poll: a plain interval, first tick one period out so the
        // fresh run (which has zero metrics) isn't queried the instant it
        // starts.
        let mut error_poll = auto_restart.error_rate.as_ref().map(|er| {
            let period = Duration::from_millis(er.poll_interval_ms);
            tokio::time::interval_at(tokio::time::Instant::now() + period, period)
        });

        // Periodic timer: the next instant at which the periodic trigger is
        // evaluated. Starts at `running_since + interval`; the on-idle mode
        // reuses it as a short re-poll while it waits for a quiet window.
        let mut periodic_deadline = auto_restart
            .periodic
            .as_ref()
            .map(|p| running_since + Duration::from_millis(p.interval_ms));

        loop {
            tokio::select! {
                exit = child.wait() => {
                    warn!(?exit, "child exited from running");
                    self.record_drain_complete();
                    self.set_state(ServiceState::Failed { retry_count: 0 });
                    return StartingOutcome::Break;
                }
                // Error-rate watchdog poll. The branch future borrows only the
                // local `error_poll`; the decision + restart run in the handler.
                _ = async {
                    match error_poll.as_mut() {
                        Some(p) => { p.tick().await; }
                        None => std::future::pending::<()>().await,
                    }
                } => {
                    if let Some(detail) = self.evaluate_error_rate(run_id, running_since).await
                        && matches!(
                            self.perform_error_rate_restart(child, run_id, detail).await,
                            AutoRestartOutcome::Restarted | AutoRestartOutcome::Disabled,
                        )
                    {
                        return StartingOutcome::Break;
                    }
                }
                // Periodic-restart timer. `periodic_deadline` is a plain
                // `Instant` computed eagerly, so the branch future holds no
                // borrow of `self`.
                _ = async {
                    match periodic_deadline {
                        Some(d) => tokio::time::sleep_until(d).await,
                        None => std::future::pending::<()>().await,
                    }
                } => {
                    if matches!(
                        self.on_periodic_tick(&mut periodic_deadline, child, run_id).await,
                        PeriodicOutcome::Restarted,
                    ) {
                        return StartingOutcome::Break;
                    }
                }
                // Persistent services ignore the idle timeout entirely — by
                // definition they stay loaded until evicted or shut down.
                // Without this guard, a persistent service that respawns
                // without receiving traffic would idle-time-out on entry to
                // Running (its `last_activity` stamp is stale), drain to
                // Idle, then get re-ensured by `persistent_watcher` in an
                // endless ~15 s loop.
                _ = tokio::time::sleep_until(idle_deadline_for(&self.init.last_activity, self.current_svc().idle_timeout_ms)), if self.current_svc().lifecycle != crate::config::Lifecycle::Persistent => {
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
                    self.end_run(run_id).await;
                    self.record_drain_complete();
                    self.set_state(ServiceState::Idle);
                    return StartingOutcome::Break;
                }
                cmd = self.rx.recv() => {
                    match self.on_running_command(cmd, child, run_id).await {
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
        self.drain_now_bounded(child, run_id, reason, None).await;
    }

    /// As [`Self::drain_now`], but with an optional override for how long the
    /// pipeline waits for in-flight requests to finish before SIGTERM. The
    /// stall watchdog passes a short bound: its whole premise is that the run
    /// is producing nothing, so the default `max_request_duration` (which can
    /// be many minutes) would just make the daemon wait on a request that will
    /// never complete.
    async fn drain_now_bounded(
        &mut self,
        child: &mut dyn ManagedChild,
        run_id: i64,
        reason: crate::supervise::drain::DrainReason,
        inflight_wait: Option<Duration>,
    ) {
        self.set_state(ServiceState::Draining);
        let current = self.current_svc();
        let cfg = DrainConfig {
            max_request_duration: inflight_wait
                .unwrap_or_else(|| Duration::from_millis(current.max_request_duration_ms)),
            drain_timeout: Duration::from_millis(current.drain_timeout_ms),
            extended_stream_drain: Duration::from_millis(current.extended_stream_drain_ms),
            sigterm_grace: RUNNING_SIGTERM_GRACE,
        };
        drain_pipeline(child, &cfg, self.init.inflight.clone(), reason).await;
        self.run_shutdown_command().await;
        self.end_run(run_id).await;
        self.record_drain_complete();
    }

    /// Query the current run's recent error rate and decide whether the
    /// error-rate watchdog should fire. Returns a human-readable detail
    /// string when it should, else `None`.
    ///
    /// Gated on the `min_uptime` cooldown: a freshly (re)started run must
    /// live that long before the watchdog can fire. That, together with the
    /// run-scoped, windowed query — which starts from zero metrics on every
    /// respawn — is what stops restart flapping.
    async fn evaluate_error_rate(
        &self,
        run_id: i64,
        running_since: tokio::time::Instant,
    ) -> Option<String> {
        let ar = self.current_svc().auto_restart;
        let er = ar.error_rate.as_ref()?;
        if tokio::time::Instant::now().duration_since(running_since)
            < Duration::from_millis(ar.min_uptime_ms)
        {
            return None;
        }
        let since_ms = crate::tracking::now_unix_ms() - er.window_ms as i64;
        let (total, errors) = match self
            .deps
            .db
            .error_rate_since(
                self.init.service_id,
                run_id,
                since_ms,
                er.statuses.min_status_code(),
            )
            .await
        {
            Ok(v) => v,
            Err(e) => {
                warn!(service = %self.init.identity.name, error = %e, "auto-restart: error-rate query failed");
                return None;
            }
        };
        let rate = error_rate_trips(total, errors, er)?;
        Some(format!(
            "error rate {:.0}% ({errors}/{total} requests over {}s) ≥ threshold {:.0}%",
            rate * 100.0,
            er.window_ms / 1000,
            er.max_error_rate * 100.0,
        ))
    }

    /// Perform an error-rate-triggered restart. Thin wrapper over
    /// [`Self::perform_auto_restart`] that labels the trigger.
    async fn perform_error_rate_restart(
        &mut self,
        child: &mut dyn ManagedChild,
        run_id: i64,
        detail: String,
    ) -> AutoRestartOutcome {
        self.perform_auto_restart(child, run_id, "error_rate", detail)
            .await
    }

    /// Drain the current run and return to Idle for a watchdog-triggered
    /// restart, or disable the service if the flap cap has been reached.
    /// Either way the child is drained; the caller breaks out of the Running
    /// loop afterward. Shared by the error-rate and stall watchdogs — both
    /// count toward the same flap cap.
    async fn perform_auto_restart(
        &mut self,
        child: &mut dyn ManagedChild,
        run_id: i64,
        trigger: &'static str,
        detail: String,
    ) -> AutoRestartOutcome {
        let ar = self.current_svc().auto_restart;
        // A stall restart drains a run that is by definition producing nothing,
        // so waiting the full `max_request_duration` for its wedged in-flight
        // request to finish is pointless — bound the wait to a short grace.
        let inflight_wait = (trigger == "ttft_stall").then_some(STALL_DRAIN_INFLIGHT_WAIT);
        let now = tokio::time::Instant::now();
        let window = Duration::from_millis(ar.flap_window_ms);
        self.auto_restart_history
            .retain(|t| now.duration_since(*t) < window);
        if self.auto_restart_history.len() as u32 >= ar.max_restarts {
            warn!(
                service = %self.init.identity.name,
                restarts = self.auto_restart_history.len(),
                trigger,
                detail = %detail,
                "auto-restart flap cap reached; disabling instead of restarting"
            );
            self.drain_now_bounded(
                child,
                run_id,
                drain::DrainReason::AutoRestart,
                inflight_wait,
            )
            .await;
            self.set_state(ServiceState::Disabled {
                reason: DisableReason::AutoRestartLoop,
            });
            return AutoRestartOutcome::Disabled;
        }
        self.auto_restart_history.push(now);
        warn!(service = %self.init.identity.name, trigger, detail = %detail, "auto-restart: watchdog firing");
        self.emit_auto_restarted(trigger, detail);
        self.drain_now_bounded(
            child,
            run_id,
            drain::DrainReason::AutoRestart,
            inflight_wait,
        )
        .await;
        self.set_state(ServiceState::Idle);
        AutoRestartOutcome::Restarted
    }

    /// Evaluate the periodic timer when its deadline elapses. `deadline` is
    /// rewritten in place: cleared once the trigger has fired or handed off to
    /// the request path, or set to a short re-poll while `on-idle` waits for a
    /// quiet window.
    async fn on_periodic_tick(
        &mut self,
        deadline: &mut Option<tokio::time::Instant>,
        child: &mut dyn ManagedChild,
        run_id: i64,
    ) -> PeriodicOutcome {
        let ar = self.current_svc().auto_restart;
        let Some(periodic) = ar.periodic.as_ref() else {
            *deadline = None;
            return PeriodicOutcome::Continue;
        };
        match periodic.mode {
            PeriodicMode::Immediate => {
                self.perform_periodic_restart(child, run_id, "interval elapsed (immediate)".into())
                    .await;
                PeriodicOutcome::Restarted
            }
            PeriodicMode::OnRequest => {
                // Arm the flag and disarm the timer; the next Ensure drives the
                // drain → respawn (see `on_running_command`).
                self.restart_pending = true;
                *deadline = None;
                info!(
                    service = %self.init.identity.name,
                    "periodic interval elapsed; will restart on next request"
                );
                PeriodicOutcome::Continue
            }
            PeriodicMode::OnIdle => {
                if self
                    .init
                    .inflight
                    .load(std::sync::atomic::Ordering::Relaxed)
                    == 0
                {
                    self.perform_periodic_restart(
                        child,
                        run_id,
                        "interval elapsed (idle window)".into(),
                    )
                    .await;
                    PeriodicOutcome::Restarted
                } else {
                    // Still serving; re-check after a short poll.
                    *deadline = Some(tokio::time::Instant::now() + PERIODIC_IDLE_POLL);
                    PeriodicOutcome::Continue
                }
            }
        }
    }

    /// Drain the child and return to Idle for a periodic restart. Unlike the
    /// error-rate path, periodic restarts are intentional maintenance and do
    /// not count toward the flap cap.
    async fn perform_periodic_restart(
        &mut self,
        child: &mut dyn ManagedChild,
        run_id: i64,
        detail: String,
    ) {
        info!(service = %self.init.identity.name, detail = %detail, "auto-restart: periodic timer firing");
        self.emit_auto_restarted("periodic", detail);
        self.drain_now(child, run_id, drain::DrainReason::AutoRestart)
            .await;
        self.set_state(ServiceState::Idle);
    }

    /// Publish an [`Event::AutoRestarted`] to the daemon event stream.
    fn emit_auto_restarted(&self, trigger: &str, detail: String) {
        self.deps.events.publish(Event::AutoRestarted {
            service: self.init.identity.name.clone(),
            trigger: trigger.to_string(),
            detail,
            at_ms: crate::tracking::now_unix_ms(),
        });
    }

    /// Dispatch a command received while the service is Running.
    async fn on_running_command(
        &mut self,
        cmd: Option<SupervisorCommand>,
        child: &mut dyn ManagedChild,
        run_id: i64,
    ) -> RunningOutcome {
        match cmd {
            Some(SupervisorCommand::Shutdown { ack }) => {
                info!(service = %self.init.identity.name, "draining");
                let next = transition(&self.read_state(), StateEvent::DrainRequested);
                self.set_state(next);
                let _ = self.cancel_tx.send(true);
                drain::sigterm_then_sigkill(child, RUNNING_SIGTERM_GRACE).await;
                self.run_shutdown_command().await;
                self.end_run(run_id).await;
                self.record_drain_complete();
                let _ = ack.send(());
                RunningOutcome::Exit
            }
            Some(SupervisorCommand::Ensure { ack, source }) => {
                if self.restart_pending {
                    // Periodic on-request restart is armed: this request is the
                    // trigger. Drain now and carry its Ensure through to the
                    // idle-ensure spawn path so the caller blocks on the fresh
                    // process. Falls back to a plain restart if the child is
                    // already gone.
                    self.restart_pending = false;
                    info!(
                        service = %self.init.identity.name,
                        "periodic on-request restart: draining for incoming request"
                    );
                    self.deferred_ensure = Some((ack, source));
                    self.emit_auto_restarted("periodic", "on-request interval elapsed".into());
                    self.drain_now(child, run_id, drain::DrainReason::AutoRestart)
                        .await;
                    self.set_state(ServiceState::Idle);
                    return RunningOutcome::Break;
                }
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

                self.end_run(run_id).await;
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
            Some(SupervisorCommand::WatchdogStall {
                run_id: stalled_run,
            }) => {
                // Ignore a stall reported against a run that has already been
                // replaced — its restart, if any, has already happened.
                if stalled_run != run_id {
                    return RunningOutcome::Continue;
                }
                let timeout_ms = self
                    .current_svc()
                    .auto_restart
                    .ttft_stall
                    .map(|t| t.timeout_ms)
                    .unwrap_or(0);
                let detail = format!(
                    "no response frame for {:.0}s across the whole run (upstream stall)",
                    timeout_ms as f64 / 1000.0
                );
                match self
                    .perform_auto_restart(child, run_id, "ttft_stall", detail)
                    .await
                {
                    AutoRestartOutcome::Restarted | AutoRestartOutcome::Disabled => {
                        RunningOutcome::Break
                    }
                }
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
            Some(SupervisorCommand::Ensure { ack, .. }) => {
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
            // Not yet Running: any stall report is stale.
            Some(SupervisorCommand::WatchdogStall { .. }) => StartingOutcome::Continue,
            // Drain request while the child is still starting: abort the
            // spawn, release the allocation, and drop back to Idle. The
            // caller (retry_pack_with_eviction / try_eviction_to_fit /
            // ShutdownDrain) needs the VRAM; a no-op ack would pretend
            // it had been freed when it hasn't.
            Some(SupervisorCommand::BeginDrain { reason, ack }) => {
                info!(
                    service = %self.init.identity.name,
                    ?reason,
                    "BeginDrain while starting; aborting in-progress spawn"
                );
                let _ = self.cancel_tx.send(true);
                drain::sigterm_then_sigkill(child, STARTING_SIGTERM_GRACE).await;
                self.run_shutdown_command().await;
                self.end_run(run_id).await;
                self.deps
                    .allocations
                    .lock()
                    .remove(&self.init.identity.name);
                self.deps.observation.clear(&self.init.identity.name);
                self.emit_allocation_changed();
                if let Some(bus) = self.start_bus_carry.take() {
                    let _ = bus.send(StartOutcome::Err(StartFailure {
                        kind: StartFailureKind::LaunchFailed,
                        message: format!("start aborted by drain ({reason:?})"),
                    }));
                }
                self.set_state(ServiceState::Idle);
                let _ = ack.send(());
                StartingOutcome::Break
            }
            Some(SupervisorCommand::FastKill { reason, ack }) => {
                info!(
                    service = %self.init.identity.name,
                    ?reason,
                    "FastKill while starting; aborting in-progress spawn"
                );
                let _ = self.cancel_tx.send(true);
                drain::sigterm_then_sigkill(child, STARTING_SIGTERM_GRACE).await;
                self.run_shutdown_command().await;
                self.end_run(run_id).await;
                self.deps
                    .allocations
                    .lock()
                    .remove(&self.init.identity.name);
                self.deps.observation.clear(&self.init.identity.name);
                self.emit_allocation_changed();
                if let Some(bus) = self.start_bus_carry.take() {
                    let _ = bus.send(StartOutcome::Err(StartFailure {
                        kind: StartFailureKind::LaunchFailed,
                        message: format!("start fast-killed ({reason:?})"),
                    }));
                }
                self.set_state(ServiceState::Idle);
                let _ = ack.send(());
                StartingOutcome::Break
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
                self.end_run(run_id).await;
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
                let next = transition(&self.read_state(), StateEvent::RetryAfterBackoff);
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
                        Step::Exit
                    }
                    Some(SupervisorCommand::Ensure { ack, .. }) => {
                        // Surface a meaningful failure instead of dropping
                        // the ack: the client would otherwise see
                        // "supervisor unreachable", which doesn't tell them
                        // the service is in retry backoff.
                        let _ = ack.send(EnsureResponse::Unavailable(
                            EnsureFailure::StartFailed(format!(
                                "service {} is in Failed state; awaiting retry backoff",
                                self.init.identity.name
                            )),
                        ));
                        Step::Continue
                    }
                    // Failed means there's no child and no allocation to
                    // release, so drain/kill are instant no-ops but must
                    // still ack so the caller's `begin_drain.await` returns.
                    Some(SupervisorCommand::BeginDrain { ack, .. })
                    | Some(SupervisorCommand::FastKill { ack, .. }) => {
                        let _ = ack.send(());
                        Step::Continue
                    }
                    Some(SupervisorCommand::ActivityPing) => Step::Continue,
                    // No running child; a stall report is stale.
                    Some(SupervisorCommand::WatchdogStall { .. }) => Step::Continue,
                    Some(SupervisorCommand::Enable { ack }) => {
                        // Failed is not disabled; enable is a no-op.
                        let _ = ack.send(EnableResult::NotDisabled);
                        Step::Continue
                    }
                    Some(SupervisorCommand::Disable { ack }) => {
                        // Disable a failed service: skip the retry and go to Disabled.
                        self.set_state(ServiceState::Disabled {
                            reason: DisableReason::UserDisabled,
                        });
                        let _ = ack.send(DisableResult::Disabled);
                        Step::Continue
                    }
                    None => Step::Exit,
                }
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
                Some(SupervisorCommand::Ensure { ack, .. }) => {
                    let _ = ack.send(EnsureResponse::Unavailable(EnsureFailure::ServiceDisabled(
                        "service disabled".into(),
                    )));
                }
                Some(SupervisorCommand::ActivityPing) => {}
                // Disabled: no running child, so a stall report is stale.
                Some(SupervisorCommand::WatchdogStall { .. }) => {}
                // Service is disabled; drain/kill are no-ops.
                Some(SupervisorCommand::BeginDrain { ack, .. }) => {
                    let _ = ack.send(());
                }
                Some(SupervisorCommand::FastKill { ack, .. }) => {
                    let _ = ack.send(());
                }
                Some(SupervisorCommand::Enable { ack }) => {
                    // Transition back to Idle so the next Ensure can start it.
                    // Clear the auto-restart flap history: a manual re-enable is
                    // an operator override that grants a fresh restart budget.
                    // Without this, a service disabled with `AutoRestartLoop`
                    // (whose history is full by construction, all within the
                    // flap window) would be re-disabled on its very first
                    // error-rate trip after re-enable.
                    self.auto_restart_history.clear();
                    let next = transition(&self.read_state(), StateEvent::UserEnable);
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

/// Outcome of a single command dispatch inside the Running inner loop.
enum RunningOutcome {
    Continue,
    Break,
    Exit,
}

/// Outcome of an error-rate watchdog firing: the service was either drained
/// and returned to Idle for respawn, or disabled after tripping the flap cap.
/// Both leave the Running loop.
enum AutoRestartOutcome {
    Restarted,
    Disabled,
}

/// Outcome of a periodic-timer tick: either the loop keeps running (the
/// trigger armed the request path, or `on-idle` is still waiting for a quiet
/// window) or the service was drained and must leave the Running loop.
enum PeriodicOutcome {
    Continue,
    Restarted,
}

/// Re-poll cadence for the `on-idle` periodic mode while it waits for the
/// service to fall quiet after its interval has elapsed.
const PERIODIC_IDLE_POLL: Duration = Duration::from_secs(1);

/// Pure verdict for the error-rate watchdog: `Some(rate)` when the window's
/// error ratio meets or exceeds the threshold with enough requests to trust
/// it, else `None`. Kept free of I/O so it can be unit-tested directly.
fn error_rate_trips(total: u64, errors: u64, cfg: &ErrorRateTrigger) -> Option<f64> {
    if total < cfg.min_requests as u64 {
        return None;
    }
    let rate = errors as f64 / total as f64;
    (rate >= cfg.max_error_rate).then_some(rate)
}

/// Convert a `DeviceSlot` to the canonical string key used in
/// `AllocationChanged` reservations (`"cpu"` or `"gpu:N"`).
pub(crate) fn slot_to_key(slot: &crate::config::DeviceSlot) -> String {
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

#[cfg(test)]
mod auto_restart_tests {
    use super::*;
    use crate::config::validate::ErrorStatusClass;

    fn cfg(max_error_rate: f64, min_requests: u32) -> ErrorRateTrigger {
        ErrorRateTrigger {
            window_ms: 120_000,
            max_error_rate,
            min_requests,
            poll_interval_ms: 30_000,
            statuses: ErrorStatusClass::ServerOnly,
        }
    }

    #[test]
    fn trips_when_rate_meets_threshold_with_enough_requests() {
        // 24/42 ≈ 57% ≥ 50% with 42 ≥ 20 requests — the production wedge shape.
        let rate = error_rate_trips(42, 24, &cfg(0.5, 20)).expect("should trip");
        assert!((rate - 24.0 / 42.0).abs() < 1e-9);
    }

    #[test]
    fn does_not_trip_below_min_requests() {
        // 2/2 = 100% but only 2 requests — the floor must suppress it.
        assert!(error_rate_trips(2, 2, &cfg(0.5, 20)).is_none());
    }

    #[test]
    fn does_not_trip_below_threshold() {
        // 5/40 = 12.5% < 50%.
        assert!(error_rate_trips(40, 5, &cfg(0.5, 20)).is_none());
    }

    #[test]
    fn trips_exactly_at_threshold() {
        // Boundary: exactly 50% counts as tripping (>=).
        assert!(error_rate_trips(20, 10, &cfg(0.5, 20)).is_some());
    }

    #[test]
    fn status_class_error_boundaries() {
        assert!(!ErrorStatusClass::ServerOnly.is_error(499));
        assert!(ErrorStatusClass::ServerOnly.is_error(500));
        assert!(!ErrorStatusClass::ServerOnly.is_error(400));
        assert!(ErrorStatusClass::ClientAndServer.is_error(400));
        assert!(ErrorStatusClass::ClientAndServer.is_error(503));
        assert_eq!(ErrorStatusClass::ServerOnly.min_status_code(), 500);
        assert_eq!(ErrorStatusClass::ClientAndServer.min_status_code(), 400);
    }
}
