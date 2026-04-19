//! Service supervision: per-service tokio tasks, child lifetimes, health loops.
//!
//! Linux-coupled via `os::unix::process::ExitStatusExt` (signal() on ExitStatus)
//! and the submodules it delegates to (`drain`, `orphans`, `spawn`).

pub mod drain;
pub mod health;
pub mod logs;
pub mod orphans;
pub mod registry;
pub mod spawn;
pub mod state;

use std::{
    os::unix::process::ExitStatusExt,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

pub use orphans::{OrphanDisposition, reconcile};
use parking_lot::Mutex as SyncMutex;
pub use spawn::{SpawnConfig, render_argv};
use tokio::{
    process::Child,
    sync::{broadcast, mpsc, watch},
    task::JoinHandle,
};
use tracing::{error, info, warn};

use crate::{
    allocator::placement::Packed,
    config::validate::{EffectiveConfig, ServiceConfig},
    db::{Database, logs::BatcherHandle},
    devices::Allocation,
    supervise::{
        drain::{DrainConfig, drain_pipeline, fast_kill},
        health::{HealthConfig, HealthOutcome, wait_healthy},
        logs::{spawn_pump_stderr, spawn_pump_stdout},
        registry::ServiceRegistry,
        spawn::spawn_child,
        state::{DisableReason, Event as StateEvent, ServiceState, transition},
    },
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
    /// Service is disabled or stopped; cannot start.
    Unavailable { reason: String },
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
}

/// Daemon-wide shared state every supervisor borrows. Cloning it is cheap
/// (every field is `Arc`-backed).
#[derive(Clone)]
pub struct SupervisorDeps {
    pub db: Database,
    pub batcher: BatcherHandle,
    pub snapshot: crate::devices::snapshotter::SharedSnapshot,
    pub allocations: Arc<parking_lot::Mutex<crate::allocator::AllocationTable>>,
    pub rolling: RollingTable,
    pub observation: ObservationTable,
    pub registry: ServiceRegistry,
    pub effective: Arc<EffectiveConfig>,
}

/// Per-service initialisation for a single supervisor task.
pub struct SupervisorInit {
    pub svc: ServiceConfig,
    pub allocation: Allocation,
    pub service_id: i64,
    pub last_activity: Arc<AtomicU64>,
    pub inflight: Arc<AtomicU64>,
}

pub fn spawn_supervisor(init: SupervisorInit, deps: SupervisorDeps) -> SupervisorHandle {
    let (tx, rx) = mpsc::channel(SUPERVISOR_COMMAND_MAILBOX);
    let name = init.svc.name.clone();
    let join = tokio::spawn(run(init, deps, rx));
    SupervisorHandle {
        name,
        tx,
        join: tokio::sync::Mutex::new(Some(join)),
    }
}

/// Mailbox depth for per-supervisor command channels.
const SUPERVISOR_COMMAND_MAILBOX: usize = 32;

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

async fn run(init: SupervisorInit, deps: SupervisorDeps, rx: mpsc::Receiver<SupervisorCommand>) {
    let mut loop_state = RunLoop::new(init, deps, rx);
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
}

impl RunLoop {
    fn new(
        init: SupervisorInit,
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
        }
    }

    fn set_state(&mut self, new_state: ServiceState) {
        self.state = new_state;
        *self.state_mirror.lock() = self.state.clone();
    }

    /// Remove this service's reservation, clear observation, and finalise the
    /// rolling correction. Used when the child exits or is drained back to
    /// Idle. Logs nothing itself; callers emit the tracing event that fits
    /// their context.
    fn record_drain_complete(&mut self) {
        self.deps.rolling.update(
            &self.init.svc.name,
            self.deps.observation.read_peak(&self.init.svc.name),
            self.base_total_bytes_for_rolling,
        );
        self.deps.observation.clear(&self.init.svc.name);
        self.deps.allocations.lock().remove(&self.init.svc.name);
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
                        name: self.init.svc.name.clone(),
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

        let want = match self.compute_reservation_map(&snap, &table) {
            Ok(w) => w,
            Err(reason) => {
                let _ = ack.send(EnsureResponse::Unavailable { reason });
                return false;
            }
        };

        if let Err(nofit) =
            crate::allocator::can_fit(&want, &snap, &table, Some(&self.init.svc.name))
            && let Err(reason) = self.try_eviction_to_fit(&want, &nofit).await
        {
            let _ = ack.send(EnsureResponse::Unavailable { reason });
            return false;
        }

        // Reserve in the allocation table before spawning. Capture the total
        // reserved bytes (MB → bytes) for the rolling update that fires when
        // the service later drains back to Idle.
        self.base_total_bytes_for_rolling = want.values().sum::<u64>() * 1024 * 1024;
        self.deps
            .allocations
            .lock()
            .insert(self.init.svc.name.clone(), want);

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
    /// eventual `render_argv` call. The returned `Err(String)` is the reason
    /// an `EnsureResponse::Unavailable` should carry.
    fn compute_reservation_map(
        &mut self,
        snap: &crate::devices::DeviceSnapshot,
        table: &crate::allocator::AllocationTable,
    ) -> Result<std::collections::BTreeMap<crate::config::DeviceSlot, u64>, String> {
        let svc = &self.init.svc;
        if matches!(svc.template, crate::config::Template::Command) {
            self.packed_for_spawn = None;
            let bytes_mb = match svc.allocation_mode {
                crate::config::AllocationMode::Static { vram_mb } => vram_mb,
                crate::config::AllocationMode::Dynamic { min_mb, .. } => min_mb,
                crate::config::AllocationMode::None => 0,
            };
            let target_gpu: Option<u32> = svc
                .raw
                .devices
                .as_ref()
                .and_then(|d| d.gpu_allow.as_ref())
                .and_then(|list| list.first().copied())
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
        let model_path = svc
            .raw
            .model
            .as_ref()
            .ok_or_else(|| "no model path configured".to_string())?
            .clone();
        let mut est = crate::estimator::estimate_from_path(&model_path, svc)
            .map_err(|e| format!("estimator: {e}"))?;
        // Apply rolling correction to weights_bytes.
        let rc = self.deps.rolling.get(&svc.name);
        est.weights_bytes = (est.weights_bytes as f64 * rc.rolling_mean) as u64;

        let packed = crate::allocator::placement::pack(&est, svc, snap, table)
            .map_err(|e| format!("placement: {e}"))?;
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
            self.init.svc.priority,
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

        warn!(service = %self.init.svc.name, evict_count = to_evict.len(), "eviction planned to make room");
        for victim in &to_evict {
            if let Some(handle) = self.deps.registry.get(victim) {
                handle
                    .begin_drain(crate::supervise::drain::DrainReason::Eviction)
                    .await;
            }
        }

        // Re-attempt can_fit after evictions.
        let snap2 = self.deps.snapshot.read().clone();
        let table2 = self.deps.allocations.lock().clone();
        if let Err(again) =
            crate::allocator::can_fit(want, &snap2, &table2, Some(&self.init.svc.name))
        {
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
        let mut out = Vec::new();
        for (_name, handle) in all_services {
            if handle.name.as_str() == self.init.svc.name.as_str() {
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
            let priority = self
                .deps
                .effective
                .services
                .iter()
                .find(|s| s.name == handle.name)
                .map(|c| c.priority)
                .unwrap_or(50);
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
        let spawn_cfg = render_argv(
            &self.init.svc,
            &self.init.allocation,
            self.packed_for_spawn.as_ref().map(|p| &p.args),
        );
        let cmdline = format!("{} {}", spawn_cfg.binary, spawn_cfg.args.join(" "));
        let mut child = match spawn_child(&spawn_cfg).await {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "spawn failed");
                if let Some(bus) = self.start_bus_carry.take() {
                    let _ = bus.send(StartOutcome::Err(StartFailure {
                        kind: StartFailureKind::LaunchFailed,
                        message: format!("{e}"),
                    }));
                }
                self.deps.allocations.lock().remove(&self.init.svc.name);
                self.set_state(ServiceState::Failed { retry_count: 0 });
                return Step::Continue;
            }
        };

        let pid = child.id().unwrap_or(0) as i32;
        self.deps
            .observation
            .register(&self.init.svc.name, pid as u32);
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

        if let Some(stdout) = child.stdout.take() {
            spawn_pump_stdout(
                stdout,
                self.init.service_id,
                run_id,
                self.deps.batcher.clone(),
            );
        }
        if let Some(stderr) = child.stderr.take() {
            spawn_pump_stderr(
                stderr,
                self.init.service_id,
                run_id,
                self.deps.batcher.clone(),
            );
        }

        let health_cfg = HealthConfig {
            url: format!(
                "http://127.0.0.1:{}{}",
                self.init.svc.private_port, self.init.svc.health.http_path
            ),
            probe_interval: Duration::from_millis(self.init.svc.health.probe_interval_ms),
            timeout: Duration::from_millis(self.init.svc.health.timeout_ms),
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
                    match self.on_health_outcome(outcome, &mut child, run_id).await {
                        StartingOutcome::Continue => {}
                        StartingOutcome::Break => break,
                        StartingOutcome::Exit => return Step::Exit,
                    }
                }
                cmd = self.rx.recv() => {
                    match self.on_starting_command(cmd, &mut child).await {
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
        self.deps.allocations.lock().remove(&self.init.svc.name);
        self.deps.observation.clear(&self.init.svc.name);

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
                warn!(service = %self.init.svc.name, attempts = self.oom_attempts, "OOM retry limit reached; disabling");
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
                warn!(service = %self.init.svc.name, "OOM kill detected; bumping rolling factor for retry");
                self.deps.rolling.bump_for_oom_retry(&self.init.svc.name);
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
        child: &mut Child,
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
                warn!(service = %self.init.svc.name, "health timed out; disabling");
                if let Some(bus) = self.start_bus_carry.take() {
                    let _ = bus.send(StartOutcome::Err(StartFailure {
                        kind: StartFailureKind::HealthTimeout,
                        message: "health check timed out".into(),
                    }));
                }
                self.deps.allocations.lock().remove(&self.init.svc.name);
                self.set_state(ServiceState::Disabled {
                    reason: DisableReason::HealthTimeout,
                });
                drain::sigterm_then_sigkill(child, STARTING_SIGTERM_GRACE).await;
                StartingOutcome::Break
            }
            Ok(HealthOutcome::Cancelled) | Err(_) => {
                self.deps.allocations.lock().remove(&self.init.svc.name);
                drain::sigterm_then_sigkill(child, STARTING_SIGTERM_GRACE).await;
                StartingOutcome::Exit
            }
        }
    }

    /// Warming grace: sleep `warming_grace_ms` while also watching for child
    /// exit and Shutdown.
    async fn run_warming_grace(&mut self, child: &mut Child, run_id: i64) -> WarmingOutcome {
        let grace = Duration::from_millis(self.init.svc.warming_grace_ms);
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
                self.deps.allocations.lock().remove(&self.init.svc.name);
                self.set_state(ServiceState::Failed { retry_count: 0 });
                WarmingOutcome::ChildExited
            }
            cmd = self.rx.recv() => {
                if let Some(SupervisorCommand::Shutdown { ack }) = cmd {
                    info!(service = %self.init.svc.name, "draining during warming");
                    let _ = self.cancel_tx.send(true);
                    drain::sigterm_then_sigkill(child, RUNNING_SIGTERM_GRACE).await;
                    delete_running_row(&self.deps.db, self.init.service_id, run_id).await;
                    self.deps.allocations.lock().remove(&self.init.svc.name);
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
    async fn run_running_loop(&mut self, child: &mut Child, run_id: i64) -> StartingOutcome {
        let pid = child.id().unwrap_or(0) as i32;
        loop {
            tokio::select! {
                exit = child.wait() => {
                    warn!(?exit, "child exited from running");
                    self.record_drain_complete();
                    self.set_state(ServiceState::Failed { retry_count: 0 });
                    return StartingOutcome::Break;
                }
                _ = tokio::time::sleep_until(idle_deadline_for(&self.init.last_activity, self.init.svc.idle_timeout_ms)) => {
                    // Re-check the atomic; a recent ping may have extended the deadline.
                    let now = crate::tracking::now_unix_ms_u64();
                    let last = self.init.last_activity.load(Ordering::Relaxed);
                    if now + IDLE_DEADLINE_SKEW_MS < last + self.init.svc.idle_timeout_ms {
                        // A ping arrived; loop again with a fresh deadline.
                        continue;
                    }
                    info!(service = %self.init.svc.name, "idle timeout; draining to idle");
                    drain::sigterm_then_sigkill(child, RUNNING_SIGTERM_GRACE).await;
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

    /// Dispatch a command received while the service is Running.
    async fn on_running_command(
        &mut self,
        cmd: Option<SupervisorCommand>,
        child: &mut Child,
        run_id: i64,
        pid: i32,
    ) -> RunningOutcome {
        match cmd {
            Some(SupervisorCommand::Shutdown { ack }) => {
                info!(service = %self.init.svc.name, "draining");
                let next = transition(&self.state, StateEvent::DrainRequested);
                self.set_state(next);
                let _ = self.cancel_tx.send(true);
                drain::sigterm_then_sigkill(child, RUNNING_SIGTERM_GRACE).await;
                delete_running_row(&self.deps.db, self.init.service_id, run_id).await;
                self.record_drain_complete();
                let _ = ack.send(());
                RunningOutcome::Exit
            }
            Some(SupervisorCommand::Snapshot { ack }) => {
                let _ = ack.send(SupervisorSnapshot {
                    name: self.init.svc.name.clone(),
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
                self.init
                    .last_activity
                    .store(crate::tracking::now_unix_ms_u64(), Ordering::Relaxed);
                RunningOutcome::Continue
            }
            Some(SupervisorCommand::BeginDrain { reason, ack }) => {
                info!(service = %self.init.svc.name, ?reason, "BeginDrain received; draining");
                self.set_state(ServiceState::Draining);

                let cfg = DrainConfig {
                    max_request_duration: Duration::from_millis(
                        self.init.svc.max_request_duration_ms,
                    ),
                    drain_timeout: Duration::from_millis(self.init.svc.drain_timeout_ms),
                    extended_stream_drain: Duration::from_millis(
                        self.init.svc.extended_stream_drain_ms,
                    ),
                    sigterm_grace: RUNNING_SIGTERM_GRACE,
                };
                drain_pipeline(child, &cfg, self.init.inflight.clone(), reason).await;

                delete_running_row(&self.deps.db, self.init.service_id, run_id).await;
                self.record_drain_complete();

                let _ = ack.send(());
                self.set_state(ServiceState::Idle);
                RunningOutcome::Break
            }
            Some(SupervisorCommand::FastKill { reason, ack }) => {
                info!(service = %self.init.svc.name, ?reason, "FastKill received");
                self.set_state(ServiceState::Draining);

                fast_kill(child, reason).await;

                delete_running_row(&self.deps.db, self.init.service_id, run_id).await;
                self.deps.allocations.lock().remove(&self.init.svc.name);
                self.deps.observation.clear(&self.init.svc.name);
                let _ = ack.send(());
                self.set_state(ServiceState::Idle);
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
        child: &mut Child,
    ) -> StartingOutcome {
        let pid = child.id().unwrap_or(0) as i32;
        match cmd {
            Some(SupervisorCommand::Shutdown { ack }) => {
                let _ = self.cancel_tx.send(true);
                drain::sigterm_then_sigkill(child, STARTING_SIGTERM_GRACE).await;
                self.deps.allocations.lock().remove(&self.init.svc.name);
                let _ = ack.send(());
                StartingOutcome::Exit
            }
            Some(SupervisorCommand::Snapshot { ack }) => {
                let _ = ack.send(SupervisorSnapshot {
                    name: self.init.svc.name.clone(),
                    state: self.state.clone(),
                    run_id: None,
                    pid: Some(pid),
                });
                StartingOutcome::Continue
            }
            Some(SupervisorCommand::Ensure { ack }) => {
                // Already in Starting; subscribe to existing bus or report running.
                if let Some(sender) = self.start_bus_carry.as_ref() {
                    if sender.receiver_count() >= self.init.svc.raw.start_queue_depth() {
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
                if let Some(SupervisorCommand::Shutdown { ack }) = cmd {
                    let _ = ack.send(());
                    return Step::Exit;
                }
                Step::Continue
            }
        }
    }

    async fn handle_disabled(&mut self) -> Step {
        info!(service = %self.init.svc.name, "disabled; awaiting shutdown or enable");
        loop {
            match self.rx.recv().await {
                Some(SupervisorCommand::Shutdown { ack }) => {
                    let _ = ack.send(());
                    return Step::Exit;
                }
                Some(SupervisorCommand::Snapshot { ack }) => {
                    let _ = ack.send(SupervisorSnapshot {
                        name: self.init.svc.name.clone(),
                        state: self.state.clone(),
                        run_id: None,
                        pid: None,
                    });
                }
                Some(SupervisorCommand::Ensure { ack }) => {
                    let _ = ack.send(EnsureResponse::Unavailable {
                        reason: "service disabled".into(),
                    });
                }
                Some(SupervisorCommand::ActivityPing) => {}
                // Service is disabled; drain/kill are no-ops.
                Some(SupervisorCommand::BeginDrain { ack, .. }) => {
                    let _ = ack.send(());
                }
                Some(SupervisorCommand::FastKill { ack, .. }) => {
                    let _ = ack.send(());
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

/// Compute the tokio `Instant` at which the idle deadline fires, based on the
/// last recorded activity timestamp.
fn idle_deadline_for(last_activity: &Arc<AtomicU64>, timeout_ms: u64) -> tokio::time::Instant {
    let now = crate::tracking::now_unix_ms_u64();
    let last = last_activity.load(Ordering::Relaxed);
    let deadline_ms_from_now = (last + timeout_ms).saturating_sub(now);
    tokio::time::Instant::now() + Duration::from_millis(deadline_ms_from_now)
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

    let mut handle = db.handle();
    if let Err(e) = toasty::create!(RunningService {
        service_id,
        run_id,
        pid,
        spawned_at: crate::tracking::now_unix_ms(),
        command_line,
        allocation,
        state: "starting".to_string(),
    })
    .exec(&mut handle)
    .await
    {
        warn!(error = %e, "running_services insert failed");
    }
}

/// Delete the `running_services` row for `(service_id, run_id)` if present.
async fn delete_running_row(db: &Database, service_id: i64, run_id: i64) {
    use crate::db::models::RunningService;

    let mut handle = db.handle();
    let filter = RunningService::fields()
        .service_id()
        .eq(service_id)
        .and(RunningService::fields().run_id().eq(run_id));
    if let Err(e) = RunningService::filter(filter)
        .delete()
        .exec(&mut handle)
        .await
    {
        warn!(error = %e, "running_services delete failed");
    }
}
