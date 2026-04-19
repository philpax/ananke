//! Service supervision: per-service tokio tasks, child lifetimes, health loops.

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
    sync::{mpsc, watch},
    task::JoinHandle,
};
use tracing::{error, info, warn};

use crate::{
    config::validate::{EffectiveConfig, ServiceConfig},
    db::{Database, logs::BatcherHandle},
    devices::Allocation,
    supervise::{
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

#[allow(clippy::too_many_arguments)]
pub fn spawn_supervisor(
    svc: ServiceConfig,
    allocation: Allocation,
    db: Database,
    batcher: BatcherHandle,
    service_id: i64,
    last_activity: Arc<AtomicU64>,
    snapshot: crate::devices::snapshotter::SharedSnapshot,
    allocations: Arc<parking_lot::Mutex<crate::allocator::AllocationTable>>,
    rolling: RollingTable,
    observation: ObservationTable,
    inflight: Arc<AtomicU64>,
    registry: ServiceRegistry,
    effective: Arc<EffectiveConfig>,
) -> SupervisorHandle {
    let (tx, rx) = mpsc::channel(32);
    let name = svc.name.clone();
    let join = tokio::spawn(run(
        svc,
        allocation,
        db,
        batcher,
        service_id,
        last_activity,
        snapshot,
        allocations,
        rolling,
        observation,
        inflight,
        registry,
        effective,
        rx,
    ));
    SupervisorHandle {
        name,
        tx,
        join: tokio::sync::Mutex::new(Some(join)),
    }
}

#[allow(clippy::too_many_arguments)]
async fn run(
    svc: ServiceConfig,
    allocation: Allocation,
    db: Database,
    batcher: BatcherHandle,
    service_id: i64,
    last_activity: Arc<AtomicU64>,
    snapshot: crate::devices::snapshotter::SharedSnapshot,
    allocations: Arc<parking_lot::Mutex<crate::allocator::AllocationTable>>,
    rolling: RollingTable,
    observation: ObservationTable,
    inflight: Arc<AtomicU64>,
    registry: ServiceRegistry,
    effective: Arc<EffectiveConfig>,
    mut rx: mpsc::Receiver<SupervisorCommand>,
) {
    let mut state = ServiceState::Idle;
    let state_mirror = Arc::new(SyncMutex::new(state.clone()));
    let (cancel_tx, cancel_rx) = watch::channel(false);
    // Carries a broadcast sender from Idle through to Starting so waiters can
    // be notified of the outcome once the child finishes warming.
    let mut start_bus_carry: Option<tokio::sync::broadcast::Sender<StartOutcome>> = None;
    // Carries the placement-derived CommandArgs from Idle (where they are
    // computed) into Starting (where render_argv consumes them).
    let mut packed_for_spawn: Option<crate::allocator::placement::Packed> = None;
    // OOM retry state: counts consecutive OOM kills for the current service.
    let mut oom_attempts: u32 = 0;
    // Stash the total reserved bytes (in bytes) from the Ensure path so the
    // rolling update on drain can use the original estimate as the base.
    let mut base_total_bytes_for_rolling: u64 = 0;

    loop {
        match &state {
            ServiceState::Idle => {
                // For on_demand services we wait here for an Ensure. Persistent
                // services have the daemon call ensure() synthetically at boot.
                loop {
                    let cmd = rx.recv().await;
                    match cmd {
                        Some(SupervisorCommand::Shutdown { ack }) => {
                            let _ = ack.send(());
                            return;
                        }
                        Some(SupervisorCommand::Snapshot { ack }) => {
                            let _ = ack.send(SupervisorSnapshot {
                                name: svc.name.clone(),
                                state: state.clone(),
                                run_id: None,
                                pid: None,
                            });
                        }
                        Some(SupervisorCommand::Ensure { ack }) => {
                            let snap = snapshot.read().clone();
                            let table = allocations.lock().clone();

                            // Determine the allocation to reserve. Command-template
                            // services use their declared `allocation.{static|dynamic}`
                            // sizing; llama-cpp services use `placement_override` as
                            // the escape hatch or fall back to the GGUF estimator +
                            // layer-aware packer.
                            let want = if matches!(svc.template, crate::config::Template::Command) {
                                packed_for_spawn = None;
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
                                        crate::config::PlacementPolicy::CpuOnly => {
                                            crate::config::DeviceSlot::Cpu
                                        }
                                        _ => match target_gpu {
                                            Some(id) => crate::config::DeviceSlot::Gpu(id),
                                            None => crate::config::DeviceSlot::Cpu,
                                        },
                                    };
                                    map.insert(slot, bytes_mb);
                                }
                                map
                            } else if !svc.placement_override.is_empty() {
                                packed_for_spawn = None;
                                svc.placement_override.clone()
                            } else {
                                // Attempt estimator + placement; fall back to unavailable on error.
                                let model_path = match svc.raw.model.as_ref() {
                                    Some(p) => p.clone(),
                                    None => {
                                        let _ = ack.send(EnsureResponse::Unavailable {
                                            reason: "no model path configured".into(),
                                        });
                                        continue;
                                    }
                                };
                                let mut est =
                                    match crate::estimator::estimate_from_path(&model_path, &svc) {
                                        Ok(e) => e,
                                        Err(e) => {
                                            let _ = ack.send(EnsureResponse::Unavailable {
                                                reason: format!("estimator: {e}"),
                                            });
                                            continue;
                                        }
                                    };
                                // Apply rolling correction to weights_bytes.
                                let rc = rolling.get(&svc.name);
                                est.weights_bytes =
                                    (est.weights_bytes as f64 * rc.rolling_mean) as u64;

                                let packed = match crate::allocator::placement::pack(
                                    &est, &svc, &snap, &table,
                                ) {
                                    Ok(p) => p,
                                    Err(e) => {
                                        let _ = ack.send(EnsureResponse::Unavailable {
                                            reason: format!("placement: {e}"),
                                        });
                                        continue;
                                    }
                                };
                                // Convert Allocation bytes (per-DeviceId, in bytes) to the
                                // BTreeMap<DeviceSlot, u64> in MB that can_fit + insert expects.
                                let want_mb: std::collections::BTreeMap<
                                    crate::config::DeviceSlot,
                                    u64,
                                > = packed
                                    .allocation
                                    .bytes
                                    .iter()
                                    .map(|(id, bytes)| {
                                        let slot = match id {
                                            crate::devices::DeviceId::Cpu => {
                                                crate::config::DeviceSlot::Cpu
                                            }
                                            crate::devices::DeviceId::Gpu(n) => {
                                                crate::config::DeviceSlot::Gpu(*n)
                                            }
                                        };
                                        // Convert bytes → MB, rounding up so we never under-reserve.
                                        let mb = bytes.div_ceil(1024 * 1024);
                                        (slot, mb)
                                    })
                                    .collect();
                                packed_for_spawn = Some(packed);
                                want_mb
                            };

                            if let Err(nofit) =
                                crate::allocator::can_fit(&want, &snap, &table, Some(&svc.name))
                            {
                                // Try eviction before giving up.
                                let candidates: Vec<crate::allocator::eviction::EvictionCandidate> = {
                                    let all_services = registry.all();
                                    let mut out = Vec::new();
                                    for (_name, handle) in all_services {
                                        // Skip self: we're currently inside this supervisor's
                                        // own run loop, so awaiting our own Snapshot would
                                        // deadlock.
                                        if handle.name.as_str() == svc.name.as_str() {
                                            continue;
                                        }
                                        let Some(service_snap) = handle.snapshot().await else {
                                            continue;
                                        };
                                        let idle = matches!(service_snap.state, ServiceState::Idle);
                                        let alloc_mb = allocations
                                            .lock()
                                            .get(&handle.name)
                                            .cloned()
                                            .unwrap_or_default();
                                        let bytes = alloc_mb.values().sum::<u64>() * 1024 * 1024;
                                        let priority = effective
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
                                };

                                let reservations_now = allocations.lock().clone();
                                let free_on_slot = snap.free_bytes(&nofit.slot).unwrap_or(0);
                                let to_evict = crate::allocator::eviction::select_for_slot(
                                    nofit.needed_bytes,
                                    &nofit.slot,
                                    svc.priority,
                                    &candidates,
                                    &reservations_now,
                                    free_on_slot,
                                );

                                if to_evict.is_empty() {
                                    // Clear any computed packed args so they are not used on the
                                    // next Ensure attempt after this failure.
                                    let _ = packed_for_spawn.take();
                                    let _ = ack.send(EnsureResponse::Unavailable {
                                        reason: format!("{nofit}"),
                                    });
                                    continue;
                                }

                                warn!(service = %svc.name, evict_count = to_evict.len(), "eviction planned to make room");
                                for victim in &to_evict {
                                    if let Some(handle) = registry.get(victim) {
                                        handle
                                            .begin_drain(
                                                crate::supervise::drain::DrainReason::Eviction,
                                            )
                                            .await;
                                    }
                                }

                                // Re-attempt can_fit after evictions.
                                let snap2 = snapshot.read().clone();
                                let table2 = allocations.lock().clone();
                                if let Err(again) = crate::allocator::can_fit(
                                    &want,
                                    &snap2,
                                    &table2,
                                    Some(&svc.name),
                                ) {
                                    let _ = packed_for_spawn.take();
                                    let _ = ack.send(EnsureResponse::Unavailable {
                                        reason: format!("eviction insufficient: {again}"),
                                    });
                                    continue;
                                }
                                // Fall through to the normal reservation path.
                            }

                            // Reserve in the allocation table before spawning.
                            // Capture the total reserved bytes (MB → bytes) for the rolling
                            // update that fires when the service later drains back to Idle.
                            base_total_bytes_for_rolling = want.values().sum::<u64>() * 1024 * 1024;
                            allocations.lock().insert(svc.name.clone(), want);

                            // Create broadcast channel and subscribe the caller.
                            let sender = tokio::sync::broadcast::channel::<StartOutcome>(16).0;
                            let bus_rx = sender.subscribe();
                            let _ = ack.send(EnsureResponse::Waiting { rx: bus_rx });
                            start_bus_carry = Some(sender);

                            state = transition(&state, StateEvent::SpawnRequested).unwrap();
                            *state_mirror.lock() = state.clone();
                            break;
                        }
                        Some(SupervisorCommand::ActivityPing) => {}
                        // Service is not running; drain/kill commands are no-ops.
                        Some(SupervisorCommand::BeginDrain { ack, .. }) => {
                            let _ = ack.send(());
                        }
                        Some(SupervisorCommand::FastKill { ack, .. }) => {
                            let _ = ack.send(());
                        }
                        None => return,
                    }
                }
            }
            ServiceState::Starting => {
                let spawn_cfg = render_argv(
                    &svc,
                    &allocation,
                    packed_for_spawn.as_ref().map(|p| &p.args),
                );
                let cmdline = format!("{} {}", spawn_cfg.binary, spawn_cfg.args.join(" "));
                match spawn_child(&spawn_cfg).await {
                    Ok(mut child) => {
                        let pid = child.id().unwrap_or(0) as i32;
                        observation.register(&svc.name, pid as u32);
                        let spawn_time = Instant::now();
                        let run_id = ((std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis())
                            & 0x7FFFFFFF) as i64;
                        let allocation_json = serde_json::to_string(
                            &allocation
                                .bytes
                                .iter()
                                .map(|(k, v)| (k.as_display(), *v))
                                .collect::<std::collections::BTreeMap<_, _>>(),
                        )
                        .unwrap_or_default();
                        insert_running_row(
                            &db,
                            service_id,
                            run_id,
                            pid as i64,
                            cmdline.clone(),
                            allocation_json,
                        )
                        .await;

                        if let Some(stdout) = child.stdout.take() {
                            spawn_pump_stdout(stdout, service_id, run_id, batcher.clone());
                        }
                        if let Some(stderr) = child.stderr.take() {
                            spawn_pump_stderr(stderr, service_id, run_id, batcher.clone());
                        }

                        let health_cfg = HealthConfig {
                            url: format!(
                                "http://127.0.0.1:{}{}",
                                svc.private_port, svc.health.http_path
                            ),
                            probe_interval: Duration::from_millis(svc.health.probe_interval_ms),
                            timeout: Duration::from_millis(svc.health.timeout_ms),
                        };

                        let cancel_rx_h = cancel_rx.clone();
                        let health_task = tokio::spawn(wait_healthy(health_cfg, cancel_rx_h));

                        tokio::pin!(health_task);

                        loop {
                            tokio::select! {
                                exit = child.wait() => {
                                    warn!(?exit, "child exited during starting/warming");
                                    allocations.lock().remove(&svc.name);
                                    observation.clear(&svc.name);

                                    // Detect OOM kill: process died within 30 s and was
                                    // killed by SIGKILL (kernel OOM killer or cgroup limit).
                                    let runtime = spawn_time.elapsed();
                                    let was_sigkill = exit
                                        .as_ref()
                                        .ok()
                                        .and_then(|s| s.signal())
                                        .map(|sig| sig == libc::SIGKILL)
                                        .unwrap_or(false);
                                    if runtime < Duration::from_secs(30) && was_sigkill {
                                        oom_attempts += 1;
                                        if oom_attempts >= 2 {
                                            warn!(service = %svc.name, attempts = oom_attempts, "OOM retry limit reached; disabling");
                                            if let Some(bus) = start_bus_carry.take() {
                                                let _ = bus.send(StartOutcome::Err(StartFailure {
                                                    kind: StartFailureKind::Oom,
                                                    message: "disabled after repeated OOM kills".into(),
                                                }));
                                            }
                                            state = ServiceState::Disabled { reason: DisableReason::Oom };
                                            *state_mirror.lock() = state.clone();
                                        } else {
                                            warn!(service = %svc.name, "OOM kill detected; bumping rolling factor for retry");
                                            rolling.bump_for_oom_retry(&svc.name);
                                            // Return to Idle so the next Ensure triggers a
                                            // re-estimated start with the bumped safety factor.
                                            if let Some(bus) = start_bus_carry.take() {
                                                let _ = bus.send(StartOutcome::Err(StartFailure {
                                                    kind: StartFailureKind::Oom,
                                                    message: "OOM kill; retrying with larger reservation".into(),
                                                }));
                                            }
                                            state = ServiceState::Idle;
                                            *state_mirror.lock() = state.clone();
                                        }
                                        break;
                                    }

                                    if let Some(bus) = start_bus_carry.take() {
                                        let _ = bus.send(StartOutcome::Err(StartFailure {
                                            kind: StartFailureKind::LaunchFailed,
                                            message: "child exited during starting".into(),
                                        }));
                                    }
                                    state = ServiceState::Failed { retry_count: 0 };
                                    *state_mirror.lock() = state.clone();
                                    break;
                                }
                                outcome = &mut health_task => {
                                    match outcome {
                                        Ok(HealthOutcome::Healthy) => {
                                            state = transition(&state, StateEvent::HealthPassed).unwrap();
                                            *state_mirror.lock() = state.clone();
                                            // Warming grace — also listen for shutdown and child exit.
                                            let grace = Duration::from_millis(svc.warming_grace_ms);
                                            let mut bail = false;
                                            tokio::select! {
                                                _ = tokio::time::sleep(grace) => {
                                                    state = transition(&state, StateEvent::WarmingComplete).unwrap();
                                                    *state_mirror.lock() = state.clone();
                                                }
                                                _ = child.wait() => {
                                                    warn!("child exited during warming grace");
                                                    if let Some(bus) = start_bus_carry.take() {
                                                        let _ = bus.send(StartOutcome::Err(StartFailure {
                                                            kind: StartFailureKind::LaunchFailed,
                                                            message: "child exited during warming".into(),
                                                        }));
                                                    }
                                                    allocations.lock().remove(&svc.name);
                                                    state = ServiceState::Failed { retry_count: 0 };
                                                    *state_mirror.lock() = state.clone();
                                                    bail = true;
                                                }
                                                cmd = rx.recv() => {
                                                    if let Some(SupervisorCommand::Shutdown { ack }) = cmd {
                                                        info!(service = %svc.name, "draining during warming");
                                                        let _ = cancel_tx.send(true);
                                                        send_sigterm_and_wait(&mut child, Duration::from_secs(10)).await;
                                                        delete_running_row(&db, service_id, run_id).await;
                                                        allocations.lock().remove(&svc.name);
                                                        let _ = ack.send(());
                                                        return;
                                                    }
                                                    // Snapshot or channel-closed: fall through to warming complete.
                                                    state = transition(&state, StateEvent::WarmingComplete).unwrap();
                                                    *state_mirror.lock() = state.clone();
                                                }
                                            }
                                            if bail { break; }

                                            // Notify waiters that the service is now running.
                                            if let Some(bus) = start_bus_carry.take() {
                                                let _ = bus.send(StartOutcome::Ok);
                                            }

                                            // Running: wait for child exit, idle timeout, or commands.
                                            loop {
                                                tokio::select! {
                                                    exit = child.wait() => {
                                                        warn!(?exit, "child exited from running");
                                                        rolling.update(&svc.name, observation.read_peak(&svc.name), base_total_bytes_for_rolling);
                                                        observation.clear(&svc.name);
                                                        allocations.lock().remove(&svc.name);
                                                        state = ServiceState::Failed { retry_count: 0 };
                                                        *state_mirror.lock() = state.clone();
                                                        break;
                                                    }
                                                    _ = tokio::time::sleep_until(idle_deadline_for(&last_activity, svc.idle_timeout_ms)) => {
                                                        // Re-check the atomic; a recent ping may have extended the deadline.
                                                        let now = std::time::SystemTime::now()
                                                            .duration_since(std::time::UNIX_EPOCH)
                                                            .unwrap_or_default()
                                                            .as_millis() as u64;
                                                        let last = last_activity.load(Ordering::Relaxed);
                                                        if now + 100 < last + svc.idle_timeout_ms {
                                                            // A ping arrived; loop again with a fresh deadline.
                                                            continue;
                                                        }
                                                        info!(service = %svc.name, "idle timeout; draining to idle");
                                                        send_sigterm_and_wait(&mut child, Duration::from_secs(10)).await;
                                                        delete_running_row(&db, service_id, run_id).await;
                                                        rolling.update(&svc.name, observation.read_peak(&svc.name), base_total_bytes_for_rolling);
                                                        observation.clear(&svc.name);
                                                        allocations.lock().remove(&svc.name);
                                                        state = ServiceState::Idle;
                                                        *state_mirror.lock() = state.clone();
                                                        break;
                                                    }
                                                    cmd = rx.recv() => {
                                                        match cmd {
                                                            Some(SupervisorCommand::Shutdown { ack }) => {
                                                                info!(service = %svc.name, "draining");
                                                                state = transition(&state, StateEvent::DrainRequested).unwrap();
                                                                *state_mirror.lock() = state.clone();
                                                                let _ = cancel_tx.send(true);
                                                                send_sigterm_and_wait(&mut child, Duration::from_secs(10)).await;
                                                                delete_running_row(&db, service_id, run_id).await;
                                                                rolling.update(&svc.name, observation.read_peak(&svc.name), base_total_bytes_for_rolling);
                                                                observation.clear(&svc.name);
                                                                allocations.lock().remove(&svc.name);
                                                                let _ = ack.send(());
                                                                return;
                                                            }
                                                            Some(SupervisorCommand::Snapshot { ack }) => {
                                                                let _ = ack.send(SupervisorSnapshot {
                                                                    name: svc.name.clone(),
                                                                    state: state.clone(),
                                                                    run_id: Some(run_id),
                                                                    pid: Some(pid),
                                                                });
                                                            }
                                                            Some(SupervisorCommand::Ensure { ack }) => {
                                                                let _ = ack.send(EnsureResponse::AlreadyRunning);
                                                            }
                                                            Some(SupervisorCommand::ActivityPing) => {
                                                                let now = std::time::SystemTime::now()
                                                                    .duration_since(std::time::UNIX_EPOCH)
                                                                    .unwrap_or_default()
                                                                    .as_millis() as u64;
                                                                last_activity.store(now, Ordering::Relaxed);
                                                            }
                                                            Some(SupervisorCommand::BeginDrain { reason, ack }) => {
                                                                info!(service = %svc.name, ?reason, "BeginDrain received; draining");
                                                                state = crate::supervise::state::ServiceState::Draining;
                                                                *state_mirror.lock() = state.clone();

                                                                let cfg = crate::supervise::drain::DrainConfig {
                                                                    max_request_duration: Duration::from_millis(svc.max_request_duration_ms),
                                                                    drain_timeout: Duration::from_millis(svc.drain_timeout_ms),
                                                                    extended_stream_drain: Duration::from_millis(svc.extended_stream_drain_ms),
                                                                    sigterm_grace: Duration::from_secs(10),
                                                                };
                                                                crate::supervise::drain::drain_pipeline(&mut child, &cfg, inflight.clone(), reason).await;

                                                                delete_running_row(&db, service_id, run_id).await;
                                                                rolling.update(&svc.name, observation.read_peak(&svc.name), base_total_bytes_for_rolling);
                                                                observation.clear(&svc.name);
                                                                allocations.lock().remove(&svc.name);

                                                                let _ = ack.send(());
                                                                state = ServiceState::Idle;
                                                                *state_mirror.lock() = state.clone();
                                                                break;
                                                            }
                                                            Some(SupervisorCommand::FastKill { reason, ack }) => {
                                                                info!(service = %svc.name, ?reason, "FastKill received");
                                                                state = crate::supervise::state::ServiceState::Draining;
                                                                *state_mirror.lock() = state.clone();

                                                                crate::supervise::drain::fast_kill(&mut child, reason).await;

                                                                delete_running_row(&db, service_id, run_id).await;
                                                                allocations.lock().remove(&svc.name);
                                                                observation.clear(&svc.name);
                                                                let _ = ack.send(());
                                                                state = ServiceState::Idle;
                                                                *state_mirror.lock() = state.clone();
                                                                break;
                                                            }
                                                            None => return,
                                                        }
                                                    }
                                                }
                                            }
                                            break;
                                        }
                                        Ok(HealthOutcome::TimedOut) => {
                                            warn!(service = %svc.name, "health timed out; disabling");
                                            if let Some(bus) = start_bus_carry.take() {
                                                let _ = bus.send(StartOutcome::Err(StartFailure {
                                                    kind: StartFailureKind::HealthTimeout,
                                                    message: "health check timed out".into(),
                                                }));
                                            }
                                            allocations.lock().remove(&svc.name);
                                            state = ServiceState::Disabled { reason: DisableReason::HealthTimeout };
                                            *state_mirror.lock() = state.clone();
                                            send_sigterm_and_wait(&mut child, Duration::from_secs(5)).await;
                                            break;
                                        }
                                        Ok(HealthOutcome::Cancelled) | Err(_) => {
                                            allocations.lock().remove(&svc.name);
                                            send_sigterm_and_wait(&mut child, Duration::from_secs(5)).await;
                                            return;
                                        }
                                    }
                                }
                                cmd = rx.recv() => {
                                    match cmd {
                                        Some(SupervisorCommand::Shutdown { ack }) => {
                                            let _ = cancel_tx.send(true);
                                            send_sigterm_and_wait(&mut child, Duration::from_secs(5)).await;
                                            allocations.lock().remove(&svc.name);
                                            let _ = ack.send(());
                                            return;
                                        }
                                        Some(SupervisorCommand::Snapshot { ack }) => {
                                            let _ = ack.send(SupervisorSnapshot {
                                                name: svc.name.clone(),
                                                state: state.clone(),
                                                run_id: None,
                                                pid: Some(pid),
                                            });
                                        }
                                        Some(SupervisorCommand::Ensure { ack }) => {
                                            // Already in Starting; subscribe to existing bus or report running.
                                            if let Some(sender) = start_bus_carry.as_ref() {
                                                if sender.receiver_count() >= svc.raw.start_queue_depth() {
                                                    let _ = ack.send(EnsureResponse::QueueFull);
                                                } else {
                                                    let bus_rx = sender.subscribe();
                                                    let _ = ack.send(EnsureResponse::Waiting { rx: bus_rx });
                                                }
                                            } else {
                                                // No bus; best-effort.
                                                let _ = ack.send(EnsureResponse::AlreadyRunning);
                                            }
                                        }
                                        Some(SupervisorCommand::ActivityPing) => {}
                                        // Service is not yet running; drain/kill are no-ops during starting.
                                        Some(SupervisorCommand::BeginDrain { ack, .. }) => {
                                            let _ = ack.send(());
                                        }
                                        Some(SupervisorCommand::FastKill { ack, .. }) => {
                                            let _ = ack.send(());
                                        }
                                        None => return,
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!(error = %e, "spawn failed");
                        if let Some(bus) = start_bus_carry.take() {
                            let _ = bus.send(StartOutcome::Err(StartFailure {
                                kind: StartFailureKind::LaunchFailed,
                                message: format!("{e}"),
                            }));
                        }
                        allocations.lock().remove(&svc.name);
                        state = ServiceState::Failed { retry_count: 0 };
                        *state_mirror.lock() = state.clone();
                    }
                }
            }
            ServiceState::Failed { retry_count } => {
                let delay = match *retry_count {
                    0 => Duration::from_secs(2),
                    1 => Duration::from_secs(5),
                    _ => Duration::from_secs(15),
                };
                tokio::select! {
                    _ = tokio::time::sleep(delay) => {
                        state = transition(&state, StateEvent::RetryAfterBackoff).unwrap_or(ServiceState::Disabled { reason: DisableReason::LaunchFailed });
                        if !matches!(state, ServiceState::Disabled { .. }) {
                            // Move back to Idle so the next Ensure triggers a fresh start.
                            state = ServiceState::Idle;
                        }
                        *state_mirror.lock() = state.clone();
                    }
                    cmd = rx.recv() => {
                        if let Some(SupervisorCommand::Shutdown { ack }) = cmd {
                            let _ = ack.send(());
                            return;
                        }
                    }
                }
            }
            ServiceState::Disabled { .. } => {
                info!(service = %svc.name, "disabled; awaiting shutdown or enable");
                loop {
                    match rx.recv().await {
                        Some(SupervisorCommand::Shutdown { ack }) => {
                            let _ = ack.send(());
                            return;
                        }
                        Some(SupervisorCommand::Snapshot { ack }) => {
                            let _ = ack.send(SupervisorSnapshot {
                                name: svc.name.clone(),
                                state: state.clone(),
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
                        None => return,
                    }
                }
            }
            _ => {
                warn!(?state, "unexpected state in supervisor loop");
                return;
            }
        }
    }
}

/// Compute the tokio `Instant` at which the idle deadline fires, based on the
/// last recorded activity timestamp.
fn idle_deadline_for(last_activity: &Arc<AtomicU64>, timeout_ms: u64) -> tokio::time::Instant {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let last = last_activity.load(Ordering::Relaxed);
    let deadline_ms_from_now = (last + timeout_ms).saturating_sub(now);
    tokio::time::Instant::now() + Duration::from_millis(deadline_ms_from_now)
}

async fn send_sigterm_and_wait(child: &mut tokio::process::Child, grace: Duration) {
    if let Some(pid) = child.id() {
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(pid as i32),
            nix::sys::signal::Signal::SIGTERM,
        );
    }
    match tokio::time::timeout(grace, child.wait()).await {
        Ok(_) => {}
        Err(_) => {
            let _ = child.kill().await;
        }
    }
}

fn chrono_like_now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
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
        spawned_at: chrono_like_now_ms(),
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
