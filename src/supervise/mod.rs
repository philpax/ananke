//! Service supervision: per-service tokio tasks, child lifetimes, health loops.

pub mod health;
pub mod logs;
pub mod orphans;
pub mod spawn;

pub use orphans::{OrphanDisposition, reconcile};
pub use spawn::{SpawnConfig, render_argv};

use std::os::unix::process::ExitStatusExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex as SyncMutex;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

use crate::config::validate::ServiceConfig;
use crate::db::Database;
use crate::db::logs::BatcherHandle;
use crate::devices::Allocation;
use crate::observation::ObservationTable;
use crate::rolling::RollingTable;
use crate::state::{DisableReason, Event as StateEvent, ServiceState, transition};
use crate::supervise::health::{HealthConfig, HealthOutcome, wait_healthy};
use crate::supervise::logs::{spawn_pump_stderr, spawn_pump_stdout};
use crate::supervise::spawn::spawn_child;

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
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_supervisor(
    svc: ServiceConfig,
    allocation: Allocation,
    db: Database,
    batcher: BatcherHandle,
    service_id: i64,
    last_activity: Arc<AtomicU64>,
    snapshot: crate::snapshotter::SharedSnapshot,
    allocations: Arc<parking_lot::Mutex<crate::allocator::AllocationTable>>,
    rolling: RollingTable,
    observation: ObservationTable,
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
    snapshot: crate::snapshotter::SharedSnapshot,
    allocations: Arc<parking_lot::Mutex<crate::allocator::AllocationTable>>,
    rolling: RollingTable,
    observation: ObservationTable,
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
    let mut packed_for_spawn: Option<crate::placement::Packed> = None;
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

                            // Determine the allocation to reserve. If placement_override is
                            // non-empty, use it as the escape hatch (no estimator/placement).
                            // Otherwise, run the estimator + placement engine.
                            let want = if !svc.placement_override.is_empty() {
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
                                let mut est = match crate::estimator::estimate_from_path(
                                    &model_path,
                                    &svc,
                                ) {
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

                                let packed = match crate::placement::pack(&est, &svc, &snap, &table)
                                {
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
                                let msg = format!("{nofit}");
                                // Clear any computed packed args so they are not used on the next
                                // Ensure attempt after this failure.
                                let _ = packed_for_spawn.take();
                                let _ = ack.send(EnsureResponse::Unavailable { reason: msg });
                                continue;
                            }

                            // Reserve in the allocation table before spawning.
                            // Capture the total reserved bytes (MB → bytes) for the rolling
                            // update that fires when the service later drains back to Idle.
                            base_total_bytes_for_rolling =
                                want.values().sum::<u64>() * 1024 * 1024;
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
                        let _ = db.with_conn(|c| {
                            c.execute(
                                "INSERT INTO running_services(service_id, run_id, pid, spawned_at, command_line, allocation, state) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'starting')",
                                (
                                    service_id,
                                    run_id,
                                    pid as i64,
                                    chrono_like_now_ms(),
                                    cmdline.clone(),
                                    serde_json::to_string(
                                        &allocation
                                            .bytes
                                            .iter()
                                            .map(|(k, v)| (k.as_display(), *v))
                                            .collect::<std::collections::BTreeMap<_, _>>(),
                                    )
                                    .unwrap_or_default(),
                                ),
                            )
                        });

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
                                                    kind: StartFailureKind::LaunchFailed,
                                                    message: "disabled after repeated OOM kills".into(),
                                                }));
                                            }
                                            state = ServiceState::Disabled { reason: DisableReason::LaunchFailed };
                                            *state_mirror.lock() = state.clone();
                                        } else {
                                            warn!(service = %svc.name, "OOM kill detected; bumping rolling factor for retry");
                                            rolling.bump_for_oom_retry(&svc.name);
                                            // Return to Idle so the next Ensure triggers a
                                            // re-estimated start with the bumped safety factor.
                                            if let Some(bus) = start_bus_carry.take() {
                                                let _ = bus.send(StartOutcome::Err(StartFailure {
                                                    kind: StartFailureKind::LaunchFailed,
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
                                                        let _ = db.with_conn(|c| c.execute(
                                                            "DELETE FROM running_services WHERE service_id = ?1 AND run_id = ?2",
                                                            (service_id, run_id),
                                                        ));
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
                                                        let _ = db.with_conn(|c| c.execute(
                                                            "DELETE FROM running_services WHERE service_id = ?1 AND run_id = ?2",
                                                            (service_id, run_id),
                                                        ));
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
                                                                let _ = db.with_conn(|c| c.execute(
                                                                    "DELETE FROM running_services WHERE service_id = ?1 AND run_id = ?2",
                                                                    (service_id, run_id),
                                                                ));
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
