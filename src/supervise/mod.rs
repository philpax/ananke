//! Service supervision: per-service tokio tasks, child lifetimes, health loops.

pub mod health;
pub mod logs;
pub mod orphans;
pub mod spawn;

pub use spawn::{SpawnConfig, render_argv};

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex as SyncMutex;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

use crate::config::validate::ServiceConfig;
use crate::db::Database;
use crate::db::logs::BatcherHandle;
use crate::devices::Allocation;
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
    join: JoinHandle<()>,
}

impl SupervisorHandle {
    pub async fn shutdown(self) {
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        let _ = self
            .tx
            .send(SupervisorCommand::Shutdown { ack: ack_tx })
            .await;
        let _ = ack_rx.await;
        let _ = self.join.await;
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
}

pub fn spawn_supervisor(
    svc: ServiceConfig,
    allocation: Allocation,
    db: Database,
    batcher: BatcherHandle,
    service_id: i64,
) -> SupervisorHandle {
    let (tx, rx) = mpsc::channel(32);
    let name = svc.name.clone();
    let join = tokio::spawn(run(svc, allocation, db, batcher, service_id, rx));
    SupervisorHandle { name, tx, join }
}

async fn run(
    svc: ServiceConfig,
    allocation: Allocation,
    db: Database,
    batcher: BatcherHandle,
    service_id: i64,
    mut rx: mpsc::Receiver<SupervisorCommand>,
) {
    let mut state = ServiceState::Idle;
    let state_mirror = Arc::new(SyncMutex::new(state.clone()));
    let (cancel_tx, cancel_rx) = watch::channel(false);

    loop {
        match &state {
            ServiceState::Idle => {
                let next = transition(&state, StateEvent::SpawnRequested).unwrap();
                state = next;
                *state_mirror.lock() = state.clone();
            }
            ServiceState::Starting => {
                let spawn_cfg = render_argv(&svc, &allocation);
                let cmdline = format!("{} {}", spawn_cfg.binary, spawn_cfg.args.join(" "));
                match spawn_child(&spawn_cfg).await {
                    Ok(mut child) => {
                        let pid = child.id().unwrap_or(0) as i32;
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
                                    state = ServiceState::Failed { retry_count: 0 };
                                    *state_mirror.lock() = state.clone();
                                    break;
                                }
                                outcome = &mut health_task => {
                                    match outcome {
                                        Ok(HealthOutcome::Healthy) => {
                                            state = transition(&state, StateEvent::HealthPassed).unwrap();
                                            *state_mirror.lock() = state.clone();
                                            // Warming grace.
                                            let grace = Duration::from_millis(svc.warming_grace_ms);
                                            tokio::select! {
                                                _ = tokio::time::sleep(grace) => {
                                                    state = transition(&state, StateEvent::WarmingComplete).unwrap();
                                                    *state_mirror.lock() = state.clone();
                                                }
                                                _ = child.wait() => {
                                                    warn!("child exited during warming grace");
                                                    state = ServiceState::Failed { retry_count: 0 };
                                                    *state_mirror.lock() = state.clone();
                                                    break;
                                                }
                                            }

                                            // Running: wait for child exit or shutdown command.
                                            tokio::select! {
                                                exit = child.wait() => {
                                                    warn!(?exit, "child exited from running");
                                                    state = ServiceState::Failed { retry_count: 0 };
                                                    *state_mirror.lock() = state.clone();
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
                                                        None => return,
                                                    }
                                                }
                                            }
                                            break;
                                        }
                                        Ok(HealthOutcome::TimedOut) => {
                                            warn!(service = %svc.name, "health timed out; disabling");
                                            state = ServiceState::Disabled { reason: DisableReason::HealthTimeout };
                                            *state_mirror.lock() = state.clone();
                                            send_sigterm_and_wait(&mut child, Duration::from_secs(5)).await;
                                            break;
                                        }
                                        Ok(HealthOutcome::Cancelled) | Err(_) => {
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
                                        None => return,
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!(error = %e, "spawn failed");
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
                            // Move back to Idle → Starting on next loop iteration.
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
                if let Some(SupervisorCommand::Shutdown { ack }) = rx.recv().await {
                    let _ = ack.send(());
                    return;
                }
            }
            _ => {
                warn!(?state, "unexpected state in supervisor loop");
                return;
            }
        }
    }
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
