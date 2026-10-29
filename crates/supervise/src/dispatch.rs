//! [`RunLoop`]'s command dispatch for the `Running` and `Starting` phases:
//! translating each [`SupervisorCommand`] into the right drain/kill/state
//! transition for the phase currently in progress.

use tracing::info;

use crate::{
    AutoRestartOutcome, RunLoop, RunningOutcome, STARTING_SIGTERM_GRACE, StartingOutcome,
    drain::{self, fast_kill},
    handle::{
        DisableResult, EnableResult, EnsureResponse, StartFailure, StartFailureKind, StartOutcome,
        SupervisorCommand,
    },
    state::{DisableReason, Event as StateEvent, ServiceState, transition},
    workload::ManagedWorkload,
};

impl RunLoop {
    /// Dispatch a command received while the service is Running.
    pub(crate) async fn on_running_command(
        &mut self,
        cmd: Option<SupervisorCommand>,
        workload: &mut ManagedWorkload,
        run_id: i64,
    ) -> RunningOutcome {
        match cmd {
            Some(SupervisorCommand::Shutdown { ack }) => {
                info!(service = %self.init.identity.name, "draining");
                let next = transition(&self.read_state(), StateEvent::DrainRequested);
                self.set_state(next);
                let _ = self.cancel_tx.send(true);
                drain::sigterm_then_sigkill(
                    workload,
                    std::time::Duration::from_millis(self.current_svc().stop_grace_ms),
                )
                .await;
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
                    self.emit_auto_restarted(
                        "periodic",
                        "on-request interval elapsed".into(),
                        run_id,
                    )
                    .await;
                    self.drain_now(workload, run_id, drain::DrainReason::AutoRestart)
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
                self.drain_now(workload, run_id, reason).await;
                let _ = ack.send(());
                self.set_state(ServiceState::Idle);
                RunningOutcome::Break
            }
            Some(SupervisorCommand::FastKill { reason, ack }) => {
                info!(service = %self.init.identity.name, ?reason, "FastKill received");
                self.set_state(ServiceState::Draining);

                fast_kill(workload, reason).await;
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
                // Already running; enable only clears a stale stored mark.
                let _ = ack.send(match self.persist_operator_disable(false).await {
                    Ok(()) => EnableResult::NotDisabled,
                    Err(reason) => EnableResult::PersistFailed(reason),
                });
                RunningOutcome::Continue
            }
            Some(SupervisorCommand::Disable { ack }) => {
                if let Err(reason) = self.persist_operator_disable(true).await {
                    let _ = ack.send(DisableResult::PersistFailed(reason));
                    return RunningOutcome::Continue;
                }
                info!(service = %self.init.identity.name, "Disable received; draining then disabling");
                self.drain_now(workload, run_id, drain::DrainReason::UserKilled)
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
                    .perform_auto_restart(workload, run_id, "ttft_stall", detail)
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
    pub(crate) async fn on_starting_command(
        &mut self,
        cmd: Option<SupervisorCommand>,
        workload: &mut ManagedWorkload,
        run_id: i64,
    ) -> StartingOutcome {
        match cmd {
            Some(SupervisorCommand::Shutdown { ack }) => {
                let _ = self.cancel_tx.send(true);
                drain::sigterm_then_sigkill(workload, STARTING_SIGTERM_GRACE).await;
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
                drain::sigterm_then_sigkill(workload, STARTING_SIGTERM_GRACE).await;
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
                drain::sigterm_then_sigkill(workload, STARTING_SIGTERM_GRACE).await;
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
                // Already starting; enable only clears a stale stored mark.
                let _ = ack.send(match self.persist_operator_disable(false).await {
                    Ok(()) => EnableResult::NotDisabled,
                    Err(reason) => EnableResult::PersistFailed(reason),
                });
                StartingOutcome::Continue
            }
            Some(SupervisorCommand::Disable { ack }) => {
                // Disable during starting: drain the child, clean up, and
                // transition to Disabled.
                if let Err(reason) = self.persist_operator_disable(true).await {
                    let _ = ack.send(DisableResult::PersistFailed(reason));
                    return StartingOutcome::Continue;
                }
                let _ = self.cancel_tx.send(true);
                drain::sigterm_then_sigkill(workload, STARTING_SIGTERM_GRACE).await;
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
}
