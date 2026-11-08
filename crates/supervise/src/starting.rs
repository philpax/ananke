//! [`RunLoop`]'s `Starting` phase: spawn the child, pump its stdout/stderr,
//! race the health probe against an early exit or an incoming command, and
//! hand off to [`RunLoop::run_running_loop`] once it passes.

use std::time::{Duration, Instant};

use tracing::{error, info, warn};

use crate::{
    RunLoop, STARTING_SIGTERM_GRACE, StartingOutcome, Step, drain,
    handle::{StartFailure, StartFailureKind, StartOutcome},
    health::{HealthConfig, HealthOutcome, wait_healthy},
    launch::launch_workload,
    logs::{spawn_pump_combined, spawn_pump_stderr, spawn_pump_stdout},
    state::{DisableReason, Event as StateEvent, ServiceState, transition},
    workload::ManagedWorkload,
};

/// If a child dies with SIGKILL within this window from spawn, we treat it
/// as an OOM kill and bump the rolling safety factor for the next attempt.
const OOM_KILL_WINDOW: Duration = Duration::from_secs(30);

impl RunLoop {
    /// The whole Starting → Running → (Draining|Idle|Failed|Disabled)
    /// pipeline. State transitions within this body never escape back to the
    /// outer dispatcher; we only return when the child has been cleaned up and
    /// the next outer-loop state is either a terminal variant or Idle.
    pub(crate) async fn handle_active_lifecycle(&mut self) -> Step {
        // Pull the latest ServiceConfig for this spawn. The launch helper,
        // `HealthConfig`, and the per-command branches below all read fields
        // that a reload may have changed.
        let current = self.current_svc();
        // When the placement engine has run (`packed_for_spawn = Some`), use
        // its computed Allocation for rendering; `self.init.allocation` is
        // otherwise the placement-override source.
        let spawn_alloc = self
            .packed_for_spawn
            .as_ref()
            .map(|p| &p.allocation)
            .unwrap_or(&self.init.allocation);

        let launched = match launch_workload(
            &current,
            spawn_alloc,
            self.packed_for_spawn.as_ref().map(|p| &p.args),
            self.init.service_id,
            &self.deps.system,
            &self.deps.db,
        )
        .await
        {
            Ok(l) => l,
            Err(e) => {
                error!(error = %e, "launch failed; aborting spawn");
                if let Some(bus) = self.start_bus_carry.take() {
                    let _ = bus.send(StartOutcome::Err(StartFailure {
                        kind: StartFailureKind::LaunchFailed,
                        message: e.to_string(),
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

        let run_id = launched.run_id;
        let pid = launched.host_pid.unwrap_or(0);
        // Attribution walks a registered pid's descendants, and pid 0 is the
        // parent of init: registering it would sum the whole machine into
        // this service. A container whose host pid the runtime didn't report
        // is attributed by cgroup or not at all.
        if pid > 0 {
            self.deps
                .observation
                .register(&self.init.identity.name, pid as u32);
        }
        // A container's own cgroup, read back from its host pid. This is the
        // observed path rather than one derived from the runtime's
        // `--cgroup-parent` argument, so it is correct for any runtime,
        // cgroup driver, or rootless layout without being configured.
        // Attribution matches the subtree, so workers the runtime places
        // beside the main process are covered even when they are not its
        // descendants.
        let cgroup_parent = if current.container.is_some() {
            (pid > 0)
                .then(|| self.deps.system.proc.cgroup_path(pid as u32))
                .flatten()
                .map(Into::into)
        } else {
            current.tracking.cgroup_parent.clone()
        };
        if current.container.is_some() && cgroup_parent.is_none() {
            warn!(
                service = %self.init.identity.name,
                "container reported no readable cgroup; attribution falls back to the host pid's descendants"
            );
        }
        self.deps
            .observation
            .set_cgroup_parent(&self.init.identity.name, cgroup_parent);
        if launched.workload_kind == "container" {
            info!(
                service = %self.init.identity.name,
                container_id = %launched.container_id.as_deref().unwrap_or("<unknown>"),
                container_name = %launched.container_name.as_deref().unwrap_or("<unknown>"),
                command_line = %launched.command_line,
                "container workload launched"
            );
        }

        let spawn_time = Instant::now();
        let mut workload = launched.workload;

        self.set_running_ids(run_id, pid);

        if let Some(stdout) = workload.take_stdout() {
            spawn_pump_stdout(
                stdout,
                self.init.service_id,
                run_id,
                self.deps.batcher.clone(),
            );
        }
        if let Some(stderr) = workload.take_stderr() {
            spawn_pump_stderr(
                stderr,
                self.init.service_id,
                run_id,
                self.deps.batcher.clone(),
            );
        }
        for combined in workload.take_combined() {
            spawn_pump_combined(
                combined,
                self.init.service_id,
                run_id,
                self.deps.batcher.clone(),
            );
        }

        // When no health check path is configured, transition to Running
        // immediately after spawn.
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
                    exit = workload.wait() => {
                        // A workload that exits on its own never reaches the
                        // drain path, so nothing else would remove its
                        // container. Only on a confirmed exit, though: an
                        // `Err` is the runtime failing to answer, and
                        // removing on that would destroy a live container.
                        // The running row stays on an unconfirmed exit for the
                        // same reason; the mirror is cleared either way.
                        if exit.is_ok() {
                            if let Err(e) = workload.cleanup().await {
                                warn!(service = %self.init.identity.name, error = %e, "cleanup after early exit failed");
                            }
                            self.end_run(run_id).await;
                        } else {
                            self.clear_running_ids();
                        }
                        return self.on_child_exit_during_start(exit, spawn_time);
                    }
                    outcome = &mut health_task => {
                        match self.on_health_outcome(outcome, &mut workload, run_id).await {
                            StartingOutcome::Continue => {}
                            StartingOutcome::Break => break,
                            StartingOutcome::Exit => return Step::Exit,
                        }
                    }
                    cmd = self.rx.recv() => {
                        match self.on_starting_command(cmd, &mut workload, run_id).await {
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
                .on_health_outcome(Ok(HealthOutcome::Healthy), &mut workload, run_id)
                .await
            {
                StartingOutcome::Continue => {}
                StartingOutcome::Break => {}
                StartingOutcome::Exit => return Step::Exit,
            }
        }
        Step::Continue
    }

    /// Workload exited while we were still in Starting (before the health
    /// probe passed). Detects OOM, updates state, and notifies waiters.
    pub(crate) fn on_child_exit_during_start(
        &mut self,
        exit: std::io::Result<crate::workload::WorkloadExit>,
        spawn_time: Instant,
    ) -> Step {
        warn!(?exit, "workload exited during starting");
        self.deps
            .allocations
            .lock()
            .remove(&self.init.identity.name);
        self.deps.observation.clear(&self.init.identity.name);
        self.emit_allocation_changed();

        // Detect OOM kill: workload died within 30 s from a SIGKILL (native
        // process only — containers report a numeric code, and runtime OOM is
        // surfaced via the exit code, not a signal).
        let runtime = spawn_time.elapsed();
        let was_sigkill =
            matches!(exit, Ok(crate::workload::WorkloadExit::Signal(sig)) if sig == libc::SIGKILL);
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
    pub(crate) async fn on_health_outcome(
        &mut self,
        outcome: Result<HealthOutcome, tokio::task::JoinError>,
        workload: &mut ManagedWorkload,
        run_id: i64,
    ) -> StartingOutcome {
        match outcome {
            Ok(HealthOutcome::Healthy) => {
                let next = transition(&self.read_state(), StateEvent::HealthPassed);
                self.set_state(next);
                // The child answered, so its weights are resident and any
                // memory peak sampled from here on is a peak of the whole
                // model rather than of a partial load. This holds because
                // llama.cpp flips `is_ready` only after `load_model` returns
                // and 503s every endpoint until then; an older build that
                // whitelisted `/v1/models` during loading would quietly turn
                // this into "the HTTP server bound".
                self.rolling_base.run_became_ready();

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

                self.run_running_loop(workload, run_id).await
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
                drain::sigterm_then_sigkill(workload, STARTING_SIGTERM_GRACE).await;
                self.run_shutdown_command().await;
                StartingOutcome::Break
            }
            Ok(HealthOutcome::Cancelled) | Err(_) => {
                self.deps
                    .allocations
                    .lock()
                    .remove(&self.init.identity.name);
                self.emit_allocation_changed();
                drain::sigterm_then_sigkill(workload, STARTING_SIGTERM_GRACE).await;
                self.run_shutdown_command().await;
                StartingOutcome::Exit
            }
        }
    }
}
