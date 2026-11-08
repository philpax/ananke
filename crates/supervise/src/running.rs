//! [`RunLoop`]'s `Running` phase: the inner select loop that races child
//! exit, the auto-restart watchdogs, the idle timeout, and incoming
//! commands, plus the shared drain helper every exit path funnels through.

use std::time::Duration;

use tracing::{info, warn};

use crate::{
    AutoRestartOutcome, PeriodicOutcome, RunLoop, RunningOutcome, StartingOutcome,
    drain::{DrainConfig, drain_pipeline},
    state::ServiceState,
    workload::ManagedWorkload,
};

/// Clock-skew tolerance on the idle-timeout re-check. Lets a ping that raced
/// our deadline extend the idle window rather than immediately draining.
const IDLE_DEADLINE_SKEW_MS: u64 = 100;

impl RunLoop {
    /// The Running inner loop: wait for child exit, idle timeout, or commands.
    pub(crate) async fn run_running_loop(
        &mut self,
        workload: &mut ManagedWorkload,
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

        // Generation-stall watchdog: polls the child's `/metrics` progress
        // counters. Like the error-rate poll, the first tick lands one period
        // out so a fresh run isn't probed the instant it starts.
        let mut genstall = self.genstall_setup(&auto_restart);

        // Spec-collapse watchdog poll: like the error-rate poll, but only for
        // llama-cpp services that configure speculative decoding — without
        // `spec_type` no request can ever carry engine draft counts, so the
        // query would be a per-interval no-op forever.
        let mut spec_poll = auto_restart
            .spec_collapse
            .as_ref()
            .filter(|_| {
                self.current_svc()
                    .llama_cpp()
                    .is_some_and(|lc| lc.spec_type.is_some())
            })
            .map(|sc| {
                let period = Duration::from_millis(sc.poll_interval_ms);
                tokio::time::interval_at(tokio::time::Instant::now() + period, period)
            });
        // Latches once this run has been observed accepting draft tokens.
        // The SQL `run_accepted` term is the primary source, but it reads
        // `request_metrics`, which retention prunes at seven days — a run
        // older than that would lose its arming evidence and silently stop
        // being watched. The latch is per-run local state, so it resets on
        // every respawn exactly like the run-scoped query does.
        let mut spec_ever_accepted = false;

        loop {
            tokio::select! {
                exit = workload.wait() => {
                    warn!(?exit, "child exited from running");
                    // Only remove on a confirmed exit. An `Err` here means
                    // the runtime was unreachable, not that the workload
                    // stopped, and force-removing on that would destroy a
                    // healthy container over one failed query.
                    // The running row stays on an unconfirmed exit for the
                    // same reason; the mirror is cleared either way, since
                    // this supervisor no longer tracks the workload.
                    if exit.is_ok() {
                        if let Err(e) = workload.cleanup().await {
                            warn!(service = %self.init.identity.name, error = %e, "cleanup after exit failed");
                        }
                        self.end_run(run_id).await;
                    } else {
                        warn!(service = %self.init.identity.name, "exit unconfirmed; leaving the workload for reconciliation");
                        self.clear_running_ids();
                    }
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
                            self.perform_error_rate_restart(workload, run_id, detail).await,
                            AutoRestartOutcome::Restarted | AutoRestartOutcome::Disabled,
                        )
                    {
                        return StartingOutcome::Break;
                    }
                }
                // Generation-stall watchdog poll. Mirrors the error-rate
                // branch: the future borrows only the local `genstall`; the
                // fetch + decision + restart run in the handler.
                _ = async {
                    match genstall.as_mut() {
                        Some(g) => { g.poll.tick().await; }
                        None => std::future::pending::<()>().await,
                    }
                } => {
                    if let Some(g) = genstall.as_mut()
                        && let Some(detail) =
                            self.evaluate_generation_stall(g, running_since).await
                        && matches!(
                            self.perform_auto_restart(workload, run_id, "generation_stall", detail)
                                .await,
                            AutoRestartOutcome::Restarted | AutoRestartOutcome::Disabled,
                        )
                    {
                        return StartingOutcome::Break;
                    }
                }
                // Spec-collapse watchdog poll. Mirrors the error-rate branch:
                // the future borrows only the local `spec_poll`; the decision
                // + restart run in the handler.
                _ = async {
                    match spec_poll.as_mut() {
                        Some(p) => { p.tick().await; }
                        None => std::future::pending::<()>().await,
                    }
                } => {
                    if let Some(detail) = self
                        .evaluate_spec_collapse(run_id, running_since, &mut spec_ever_accepted)
                        .await
                        && matches!(
                            self.perform_auto_restart(workload, run_id, "spec_collapse", detail)
                                .await,
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
                        self.on_periodic_tick(&mut periodic_deadline, workload, run_id).await,
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
                _ = tokio::time::sleep_until(idle_deadline_for(&self.init.last_activity, self.current_svc().idle_timeout_ms)), if self.current_svc().lifecycle != ananke_config::Lifecycle::Persistent => {
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
                    crate::drain::sigterm_then_sigkill(
                        workload, Duration::from_millis(self.current_svc().stop_grace_ms),
                    ).await;
                    self.run_shutdown_command().await;
                    self.end_run(run_id).await;
                    self.record_drain_complete();
                    self.set_state(ServiceState::Idle);
                    return StartingOutcome::Break;
                }
                cmd = self.rx.recv() => {
                    match self.on_running_command(cmd, workload, run_id).await {
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
    pub(crate) async fn drain_now(
        &mut self,
        workload: &mut ManagedWorkload,
        run_id: i64,
        reason: crate::drain::DrainReason,
    ) {
        self.drain_now_bounded(workload, run_id, reason, None).await;
    }

    /// As [`Self::drain_now`], but with an optional override for how long the
    /// pipeline waits for in-flight requests to finish before SIGTERM. The
    /// stall watchdog passes a short bound: its whole premise is that the run
    /// is producing nothing, so the default `max_request_duration` (which can
    /// be many minutes) would just make the daemon wait on a request that will
    /// never complete.
    pub(crate) async fn drain_now_bounded(
        &mut self,
        workload: &mut ManagedWorkload,
        run_id: i64,
        reason: crate::drain::DrainReason,
        inflight_wait: Option<Duration>,
    ) {
        self.set_state(ServiceState::Draining);
        let current = self.current_svc();
        let cfg = DrainConfig {
            max_request_duration: inflight_wait
                .unwrap_or_else(|| Duration::from_millis(current.max_request_duration_ms)),
            drain_timeout: Duration::from_millis(current.drain_timeout_ms),
            extended_stream_drain: Duration::from_millis(current.extended_stream_drain_ms),
            sigterm_grace: Duration::from_millis(current.stop_grace_ms),
        };
        drain_pipeline(workload, &cfg, self.init.inflight.clone(), reason).await;
        self.run_shutdown_command().await;
        self.end_run(run_id).await;
        self.record_drain_complete();
    }
}

/// Compute the tokio `Instant` at which the idle deadline fires, based on the
/// last recorded activity instant. Lives entirely on the tokio monotonic
/// clock so `tokio::time::pause()` can freeze and advance it virtually.
fn idle_deadline_for(
    last_activity: &ananke_tracking::activity::ActivityStamp,
    timeout_ms: u64,
) -> tokio::time::Instant {
    let last = *last_activity.lock();
    last + Duration::from_millis(timeout_ms)
}
