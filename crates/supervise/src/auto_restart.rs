//! [`RunLoop`]'s auto-restart *actions*: draining and restarting (or
//! disabling past the flap cap) once a watchdog in
//! [`crate::watchdogs`] has decided to fire, plus the periodic
//! timer that drives scheduled restarts independent of any watchdog signal.

use std::time::Duration;

use ananke_api::events::Event;
use ananke_config::validate::PeriodicMode;
use tracing::{info, warn};

use crate::{
    AutoRestartOutcome, PeriodicOutcome, RunLoop, drain,
    state::{DisableReason, ServiceState},
    workload::ManagedWorkload,
};

/// In-flight drain grace for a stall-triggered restart. Short by design: the
/// stall watchdog only fires once the run has produced nothing for the whole
/// timeout window, so its wedged in-flight request will never complete and
/// there is no healthy traffic to preserve — a brief grace for a tailing
/// packet, then SIGTERM.
const STALL_DRAIN_INFLIGHT_WAIT: Duration = Duration::from_secs(5);

impl RunLoop {
    /// Perform an error-rate-triggered restart. Thin wrapper over
    /// [`Self::perform_auto_restart`] that labels the trigger.
    pub(crate) async fn perform_error_rate_restart(
        &mut self,
        workload: &mut ManagedWorkload,
        run_id: i64,
        detail: String,
    ) -> AutoRestartOutcome {
        self.perform_auto_restart(workload, run_id, "error_rate", detail)
            .await
    }

    /// Drain the current run and return to Idle for a watchdog-triggered
    /// restart, or disable the service if the flap cap has been reached.
    /// Either way the child is drained; the caller breaks out of the Running
    /// loop afterward. Shared by the error-rate, TTFT-stall, and
    /// generation-stall watchdogs — all count toward the same flap cap.
    pub(crate) async fn perform_auto_restart(
        &mut self,
        workload: &mut ManagedWorkload,
        run_id: i64,
        trigger: &'static str,
        detail: String,
    ) -> AutoRestartOutcome {
        let ar = self.current_svc().auto_restart;
        // A stall restart drains a run that is by definition producing nothing,
        // so waiting the full `max_request_duration` for its wedged in-flight
        // request to finish is pointless — bound the wait to a short grace.
        let inflight_wait = matches!(trigger, "ttft_stall" | "generation_stall")
            .then_some(STALL_DRAIN_INFLIGHT_WAIT);
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
            // The firing that trips the flap cap is the most important one to
            // record — it is the one that takes the service down for good.
            // Recorded but not published as `AutoRestarted`: nothing is
            // restarted here, and the `set_state` below already broadcasts
            // the transition to `Disabled`.
            self.record_auto_restart(
                trigger,
                format!("{detail} (flap cap reached; service disabled)"),
                run_id,
                ananke_tracking::now_unix_ms(),
            )
            .await;
            self.drain_now_bounded(
                workload,
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
        self.emit_auto_restarted(trigger, detail, run_id).await;
        self.drain_now_bounded(
            workload,
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
    pub(crate) async fn on_periodic_tick(
        &mut self,
        deadline: &mut Option<tokio::time::Instant>,
        workload: &mut ManagedWorkload,
        run_id: i64,
    ) -> PeriodicOutcome {
        let ar = self.current_svc().auto_restart;
        let Some(periodic) = ar.periodic.as_ref() else {
            *deadline = None;
            return PeriodicOutcome::Continue;
        };
        match periodic.mode {
            PeriodicMode::Immediate => {
                self.perform_periodic_restart(
                    workload,
                    run_id,
                    "interval elapsed (immediate)".into(),
                )
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
                if self.init.inflight.current() == 0 {
                    self.perform_periodic_restart(
                        workload,
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
        workload: &mut ManagedWorkload,
        run_id: i64,
        detail: String,
    ) {
        info!(service = %self.init.identity.name, detail = %detail, "auto-restart: periodic timer firing");
        self.emit_auto_restarted("periodic", detail, run_id).await;
        self.drain_now(workload, run_id, drain::DrainReason::AutoRestart)
            .await;
        self.set_state(ServiceState::Idle);
    }

    /// Publish an [`Event::AutoRestarted`] to the daemon event stream and
    /// persist the firing. Only for firings that actually respawn the
    /// service — the flap-cap path calls [`Self::record_auto_restart`]
    /// directly, since it disables rather than restarts.
    pub(crate) async fn emit_auto_restarted(&self, trigger: &str, detail: String, run_id: i64) {
        let at_ms = ananke_tracking::now_unix_ms();
        self.deps.events.publish(Event::AutoRestarted {
            service: self.init.identity.name.clone(),
            trigger: trigger.to_string(),
            detail: detail.clone(),
            at_ms,
        });
        self.record_auto_restart(trigger, detail, run_id, at_ms)
            .await;
    }

    /// Persist one watchdog firing to `service_restarts` so it outlives the
    /// live WebSocket: a watchdog restart with no browser attached would
    /// otherwise leave nothing behind but a daemon log line.
    ///
    /// Awaited rather than detached: the flap-cap path records the firing
    /// that takes a service down for good and then immediately drains and
    /// disables, so a spawned task racing daemon shutdown could lose exactly
    /// the record the operator most needs. The write is a couple of
    /// statements against local SQLite.
    async fn record_auto_restart(&self, trigger: &str, detail: String, run_id: i64, at_ms: i64) {
        let row = ananke_db::models::ServiceRestart {
            restart_id: 0,
            service_id: self.init.service_id,
            run_id: Some(run_id),
            at_ms,
            trigger: trigger.to_string(),
            detail,
        };
        if let Err(e) = self.deps.db.insert_service_restart(&row).await {
            warn!(service = %self.init.identity.name, error = %e, "failed to persist auto-restart record");
        }
    }
}

/// Re-poll cadence for the `on-idle` periodic mode while it waits for the
/// service to fall quiet after its interval has elapsed.
const PERIODIC_IDLE_POLL: Duration = Duration::from_secs(1);
