//! [`RunLoop`]'s auto-restart watchdog *decisions*: polling the error-rate,
//! generation-stall, and spec-collapse signals and returning a human-readable
//! detail string when one should fire. The actual restart/disable action
//! lives in [`crate::auto_restart`].
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::time::Duration;

use ananke_config::validate::{AutoRestartSettings, ErrorRateTrigger, SpecCollapseTrigger};
use ananke_db::SpecAcceptance;
use tracing::{info, warn};

use crate::{GenStallLoop, RunLoop, genstall};

/// Per-request bound on a generation-stall watchdog `/metrics` fetch. The
/// endpoint answers from memory on the loopback, so anything slower than this
/// is indistinguishable from unreachable.
const GENSTALL_FETCH_TIMEOUT: Duration = Duration::from_secs(2);

impl RunLoop {
    /// Query the current run's recent error rate and decide whether the
    /// error-rate watchdog should fire. Returns a human-readable detail
    /// string when it should, else `None`.
    ///
    /// Gated on the `min_uptime` cooldown: a freshly (re)started run must
    /// live that long before the watchdog can fire. That, together with the
    /// run-scoped, windowed query — which starts from zero metrics on every
    /// respawn — is what stops restart flapping.
    pub(crate) async fn evaluate_error_rate(
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
        let since_ms = ananke_tracking::now_unix_ms() - er.window_ms as i64;
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

    /// Build the generation-stall poll state at Running entry, or `None` when
    /// the watchdog is off. A llama-cpp service with an explicit
    /// `metrics = false` self-disables: the `--metrics` flag injection is
    /// suppressed too (see `spawn.rs`), so there would be nothing to poll.
    /// Command services keep their opt-in setting; an unreachable or
    /// non-llama.cpp `/metrics` is handled at poll time by warning once and
    /// never firing.
    pub(crate) fn genstall_setup(&self, ar: &AutoRestartSettings) -> Option<GenStallLoop> {
        let gs = ar.generation_stall.as_ref()?;
        if let Some(lc) = self.current_svc().llama_cpp()
            && lc.metrics == Some(false)
        {
            info!(
                service = %self.init.identity.name,
                "generation-stall watchdog disabled: service sets metrics = false"
            );
            return None;
        }
        let period = Duration::from_millis(gs.poll_interval_ms);
        Some(GenStallLoop {
            poll: tokio::time::interval_at(tokio::time::Instant::now() + period, period),
            state: genstall::GenStallState::new(tokio::time::Instant::now()),
            client: reqwest::Client::builder()
                .timeout(GENSTALL_FETCH_TIMEOUT)
                .build()
                // Invariant: a default builder with no custom connector cannot fail.
                .unwrap_or_else(|_| unreachable!("reqwest client from default builder builds")),
            url: format!(
                "http://127.0.0.1:{}/metrics",
                self.init.identity.private_port
            ),
            warned_unreachable: false,
        })
    }

    /// Poll the child's `/metrics` once and decide whether the
    /// generation-stall watchdog should fire. Returns the detail string when
    /// it should, else `None`. Gated on the same `min_uptime` cooldown as the
    /// error-rate watchdog; the timeout threshold is re-read live so a config
    /// edit takes effect without a respawn.
    pub(crate) async fn evaluate_generation_stall(
        &self,
        g: &mut GenStallLoop,
        running_since: tokio::time::Instant,
    ) -> Option<String> {
        let ar = self.current_svc().auto_restart;
        let gs = ar.generation_stall.as_ref()?;
        if tokio::time::Instant::now().duration_since(running_since)
            < Duration::from_millis(ar.min_uptime_ms)
        {
            return None;
        }
        let progress = genstall::fetch_progress(&g.client, &g.url).await;
        if progress.is_none() && !g.warned_unreachable {
            g.warned_unreachable = true;
            warn!(
                service = %self.init.identity.name,
                url = %g.url,
                "generation-stall watchdog cannot read progress counters from /metrics; the watchdog will not fire (is the child started with --metrics?)"
            );
        }
        let inflight = self.init.inflight.current();
        let timeout = Duration::from_millis(gs.timeout_ms);
        g.state.observe(progress, inflight, timeout).then(|| {
            format!(
                "/metrics progress counters flat for {}s with {inflight} request(s) in flight (upstream wedge)",
                gs.timeout_ms / 1000,
            )
        })
    }

    /// Query the current run's draft acceptance and decide whether the
    /// spec-collapse watchdog should fire. Returns a human-readable detail
    /// string when it should, else `None`.
    ///
    /// The trip condition requires an actual collapse: the run must have
    /// accepted draft tokens at some point, and the recent window must hold
    /// at least `min_requests` drafting requests with zero accepted between
    /// them. The prior-acceptance requirement keeps workloads that
    /// legitimately never accept (e.g. grammar-constrained speculative
    /// decoding, where drafts are rejected for violating the grammar) from
    /// tripping the watchdog: for them, zero is the workload's baseline, not
    /// a state change. It also bounds the blast radius of any residual false
    /// positive to a single restart — a fresh run cannot re-fire until it
    /// has demonstrated acceptance again, so a healthy-but-zero service
    /// never reaches the flap cap through this trigger.
    /// Gated on the same `min_uptime` cooldown as the error-rate watchdog.
    ///
    /// `ever_accepted` latches the prior-acceptance requirement across polls
    /// so it survives metrics retention pruning the run's early rows; see
    /// where it is declared in [`Self::run_running_loop`].
    pub(crate) async fn evaluate_spec_collapse(
        &self,
        run_id: i64,
        running_since: tokio::time::Instant,
        ever_accepted: &mut bool,
    ) -> Option<String> {
        let ar = self.current_svc().auto_restart;
        let sc = ar.spec_collapse.as_ref()?;
        if tokio::time::Instant::now().duration_since(running_since)
            < Duration::from_millis(ar.min_uptime_ms)
        {
            return None;
        }
        let since_ms = ananke_tracking::now_unix_ms() - sc.window_ms as i64;
        let acceptance = match self
            .deps
            .db
            .spec_acceptance_since(self.init.service_id, run_id, since_ms)
            .await
        {
            Ok(v) => v,
            Err(e) => {
                warn!(service = %self.init.identity.name, error = %e, "auto-restart: spec-collapse query failed");
                return None;
            }
        };
        *ever_accepted |= acceptance.run_accepted > 0;
        spec_collapse_trips(acceptance, sc, *ever_accepted).then(|| {
            // `run_accepted` can read zero even though the run demonstrably
            // accepted earlier, if retention pruned those rows — the latch
            // remembers what the query no longer can.
            let earlier = if acceptance.run_accepted > 0 {
                format!("{} accepted earlier in the run", acceptance.run_accepted)
            } else {
                "acceptance observed earlier in the run".to_string()
            };
            format!(
                "speculative draft acceptance collapsed: 0 of {} drafted tokens accepted across {} requests over {}s, after {earlier}",
                acceptance.window_drafted,
                acceptance.window_drafting,
                sc.window_ms / 1000,
            )
        })
    }
}

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

/// Pure verdict for the spec-collapse watchdog: `true` when the window holds
/// enough drafted tokens to trust, their accepted total is exactly zero,
/// and the run accepted draft tokens before the window — i.e. the
/// acceptance actually collapsed rather than never existing. The floor
/// counts drafted tokens, not requests: long generations arrive slowly but
/// draft thousands of tokens each, so a request floor starves on exactly
/// the traffic a garbage wedge produces. Kept free of I/O so it can be
/// unit-tested directly.
///
/// `ever_accepted` carries the caller's latched view of prior acceptance,
/// which outlives retention pruning `a.run_accepted` back to zero.
fn spec_collapse_trips(a: SpecAcceptance, cfg: &SpecCollapseTrigger, ever_accepted: bool) -> bool {
    a.window_drafted >= cfg.min_draft_tokens
        && a.window_accepted == 0
        && (ever_accepted || a.run_accepted > 0)
}

#[cfg(test)]
mod auto_restart_tests {
    use ananke_config::validate::ErrorStatusClass;

    use super::*;

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

    fn sc_cfg(min_draft_tokens: u64) -> SpecCollapseTrigger {
        SpecCollapseTrigger {
            window_ms: 120_000,
            min_draft_tokens,
            poll_interval_ms: 30_000,
        }
    }

    fn acceptance(
        window_drafting: u64,
        window_drafted: u64,
        window_accepted: u64,
        run_accepted: u64,
    ) -> SpecAcceptance {
        SpecAcceptance {
            window_drafting,
            window_drafted,
            window_accepted,
            run_accepted,
        }
    }

    #[test]
    fn spec_collapse_trips_on_collapse_after_healthy_acceptance() {
        // The production wedge shape: healthy acceptance earlier in the run,
        // then a single long aborted generation drafting thousands of tokens
        // with not one accepted.
        assert!(spec_collapse_trips(
            acceptance(1, 5_000, 0, 5_000),
            &sc_cfg(200),
            false,
        ));
    }

    #[test]
    fn spec_collapse_needs_min_draft_tokens() {
        // A couple of unlucky short generations must not restart the service.
        assert!(!spec_collapse_trips(
            acceptance(9, 199, 0, 5_000),
            &sc_cfg(200),
            false,
        ));
        assert!(spec_collapse_trips(
            acceptance(9, 200, 0, 5_000),
            &sc_cfg(200),
            false,
        ));
    }

    #[test]
    fn spec_collapse_vetoed_by_any_window_acceptance() {
        // One accepted draft token anywhere in the window proves the pairing
        // still works — even thousands of otherwise-rejected tokens must not
        // trip it.
        assert!(!spec_collapse_trips(
            acceptance(48, 5_000, 1, 5_000),
            &sc_cfg(200),
            true,
        ));
    }

    #[test]
    fn spec_collapse_needs_prior_acceptance() {
        // A run that has never accepted a draft token is a workload property
        // (e.g. grammar-constrained speculative decoding), not a collapse.
        assert!(!spec_collapse_trips(
            acceptance(48, 5_000, 0, 0),
            &sc_cfg(200),
            false,
        ));
    }

    #[test]
    fn spec_collapse_latch_survives_pruned_run_history() {
        // Retention can prune the early rows that proved this run once
        // accepted, zeroing the SQL term. The caller's latch keeps the
        // trigger armed; without it, a run older than the metrics retention
        // window would silently stop being watched.
        assert!(!spec_collapse_trips(
            acceptance(48, 5_000, 0, 0),
            &sc_cfg(200),
            false,
        ));
        assert!(spec_collapse_trips(
            acceptance(48, 5_000, 0, 0),
            &sc_cfg(200),
            true,
        ));
    }

    #[test]
    fn spec_collapse_ignores_empty_window() {
        // No drafting traffic at all (idle, or a non-spec service).
        assert!(!spec_collapse_trips(
            acceptance(0, 0, 0, 0),
            &sc_cfg(200),
            false,
        ));
        assert!(!spec_collapse_trips(
            acceptance(0, 0, 0, 5_000),
            &sc_cfg(200),
            true,
        ));
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
