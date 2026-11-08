//! Full drain pipeline. Sends SIGTERM via the child's
//! `ManagedChild::sigterm` and escalates to SIGKILL on timeout. The
//! [`ProcessSpawner`](ananke_system::ProcessSpawner) abstraction means the
//! same pipeline works against real children under `LocalSpawner` and
//! against purely virtual ones under `FakeSpawner` in tests.

use std::{sync::Arc, time::Duration};

pub use ananke_placement::DrainReason;
use ananke_proxy::InflightCounter;
use tracing::{info, warn};

use crate::workload::ManagedWorkload;

/// Polling cadence while waiting for in-flight counters to reach zero.
const INFLIGHT_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// SIGTERM grace used by `fast_kill`. Short because the caller has already
/// decided the child has misbehaved and must go; no streaming-client courtesy.
const FAST_KILL_SIGTERM_GRACE: Duration = Duration::from_secs(5);

/// Bound on how long to wait after SIGKILL for the exit status and the tail
/// of the log stream before cleaning up. Short: the workload is already
/// dead, and this only covers the runtime's own reaping latency.
const POST_KILL_GRACE: Duration = Duration::from_secs(5);

#[derive(Debug, Clone)]
pub struct DrainConfig {
    pub max_request_duration: Duration,
    pub drain_timeout: Duration,
    pub extended_stream_drain: Duration,
    pub sigterm_grace: Duration,
}

/// Run the full drain pipeline against `child`. Caller is expected to
/// have already transitioned the service state to `Draining` and
/// refused new requests.
pub async fn drain_pipeline(
    workload: &mut ManagedWorkload,
    cfg: &DrainConfig,
    inflight: Arc<InflightCounter>,
    reason: DrainReason,
) {
    let initial_inflight = inflight.current();
    if initial_inflight > 0 {
        info!(
            ?reason,
            inflight = initial_inflight,
            "drain: waiting for in-flight requests"
        );
        let timed_out = wait_inflight_zero(&inflight, cfg.max_request_duration).await;
        if timed_out {
            warn!(
                ?reason,
                inflight = inflight.current(),
                "drain: max_request_duration elapsed with requests still in flight"
            );
        }

        // drain_timeout grace: give tailing SSE packets a chance to flush
        // after the HTTP body has ended but before we SIGTERM. Only
        // relevant if we actually had traffic — a drain triggered on a
        // quiescent service (inflight == 0 from the start) has nothing
        // to tail and a blanket sleep here just adds 30s to every idle
        // eviction. See the eviction cascade timings if you're tempted
        // to make this unconditional.
        info!(?reason, "drain: drain_timeout grace");
        tokio::time::sleep(cfg.drain_timeout).await;

        // Extended SSE drain only if there are still requests active —
        // they are very likely streaming clients (the non-streaming
        // path decrements the guard on response end).
        if inflight.current() > 0 {
            info!(?reason, "drain: extended stream drain");
            let _ = wait_inflight_zero(&inflight, cfg.extended_stream_drain).await;
        }
    } else {
        info!(?reason, "drain: inflight already zero, skipping grace");
    }

    info!(?reason, "drain: SIGTERM");
    match sigterm_then_sigkill(workload, cfg.sigterm_grace).await {
        SigtermOutcome::Exited => info!(?reason, "drain: workload exited gracefully"),
        SigtermOutcome::Killed => warn!(?reason, "drain: SIGKILL after grace"),
    }
}

/// Balloon fast-path: short SIGTERM grace then SIGKILL; no inflight wait.
pub async fn fast_kill(workload: &mut ManagedWorkload, reason: DrainReason) {
    warn!(?reason, "fast_kill: SIGTERM + short grace");
    let _ = sigterm_then_sigkill(workload, FAST_KILL_SIGTERM_GRACE).await;
}

/// Send SIGTERM to `workload` and wait up to `grace` for it to exit.
/// Escalates to SIGKILL on timeout, then cleans the workload up.
///
/// Every terminal path funnels through here — `drain_pipeline`, `fast_kill`,
/// and the supervisor's starting/running abort paths — which is why the
/// cleanup belongs here rather than at each call site. For a native process
/// it is a no-op (the child's drop handle reaps it); for a container it is
/// the explicit removal that stands in for the `--rm` this design
/// deliberately does not use, and it runs only after the exit status has
/// been read so the final logs are already on their way to the batcher.
pub async fn sigterm_then_sigkill(
    workload: &mut ManagedWorkload,
    grace: Duration,
) -> SigtermOutcome {
    let _ = workload.terminate().await;
    let outcome = match tokio::time::timeout(grace, workload.wait()).await {
        // Only an actual exit status counts as having stopped. An `Err`
        // means the runtime could not tell us — treating that as graceful
        // would skip the escalation and leave the workload running while
        // the supervisor frees its reservation.
        Ok(Ok(_)) => SigtermOutcome::Exited,
        Ok(Err(e)) => {
            warn!(error = %e, "could not confirm exit; escalating");
            let _ = workload.kill().await;
            SigtermOutcome::Killed
        }
        Err(_) => {
            let _ = workload.kill().await;
            // A killed workload still has an exit status to collect and a
            // log follower to let finish; removing before that truncates
            // the record of why it had to be killed.
            let _ = tokio::time::timeout(POST_KILL_GRACE, workload.wait()).await;
            SigtermOutcome::Killed
        }
    };
    if let Err(e) = workload.cleanup().await {
        warn!(error = %e, "workload cleanup failed; startup reconciliation will retry");
    }
    outcome
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SigtermOutcome {
    /// Child exited within the grace window.
    Exited,
    /// Grace elapsed; SIGKILL issued.
    Killed,
}

/// Poll `inflight` until it reaches zero or `bound` elapses. Returns `true`
/// if the bound expired before the counter reached zero.
async fn wait_inflight_zero(inflight: &InflightCounter, bound: Duration) -> bool {
    let deadline = tokio::time::Instant::now() + bound;
    loop {
        if inflight.current() == 0 {
            return false;
        }
        if tokio::time::Instant::now() >= deadline {
            return true;
        }
        tokio::time::sleep(INFLIGHT_POLL_INTERVAL).await;
    }
}
