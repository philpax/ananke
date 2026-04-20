//! Full drain pipeline per spec §10.3. Sends SIGTERM via the child's
//! `ManagedChild::sigterm` and escalates to SIGKILL on timeout. The
//! [`ProcessSpawner`](crate::system::ProcessSpawner) abstraction means the
//! same pipeline works against real children under `LocalSpawner` and
//! against purely virtual ones under `FakeSpawner` in tests.

use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use tracing::{info, warn};

use crate::system::ManagedChild;

/// Polling cadence while waiting for in-flight counters to reach zero.
const INFLIGHT_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// SIGTERM grace used by `fast_kill`. Short because the caller has already
/// decided the child has misbehaved and must go; no streaming-client courtesy.
const FAST_KILL_SIGTERM_GRACE: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainReason {
    Shutdown,
    IdleTimeout,
    Eviction,
    TtlExpired,
    UserKilled,
    ConfigChanged,
}

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
    child: &mut dyn ManagedChild,
    cfg: &DrainConfig,
    inflight: Arc<AtomicU64>,
    reason: DrainReason,
) {
    let initial_inflight = inflight.load(Ordering::Relaxed);
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
                inflight = inflight.load(Ordering::Relaxed),
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
        if inflight.load(Ordering::Relaxed) > 0 {
            info!(?reason, "drain: extended stream drain");
            let _ = wait_inflight_zero(&inflight, cfg.extended_stream_drain).await;
        }
    } else {
        info!(?reason, "drain: inflight already zero, skipping grace");
    }

    info!(?reason, "drain: SIGTERM");
    match sigterm_then_sigkill(child, cfg.sigterm_grace).await {
        SigtermOutcome::Exited => info!(?reason, "drain: child exited gracefully"),
        SigtermOutcome::Killed => warn!(?reason, "drain: SIGKILL after grace"),
    }
}

/// Balloon fast-path: short SIGTERM grace then SIGKILL; no inflight wait.
pub async fn fast_kill(child: &mut dyn ManagedChild, reason: DrainReason) {
    warn!(?reason, "fast_kill: SIGTERM + short grace");
    let _ = sigterm_then_sigkill(child, FAST_KILL_SIGTERM_GRACE).await;
}

/// Send SIGTERM to `child` and wait up to `grace` for it to exit. Escalates
/// to SIGKILL on timeout. Shared between `drain_pipeline`, `fast_kill`, and
/// the supervisor's starting/running abort paths.
pub async fn sigterm_then_sigkill(child: &mut dyn ManagedChild, grace: Duration) -> SigtermOutcome {
    let _ = child.sigterm().await;
    match tokio::time::timeout(grace, child.wait()).await {
        Ok(_) => SigtermOutcome::Exited,
        Err(_) => {
            let _ = child.sigkill().await;
            SigtermOutcome::Killed
        }
    }
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
async fn wait_inflight_zero(inflight: &AtomicU64, bound: Duration) -> bool {
    let deadline = tokio::time::Instant::now() + bound;
    loop {
        if inflight.load(Ordering::Relaxed) == 0 {
            return false;
        }
        if tokio::time::Instant::now() >= deadline {
            return true;
        }
        tokio::time::sleep(INFLIGHT_POLL_INTERVAL).await;
    }
}
