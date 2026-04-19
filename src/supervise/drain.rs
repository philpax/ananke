//! Full drain pipeline per spec §10.3.

use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use tokio::process::Child;
use tracing::{info, warn};

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
    child: &mut Child,
    cfg: &DrainConfig,
    inflight: Arc<AtomicU64>,
    reason: DrainReason,
) {
    info!(?reason, "drain: waiting for in-flight requests");
    let deadline = tokio::time::Instant::now() + cfg.max_request_duration;
    loop {
        if inflight.load(Ordering::Relaxed) == 0 {
            break;
        }
        if tokio::time::Instant::now() >= deadline {
            warn!(
                ?reason,
                inflight = inflight.load(Ordering::Relaxed),
                "drain: max_request_duration elapsed with requests still in flight"
            );
            break;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }

    info!(?reason, "drain: drain_timeout grace");
    tokio::time::sleep(cfg.drain_timeout).await;

    // Extended SSE drain only if there are still requests active — they
    // are very likely streaming clients (the non-streaming path decrements
    // the guard on response end).
    if inflight.load(Ordering::Relaxed) > 0 {
        info!(?reason, "drain: extended stream drain");
        let stream_deadline = tokio::time::Instant::now() + cfg.extended_stream_drain;
        loop {
            if inflight.load(Ordering::Relaxed) == 0 {
                break;
            }
            if tokio::time::Instant::now() >= stream_deadline {
                break;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    info!(?reason, "drain: SIGTERM");
    if let Some(pid) = child.id() {
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(pid as i32),
            nix::sys::signal::Signal::SIGTERM,
        );
    }
    match tokio::time::timeout(cfg.sigterm_grace, child.wait()).await {
        Ok(_) => info!(?reason, "drain: child exited gracefully"),
        Err(_) => {
            warn!(?reason, "drain: SIGKILL after grace");
            let _ = child.kill().await;
        }
    }
}

/// Balloon fast-path: 5 s SIGTERM grace then SIGKILL; no inflight wait.
pub async fn fast_kill(child: &mut Child, reason: DrainReason) {
    warn!(?reason, "fast_kill: SIGTERM + 5s grace");
    if let Some(pid) = child.id() {
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(pid as i32),
            nix::sys::signal::Signal::SIGTERM,
        );
    }
    match tokio::time::timeout(Duration::from_secs(5), child.wait()).await {
        Ok(_) => {}
        Err(_) => {
            let _ = child.kill().await;
        }
    }
}
