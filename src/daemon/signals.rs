//! Linux-only: signal handling via `tokio::signal::unix`.
//! SIGTERM/SIGINT → graceful drain, SIGQUIT → emergency.

use std::time::Duration;

use tokio::signal::unix::{SignalKind, signal};
use tracing::info;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownKind {
    Graceful,
    Emergency,
}

/// Blocks until a shutdown signal arrives.
pub async fn await_shutdown() -> ShutdownKind {
    let mut term = signal(SignalKind::terminate()).expect("SIGTERM handler");
    let mut int = signal(SignalKind::interrupt()).expect("SIGINT handler");
    let mut quit = signal(SignalKind::quit()).expect("SIGQUIT handler");
    tokio::select! {
        _ = term.recv() => { info!("SIGTERM received"); ShutdownKind::Graceful }
        _ = int.recv() => { info!("SIGINT received"); ShutdownKind::Graceful }
        _ = quit.recv() => { info!("SIGQUIT received"); ShutdownKind::Emergency }
    }
}

pub fn grace_for(kind: ShutdownKind) -> Duration {
    match kind {
        ShutdownKind::Graceful => Duration::from_secs(10),
        ShutdownKind::Emergency => Duration::from_secs(5),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grace_is_shorter_for_emergency() {
        assert!(grace_for(ShutdownKind::Emergency) < grace_for(ShutdownKind::Graceful));
    }
}
