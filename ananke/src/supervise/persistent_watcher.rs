//! Background loop that re-ensures every persistent service sitting in
//! a dormant state (Idle / Evicted / Disabled{NoFit}).
//!
//! Per spec §5.1, a persistent service "auto-respawns when VRAM
//! permits" — the corollary of making persistent services evictable
//! while idle. The cheapest implementation is a short poll: tick every
//! `TICK` seconds, snapshot each persistent supervisor, and fire
//! `enable()` + `ensure()` on anything dormant. If the ensure itself
//! hits `NoFit` (capacity still not available), it drops the service
//! back into `Disabled{NoFit}` and we try again on the next tick — no
//! special-case needed for "wait until VRAM frees".
//!
//! The tick is deliberately slower than the supervisor command
//! channel is fast, so we don't meaningfully load the registry.
//! `ensure()` is idempotent once a service is already
//! Starting/Warming/Running, so racing it against a user-driven
//! `/v1/chat/completions` causes no harm.

use std::time::Duration;

use tokio::sync::watch;
use tracing::info;

use crate::{
    config::Lifecycle,
    daemon::app_state::AppState,
    supervise::state::{DisableReason, ServiceState},
};

const TICK: Duration = Duration::from_secs(5);

pub async fn run_loop(state: AppState, mut shutdown: watch::Receiver<bool>) {
    let mut tick = tokio::time::interval(TICK);
    // Absorb the immediate tick so we pace from `TICK` onward; the
    // daemon's boot-time `provision::boot` already ensures every
    // persistent service once, and re-firing at t=0 would race with it.
    tick.tick().await;

    loop {
        tokio::select! {
            _ = shutdown.changed() => {
                if *shutdown.borrow() { return; }
            }
            _ = tick.tick() => {
                re_ensure_persistent(&state).await;
            }
        }
    }
}

async fn re_ensure_persistent(state: &AppState) {
    let effective = state.config.effective();
    for svc_cfg in effective.services.iter() {
        if svc_cfg.lifecycle != Lifecycle::Persistent {
            continue;
        }
        let Some(handle) = state.registry.get(&svc_cfg.name) else {
            continue;
        };
        let Some(snap) = handle.snapshot().await else {
            continue;
        };
        if !dormant(&snap.state) {
            continue;
        }
        let h = handle.clone();
        let name = svc_cfg.name.clone();
        tokio::spawn(async move {
            info!(service = %name, dormant_state = %snap.state.name(), "re-ensuring persistent service");
            // Disabled{NoFit} supervisors reject Ensure outright; Enable
            // transitions back to Idle so the next Ensure can run. Enable
            // is a no-op on non-Disabled states so it's safe to fire
            // unconditionally.
            let _ = h.enable().await;
            let _ = h.ensure().await;
        });
    }
}

/// States from which a persistent service can be legally restarted
/// without operator intervention. `Failed` needs its retry backoff;
/// `Disabled` for reasons other than `NoFit` (ConfigError, CrashLoop,
/// UserDisabled, etc.) stays disabled until something human-shaped
/// intervenes.
fn dormant(state: &ServiceState) -> bool {
    matches!(
        state,
        ServiceState::Idle
            | ServiceState::Evicted
            | ServiceState::Disabled {
                reason: DisableReason::NoFit,
            }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dormant_matches_expected_states() {
        assert!(dormant(&ServiceState::Idle));
        assert!(dormant(&ServiceState::Evicted));
        assert!(dormant(&ServiceState::Disabled {
            reason: DisableReason::NoFit,
        }));

        assert!(!dormant(&ServiceState::Starting));
        assert!(!dormant(&ServiceState::Warming));
        assert!(!dormant(&ServiceState::Running));
        assert!(!dormant(&ServiceState::Draining));
        assert!(!dormant(&ServiceState::Stopped));
        assert!(!dormant(&ServiceState::Failed { retry_count: 0 }));
        assert!(!dormant(&ServiceState::Disabled {
            reason: DisableReason::UserDisabled,
        }));
        assert!(!dormant(&ServiceState::Disabled {
            reason: DisableReason::CrashLoop,
        }));
        assert!(!dormant(&ServiceState::Disabled {
            reason: DisableReason::ConfigError("x".into()),
        }));
    }
}
