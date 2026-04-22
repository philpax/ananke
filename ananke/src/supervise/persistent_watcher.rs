//! Background loop that re-ensures every persistent service sitting in
//! a dormant state (Idle / Evicted / Disabled{NoFit}).
//!
//! A persistent service auto-respawns when VRAM permits — the corollary
//! of making persistent services evictable while idle. The cheapest
//! implementation is a short poll: tick every
//! `TICK` seconds, snapshot each persistent supervisor, and fire
//! `enable()` + `ensure()` on anything dormant. If the ensure itself
//! hits `NoFit` (capacity still not available), it drops the service
//! back into `Disabled{NoFit}` and we try again on the next tick — no
//! special-case needed for "wait until VRAM frees".
//!
//! The tick is deliberately slower than the supervisor command
//! channel is fast, so we don't meaningfully load the registry.
//! `ensure()` is idempotent once a service is already
//! Starting/Running, so racing it against a user-driven
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
    // Skip rather than burst on missed ticks: if a prior `tick.tick()`
    // was blocked (e.g. awaiting a peer snapshot that queued behind a
    // long drain), we don't want to fire N back-to-back re-ensures —
    // one catch-up is enough.
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
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

    // Passive: if *any* other service is currently active
    // (Starting/Running), defer. The watcher's job is to
    // reclaim VRAM that's going unused, not to fight on-demand traffic
    // for it. If an on-demand service idle-drains out, the next tick
    // finds the pool quiet and re-ensures; if the operator stops the
    // on-demand service explicitly, same. This keeps the persistent
    // service's respawn from ping-ponging against every peer start.
    let any_peer_active = state.registry.all().into_iter().any(|(name, handle)| {
        let is_persistent = effective
            .services
            .iter()
            .any(|s| s.name == name && s.lifecycle == Lifecycle::Persistent);
        if is_persistent {
            return false;
        }
        matches!(
            handle.peek_state(),
            ServiceState::Starting | ServiceState::Running,
        )
    });
    if any_peer_active {
        return;
    }

    for svc_cfg in effective.services.iter() {
        if svc_cfg.lifecycle != Lifecycle::Persistent {
            continue;
        }
        let Some(handle) = state.registry.get(&svc_cfg.name) else {
            continue;
        };
        // Read state via the lock-free mirror so the watcher doesn't
        // stack commands into the supervisor's bounded mailbox on
        // every tick. Without this, a supervisor backed up inside its
        // own `handle_idle_ensure` (serial drains, etc.) would deadlock
        // with the watcher when the mailbox fills.
        let peer_state = handle.peek_state();
        if !dormant(&peer_state) {
            continue;
        }
        let h = handle.clone();
        let name = svc_cfg.name.clone();
        tokio::spawn(async move {
            info!(service = %name, dormant_state = %peer_state.name(), "re-ensuring persistent service");
            // Disabled{NoFit} supervisors reject Ensure outright; Enable
            // transitions back to Idle so the next Ensure can run. Enable
            // is a no-op on non-Disabled states so it's safe to fire
            // unconditionally.
            let _ = h.enable().await;
            let _ = h.ensure(crate::supervise::EnsureSource::BackgroundWatcher).await;
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
