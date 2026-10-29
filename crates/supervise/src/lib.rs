//! Service supervision: per-service tokio tasks, child lifetimes, health loops.
//!
//! Linux-coupled via `os::unix::process::ExitStatusExt` (signal() on ExitStatus)
//! and the submodules it delegates to (`drain`, `orphans`, `spawn`).
//!
//! [`handle`] is the client-facing entry point (the command/response types and
//! [`spawn_supervisor`]); this file holds the actor loop itself — [`RunLoop`],
//! its dispatcher [`run`], and the `Step`/`*Outcome` vocabulary shared across
//! every phase module ([`idle`], [`starting`], [`running`], [`dispatch`],
//! [`terminal`]) plus the supporting [`context`], [`reservation`],
//! [`eviction`], [`ensure`], [`watchdogs`], and [`auto_restart`] modules.

pub mod drain;
pub mod ensure;
pub mod estimate_cache;
pub mod genstall;
pub mod handle;
pub mod health;
pub mod logs;
pub mod manual_reservation;
pub mod orphans;
pub mod persistent_watcher;
pub mod preview;
pub mod provision;
pub mod reconciler;
pub mod registry;
pub mod spawn;
pub mod state;
pub mod workload;

mod auto_restart;
mod context;
mod dispatch;
mod eviction;
mod idle;
pub mod launch;
mod reservation;
mod rolling;
mod running;
mod starting;
mod terminal;
mod watchdogs;

use std::{sync::Arc, time::Duration};

use ananke_allocator::placement::Packed;
use ananke_config::validate::ServiceConfig;
pub use ananke_placement::{DrainReason, KillHandle};
pub use orphans::{OrphanDisposition, reconcile};
use parking_lot::Mutex as SyncMutex;
pub use persistent_watcher::PersistentWatchDeps;
use tokio::sync::{broadcast, mpsc, watch};
use tracing::warn;

use crate::handle::MirroredState;
pub use crate::{
    context::slot_to_key,
    ensure::{EnsureFailure, EnsureOutcome, await_ensure},
    eviction::eviction_candidates,
    handle::{
        DisableResult, EnableResult, EnsureResponse, EnsureSource, ServiceIdentity, StartFailure,
        StartFailureKind, StartOutcome, SupervisorCommand, SupervisorDeps, SupervisorHandle,
        SupervisorInit, SupervisorSnapshot, spawn_supervisor,
    },
    preview::{
        PlacementOutcome, PreviewError, preview_command, preview_command_placement,
        preview_container_command, preview_override_placement, preview_placement,
    },
    spawn::{SpawnConfig, render_argv},
};

/// SIGTERM grace during Starting where the child may not yet be ready to
/// drain gracefully. Short so we do not block shutdown on a half-loaded
/// child.
const STARTING_SIGTERM_GRACE: Duration = Duration::from_secs(5);

/// Live state for the generation-stall watchdog across one run: the poll
/// cadence, the pure stall-decision state, and the HTTP bits for reaching the
/// child's `/metrics`. Seeded at Running entry by `genstall_setup`.
struct GenStallLoop {
    poll: tokio::time::Interval,
    state: genstall::GenStallState,
    client: reqwest::Client,
    url: String,
    /// Whether the one-time "cannot read /metrics" warning has been logged.
    warned_unreachable: bool,
}

async fn run(
    init: SupervisorInit,
    boot_svc: ServiceConfig,
    deps: SupervisorDeps,
    rx: mpsc::Receiver<SupervisorCommand>,
    mirror: Arc<SyncMutex<MirroredState>>,
) {
    let mut loop_state = RunLoop::new(init, boot_svc, deps, rx, mirror);
    loop {
        let step = match loop_state.read_state() {
            state::ServiceState::Idle => loop_state.handle_idle().await,
            state::ServiceState::Starting => loop_state.handle_active_lifecycle().await,
            state::ServiceState::Failed { retry_count } => {
                loop_state.handle_failed(retry_count).await
            }
            state::ServiceState::Disabled { .. } => loop_state.handle_disabled().await,
            other => {
                warn!(state = ?other, "unexpected state in supervisor loop");
                return;
            }
        };
        if matches!(step, Step::Exit) {
            return;
        }
    }
}

/// Result of a `handle_*` method: either continue the outer dispatcher loop
/// (consulting the updated `state`) or exit the supervisor task entirely.
enum Step {
    Continue,
    Exit,
}

/// Mutable context threaded through every `handle_*` method. Owns every
/// binding that outlives a single state's body and is read or mutated across
/// transitions. The lifecycle phase, run_id, and pid all live in the shared
/// `mirror` cell; there is no local `state` field — readers call
/// [`Self::read_state`] or [`Self::read_full`].
struct RunLoop {
    init: SupervisorInit,
    deps: SupervisorDeps,
    rx: mpsc::Receiver<SupervisorCommand>,
    mirror: Arc<SyncMutex<MirroredState>>,
    cancel_tx: watch::Sender<bool>,
    cancel_rx: watch::Receiver<bool>,
    /// Carries a broadcast sender from Idle through to Starting so waiters can
    /// be notified of the outcome once the child passes the health probe.
    start_bus_carry: Option<broadcast::Sender<StartOutcome>>,
    /// When Some, the supervisor is parked in Idle waiting for a busy peer
    /// to idle out so we can evict it. The bus is shared with the original
    /// caller (via `EnsureResponse::Waiting`) and resolves to `Ok` once the
    /// fit succeeds or to `Err` on shutdown / disable / hard-reject.
    pending_ensure_bus: Option<broadcast::Sender<StartOutcome>>,
    /// The source of the currently parked Ensure (matches the original
    /// `EnsureSource` that entered the queue). Consulted by
    /// `retry_queued_ensure` for the yield-to-nonpersistent check.
    pending_ensure_source: EnsureSource,
    /// Busy peers we're waiting on for the queued Ensure. The poll-tick
    /// branch of the Idle loop does a cheap atomic-load check against
    /// these peers' inflight counters and skips the expensive estimator-
    /// plus-packer path entirely while they're all still above zero.
    /// Updated every time the retry returns `WaitForBusy` with a
    /// potentially different set of peers.
    queued_watch: Vec<smol_str::SmolStr>,
    /// Wall-monotonic stamp captured the first time the current Ensure
    /// entered the queue. `QUEUE_BLOCKED_GRACE` is measured against
    /// this; the value is preserved across retry ticks (only cleared
    /// when the queue resolves, succeeds, or hard-fails) so a flapping
    /// `busy_peers` set doesn't reset the wait clock and let the client
    /// hang forever.
    queued_since: Option<tokio::time::Instant>,
    /// Carries the placement-derived `CommandArgs` from Idle (where they are
    /// computed) into Starting (where `render_argv` consumes them).
    packed_for_spawn: Option<Packed>,
    /// Counts consecutive OOM kills for the current service.
    oom_attempts: u32,
    /// The placement inputs the rolling correction needs, captured when this
    /// run's reservation was committed and consumed when the service drains
    /// back to Idle. See [`rolling::RollingBase`].
    rolling_base: rolling::RollingBase,
    /// Boot-time snapshot of this service's config, consulted by
    /// [`Self::current_svc`] as a fallback only when the live config has
    /// dropped the service (mid-reload). Never mutated — the live
    /// [`ConfigManager`] is always preferred.
    boot_svc: ServiceConfig,
    /// Monotonic timestamps of recent error-rate auto-restarts, pruned to the
    /// flap window. Lives on the supervisor (not the run) so it survives the
    /// drain → respawn cycle and can trip the flap cap. Only the error-rate
    /// trigger touches this — periodic restarts are intentional maintenance
    /// and must not count toward the cap.
    auto_restart_history: Vec<tokio::time::Instant>,
    /// Set by the periodic `on-request` trigger when the interval elapses: the
    /// current run is stale and the next incoming request drives a drain →
    /// respawn (carried through [`Self::deferred_ensure`]) instead of hitting
    /// the wedged child.
    restart_pending: bool,
    /// An `Ensure` whose reply is deferred across an on-request restart drain.
    /// Held from the Running `Ensure` handler through the drain, then replayed
    /// at the top of [`Self::handle_idle`] so the triggering request blocks on
    /// the fresh process via the normal idle-ensure spawn path.
    deferred_ensure: Option<(tokio::sync::oneshot::Sender<EnsureResponse>, EnsureSource)>,
}

/// Outcome of a sub-step inside the Starting-through-Draining pipeline.
enum StartingOutcome {
    /// Keep spinning the current inner select.
    Continue,
    /// Fall out of the Starting outer select (back to the dispatcher).
    Break,
    /// Exit the supervisor task entirely.
    Exit,
}

/// Outcome of a single command dispatch inside the Running inner loop.
enum RunningOutcome {
    Continue,
    Break,
    Exit,
}

/// Outcome of an error-rate watchdog firing: the service was either drained
/// and returned to Idle for respawn, or disabled after tripping the flap cap.
/// Both leave the Running loop.
enum AutoRestartOutcome {
    Restarted,
    Disabled,
}

/// Outcome of a periodic-timer tick: either the loop keeps running (the
/// trigger armed the request path, or `on-idle` is still waiting for a quiet
/// window) or the service was drained and must leave the Running loop.
enum PeriodicOutcome {
    Continue,
    Restarted,
}
