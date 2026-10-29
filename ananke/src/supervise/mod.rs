//! Service supervision: per-service tokio tasks, child lifetimes, health loops.
//!
//! Re-exported from `ananke-supervise` so `crate::supervise::…` paths
//! inside the daemon are unchanged by the split.

pub use ananke_supervise::{
    DisableResult, EnableResult, EnsureFailure, EnsureOutcome, EnsureResponse, EnsureSource,
    KillHandle, OrphanDisposition, PersistentWatchDeps, PlacementOutcome, PreviewError,
    ServiceIdentity, SpawnConfig, StartFailure, StartFailureKind, StartOutcome, SupervisorCommand,
    SupervisorDeps, SupervisorHandle, SupervisorInit, SupervisorSnapshot, await_ensure, drain,
    ensure, estimate_cache, eviction_candidates, genstall, handle, health, launch, logs,
    manual_reservation, orphans, persistent_watcher, preview, preview_command,
    preview_command_placement, preview_container_command, preview_override_placement,
    preview_placement, provision, reconcile, reconciler, registry, render_argv, spawn,
    spawn_supervisor, state, workload,
};
