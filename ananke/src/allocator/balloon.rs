//! Per-dynamic-service balloon resolver.
//!
//! Each dynamic service gets one resolver task. The task samples observed
//! VRAM every `SAMPLE_INTERVAL`, maintains a rolling window, and takes
//! action when either the max-VRAM ceiling is breached for too long or
//! growth pressure is detected while contention with a borrower exists.

use std::{collections::VecDeque, time::Duration};

use parking_lot::Mutex;
use smol_str::SmolStr;
use tokio::sync::watch;
use tracing::{debug, info, warn};

use crate::{
    allocator::AllocationTable,
    config::{
        Lifecycle,
        manager::ConfigManager,
        validate::{DEFAULT_SERVICE_PRIORITY, DeviceSlot},
    },
    daemon::events::EventBus,
    devices::snapshotter::SharedSnapshot,
    supervise::{drain::DrainReason, registry::ServiceRegistry, slot_to_key},
    tracking::observation::ObservationTable,
};

const WINDOW_SIZE: usize = 6;
const SAMPLE_INTERVAL: Duration = Duration::from_secs(2);

/// If the dynamic service exceeds `max_mb * 110 %` for this long, fast-kill it.
const OVER_CEILING_GRACE: Duration = Duration::from_secs(30);

/// Headroom above `max_mb` tolerated before `OVER_CEILING_GRACE` applies, as
/// permille (110 ‰ = +10 %, i.e. 1.10 ×).
const OVER_CEILING_PERMILLE: u64 = 1100;

/// Pledge update rate-limit: ignore deltas smaller than this many MiB. A
/// dynamic service drifting by a few hundred MiB shouldn't churn the event
/// stream; a 5 % relative threshold catches the same pattern at higher pledges
/// (a 12 GiB pledge → 600 MiB sensitivity).
const PLEDGE_DELTA_FLOOR_MB: u64 = 256;
const PLEDGE_DELTA_PERMILLE: u64 = 50; // 5.0 %

/// Detect growth in a sample window using linear-regression slope and a
/// majority-non-decreasing jitter check.
///
/// Returns `true` when:
/// 1. The window has at least `WINDOW_SIZE / 2 + 1` samples.
/// 2. The OLS slope over the window is strictly positive.
/// 3. A majority of adjacent-sample deltas are non-negative (jitter tolerance).
/// 4. The slope-projected next sample would exceed `floor_bytes`.
pub fn detect_growth(window: &VecDeque<u64>, floor_bytes: u64) -> bool {
    let min_samples = WINDOW_SIZE / 2 + 1;
    if window.len() < min_samples {
        return false;
    }

    let n = window.len() as i64;
    let mut sum_x: i64 = 0;
    let mut sum_y: i64 = 0;
    let mut sum_xy: i64 = 0;
    let mut sum_xx: i64 = 0;
    for (i, v) in window.iter().enumerate() {
        let x = i as i64;
        let y = *v as i64;
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }
    let denom = n * sum_xx - sum_x * sum_x;
    if denom == 0 {
        return false;
    }
    // Integer slope; positive means the fit line is rising.
    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    if slope <= 0 {
        return false;
    }

    // Majority of consecutive pairs must be non-decreasing.
    let samples: Vec<u64> = window.iter().copied().collect();
    let total = samples.len().saturating_sub(1);
    if total == 0 {
        return false;
    }
    let non_neg = samples.windows(2).filter(|pair| pair[1] >= pair[0]).count();
    if non_neg * 2 <= total {
        return false;
    }

    // The slope-projected next value must exceed the floor.
    let last = *window.back().unwrap() as i64;
    let projected = last + slope;
    projected as u64 > floor_bytes
}

#[derive(Debug, Clone)]
pub struct BalloonConfig {
    pub min_mb: u64,
    pub max_mb: u64,
    /// Minimum time a borrower must have been alive before we fast-kill it.
    pub min_borrower_runtime: Duration,
    /// Extra headroom added to the `min_mb` floor for growth detection.
    pub margin_bytes: u64,
}

/// Compute the pledge a dynamic service should hold given a recent
/// observation window. The window's max acts as a "recent peak" — transient
/// spikes lift the pledge while the corresponding sample is in the window
/// and decay back as samples roll out, so a one-time burst doesn't pin the
/// reservation forever. Always clamped to `[min_mb, max_mb]` per the
/// dynamic-allocation contract.
///
/// Returns `None` when the window is empty (no sample has been taken yet).
pub fn pledge_from_window(window: &VecDeque<u64>, min_mb: u64, max_mb: u64) -> Option<u64> {
    let peak_bytes = *window.iter().max()?;
    let peak_mb = peak_bytes / (1024 * 1024);
    Some(peak_mb.clamp(min_mb, max_mb))
}

/// Decide whether a pledge change is meaningful enough to push to the
/// allocation table (and out via `AllocationChanged`).
///
/// Rules:
/// - First-ever update (no current pledge) always emits.
/// - No change → don't emit.
/// - Crossing the `min_mb` boundary (baseline ↔ above-baseline) always
///   emits; peers should know immediately when slack appears or vanishes.
/// - Otherwise gate on a delta of at least 5 % of the current pledge or
///   `PLEDGE_DELTA_FLOOR_MB`, whichever is larger.
pub fn should_update_pledge(current_mb: Option<u64>, desired_mb: u64, min_mb: u64) -> bool {
    let Some(prev) = current_mb else {
        return true;
    };
    if prev == desired_mb {
        return false;
    }
    if (prev > min_mb) != (desired_mb > min_mb) {
        return true;
    }
    let delta = prev.abs_diff(desired_mb);
    let percent_threshold = (prev * PLEDGE_DELTA_PERMILLE) / 1000;
    delta >= percent_threshold.max(PLEDGE_DELTA_FLOOR_MB)
}

/// Inputs to [`spawn_resolver`] beyond the per-service `BalloonConfig` —
/// the shared collaborators every resolver task needs. Bundled into a
/// struct so the spawn signature stays under clippy's argument limit and
/// callers can build the inputs once and clone.
pub struct ResolverDeps {
    pub observation: ObservationTable,
    pub registry: ServiceRegistry,
    pub allocations: std::sync::Arc<Mutex<AllocationTable>>,
    pub events: EventBus,
    /// Live snapshot used to compute per-GPU pledge totals against
    /// physical capacity — the contention resolver only fires when a GPU
    /// the service holds is actually over-pledged.
    pub snapshot: SharedSnapshot,
    /// Live `ConfigManager` so the contention resolver can look up a
    /// peer's lifecycle to apply the persistent-vs-on-demand tie-break.
    pub config: std::sync::Arc<ConfigManager>,
    pub shutdown: watch::Receiver<bool>,
}

/// Spawn a balloon-resolver task for a dynamic service.
///
/// The task runs until `deps.shutdown` fires `true`. Each `SAMPLE_INTERVAL` it:
/// 1. Reads the observed VRAM peak and slides the sample window forward.
/// 2. Reconciles the pledge book against the recent peak so other services'
///    fit decisions see realistic usage rather than the stale `min_mb`
///    floor (rate-limited by [`should_update_pledge`]).
/// 3. Enforces the `max_mb * 110 %` ceiling for `OVER_CEILING_GRACE`.
/// 4. Resolves contention by fast-killing the lower-priority side when
///    growth pressure is detected and a borrower is present.
pub fn spawn_resolver(
    service_name: SmolStr,
    cfg: BalloonConfig,
    svc_priority: u8,
    svc_lifecycle: Lifecycle,
    deps: ResolverDeps,
) -> tokio::task::JoinHandle<()> {
    let ResolverDeps {
        observation,
        registry,
        allocations,
        events,
        snapshot,
        config,
        mut shutdown,
    } = deps;
    tokio::spawn(async move {
        let mut window: VecDeque<u64> = VecDeque::with_capacity(WINDOW_SIZE);
        let mut exceeded_since: Option<std::time::Instant> = None;
        let mut tick = tokio::time::interval(SAMPLE_INTERVAL);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        loop {
            tokio::select! {
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        return;
                    }
                }
                _ = tick.tick() => {}
            }

            // VRAM-only peak: the dynamic pledge models GPU bytes, not
            // the python interpreter's RSS. Combining the two used to
            // inflate the pledge with multi-GB CPU footprint and trigger
            // false over-commit signals during normal SDXL inference.
            let observed = observation.read_peak_vram(&service_name);
            if window.len() == WINDOW_SIZE {
                window.pop_front();
            }
            window.push_back(observed);

            reconcile_pledge(&service_name, &window, &cfg, &allocations, &events);

            // If observed > max_mb by >10% for >30 s, fast-kill self.
            //
            // The resolver does NOT terminate after firing — fast_kill
            // drains the supervisor, which then re-ensures via its
            // normal lifecycle. The resolver task lives for the whole
            // daemon run (one task per service, spawned at provision
            // time) and re-arms across the kill: window cleared, grace
            // timer reset, observation cleared by the drain itself.
            // Returning here would orphan the resolver for the rest of
            // the daemon's lifetime, leaving subsequent runs of the
            // service without pledge tracking or contention guarding.
            let ceiling = cfg.max_mb * 1024 * 1024 * OVER_CEILING_PERMILLE / 1000;
            if observed > ceiling {
                if let Some(since) = exceeded_since {
                    if since.elapsed() > OVER_CEILING_GRACE {
                        warn!(
                            service = %service_name,
                            observed,
                            max_bytes = cfg.max_mb * 1024 * 1024,
                            "balloon: max_vram_gb exceeded by >10% for >30s; fast-killing dynamic service",
                        );
                        if let Some(handle) = registry.get(&service_name) {
                            handle.fast_kill(DrainReason::Eviction).await;
                        }
                        window.clear();
                        exceeded_since = None;
                        continue;
                    }
                } else {
                    exceeded_since = Some(std::time::Instant::now());
                }
            } else {
                exceeded_since = None;
            }

            // Contention resolver. Pre-condition for firing:
            // 1. Growth detected in the recent sample window.
            // 2. A GPU the service holds is OOM-pressured — physical
            //    NVML free has dropped below `OOM_MARGIN_BYTES`. The
            //    earlier "pledge sum > total" gate over-fired: pledges
            //    are upper-bound reservations (estimator predictions
            //    pad ~10–20 %, dynamic services hold the recent-peak
            //    high-water mark), so two services can pledge to >100 %
            //    without ever actually filling the GPU. Only the kernel
            //    knows when the next allocation is about to fail, and
            //    NVML free is the signal it gives us.
            //
            // When the gate fires, we identify a peer to resolve against
            // using priority + lifecycle: strict numeric priority always
            // wins; at tied priority, on-demand yields to persistent.
            let floor = cfg.min_mb * 1024 * 1024 + cfg.margin_bytes;
            if detect_growth(&window, floor) {
                let reservations = allocations.lock().clone();
                let snap_now = snapshot.read().clone();
                let overcommitted = overcommitted_gpus_for(&service_name, &reservations, &snap_now);
                if overcommitted.is_empty() {
                    debug!(
                        service = %service_name,
                        observed,
                        "balloon: growth detected but no GPU is over-committed; deferring to pledge book",
                    );
                } else {
                    let cfg_snapshot = config.effective();
                    let resolution = resolve_contention(
                        &service_name,
                        svc_priority,
                        svc_lifecycle,
                        &reservations,
                        &overcommitted,
                        &registry,
                        &cfg_snapshot.services,
                    );
                    drop(cfg_snapshot);
                    match resolution {
                        ContentionAction::EvictPeer { peer } => {
                            info!(
                                service = %service_name,
                                peer = %peer,
                                gpus = ?overcommitted,
                                "balloon: over-committed GPU; evicting lower-ranked peer",
                            );
                            if let Some(handle) = registry.get(&peer) {
                                handle.fast_kill(DrainReason::Eviction).await;
                            }
                            window.clear();
                        }
                        ContentionAction::YieldSelf { to } => {
                            info!(
                                service = %service_name,
                                to = %to,
                                gpus = ?overcommitted,
                                "balloon: over-committed GPU; yielding to higher-ranked peer",
                            );
                            if let Some(handle) = registry.get(&service_name) {
                                handle.fast_kill(DrainReason::Eviction).await;
                            }
                            // Don't terminate — see the over-ceiling
                            // branch above. The resolver re-arms once
                            // the supervisor finishes draining; the
                            // window resets so we don't immediately
                            // re-fire on the same stale samples.
                            window.clear();
                        }
                        ContentionAction::NoCandidate => {
                            debug!(
                                service = %service_name,
                                gpus = ?overcommitted,
                                "balloon: over-committed GPU but no peer to resolve against",
                            );
                        }
                    }
                }
            } else {
                debug!(
                    service = %service_name,
                    observed,
                    window_len = window.len(),
                    "balloon: no growth detected",
                );
            }
        }
    })
}

/// Outcome of the contention resolver's peer pick.
#[derive(Debug, PartialEq, Eq)]
enum ContentionAction {
    /// We outrank a peer on the over-committed GPU; evict it.
    EvictPeer { peer: SmolStr },
    /// A peer outranks us; yield by fast-killing self.
    YieldSelf { to: SmolStr },
    /// GPU is over-committed but no peer is a valid contention partner
    /// (e.g. we're alone on it, or every peer is itself this service).
    NoCandidate,
}

/// Physical free-VRAM headroom (in bytes) below which the resolver
/// considers a GPU "OOM-pressured" — there isn't room for the next
/// growth tick, so a contention resolution is justified.
///
/// Aligned with [`BalloonConfig::margin_bytes`] (512 MiB) by
/// convention: the same magnitude already used as the growth-detection
/// floor. Smaller and we'd let the GPU OOM before reacting; larger and
/// we'd kill services preemptively when there's still room to grow.
const OOM_MARGIN_BYTES: u64 = 512 * 1024 * 1024;

/// GPUs the named service holds an allocation on whose physical NVML
/// free-VRAM has fallen below [`OOM_MARGIN_BYTES`]. Empty when there's
/// still slack on every GPU we touch.
///
/// A pledge-sum-vs-total check (the original formulation) over-fires:
/// pledges over-state actual usage (the estimator may predict more than
/// what loads, and a dynamic service's pledge is the recent-peak
/// high-water mark, not its current footprint). Two services whose
/// pledge sums to >100 % of the GPU might in fact be using <100 %; the
/// kernel won't OOM unless the *physical* allocation overflows. NVML
/// free-bytes is the immediate signal.
fn overcommitted_gpus_for(
    service_name: &SmolStr,
    reservations: &AllocationTable,
    snapshot: &crate::devices::DeviceSnapshot,
) -> Vec<u32> {
    let mine: std::collections::BTreeSet<u32> = reservations
        .get(service_name)
        .map(|row| {
            row.keys()
                .filter_map(|s| match s {
                    DeviceSlot::Gpu(id) => Some(*id),
                    DeviceSlot::Cpu => None,
                })
                .collect()
        })
        .unwrap_or_default();
    if mine.is_empty() {
        return Vec::new();
    }
    snapshot
        .gpus
        .iter()
        .filter(|g| mine.contains(&g.id))
        .filter(|g| g.free_bytes < OOM_MARGIN_BYTES)
        .map(|g| g.id)
        .collect()
}

/// Pick a contention peer on one of `overcommitted_gpus` and decide whose
/// turn it is to leave. Pure-data over the inputs so unit tests can drive
/// every branch without spawning supervisors.
fn resolve_contention(
    service_name: &SmolStr,
    svc_priority: u8,
    svc_lifecycle: Lifecycle,
    reservations: &AllocationTable,
    overcommitted_gpus: &[u32],
    registry: &ServiceRegistry,
    services: &[crate::config::ServiceConfig],
) -> ContentionAction {
    let lifecycle_of = |name: &SmolStr| -> Lifecycle {
        services
            .iter()
            .find(|s| s.name.as_str() == name.as_str())
            .map(|s| s.lifecycle)
            .unwrap_or(Lifecycle::OnDemand)
    };
    let priority_of = |name: &SmolStr| -> u8 {
        services
            .iter()
            .find(|s| s.name.as_str() == name.as_str())
            .map(|s| s.priority)
            .unwrap_or(DEFAULT_SERVICE_PRIORITY)
    };
    for peer_name in reservations.keys() {
        if peer_name.as_str() == service_name.as_str() {
            continue;
        }
        // Peer must hold a pledge on at least one over-committed GPU.
        let peer_row = match reservations.get(peer_name) {
            Some(r) => r,
            None => continue,
        };
        let intersects = overcommitted_gpus
            .iter()
            .any(|id| peer_row.get(&DeviceSlot::Gpu(*id)).copied().unwrap_or(0) > 0);
        if !intersects {
            continue;
        }
        if registry.get(peer_name).is_none() {
            continue;
        }
        let peer_priority = priority_of(peer_name);
        let peer_lifecycle = lifecycle_of(peer_name);
        if svc_priority > peer_priority {
            return ContentionAction::EvictPeer {
                peer: peer_name.clone(),
            };
        }
        if svc_priority < peer_priority {
            return ContentionAction::YieldSelf {
                to: peer_name.clone(),
            };
        }
        // Tied numeric priority — lifecycle breaks the tie.
        match (svc_lifecycle, peer_lifecycle) {
            (Lifecycle::OnDemand, Lifecycle::Persistent) => {
                return ContentionAction::YieldSelf {
                    to: peer_name.clone(),
                };
            }
            (Lifecycle::Persistent, Lifecycle::OnDemand) => {
                return ContentionAction::EvictPeer {
                    peer: peer_name.clone(),
                };
            }
            // Both same lifecycle — fall through to the historical default:
            // the dynamic side (us, by definition of running this resolver)
            // yields. Avoids two persistent peers oscillating; in practice
            // only one of them is dynamic, so this branch is rare.
            _ => {
                return ContentionAction::YieldSelf {
                    to: peer_name.clone(),
                };
            }
        }
    }
    ContentionAction::NoCandidate
}

/// Update the pledge book row for a dynamic service so it reflects the
/// recent observed peak (clamped to `[min_mb, max_mb]`), and emit
/// `AllocationChanged` when the change clears the rate-limit gate.
///
/// The slot the service holds is read from its own row in
/// [`AllocationTable`]. Command-template services hold a single slot at a
/// time, so the first non-CPU entry (or the only entry) is the one to
/// update. No-op when the service has no row (idle, draining, or never
/// started).
fn reconcile_pledge(
    service_name: &SmolStr,
    window: &VecDeque<u64>,
    cfg: &BalloonConfig,
    allocations: &Mutex<AllocationTable>,
    events: &EventBus,
) {
    let Some(desired_mb) = pledge_from_window(window, cfg.min_mb, cfg.max_mb) else {
        return;
    };

    let mut guard = allocations.lock();
    let Some(row) = guard.get_mut(service_name) else {
        return;
    };
    // Pick the GPU slot the service is pinned to. Falls back to the only
    // entry (which may legitimately be Cpu in test setups) so a CPU-spilled
    // dynamic service still sees its pledge tracked.
    let target_slot = row
        .keys()
        .find(|s| matches!(s, DeviceSlot::Gpu(_)))
        .or_else(|| row.keys().next())
        .cloned();
    let Some(slot) = target_slot else {
        return;
    };
    let current_mb = row.get(&slot).copied();
    if !should_update_pledge(current_mb, desired_mb, cfg.min_mb) {
        return;
    }
    debug!(
        service = %service_name,
        slot = ?slot,
        previous_mb = ?current_mb,
        desired_mb,
        "balloon: reconciling pledge to observed peak"
    );
    row.insert(slot, desired_mb);

    let reservations: std::collections::BTreeMap<String, u64> = row
        .iter()
        .map(|(s, mb)| (slot_to_key(s), mb * 1024 * 1024))
        .collect();
    drop(guard);
    events.publish(ananke_api::Event::AllocationChanged {
        service: service_name.clone(),
        reservations,
        at_ms: crate::tracking::now_unix_ms(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_window(samples: &[u64]) -> VecDeque<u64> {
        VecDeque::from(samples.to_vec())
    }

    fn mb(n: u64) -> u64 {
        n * 1024 * 1024
    }

    #[test]
    fn flat_window_no_growth() {
        let w = mk_window(&[10, 10, 10, 10, 10, 10]);
        assert!(!detect_growth(&w, 0));
    }

    #[test]
    fn monotonic_growth_detected() {
        let w = mk_window(&[10, 12, 14, 16, 18, 20]);
        assert!(detect_growth(&w, 0));
    }

    #[test]
    fn noisy_but_growing_detected() {
        let w = mk_window(&[10, 13, 12, 17, 16, 20]);
        assert!(detect_growth(&w, 0));
    }

    #[test]
    fn declining_rejected() {
        let w = mk_window(&[20, 18, 16, 14, 12, 10]);
        assert!(!detect_growth(&w, 0));
    }

    #[test]
    fn insufficient_samples_rejected() {
        let w = mk_window(&[10, 20]);
        assert!(!detect_growth(&w, 0));
    }

    #[test]
    fn floor_gate_applied() {
        // Growing, but projected stays below floor.
        let w = mk_window(&[10, 11, 12, 13, 14, 15]);
        assert!(!detect_growth(&w, 1000));
    }

    // pledge_from_window ----------------------------------------------------

    #[test]
    fn pledge_from_empty_window_is_none() {
        let w = VecDeque::new();
        assert_eq!(pledge_from_window(&w, 2 * 1024, 20 * 1024), None);
    }

    #[test]
    fn pledge_clamps_to_min_when_observed_below() {
        // Observed peak < min_mb (e.g. process just started, hasn't allocated
        // anything yet). Pledge must not fall below the declared floor.
        let w = mk_window(&[mb(500), mb(800)]);
        assert_eq!(pledge_from_window(&w, 2 * 1024, 20 * 1024), Some(2 * 1024));
    }

    #[test]
    fn pledge_clamps_to_max_when_observed_above() {
        // Observed peak > max_mb (transient overshoot — the ceiling watchdog
        // will fast-kill if it persists). Pledge stays at max_mb.
        let w = mk_window(&[mb(2 * 1024), mb(25 * 1024)]);
        assert_eq!(pledge_from_window(&w, 2 * 1024, 20 * 1024), Some(20 * 1024));
    }

    #[test]
    fn pledge_tracks_window_max_within_range() {
        // Mid-range observations: the pledge mirrors the window max so a
        // transient spike to 12 GiB lifts the pledge to 12 GiB until it rolls
        // out of the window.
        let w = mk_window(&[mb(3 * 1024), mb(12 * 1024), mb(7 * 1024)]);
        assert_eq!(pledge_from_window(&w, 2 * 1024, 20 * 1024), Some(12 * 1024));
    }

    #[test]
    fn pledge_decays_as_spikes_roll_out() {
        // Once the spike-sample is no longer in the window, the pledge falls
        // back to the new max — peers can reclaim the headroom they were
        // previously locked out of.
        let w = mk_window(&[mb(7 * 1024), mb(6 * 1024), mb(5 * 1024)]);
        assert_eq!(pledge_from_window(&w, 2 * 1024, 20 * 1024), Some(7 * 1024));
    }

    // should_update_pledge --------------------------------------------------

    #[test]
    fn first_pledge_always_emits() {
        assert!(should_update_pledge(None, 2 * 1024, 2 * 1024));
    }

    #[test]
    fn no_change_does_not_emit() {
        assert!(!should_update_pledge(Some(8 * 1024), 8 * 1024, 2 * 1024));
    }

    #[test]
    fn baseline_to_above_baseline_always_emits() {
        // Service grew from min_mb (2 GiB floor) to 3 GiB. Even though 1 GiB
        // is below the absolute floor, the boundary transition matters: peers
        // need to know slack on the device just shrank.
        assert!(should_update_pledge(Some(2 * 1024), 3 * 1024, 2 * 1024));
    }

    #[test]
    fn return_to_baseline_always_emits() {
        // Service shrunk back to min_mb. Peers that were locked out should
        // see the headroom return immediately.
        assert!(should_update_pledge(Some(8 * 1024), 2 * 1024, 2 * 1024));
    }

    #[test]
    fn small_drift_does_not_emit() {
        // 50 MiB drift on a 12 GiB pledge: well below 5 % (614 MiB) and the
        // 256 MiB absolute floor. Don't churn the event stream.
        assert!(!should_update_pledge(
            Some(12 * 1024),
            12 * 1024 + 50,
            2 * 1024
        ));
    }

    #[test]
    fn meaningful_growth_emits() {
        // 700 MiB growth on a 12 GiB pledge: above both 5 % (614 MiB) and the
        // 256 MiB absolute floor.
        assert!(should_update_pledge(
            Some(12 * 1024),
            12 * 1024 + 700,
            2 * 1024
        ));
    }

    #[test]
    fn small_pledge_uses_absolute_floor() {
        // 200 MiB drift on a 4 GiB pledge: 5 % is only 204 MiB, but the
        // absolute floor is 256 MiB → don't emit.
        assert!(!should_update_pledge(
            Some(4 * 1024),
            4 * 1024 + 200,
            2 * 1024
        ));
        // 300 MiB drift on the same pledge: above the 256 MiB absolute floor.
        assert!(should_update_pledge(
            Some(4 * 1024),
            4 * 1024 + 300,
            2 * 1024
        ));
    }

    // overcommitted_gpus_for ---------------------------------------------

    fn snap_with_free(id: u32, total_gb: u64, free_bytes: u64) -> crate::devices::DeviceSnapshot {
        crate::devices::DeviceSnapshot {
            gpus: vec![crate::devices::GpuSnapshot {
                id,
                name: format!("GPU {id}"),
                total_bytes: total_gb * 1024 * 1024 * 1024,
                free_bytes,
            }],
            cpu: None,
            taken_at_ms: 0,
        }
    }

    fn alloc_table(entries: &[(&str, u32, u64)]) -> AllocationTable {
        let mut t = AllocationTable::new();
        for (name, gpu, mb) in entries {
            t.entry(SmolStr::new(*name))
                .or_default()
                .insert(DeviceSlot::Gpu(*gpu), *mb);
        }
        t
    }

    /// Pledge sum exceeds GPU total but NVML still reports plenty of free
    /// VRAM — the kernel won't OOM, so the resolver must NOT fire. This
    /// is the user-reported false-positive: comfy's high-water pledge
    /// (14 GB) plus qwen's estimator-prediction pledge (12 GB) sums to
    /// 26 GB on a 24 GB card, but actual usage was ~22 GB and there was
    /// real headroom — the resolver shouldn't kill anyone.
    #[test]
    fn overcommit_no_when_physical_free_is_ample() {
        let snap = snap_with_free(1, 24, 2 * 1024 * 1024 * 1024);
        let table = alloc_table(&[("comfy", 1, 14 * 1024), ("qwen", 1, 12 * 1024)]);
        let gpus = overcommitted_gpus_for(&SmolStr::new("comfy"), &table, &snap);
        assert!(
            gpus.is_empty(),
            "pledge sum > total but NVML free is fine — no OOM imminent"
        );
    }

    /// NVML reports free VRAM below `OOM_MARGIN_BYTES`. The next growth
    /// tick would OOM the kernel — fire the resolver.
    #[test]
    fn overcommit_yes_when_physical_free_below_margin() {
        let snap = snap_with_free(1, 24, 100 * 1024 * 1024); // 100 MiB free
        let table = alloc_table(&[("comfy", 1, 14 * 1024), ("qwen", 1, 12 * 1024)]);
        let gpus = overcommitted_gpus_for(&SmolStr::new("comfy"), &table, &snap);
        assert_eq!(gpus, vec![1]);
    }

    /// Pressure on a GPU we don't hold isn't our problem.
    #[test]
    fn overcommit_filters_to_held_gpus() {
        let snap = crate::devices::DeviceSnapshot {
            gpus: vec![
                crate::devices::GpuSnapshot {
                    id: 1,
                    name: "GPU 1".into(),
                    total_bytes: 24 * 1024 * 1024 * 1024,
                    free_bytes: 10 * 1024 * 1024 * 1024, // plenty
                },
                crate::devices::GpuSnapshot {
                    id: 2,
                    name: "GPU 2".into(),
                    total_bytes: 24 * 1024 * 1024 * 1024,
                    free_bytes: 50 * 1024 * 1024, // pressured, but not ours
                },
            ],
            cpu: None,
            taken_at_ms: 0,
        };
        let table = alloc_table(&[
            ("comfy", 1, 4 * 1024),
            ("a", 2, 20 * 1024),
            ("b", 2, 10 * 1024),
        ]);
        let gpus = overcommitted_gpus_for(&SmolStr::new("comfy"), &table, &snap);
        assert!(gpus.is_empty());
    }

    /// A service with no allocation has nothing to defend.
    #[test]
    fn overcommit_empty_when_service_has_no_allocation() {
        let snap = snap_with_free(1, 24, 50 * 1024 * 1024);
        let table = alloc_table(&[("a", 1, 30 * 1024)]);
        let gpus = overcommitted_gpus_for(&SmolStr::new("ghost"), &table, &snap);
        assert!(gpus.is_empty());
    }

    // resolve_contention --------------------------------------------------

    fn svc(name: &str, priority: u8, lifecycle: Lifecycle) -> crate::config::ServiceConfig {
        let mut s = crate::config::validate::test_fixtures::minimal_service(name);
        s.priority = priority;
        s.lifecycle = lifecycle;
        s
    }

    /// Strict numeric priority always wins: an on-demand requester at
    /// priority 70 evicts a tied-lifecycle peer at priority 50.
    #[test]
    fn resolve_strict_priority_wins() {
        let services = vec![
            svc("hi-prio", 70, Lifecycle::OnDemand),
            svc("low-prio", 50, Lifecycle::Persistent),
        ];
        let table = alloc_table(&[("hi-prio", 1, 14 * 1024), ("low-prio", 1, 12 * 1024)]);
        let registry = with_handles(&["hi-prio", "low-prio"]);
        let action = resolve_contention(
            &SmolStr::new("hi-prio"),
            70,
            Lifecycle::OnDemand,
            &table,
            &[1],
            &registry,
            &services,
        );
        assert_eq!(
            action,
            ContentionAction::EvictPeer {
                peer: SmolStr::new("low-prio"),
            }
        );
    }

    /// At tied numeric priority, on-demand requester yields to a
    /// persistent peer (the lifecycle tie-break). Reproduces the user's
    /// scenario: ComfyUI (on-demand) vs Qwen (persistent), both priority 50.
    #[test]
    fn resolve_on_demand_yields_to_persistent_at_tied_priority() {
        let services = vec![
            svc("comfy", 50, Lifecycle::OnDemand),
            svc("qwen", 50, Lifecycle::Persistent),
        ];
        let table = alloc_table(&[("comfy", 1, 10 * 1024), ("qwen", 1, 12 * 1024)]);
        let registry = with_handles(&["comfy", "qwen"]);
        let action = resolve_contention(
            &SmolStr::new("comfy"),
            50,
            Lifecycle::OnDemand,
            &table,
            &[1],
            &registry,
            &services,
        );
        assert_eq!(
            action,
            ContentionAction::YieldSelf {
                to: SmolStr::new("qwen")
            }
        );
    }

    /// Reverse: a persistent dynamic service (rare but possible) at tied
    /// priority evicts an on-demand peer.
    #[test]
    fn resolve_persistent_evicts_on_demand_at_tied_priority() {
        let services = vec![
            svc("dyn-persistent", 50, Lifecycle::Persistent),
            svc("on-demand-peer", 50, Lifecycle::OnDemand),
        ];
        let table = alloc_table(&[
            ("dyn-persistent", 1, 14 * 1024),
            ("on-demand-peer", 1, 12 * 1024),
        ]);
        let registry = with_handles(&["dyn-persistent", "on-demand-peer"]);
        let action = resolve_contention(
            &SmolStr::new("dyn-persistent"),
            50,
            Lifecycle::Persistent,
            &table,
            &[1],
            &registry,
            &services,
        );
        assert_eq!(
            action,
            ContentionAction::EvictPeer {
                peer: SmolStr::new("on-demand-peer"),
            }
        );
    }

    /// No peer holds an allocation on the over-committed GPU → NoCandidate
    /// (the resolver leaves the situation alone; the kernel will OOM
    /// whichever side is allocating, which is semantically correct).
    #[test]
    fn resolve_no_candidate_when_alone_on_overcommit_gpu() {
        let services = vec![svc("comfy", 50, Lifecycle::OnDemand)];
        let table = alloc_table(&[("comfy", 1, 26 * 1024)]); // self-overcommitting somehow
        let registry = with_handles(&["comfy"]);
        let action = resolve_contention(
            &SmolStr::new("comfy"),
            50,
            Lifecycle::OnDemand,
            &table,
            &[1],
            &registry,
            &services,
        );
        assert_eq!(action, ContentionAction::NoCandidate);
    }

    /// Synthesise a minimal `ServiceRegistry` with present-but-dead
    /// handles for the named services. The contention resolver only
    /// checks registry membership (`registry.get(...).is_some()`), not
    /// handle health, so this is sufficient for the unit-level pure-data
    /// tests.
    fn with_handles(names: &[&str]) -> ServiceRegistry {
        // We can't construct a real SupervisorHandle here without the full
        // supervisor stack; build a registry with synthetic entries by
        // taking handles from a tiny side-helper that spawns a no-op
        // supervisor. For the pure-data scope of these tests, presence is
        // all that matters, so just clone the same handle into each slot.
        let registry = ServiceRegistry::new();
        let handle = std::sync::Arc::new(crate::supervise::SupervisorHandle::stub_for_test());
        for n in names {
            registry.insert(SmolStr::new(*n), handle.clone());
        }
        registry
    }
}
