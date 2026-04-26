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
    config::validate::{DEFAULT_SERVICE_PRIORITY, DeviceSlot},
    daemon::events::EventBus,
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
    deps: ResolverDeps,
) -> tokio::task::JoinHandle<()> {
    let ResolverDeps {
        observation,
        registry,
        allocations,
        events,
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

            let observed = observation.read_peak(&service_name);
            if window.len() == WINDOW_SIZE {
                window.pop_front();
            }
            window.push_back(observed);

            reconcile_pledge(&service_name, &window, &cfg, &allocations, &events);

            // If observed > max_mb by >10% for >30 s, fast-kill self.
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
                        return;
                    }
                } else {
                    exceeded_since = Some(std::time::Instant::now());
                }
            } else {
                exceeded_since = None;
            }

            // Look for a borrower currently holding an allocation on the same
            // device. For now, priority is the service default for all
            // borrowers; a follow-up can wire real priority through the
            // registry.
            let reservations = allocations.lock().clone();
            let mut candidate_borrower: Option<(SmolStr, u8)> = None;
            for name in reservations.keys() {
                if name.as_str() == service_name.as_str() {
                    continue;
                }
                if registry.get(name).is_some() {
                    // Handle presence is enough — the allocation table only
                    // carries live services, so existence in the registry
                    // implies a borrower worth considering.
                    // Borrower priority defaults to the service default until
                    // a real lookup is wired.
                    candidate_borrower = Some((name.clone(), DEFAULT_SERVICE_PRIORITY));
                    break;
                }
            }

            let floor = cfg.min_mb * 1024 * 1024 + cfg.margin_bytes;
            if detect_growth(&window, floor) {
                if let Some((borrower, borrower_priority)) = candidate_borrower {
                    info!(
                        service = %service_name,
                        borrower = %borrower,
                        "balloon: growth detected; resolving contention",
                    );
                    if svc_priority > borrower_priority {
                        // We outrank the borrower; evict it.
                        if let Some(handle) = registry.get(&borrower) {
                            handle.fast_kill(DrainReason::Eviction).await;
                        }
                    } else {
                        // Borrower outranks or ties us; yield by fast-killing self.
                        if let Some(handle) = registry.get(&service_name) {
                            handle.fast_kill(DrainReason::Eviction).await;
                        }
                        return;
                    }
                    // Reset window after action so transient spikes don't re-trigger immediately.
                    window.clear();
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
}
