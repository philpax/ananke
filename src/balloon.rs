//! Per-dynamic-service balloon resolver (spec §8.4).
//!
//! Each dynamic service gets one resolver task. The task samples observed
//! VRAM every `SAMPLE_INTERVAL`, maintains a rolling window, and takes
//! action when either the max-VRAM ceiling is breached for too long or
//! growth pressure is detected while contention with a borrower exists.

use std::collections::VecDeque;
use std::time::Duration;

use parking_lot::Mutex;
use smol_str::SmolStr;
use tokio::sync::watch;
use tracing::{debug, info, warn};

use crate::allocator::AllocationTable;
use crate::drain::DrainReason;
use crate::observation::ObservationTable;
use crate::service_registry::ServiceRegistry;

const WINDOW_SIZE: usize = 6;
const SAMPLE_INTERVAL: Duration = Duration::from_secs(2);

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
    let non_neg = samples
        .windows(2)
        .filter(|pair| pair[1] >= pair[0])
        .count();
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

/// Spawn a balloon-resolver task for a dynamic service.
///
/// The task runs until `shutdown` fires `true`. It samples the observed
/// peak every `SAMPLE_INTERVAL`, enforces the max-VRAM ceiling, and
/// resolves contention by fast-killing the lower-priority side when growth
/// is detected.
pub fn spawn_resolver(
    service_name: SmolStr,
    cfg: BalloonConfig,
    svc_priority: u8,
    observation: ObservationTable,
    registry: ServiceRegistry,
    allocations: std::sync::Arc<Mutex<AllocationTable>>,
    mut shutdown: watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
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

            // Spec §8.4: if observed > max_mb by >10% for >30 s, fast-kill self.
            let ceiling = cfg.max_mb * 1024 * 1024 * 110 / 100;
            if observed > ceiling {
                if let Some(since) = exceeded_since {
                    if since.elapsed() > Duration::from_secs(30) {
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
            // device. For phase 4, priority is 50 for all borrowers; a later
            // phase can wire real priority through the registry.
            let reservations = allocations.lock().clone();
            let mut candidate_borrower: Option<(SmolStr, u8)> = None;
            for (name, _) in &reservations {
                if name.as_str() == service_name.as_str() {
                    continue;
                }
                if let Some(handle) = registry.get(name) {
                    // Snapshot is cheap; we just need the service to be alive.
                    if let Some(_snap) = handle.snapshot().await {
                        // Borrower priority defaults to 50 until a real lookup is wired.
                        candidate_borrower = Some((name.clone(), 50));
                        break;
                    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_window(samples: &[u64]) -> VecDeque<u64> {
        VecDeque::from(samples.to_vec())
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
}
