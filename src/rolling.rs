//! Per-service rolling correction (spec §8.3).

use std::{collections::BTreeMap, sync::Arc};

use parking_lot::RwLock;
use smol_str::SmolStr;
use tracing::{info, warn};

#[derive(Debug, Clone, Copy)]
pub struct RollingCorrection {
    pub rolling_mean: f64,
    pub sample_count: u32,
    /// Count of consecutive samples with |mean-1.0| > 0.3.
    pub drift_samples: u32,
}

impl Default for RollingCorrection {
    fn default() -> Self {
        Self {
            rolling_mean: 1.0,
            sample_count: 0,
            drift_samples: 0,
        }
    }
}

#[derive(Clone, Default)]
pub struct RollingTable {
    inner: Arc<RwLock<BTreeMap<SmolStr, RollingCorrection>>>,
}

impl RollingTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, name: &SmolStr) -> RollingCorrection {
        self.inner.read().get(name).copied().unwrap_or_default()
    }

    /// Inject a synthetic 1.4× correction after an OOM kill to force the next
    /// estimate to reserve more memory before retrying.
    pub fn bump_for_oom_retry(&self, name: &SmolStr) {
        // A ratio of 1.4 signals the estimator was 40% short, which is the
        // maximum useful nudge before triggering the drift warning path.
        self.update(name, 140, 100);
    }

    pub fn update(&self, name: &SmolStr, observed_peak_bytes: u64, base_estimate_bytes: u64) {
        if base_estimate_bytes == 0 {
            return;
        }
        let ratio = observed_peak_bytes as f64 / base_estimate_bytes as f64;
        let mut guard = self.inner.write();
        let entry = guard.entry(name.clone()).or_default();
        let n = entry.sample_count as f64 + 1.0;
        let new_mean = (entry.rolling_mean * (n - 1.0) + ratio) / n;
        entry.rolling_mean = new_mean.clamp(0.8, 1.5);
        entry.sample_count = entry.sample_count.saturating_add(1);

        if (entry.rolling_mean - 1.0).abs() > 0.3 {
            entry.drift_samples = entry.drift_samples.saturating_add(1);
            if entry.drift_samples >= 5 {
                warn!(
                    service = %name,
                    mean = entry.rolling_mean,
                    "estimator_drift: rolling_mean has been >0.3 away from 1.0 for 5+ runs"
                );
            }
        } else {
            entry.drift_samples = 0;
        }

        if entry.rolling_mean > 1.2 {
            warn!(
                service = %name,
                mean = entry.rolling_mean,
                "rolling correction: under-estimation"
            );
        } else if entry.rolling_mean < 0.85 {
            warn!(
                service = %name,
                mean = entry.rolling_mean,
                "rolling correction: over-reservation"
            );
        } else {
            info!(
                service = %name,
                mean = entry.rolling_mean,
                sample = entry.sample_count,
                "rolling correction updated"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_converges_to_observed_ratio() {
        let t = RollingTable::new();
        let svc = SmolStr::new("demo");
        for _ in 0..5 {
            t.update(&svc, 120, 100); // observed=120, base=100, ratio=1.2
        }
        let rc = t.get(&svc);
        assert!((rc.rolling_mean - 1.2).abs() < 0.05);
    }

    #[test]
    fn mean_clamps_high() {
        let t = RollingTable::new();
        let svc = SmolStr::new("demo");
        t.update(&svc, 1000, 100); // ratio = 10
        assert_eq!(t.get(&svc).rolling_mean, 1.5);
    }

    #[test]
    fn mean_clamps_low() {
        let t = RollingTable::new();
        let svc = SmolStr::new("demo");
        t.update(&svc, 10, 100); // ratio = 0.1
        assert_eq!(t.get(&svc).rolling_mean, 0.8);
    }

    #[test]
    fn zero_base_is_noop() {
        let t = RollingTable::new();
        let svc = SmolStr::new("demo");
        t.update(&svc, 100, 0);
        assert_eq!(t.get(&svc).sample_count, 0);
    }
}
