//! Integration test: feeding repeated under-estimate samples to a `RollingTable`
//! converges the rolling mean above the 1.2 warning threshold.
#![cfg(feature = "test-fakes")]

use ananke::tracking::rolling::RollingTable;
use smol_str::SmolStr;

#[test]
fn rolling_mean_converges_above_threshold_warns() {
    let t = RollingTable::new();
    let svc = SmolStr::new("demo");
    // Observed peak is 130, base estimate is 100 → ratio = 1.3.
    // After three samples the running mean should exceed 1.2.
    for _ in 0..3 {
        t.update(&svc, 130, 100);
    }
    let rc = t.get(&svc);
    assert!(
        rc.rolling_mean > 1.2,
        "rolling_mean ({}) must exceed 1.2 after three 1.3× samples",
        rc.rolling_mean
    );
}
