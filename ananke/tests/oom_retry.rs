//! Integration test: `bump_for_oom_retry` increases the rolling mean so the
//! next spawn attempt reserves more memory.
#![cfg(feature = "test-fakes")]

use ananke::tracking::rolling::RollingTable;
use smol_str::SmolStr;

#[test]
fn oom_bump_increases_rolling_mean() {
    let t = RollingTable::new();
    let svc = SmolStr::new("demo");
    let before = t.get(&svc).rolling_mean;
    t.bump_for_oom_retry(&svc);
    let after = t.get(&svc).rolling_mean;
    assert!(
        after > before,
        "rolling_mean after OOM bump ({after}) must exceed initial value ({before})"
    );
}
