#![cfg(feature = "test-fakes")]
use std::collections::BTreeMap;

use ananke::{
    allocator::eviction::{EvictionCandidate, select_for_slot},
    config::validate::DeviceSlot,
};
use smol_str::SmolStr;

#[test]
fn low_prio_evicted_for_higher_prio_placement() {
    let mut res = BTreeMap::new();
    let mut r1 = BTreeMap::new();
    r1.insert(DeviceSlot::Gpu(0), 4096u64);
    res.insert(SmolStr::new("low"), r1);

    let cands = vec![EvictionCandidate {
        name: SmolStr::new("low"),
        priority: 30,
        idle: false,
        allocation_bytes: 4 * 1024 * 1024 * 1024,
    }];
    let sel = select_for_slot(
        4 * 1024 * 1024 * 1024,
        &DeviceSlot::Gpu(0),
        70,
        &cands,
        &res,
        0,
    );
    assert_eq!(sel, vec![SmolStr::new("low")]);
}
