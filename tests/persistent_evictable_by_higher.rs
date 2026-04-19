use std::collections::BTreeMap;

use ananke::{
    allocator::eviction::{EvictionCandidate, select_for_slot},
    config::validate::DeviceSlot,
};
use smol_str::SmolStr;

#[test]
fn persistent_not_special_cased() {
    // Even a "persistent" service (just a lifecycle label) is evictable
    // if its priority is lower. The planner only looks at priority + idle.
    let mut res = BTreeMap::new();
    let mut r = BTreeMap::new();
    r.insert(DeviceSlot::Gpu(0), 4096u64);
    res.insert(SmolStr::new("persistent-svc"), r);

    let cands = vec![EvictionCandidate {
        name: SmolStr::new("persistent-svc"),
        priority: 50,
        idle: false,
        allocation_bytes: 4 * 1024 * 1024 * 1024,
    }];
    let sel = select_for_slot(
        4 * 1024 * 1024 * 1024,
        &DeviceSlot::Gpu(0),
        80,
        &cands,
        &res,
        0,
    );
    assert_eq!(sel, vec![SmolStr::new("persistent-svc")]);

    // A service at the same priority as the requester cannot be evicted.
    let cands_pinned = vec![EvictionCandidate {
        name: SmolStr::new("pinned"),
        priority: 100,
        idle: false,
        allocation_bytes: 4 * 1024 * 1024 * 1024,
    }];
    let pinned = select_for_slot(
        4 * 1024 * 1024 * 1024,
        &DeviceSlot::Gpu(0),
        100,
        &cands_pinned,
        &res,
        0,
    );
    assert!(pinned.is_empty(), "same priority should not evict");
}
