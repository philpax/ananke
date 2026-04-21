//! Eviction planner.
//!
//! An idle service (no in-flight requests) is always a valid eviction
//! target regardless of priority — if nothing's using it, displacing it
//! is free. A busy service still needs strictly-higher incoming priority
//! to be displaced, so an in-flight request isn't interrupted by an
//! equal-priority competitor.
//!
//! Priority therefore only matters as (a) a tie-breaker among idle
//! candidates, and (b) a protection for busy services against peer
//! competitors. For single-workload setups where nothing runs
//! concurrently, idle-first scoring does all the useful work and
//! priority is effectively unused.

use std::collections::BTreeMap;

use smol_str::SmolStr;

use crate::config::{DeviceSlot, ServiceConfig};

#[derive(Debug, Clone)]
pub struct EvictionCandidate {
    pub name: SmolStr,
    pub priority: u8,
    pub idle: bool,
    pub allocation_bytes: u64,
}

impl EvictionCandidate {
    /// Whether this candidate can be displaced by an incoming request
    /// at `incoming_priority`. Single source of truth for the rule; every
    /// other eviction decision must go through this.
    pub fn is_evictable_by(&self, incoming_priority: u8) -> bool {
        self.idle || self.priority < incoming_priority
    }
}

/// Select the minimum set of services whose eviction would free enough
/// capacity for `want` on `want_slot`.
///
/// Eligibility: idle candidates are always evictable; busy candidates
/// require strictly-higher incoming priority. Sort order within the
/// eligible set is idle-first, then lowest priority, then smallest
/// allocation. Stops as soon as cumulative freed bytes ≥ want bytes.
/// Returns empty if no set suffices (caller falls back to the
/// NoFit path).
pub fn select_for_slot(
    want_bytes: u64,
    want_slot: &DeviceSlot,
    want_priority: u8,
    running: &[EvictionCandidate],
    reservations: &BTreeMap<SmolStr, BTreeMap<DeviceSlot, u64>>,
    free_bytes: u64,
) -> Vec<SmolStr> {
    if free_bytes >= want_bytes {
        return Vec::new();
    }
    let needed = want_bytes - free_bytes;

    let mut candidates: Vec<&EvictionCandidate> = running
        .iter()
        .filter(|c| c.is_evictable_by(want_priority))
        .filter(|c| {
            reservations
                .get(&c.name)
                .and_then(|r| r.get(want_slot))
                .copied()
                .unwrap_or(0)
                > 0
        })
        .collect();

    // Idle first, then lowest priority, then smallest allocation.
    candidates.sort_by(|a, b| {
        b.idle
            .cmp(&a.idle)
            .then(a.priority.cmp(&b.priority))
            .then(a.allocation_bytes.cmp(&b.allocation_bytes))
    });

    let mut out = Vec::new();
    let mut freed = 0u64;
    for c in candidates {
        let bytes = reservations
            .get(&c.name)
            .and_then(|r| r.get(want_slot))
            .copied()
            .unwrap_or(0)
            * 1024
            * 1024;
        freed += bytes;
        out.push(c.name.clone());
        if freed >= needed {
            return out;
        }
    }
    // Not enough even if we evict everything eligible.
    Vec::new()
}

/// Summarise a running service into an `EvictionCandidate`. Used by the
/// scheduler/allocator to construct the input list.
pub fn summarise(svc: &ServiceConfig, idle: bool, allocation_bytes: u64) -> EvictionCandidate {
    EvictionCandidate {
        name: svc.name.clone(),
        priority: svc.priority,
        idle,
        allocation_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(name: &str, prio: u8, idle: bool, bytes: u64) -> EvictionCandidate {
        EvictionCandidate {
            name: SmolStr::new(name),
            priority: prio,
            idle,
            allocation_bytes: bytes,
        }
    }

    fn res(entries: &[(&str, u64)]) -> BTreeMap<SmolStr, BTreeMap<DeviceSlot, u64>> {
        let mut map = BTreeMap::new();
        for (n, mb) in entries {
            let mut inner = BTreeMap::new();
            inner.insert(DeviceSlot::Gpu(0), *mb);
            map.insert(SmolStr::new(*n), inner);
        }
        map
    }

    #[test]
    fn is_evictable_by_idle_always_yes() {
        let c = cand("idle", 100, true, 1);
        assert!(c.is_evictable_by(0));
        assert!(c.is_evictable_by(50));
        assert!(c.is_evictable_by(100));
    }

    #[test]
    fn is_evictable_by_busy_needs_strictly_higher() {
        let c = cand("busy", 50, false, 1);
        assert!(!c.is_evictable_by(0));
        assert!(!c.is_evictable_by(49));
        assert!(!c.is_evictable_by(50));
        assert!(c.is_evictable_by(51));
        assert!(c.is_evictable_by(100));
    }

    #[test]
    fn no_eviction_when_enough_free() {
        let sel = select_for_slot(1000, &DeviceSlot::Gpu(0), 50, &[], &BTreeMap::new(), 2000);
        assert!(sel.is_empty());
    }

    #[test]
    fn picks_idle_before_running() {
        let cands = vec![
            cand("a-idle", 40, true, 4 * 1024 * 1024 * 1024),
            cand("b-live", 30, false, 4 * 1024 * 1024 * 1024),
        ];
        let r = res(&[("a-idle", 4096), ("b-live", 4096)]);
        let sel = select_for_slot(
            4 * 1024 * 1024 * 1024,
            &DeviceSlot::Gpu(0),
            70,
            &cands,
            &r,
            0,
        );
        assert_eq!(sel, vec![SmolStr::new("a-idle")]);
    }

    #[test]
    fn picks_lowest_priority() {
        let cands = vec![
            cand("low", 20, false, 4 * 1024 * 1024 * 1024),
            cand("mid", 50, false, 4 * 1024 * 1024 * 1024),
        ];
        let r = res(&[("low", 4096), ("mid", 4096)]);
        let sel = select_for_slot(
            4 * 1024 * 1024 * 1024,
            &DeviceSlot::Gpu(0),
            70,
            &cands,
            &r,
            0,
        );
        assert_eq!(sel, vec![SmolStr::new("low")]);
    }

    #[test]
    fn evicts_multiple_if_one_insufficient() {
        let cands = vec![
            cand("a", 30, false, 2 * 1024 * 1024 * 1024),
            cand("b", 30, false, 2 * 1024 * 1024 * 1024),
            cand("c", 30, false, 2 * 1024 * 1024 * 1024),
        ];
        let r = res(&[("a", 2048), ("b", 2048), ("c", 2048)]);
        let sel = select_for_slot(
            5 * 1024 * 1024 * 1024,
            &DeviceSlot::Gpu(0),
            70,
            &cands,
            &r,
            0,
        );
        assert_eq!(sel.len(), 3);
    }

    #[test]
    fn busy_same_priority_not_evictable() {
        // Peer at the same priority with an in-flight request: untouchable
        // until it idles out or the incoming request carries strictly higher
        // priority.
        let cands = vec![cand("peer", 70, false, 4 * 1024 * 1024 * 1024)];
        let r = res(&[("peer", 4096)]);
        let sel = select_for_slot(
            4 * 1024 * 1024 * 1024,
            &DeviceSlot::Gpu(0),
            70,
            &cands,
            &r,
            0,
        );
        assert!(sel.is_empty());
    }

    #[test]
    fn idle_same_priority_is_evictable() {
        // Regression for the "persistent Qwen blocks persistent Gemma at
        // default priority" deadlock. Idle candidates should yield even
        // without a strict priority advantage.
        let cands = vec![cand("peer", 70, true, 4 * 1024 * 1024 * 1024)];
        let r = res(&[("peer", 4096)]);
        let sel = select_for_slot(
            4 * 1024 * 1024 * 1024,
            &DeviceSlot::Gpu(0),
            70,
            &cands,
            &r,
            0,
        );
        assert_eq!(sel, vec![SmolStr::new("peer")]);
    }

    #[test]
    fn idle_lower_priority_is_evictable() {
        // Same as above but the idle candidate is at a lower priority
        // than the incoming request — still evictable.
        let cands = vec![cand("idle-low", 20, true, 4 * 1024 * 1024 * 1024)];
        let r = res(&[("idle-low", 4096)]);
        let sel = select_for_slot(
            4 * 1024 * 1024 * 1024,
            &DeviceSlot::Gpu(0),
            70,
            &cands,
            &r,
            0,
        );
        assert_eq!(sel, vec![SmolStr::new("idle-low")]);
    }

    #[test]
    fn returns_empty_when_not_enough_evictable() {
        let cands = vec![cand("small", 30, false, 1024 * 1024 * 1024)];
        let r = res(&[("small", 1024)]);
        // Want 4 GB; evictable only 1 GB; cannot satisfy.
        let sel = select_for_slot(
            4 * 1024 * 1024 * 1024,
            &DeviceSlot::Gpu(0),
            70,
            &cands,
            &r,
            0,
        );
        assert!(sel.is_empty());
    }
}
