//! Eviction planner.
//!
//! An idle service (no in-flight requests) is generally a valid eviction
//! target regardless of priority — if nothing's using it, displacing it
//! is cheap. A busy service still needs strictly-higher incoming priority
//! to be displaced, so an in-flight request isn't interrupted by an
//! equal-priority competitor.
//!
//! Lifecycle is itself a priority signal at tied numeric priority: a
//! persistent peer is the operator declaring "should always be loaded,"
//! and an on-demand requester is "transient, evictable." So at tied
//! priority an on-demand request cannot evict a persistent peer (idle or
//! not); it must wait or fail. Strict numeric priority still wins —
//! raising the on-demand service's priority breaks the tie.

use std::collections::BTreeMap;

use smol_str::SmolStr;

use crate::config::{DeviceSlot, Lifecycle, ServiceConfig};

#[derive(Debug, Clone)]
pub struct EvictionCandidate {
    pub name: SmolStr,
    pub priority: u8,
    pub lifecycle: Lifecycle,
    pub idle: bool,
    pub allocation_bytes: u64,
}

impl EvictionCandidate {
    /// Whether this candidate can be displaced by an incoming request at
    /// `incoming_priority` and `incoming_lifecycle`. Single source of
    /// truth for the rule; every other eviction decision must go through
    /// this.
    ///
    /// Rules:
    /// - Strict priority advantage (`incoming > self.priority`) always wins.
    /// - At tied or lower priority, a busy candidate is untouchable.
    /// - At tied priority, an idle persistent candidate is protected from
    ///   on-demand requesters (lifecycle breaks the tie). Other tied-priority
    ///   idle combinations are still evictable.
    pub fn is_evictable_by(&self, incoming_priority: u8, incoming_lifecycle: Lifecycle) -> bool {
        if self.priority < incoming_priority {
            return true;
        }
        if !self.idle {
            return false;
        }
        // Idle, tied or higher priority — protect persistent peers from
        // on-demand requesters at the tie. Two persistent peers may still
        // displace each other (otherwise the persistent watcher deadlocks
        // with itself when restarting after a config edit).
        if matches!(self.lifecycle, Lifecycle::Persistent)
            && matches!(incoming_lifecycle, Lifecycle::OnDemand)
        {
            return false;
        }
        true
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
    want_lifecycle: Lifecycle,
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
        .filter(|c| c.is_evictable_by(want_priority, want_lifecycle))
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
        lifecycle: svc.lifecycle,
        idle,
        allocation_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(name: &str, prio: u8, idle: bool, bytes: u64) -> EvictionCandidate {
        cand_lc(name, prio, Lifecycle::OnDemand, idle, bytes)
    }

    fn cand_lc(
        name: &str,
        prio: u8,
        lifecycle: Lifecycle,
        idle: bool,
        bytes: u64,
    ) -> EvictionCandidate {
        EvictionCandidate {
            name: SmolStr::new(name),
            priority: prio,
            lifecycle,
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
    fn is_evictable_by_idle_on_demand_always_yes() {
        let c = cand("idle", 100, true, 1);
        // Idle on-demand is freely evictable by anything (the existing
        // baseline; persistent gets the lifecycle protection below).
        assert!(c.is_evictable_by(0, Lifecycle::OnDemand));
        assert!(c.is_evictable_by(50, Lifecycle::OnDemand));
        assert!(c.is_evictable_by(100, Lifecycle::OnDemand));
    }

    #[test]
    fn is_evictable_by_busy_needs_strictly_higher() {
        let c = cand("busy", 50, false, 1);
        assert!(!c.is_evictable_by(0, Lifecycle::OnDemand));
        assert!(!c.is_evictable_by(49, Lifecycle::OnDemand));
        assert!(!c.is_evictable_by(50, Lifecycle::OnDemand));
        assert!(c.is_evictable_by(51, Lifecycle::OnDemand));
        assert!(c.is_evictable_by(100, Lifecycle::OnDemand));
    }

    #[test]
    fn no_eviction_when_enough_free() {
        let sel = select_for_slot(
            1000,
            &DeviceSlot::Gpu(0),
            50,
            Lifecycle::OnDemand,
            &[],
            &BTreeMap::new(),
            2000,
        );
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
            Lifecycle::OnDemand,
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
            Lifecycle::OnDemand,
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
            Lifecycle::OnDemand,
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
            Lifecycle::OnDemand,
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
        // without a strict priority advantage when the requester is also
        // persistent (lifecycle ties don't add protection).
        let cands = vec![cand_lc(
            "peer",
            70,
            Lifecycle::Persistent,
            true,
            4 * 1024 * 1024 * 1024,
        )];
        let r = res(&[("peer", 4096)]);
        let sel = select_for_slot(
            4 * 1024 * 1024 * 1024,
            &DeviceSlot::Gpu(0),
            70,
            Lifecycle::Persistent,
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
            Lifecycle::OnDemand,
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
            Lifecycle::OnDemand,
            &cands,
            &r,
            0,
        );
        assert!(sel.is_empty());
    }

    /// Lifecycle-as-tiebreaker: an idle persistent peer at tied priority
    /// is **not** evictable by an on-demand requester. Lifecycle is itself
    /// a priority signal — the operator declared "should always be loaded."
    /// The on-demand requester must wait or fail (or raise its numeric
    /// priority to win).
    #[test]
    fn idle_persistent_peer_protected_from_on_demand_at_tied_priority() {
        let qwen = cand_lc(
            "qwen",
            50,
            Lifecycle::Persistent,
            true,
            4 * 1024 * 1024 * 1024,
        );
        assert!(!qwen.is_evictable_by(50, Lifecycle::OnDemand));
        // Strictly higher priority still wins.
        assert!(qwen.is_evictable_by(51, Lifecycle::OnDemand));
        // Persistent vs persistent at tied priority: still evictable, so
        // the persistent watcher can swap one persistent for another.
        assert!(qwen.is_evictable_by(50, Lifecycle::Persistent));
    }

    /// `select_for_slot` must respect the persistent-protection rule:
    /// an on-demand request can't pick a tied-priority persistent peer
    /// even if it's idle and would otherwise satisfy the deficit.
    #[test]
    fn select_for_slot_skips_tied_priority_persistent_for_on_demand() {
        let cands = vec![
            cand_lc(
                "qwen",
                50,
                Lifecycle::Persistent,
                true,
                12 * 1024 * 1024 * 1024,
            ),
            cand_lc(
                "alt",
                50,
                Lifecycle::OnDemand,
                true,
                12 * 1024 * 1024 * 1024,
            ),
        ];
        let r = res(&[("qwen", 12 * 1024), ("alt", 12 * 1024)]);
        let sel = select_for_slot(
            10 * 1024 * 1024 * 1024,
            &DeviceSlot::Gpu(0),
            50,
            Lifecycle::OnDemand,
            &cands,
            &r,
            0,
        );
        // Only `alt` is evictable; `qwen` is protected by lifecycle.
        assert_eq!(sel, vec![SmolStr::new("alt")]);
    }

    /// And nothing-evictable when the only candidates are tied-priority
    /// persistent peers and the requester is on-demand.
    #[test]
    fn select_for_slot_empty_when_only_protected_persistent_peers_remain() {
        let cands = vec![cand_lc(
            "qwen",
            50,
            Lifecycle::Persistent,
            true,
            12 * 1024 * 1024 * 1024,
        )];
        let r = res(&[("qwen", 12 * 1024)]);
        let sel = select_for_slot(
            10 * 1024 * 1024 * 1024,
            &DeviceSlot::Gpu(0),
            50,
            Lifecycle::OnDemand,
            &cands,
            &r,
            0,
        );
        assert!(sel.is_empty());
    }
}
