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

use ananke_config::{DeviceSlot, Lifecycle, ServiceConfig};
use ananke_placement::{AllocationTable, devices::DeviceSnapshot};
use smol_str::SmolStr;

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

/// Test eligible victim prefixes in the supplied order, stopping at the first strict fit.
pub fn plan_eviction<T, E>(
    snapshot: &DeviceSnapshot,
    table: &AllocationTable,
    candidates: &[EvictionCandidate],
    incoming: &ServiceConfig,
    mut try_fit: impl FnMut(&DeviceSnapshot, &AllocationTable) -> Result<T, E>,
) -> Result<(T, Vec<SmolStr>), E> {
    let mut result = try_fit(snapshot, table);
    if let Ok(value) = result {
        return Ok((value, Vec::new()));
    }
    let mut victims = Vec::new();
    for candidate in candidates {
        if candidate.name == incoming.name
            || !candidate.is_evictable_by(incoming.priority, incoming.lifecycle)
            || !table
                .get(&candidate.name)
                .is_some_and(|row| row.values().any(|mb| *mb > 0))
            || victims.contains(&candidate.name)
        {
            continue;
        }
        victims.push(candidate.name.clone());
        let (post_snapshot, remaining) = after_eviction(snapshot, table, &victims);
        result = try_fit(&post_snapshot, &remaining);
        if let Ok(value) = result {
            return Ok((value, victims));
        }
    }
    result.map(|value| (value, victims))
}

/// Select only peers holding pledges on currently deficient slots.
pub fn plan_fixed_eviction(
    snapshot: &DeviceSnapshot,
    table: &AllocationTable,
    candidates: &[EvictionCandidate],
    incoming: &ServiceConfig,
    want: &BTreeMap<DeviceSlot, u64>,
) -> Result<Vec<SmolStr>, crate::NoFit> {
    let deficient: Vec<_> = want
        .iter()
        .filter_map(|(slot, mb)| {
            crate::can_fit(
                &BTreeMap::from([(slot.clone(), *mb)]),
                snapshot,
                table,
                Some(&incoming.name),
            )
            .err()
            .map(|_| slot)
        })
        .collect();
    let contributing: Vec<_> = candidates
        .iter()
        .filter(|candidate| {
            table.get(&candidate.name).is_some_and(|row| {
                deficient
                    .iter()
                    .any(|slot| row.get(slot).is_some_and(|mb| *mb > 0))
            })
        })
        .cloned()
        .collect();
    plan_eviction(
        snapshot,
        table,
        &contributing,
        incoming,
        |post_snapshot, remaining| {
            crate::can_fit(want, post_snapshot, remaining, Some(&incoming.name))
        },
    )
    .map(|(_, victims)| victims)
}

/// Retain planned reclaim credit while checking fresh competing reservations.
pub fn revalidate_eviction(
    snapshot: &DeviceSnapshot,
    planned_table: &AllocationTable,
    current_table: &AllocationTable,
    victims: &[SmolStr],
    want: &BTreeMap<DeviceSlot, u64>,
    exclude: Option<&SmolStr>,
) -> Result<(), crate::NoFit> {
    let (post_snapshot, _) = after_eviction(snapshot, planned_table, victims);
    let mut remaining = current_table.clone();
    for victim in victims {
        remaining.remove(victim);
    }
    crate::can_fit(want, &post_snapshot, &remaining, exclude)
}

/// Credit only selected victims' MiB pledges, capped at each device's total capacity.
pub fn after_eviction(
    snapshot: &DeviceSnapshot,
    table: &AllocationTable,
    victims: &[SmolStr],
) -> (DeviceSnapshot, AllocationTable) {
    let mut post_snapshot = snapshot.clone();
    let mut remaining = table.clone();
    for victim in victims {
        let Some(row) = remaining.remove(victim) else {
            continue;
        };
        for (slot, mb) in row {
            let bytes = mb.saturating_mul(1024 * 1024);
            match slot {
                DeviceSlot::Cpu => {
                    if let Some(cpu) = &mut post_snapshot.cpu {
                        cpu.available_bytes = cpu
                            .available_bytes
                            .saturating_add(bytes)
                            .min(cpu.total_bytes);
                    }
                }
                DeviceSlot::Gpu(id) => {
                    if let Some(gpu) = post_snapshot.gpus.iter_mut().find(|gpu| gpu.id == id) {
                        gpu.free_bytes = gpu.free_bytes.saturating_add(bytes).min(gpu.total_bytes);
                    }
                }
            }
        }
    }
    (post_snapshot, remaining)
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
    fn bounded_plan_preserves_remaining_pledges_and_stops_at_first_prefix() {
        let snapshot = DeviceSnapshot {
            gpus: vec![ananke_placement::devices::GpuSnapshot {
                id: 0,
                name: "gpu".into(),
                total_bytes: 24 << 30,
                free_bytes: 1 << 30,
            }],
            ..DeviceSnapshot::default()
        };
        let table = res(&[("cold", 4096), ("warm", 4096), ("protected", 8192)]);
        let candidates = [
            cand("cold", 50, true, 4 << 30),
            cand("warm", 50, true, 4 << 30),
            cand("protected", 100, false, 8 << 30),
        ];
        let svc = ananke_config::validate::test_fixtures::minimal_service("incoming");
        let want = BTreeMap::from([(DeviceSlot::Gpu(0), 8192)]);
        let (_, victims) =
            plan_eviction(&snapshot, &table, &candidates, &svc, |snap, remaining| {
                crate::can_fit(&want, snap, remaining, None)
            })
            .unwrap();
        assert_eq!(victims, vec![SmolStr::new("cold"), SmolStr::new("warm")]);
        let (post, remaining) = after_eviction(&snapshot, &table, &victims);
        assert_eq!(post.gpus[0].free_bytes, 9 << 30);
        assert_eq!(remaining, res(&[("protected", 8192)]));
        assert_eq!(table.len(), 3);
        let (_, short_victims) =
            plan_eviction(&snapshot, &table, &candidates, &svc, |snap, remaining| {
                crate::can_fit(
                    &BTreeMap::from([(DeviceSlot::Gpu(0), 4096)]),
                    snap,
                    remaining,
                    None,
                )
            })
            .unwrap();
        assert_eq!(short_victims, vec![SmolStr::new("cold")]);
        assert!(
            plan_eviction(&snapshot, &table, &candidates, &svc, |snap, remaining| {
                crate::can_fit(
                    &BTreeMap::from([(DeviceSlot::Gpu(0), 12288)]),
                    snap,
                    remaining,
                    None,
                )
            })
            .is_err()
        );
        let (capped, _) = after_eviction(
            &snapshot,
            &res(&[("huge", u64::MAX)]),
            &[SmolStr::new("huge")],
        );
        assert_eq!(capped.gpus[0].free_bytes, 24 << 30);
    }

    #[test]
    fn fixed_plan_skips_wrong_device_and_retains_credit_after_drain_ack() {
        let snapshot = DeviceSnapshot {
            gpus: (0..2)
                .map(|id| ananke_placement::devices::GpuSnapshot {
                    id,
                    name: "gpu".into(),
                    total_bytes: 24 << 30,
                    free_bytes: 1 << 30,
                })
                .collect(),
            ..DeviceSnapshot::default()
        };
        let mut table = res(&[("needed", 8192)]);
        table.insert(
            "unrelated".into(),
            BTreeMap::from([(DeviceSlot::Gpu(1), 8192)]),
        );
        let candidates = [
            cand("unrelated", 50, true, 8 << 30),
            cand("needed", 50, true, 8 << 30),
        ];
        let svc = ananke_config::validate::test_fixtures::minimal_service("incoming");
        let want = BTreeMap::from([(DeviceSlot::Gpu(0), 8192)]);
        let victims = plan_fixed_eviction(&snapshot, &table, &candidates, &svc, &want).unwrap();
        assert_eq!(victims, vec![SmolStr::new("needed")]);
        let mut current = table.clone();
        current.remove("needed");
        assert!(revalidate_eviction(&snapshot, &table, &current, &victims, &want, None).is_ok());
        current.insert(
            "competitor".into(),
            BTreeMap::from([(DeviceSlot::Gpu(0), 20480)]),
        );
        assert!(revalidate_eviction(&snapshot, &table, &current, &victims, &want, None).is_err());
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
        // Idle candidates yield even without a strict priority advantage
        // when the requester is also persistent — lifecycle ties add no
        // protection. Otherwise two persistent services at the default
        // priority deadlock against each other.
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
