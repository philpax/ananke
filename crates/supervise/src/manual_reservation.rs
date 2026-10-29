//! Eviction planning for manual reservations: which services must go so an
//! operator-placed reservation fits.
//!
//! A manual reservation outranks every service. It has no priority of its
//! own: any service holding memory on a device the reservation is short of
//! may be evicted, busy or persistent alike. Busy services still drain
//! gracefully, finishing their in-flight requests first.
//!
//! A manual reservation is an allocation-table row that belongs to no
//! supervisor. Every capacity check sums it like a service pledge, and
//! because eviction picks victims from the supervisor registry, nothing can
//! evict it. Its feasibility is judged on the pledge book alone
//! (`total - pledged`): device memory already in use outside ananke is
//! assumed to be the work the reservation is for, which is the point of
//! placing one around a job that is already running.

use std::collections::BTreeMap;

use ananke_allocator::{AllocationTable, eviction::EvictionCandidate};
use ananke_config::{DeviceSlot, validate::Lifecycle};
use ananke_placement::devices::DeviceSnapshot;
use smol_str::SmolStr;

/// Prefix of the allocation-table key a manual reservation is stored under.
/// Config validation rejects service names that start with it, so the two
/// can never collide.
pub const MANUAL_RESERVATION_KEY_PREFIX: &str =
    ananke_config::validate::RESERVED_SERVICE_NAME_PREFIX;

/// The allocation-table key for manual reservation `id`.
pub fn manual_reservation_key(id: i64) -> SmolStr {
    SmolStr::new(format!("{MANUAL_RESERVATION_KEY_PREFIX}{id}"))
}

/// The manual reservation id an allocation-table key names, if it names one.
pub fn manual_reservation_id(key: &str) -> Option<i64> {
    key.strip_prefix(MANUAL_RESERVATION_KEY_PREFIX)?
        .parse()
        .ok()
}

/// Why a manual reservation cannot be placed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReservationPlanFailure {
    /// Evicting every service still leaves too little, because other
    /// manual reservations already hold the rest.
    NoFit { reason: String },
}

/// Choose the services to evict so `want` (MiB per device) fits.
///
/// Every candidate is eligible. They are taken in the order that disturbs
/// the least: idle before busy, on-demand before persistent, lower
/// priority first, then least recently used first. Each pick targets the
/// device that is still short, so a service holding memory only on devices
/// that already fit is never touched; a final pass drops any victim the
/// rest already cover. `last_used_age_ms` gives each candidate's age since
/// its last request, `None` for never used, which sorts as oldest. Returns
/// the empty set when it already fits.
pub fn plan_evictions(
    want: &BTreeMap<DeviceSlot, u64>,
    snapshot: &DeviceSnapshot,
    table: &AllocationTable,
    candidates: &[EvictionCandidate],
    last_used_age_ms: impl Fn(&str) -> Option<u128>,
) -> Result<Vec<SmolStr>, ReservationPlanFailure> {
    let fits = |victims: &[SmolStr]| {
        ananke_allocator::can_fit_after_eviction(want, snapshot, table, None, victims)
    };
    let holds = |c: &EvictionCandidate, slot: &DeviceSlot| {
        table
            .get(&c.name)
            .and_then(|row| row.get(slot))
            .is_some_and(|mb| *mb > 0)
    };
    let mut ranked: Vec<&EvictionCandidate> = candidates.iter().collect();
    ranked.sort_by_key(|c| {
        (
            !c.idle,
            matches!(c.lifecycle, Lifecycle::Persistent),
            c.priority,
            std::cmp::Reverse(last_used_age_ms(&c.name).unwrap_or(u128::MAX)),
        )
    });

    let mut victims: Vec<SmolStr> = Vec::new();
    loop {
        let nofit = match fits(&victims) {
            Ok(()) => break,
            Err(nofit) => nofit,
        };
        let next = ranked
            .iter()
            .find(|c| !victims.contains(&c.name) && holds(c, &nofit.slot));
        match next {
            Some(c) => victims.push(c.name.clone()),
            None => {
                return Err(ReservationPlanFailure::NoFit {
                    reason: nofit.to_string(),
                });
            }
        }
    }

    // A later pick on another device can cover an earlier one's deficit
    // too; keep only the victims the reservation still needs.
    let mut i = victims.len();
    while i > 0 {
        i -= 1;
        let mut without = victims.clone();
        without.remove(i);
        if fits(&without).is_ok() {
            victims = without;
        }
    }
    Ok(victims)
}

#[cfg(test)]
mod tests {
    use ananke_placement::devices::GpuSnapshot;

    use super::*;

    const GIB_MB: u64 = 1024;

    fn snapshot(total_gib: u64) -> DeviceSnapshot {
        DeviceSnapshot {
            gpus: vec![GpuSnapshot {
                id: 0,
                name: "test".into(),
                total_bytes: total_gib * GIB_MB * 1024 * 1024,
                free_bytes: total_gib * GIB_MB * 1024 * 1024,
            }],
            cpu: None,
            taken_at_ms: 0,
        }
    }

    fn cand(name: &str, priority: u8, idle: bool) -> EvictionCandidate {
        EvictionCandidate {
            name: name.into(),
            priority,
            lifecycle: Lifecycle::OnDemand,
            idle,
            allocation_bytes: 1,
        }
    }

    fn table(rows: &[(&str, u64)]) -> AllocationTable {
        rows.iter()
            .map(|(name, gib)| {
                (
                    SmolStr::new(name),
                    BTreeMap::from([(DeviceSlot::Gpu(0), gib * GIB_MB)]),
                )
            })
            .collect()
    }

    fn want(gib: u64) -> BTreeMap<DeviceSlot, u64> {
        BTreeMap::from([(DeviceSlot::Gpu(0), gib * GIB_MB)])
    }

    #[test]
    fn fits_without_evicting_anything() {
        let t = table(&[("a", 4)]);
        let plan = plan_evictions(&want(18), &snapshot(24), &t, &[cand("a", 50, true)], |_| {
            None
        });
        assert_eq!(plan, Ok(vec![]));
    }

    #[test]
    fn evicts_least_recently_used_idle_services_first() {
        let t = table(&[("old", 6), ("new", 6), ("warm", 6)]);
        let cands = [
            cand("old", 50, true),
            cand("new", 50, true),
            cand("warm", 50, true),
        ];
        let age = |name: &str| match name {
            "old" => Some(9_000),
            "new" => Some(10),
            _ => Some(500),
        };
        // 24 GiB with 18 pledged leaves 6; 12 needs one 6 GiB victim.
        let plan = plan_evictions(&want(12), &snapshot(24), &t, &cands, age).unwrap();
        assert_eq!(plan, vec![SmolStr::new("old")]);
    }

    #[test]
    fn other_manual_reservations_count_and_are_never_evicted() {
        let mut t = table(&[("a", 6)]);
        t.insert(manual_reservation_key(1), want(12));
        // 24 - 12 manual - 6 = 6 free; evicting `a` reaches 12, short of 14,
        // and the manual row is never offered as a victim.
        let plan = plan_evictions(&want(14), &snapshot(24), &t, &[cand("a", 50, true)], |_| {
            None
        });
        assert!(matches!(plan, Err(ReservationPlanFailure::NoFit { .. })));
        let plan = plan_evictions(&want(6), &snapshot(24), &t, &[cand("a", 50, true)], |_| {
            None
        });
        assert_eq!(plan, Ok(vec![]));
    }

    #[test]
    fn busy_and_persistent_services_are_evicted_too() {
        let t = table(&[("busy", 20)]);
        let plan = plan_evictions(
            &want(10),
            &snapshot(24),
            &t,
            &[cand("busy", 255, false)],
            |_| None,
        );
        assert_eq!(plan, Ok(vec![SmolStr::new("busy")]));

        let t = table(&[("p", 20)]);
        let mut c = cand("p", 255, true);
        c.lifecycle = Lifecycle::Persistent;
        let plan = plan_evictions(&want(10), &snapshot(24), &t, &[c], |_| None);
        assert_eq!(plan, Ok(vec![SmolStr::new("p")]));
    }

    #[test]
    fn least_disruptive_victims_go_first() {
        // Any one of these frees enough. Idle beats busy, on-demand beats
        // persistent, and a lower priority beats recency.
        let t = table(&[("busy", 10), ("persistent", 10), ("high", 10), ("low", 10)]);
        let mut persistent = cand("persistent", 10, true);
        persistent.lifecycle = Lifecycle::Persistent;
        let cands = [
            cand("busy", 0, false),
            persistent,
            cand("high", 90, true),
            cand("low", 20, true),
        ];
        let age = |name: &str| if name == "high" { Some(9_000) } else { Some(1) };
        let snap = snapshot(48);
        let plan = plan_evictions(&want(16), &snap, &t, &cands, age).unwrap();
        assert_eq!(plan, vec![SmolStr::new("low")]);
        let plan = plan_evictions(&want(26), &snap, &t, &cands, age).unwrap();
        assert_eq!(plan, vec![SmolStr::new("low"), SmolStr::new("high")]);
        let plan = plan_evictions(&want(36), &snap, &t, &cands, age).unwrap();
        assert_eq!(
            plan,
            vec![
                SmolStr::new("low"),
                SmolStr::new("high"),
                SmolStr::new("persistent")
            ]
        );
    }

    #[test]
    fn services_on_other_devices_are_left_alone() {
        let mut t = table(&[("a", 20)]);
        t.insert(
            "cpu-only".into(),
            BTreeMap::from([(DeviceSlot::Cpu, 64 * GIB_MB)]),
        );
        let cands = [cand("cpu-only", 10, true), cand("a", 10, true)];
        let plan = plan_evictions(&want(10), &snapshot(24), &t, &cands, |_| None).unwrap();
        assert_eq!(plan, vec![SmolStr::new("a")]);
    }

    #[test]
    fn only_services_on_the_short_device_are_evicted() {
        // gpu:0 is short; the cpu has room. `deepseek` is older but holds
        // only gpu:1 and cpu, so it must survive; `qwen` on gpu:0 goes.
        let snap = DeviceSnapshot {
            gpus: vec![
                GpuSnapshot {
                    id: 0,
                    name: "a".into(),
                    total_bytes: 24 * GIB_MB * 1024 * 1024,
                    free_bytes: 0,
                },
                GpuSnapshot {
                    id: 1,
                    name: "b".into(),
                    total_bytes: 24 * GIB_MB * 1024 * 1024,
                    free_bytes: 0,
                },
            ],
            cpu: Some(ananke_placement::devices::CpuSnapshot {
                total_bytes: 376 * GIB_MB * 1024 * 1024,
                available_bytes: 60 * GIB_MB * 1024 * 1024,
            }),
            taken_at_ms: 0,
        };
        let t: AllocationTable = [
            (
                SmolStr::new("qwen"),
                BTreeMap::from([
                    (DeviceSlot::Gpu(0), 22 * GIB_MB),
                    (DeviceSlot::Cpu, 2 * GIB_MB),
                ]),
            ),
            (
                SmolStr::new("deepseek"),
                BTreeMap::from([
                    (DeviceSlot::Gpu(1), 23 * GIB_MB),
                    (DeviceSlot::Cpu, 281 * GIB_MB),
                ]),
            ),
        ]
        .into_iter()
        .collect();
        let want = BTreeMap::from([
            (DeviceSlot::Gpu(0), 3 * GIB_MB),
            (DeviceSlot::Cpu, 4 * GIB_MB),
        ]);
        let cands = [cand("qwen", 45, true), cand("deepseek", 100, true)];
        let age = |name: &str| match name {
            "deepseek" => Some(90_000),
            _ => Some(10),
        };
        let plan = plan_evictions(&want, &snap, &t, &cands, age).unwrap();
        assert_eq!(plan, vec![SmolStr::new("qwen")]);
    }

    #[test]
    fn victims_another_pick_covers_are_dropped() {
        // Both devices short. `small` (older) frees gpu:0 only partly;
        // `big` alone covers both, so `small` is dropped again.
        let mut t = table(&[("small", 2)]);
        t.insert(
            "big".into(),
            BTreeMap::from([(DeviceSlot::Gpu(0), 20 * GIB_MB)]),
        );
        let cands = [cand("small", 10, true), cand("big", 10, true)];
        let age = |name: &str| if name == "small" { Some(900) } else { Some(1) };
        let plan = plan_evictions(&want(20), &snapshot(24), &t, &cands, age).unwrap();
        assert_eq!(plan, vec![SmolStr::new("big")]);
    }

    #[test]
    fn keys_round_trip() {
        assert_eq!(manual_reservation_id(&manual_reservation_key(42)), Some(42));
        assert_eq!(manual_reservation_id("qwen"), None);
        assert_eq!(manual_reservation_id("reservation:x"), None);
    }
}
