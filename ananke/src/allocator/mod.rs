//! Pure feasibility check for service placement.
//!
//! Phase 2 has no eviction: the allocator either admits a service whose
//! declared `placement_override` fits live free bytes minus existing
//! reservations, or reports `NoFit` with a specific slot and shortfall.
//! Future phases replace the in-crate caller with an eviction-capable
//! one but keep this function as the innermost yes/no.

pub mod balloon;
pub mod eviction;
pub mod placement;

use std::collections::BTreeMap;

use smol_str::SmolStr;

use crate::{config::validate::DeviceSlot, devices::DeviceSnapshot};

pub type AllocationTable = BTreeMap<SmolStr, BTreeMap<DeviceSlot, u64>>;

#[derive(Debug, PartialEq, Eq)]
pub struct NoFit {
    pub slot: DeviceSlot,
    pub needed_bytes: u64,
    pub available_bytes: u64,
}

impl std::fmt::Display for NoFit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let slot = match &self.slot {
            DeviceSlot::Cpu => "cpu".to_string(),
            DeviceSlot::Gpu(n) => format!("gpu:{n}"),
        };
        write!(
            f,
            "no fit on {slot}: need {} bytes, {} available",
            self.needed_bytes, self.available_bytes
        )
    }
}

impl std::error::Error for NoFit {}

/// Check whether `want` fits after treating `evicting` as already freed.
///
/// Unlike [`can_fit`], this uses the optimistic availability view
/// (`total - reserved_filtered`) rather than `min(nvml_free, …)`. Callers
/// that have issued `begin_drain` on a set of victims have committed to
/// those evictions — nvml hasn't caught up yet (the drains are in-flight),
/// so the pledge book is the right reference.
pub fn can_fit_after_eviction(
    want: &BTreeMap<DeviceSlot, u64>,
    snapshot: &DeviceSnapshot,
    reserved: &AllocationTable,
    exclude: Option<&SmolStr>,
    evicting: &[SmolStr],
) -> Result<(), NoFit> {
    let mut filtered = reserved.clone();
    for victim in evicting {
        filtered.remove(victim);
    }
    for (slot, want_mb) in want {
        let need = want_mb * 1024 * 1024;
        let free = snapshot.free_bytes(slot).unwrap_or(0);
        let total = snapshot.total_bytes(slot).unwrap_or(free);
        let reserved_bytes: u64 = filtered
            .iter()
            .filter(|(k, _)| exclude.is_none_or(|x| *k != x))
            .filter_map(|(_, alloc)| alloc.get(slot))
            .sum::<u64>()
            * 1024
            * 1024;
        let available = total.saturating_sub(reserved_bytes);
        if available < need {
            return Err(NoFit {
                slot: slot.clone(),
                needed_bytes: need,
                available_bytes: available,
            });
        }
    }
    Ok(())
}

/// Check whether `want` (per-slot MB from `placement_override`) fits in
/// the device snapshot given our pledge book.
///
/// Availability on a slot is `min(nvml_free, total - reserved)`:
///
/// - `nvml_free` is what the device actually reports unused. For services
///   that are **already running with weights loaded**, this already
///   excludes their usage.
/// - `total - reserved` is what our pledge book says *should* be free if
///   everyone played by the rules (including services whose weights are
///   still loading and have not yet hit their pledged usage).
///
/// Taking the min is conservative: it prevents over-committing to pending
/// pledges while also respecting physical pressure from external processes
/// or in-service growth beyond the pledge. An earlier formulation
/// (`free - reserved`) double-counted realized pledges — with a 20 GB
/// model already loaded on a 24 GB GPU, nvml reported 4 GB free and we
/// subtracted the same 20 GB pledge again to get 0.
pub fn can_fit(
    want: &BTreeMap<DeviceSlot, u64>,
    snapshot: &DeviceSnapshot,
    reserved: &AllocationTable,
    exclude: Option<&SmolStr>,
) -> Result<(), NoFit> {
    for (slot, want_mb) in want {
        let need = want_mb * 1024 * 1024;
        let free = snapshot.free_bytes(slot).unwrap_or(0);
        let total = snapshot.total_bytes(slot).unwrap_or(free);
        let reserved_bytes: u64 = reserved
            .iter()
            .filter(|(k, _)| exclude.is_none_or(|x| *k != x))
            .filter_map(|(_, alloc)| alloc.get(slot))
            .sum::<u64>()
            * 1024
            * 1024;
        let optimistic = total.saturating_sub(reserved_bytes);
        let available = free.min(optimistic);
        if available < need {
            return Err(NoFit {
                slot: slot.clone(),
                needed_bytes: need,
                available_bytes: available,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::{CpuSnapshot, DeviceSnapshot, GpuSnapshot};

    fn gb(n: u64) -> u64 {
        n * 1024 * 1024 * 1024
    }

    fn mb(n: u64) -> u64 {
        n
    }

    fn snapshot_with(free_gpu_gb: u64, free_cpu_gb: u64) -> DeviceSnapshot {
        DeviceSnapshot {
            gpus: vec![GpuSnapshot {
                id: 0,
                name: "Test".into(),
                total_bytes: gb(24),
                free_bytes: gb(free_gpu_gb),
            }],
            cpu: Some(CpuSnapshot {
                total_bytes: gb(128),
                available_bytes: gb(free_cpu_gb),
            }),
            taken_at_ms: 0,
        }
    }

    #[test]
    fn fits_when_below_free() {
        let mut want = BTreeMap::new();
        want.insert(DeviceSlot::Gpu(0), mb(4096));
        assert!(can_fit(&want, &snapshot_with(20, 100), &BTreeMap::new(), None).is_ok());
    }

    #[test]
    fn no_fit_on_gpu() {
        let mut want = BTreeMap::new();
        want.insert(DeviceSlot::Gpu(0), mb(30 * 1024));
        let err = can_fit(&want, &snapshot_with(10, 100), &BTreeMap::new(), None).unwrap_err();
        assert_eq!(err.slot, DeviceSlot::Gpu(0));
    }

    #[test]
    fn reservations_subtract_from_available() {
        let mut want = BTreeMap::new();
        want.insert(DeviceSlot::Gpu(0), mb(10 * 1024)); // want 10 GB

        let mut other_alloc = BTreeMap::new();
        other_alloc.insert(DeviceSlot::Gpu(0), mb(15 * 1024)); // reserved 15 GB
        let mut reserved = BTreeMap::new();
        reserved.insert(SmolStr::new("other"), other_alloc);

        // Snapshot free = 20 GB. After subtracting 15 GB reserved, 5 GB available. Want 10 GB → no fit.
        let err = can_fit(&want, &snapshot_with(20, 100), &reserved, None).unwrap_err();
        assert_eq!(err.slot, DeviceSlot::Gpu(0));
    }

    #[test]
    fn exclude_skips_own_reservation() {
        let mut want = BTreeMap::new();
        want.insert(DeviceSlot::Gpu(0), mb(10 * 1024));

        let mut self_alloc = BTreeMap::new();
        self_alloc.insert(DeviceSlot::Gpu(0), mb(15 * 1024));
        let mut reserved = BTreeMap::new();
        reserved.insert(SmolStr::new("self"), self_alloc);

        // Self's 15 GB would eat the available; excluding self means 20 GB free, want 10 GB → fit.
        let ok = can_fit(
            &want,
            &snapshot_with(20, 100),
            &reserved,
            Some(&SmolStr::new("self")),
        );
        assert!(ok.is_ok());
    }

    #[test]
    fn unknown_slot_is_no_fit_zero() {
        let mut want = BTreeMap::new();
        want.insert(DeviceSlot::Gpu(7), mb(1));
        let err = can_fit(&want, &snapshot_with(20, 100), &BTreeMap::new(), None).unwrap_err();
        assert_eq!(err.slot, DeviceSlot::Gpu(7));
        assert_eq!(err.available_bytes, 0);
    }

    #[test]
    fn realized_pledges_are_not_double_counted() {
        // Regression: before the min(free, total-reserved) fix, a service
        // already running on a 24 GB GPU with 20 GB of loaded weights hit
        // `available = 0` for any new start — nvml reported 4 GB free and we
        // subtracted the 20 GB pledge again. A 1 GB request should fit.
        let mut want = BTreeMap::new();
        want.insert(DeviceSlot::Gpu(0), mb(1024));

        let mut realized = BTreeMap::new();
        realized.insert(DeviceSlot::Gpu(0), mb(20 * 1024));
        let mut reserved = BTreeMap::new();
        reserved.insert(SmolStr::new("vl"), realized);

        // 24 GB GPU, nvml reports 4 GB free (vl's weights are loaded).
        let snap = snapshot_with(4, 100);
        assert!(can_fit(&want, &snap, &reserved, None).is_ok());
    }

    #[test]
    fn pending_pledges_still_limit_fit() {
        // A service has pledged 20 GB but has not finished loading weights
        // (nvml doesn't yet show the usage). A new start that would blow
        // the physical limit must still be rejected.
        let mut want = BTreeMap::new();
        want.insert(DeviceSlot::Gpu(0), mb(6 * 1024));

        let mut pending = BTreeMap::new();
        pending.insert(DeviceSlot::Gpu(0), mb(20 * 1024));
        let mut reserved = BTreeMap::new();
        reserved.insert(SmolStr::new("pending"), pending);

        // Nothing loaded yet → nvml reports 24 GB free. 20 GB pledged means
        // only 4 GB *should* be free. Want 6 GB → no fit.
        let snap = snapshot_with(24, 100);
        assert!(can_fit(&want, &snap, &reserved, None).is_err());
    }

    #[test]
    fn can_fit_after_eviction_treats_victims_as_freed() {
        // Regression for the scenario-03 "eviction insufficient" bug: the
        // supervisor's retry was re-running `can_fit` against the unmodified
        // AllocationTable, which still contained the in-flight drainee. With
        // that victim present, the retry always reported no-fit even when
        // the eviction plan would actually satisfy the placement.
        let mut want = BTreeMap::new();
        want.insert(DeviceSlot::Gpu(0), mb(18 * 1024)); // want 18 GB

        let mut victim_alloc = BTreeMap::new();
        victim_alloc.insert(DeviceSlot::Gpu(0), mb(15 * 1024));
        let mut reserved = BTreeMap::new();
        reserved.insert(SmolStr::new("victim"), victim_alloc);

        // Snapshot has 20 GB free but 15 GB is held by `victim`.
        let snap = snapshot_with(20, 100);

        // Without eviction: 20 - 15 = 5 GB available, want 18 → no fit.
        assert!(can_fit(&want, &snap, &reserved, None).is_err());

        // Treating `victim` as evicted: 20 GB available, want 18 → fit.
        assert!(
            can_fit_after_eviction(&want, &snap, &reserved, None, &[SmolStr::new("victim")])
                .is_ok()
        );
    }
}
