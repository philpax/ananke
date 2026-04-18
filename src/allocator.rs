//! Pure feasibility check for service placement.
//!
//! Phase 2 has no eviction: the allocator either admits a service whose
//! declared `placement_override` fits live free bytes minus existing
//! reservations, or reports `NoFit` with a specific slot and shortfall.
//! Future phases replace the in-crate caller with an eviction-capable
//! one but keep this function as the innermost yes/no.

use std::collections::BTreeMap;

use smol_str::SmolStr;

use crate::config::validate::DeviceSlot;
use crate::devices::DeviceSnapshot;

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

/// Check whether `want` (per-slot MB from `placement_override`) fits in
/// the device snapshot after subtracting the bytes already reserved by
/// other services.
pub fn can_fit(
    want: &BTreeMap<DeviceSlot, u64>,
    snapshot: &DeviceSnapshot,
    reserved: &AllocationTable,
    exclude: Option<&SmolStr>,
) -> Result<(), NoFit> {
    for (slot, want_mb) in want {
        let need = want_mb * 1024 * 1024;
        let free = snapshot.free_bytes(slot).unwrap_or(0);
        let already: u64 = reserved
            .iter()
            .filter(|(k, _)| exclude.is_none_or(|x| *k != x))
            .filter_map(|(_, alloc)| alloc.get(slot))
            .sum::<u64>()
            * 1024
            * 1024;
        let available = free.saturating_sub(already);
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
        let err =
            can_fit(&want, &snapshot_with(10, 100), &BTreeMap::new(), None).unwrap_err();
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
        let err =
            can_fit(&want, &snapshot_with(20, 100), &BTreeMap::new(), None).unwrap_err();
        assert_eq!(err.slot, DeviceSlot::Gpu(7));
        assert_eq!(err.available_bytes, 0);
    }
}
