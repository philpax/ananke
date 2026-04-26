//! Linux-only: per-service observed memory peaks. Reads NVML for GPU VRAM and
//! `/proc/{pid}/status` for CPU VmRSS.

use std::{collections::BTreeMap, sync::Arc};

use parking_lot::RwLock;
use smol_str::SmolStr;

/// Observed peak bytes per service, across the current run.
#[derive(Clone, Default)]
pub struct ObservationTable {
    inner: Arc<RwLock<BTreeMap<SmolStr, ObservedState>>>,
}

#[derive(Debug, Clone, Default)]
struct ObservedState {
    /// High-water mark of `vram_bytes + rss_bytes`. Surfaced to the
    /// `/api/services` endpoint and event stream as the operator-facing
    /// "observed footprint" of the service.
    peak_bytes: u64,
    /// High-water mark of GPU VRAM bytes alone, attributed across every
    /// pid in the per-service pid set. Tracked separately because the
    /// dynamic-allocation pledge only models VRAM — pledging combined
    /// VRAM+RSS would inflate the pledge with the python interpreter's
    /// RSS and falsely trip the over-commit check (regression: an SDXL
    /// inference's 8 GB VRAM + 12 GB RSS used to pledge as 20 GB on the
    /// GPU and trigger a self-eviction that wasn't justified).
    peak_vram_bytes: u64,
    pids: Vec<u32>,
    /// Cgroup v2 path under which this service's actual workload pids
    /// live, when the operator declared one in `[service.tracking]`.
    /// Set once at supervisor spawn alongside `register`; the snapshotter
    /// reads it to widen the per-service pid set with cgroup-resident
    /// pids that aren't descendants of the registered child.
    cgroup_parent: Option<SmolStr>,
}

impl ObservationTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, service: &SmolStr, pid: u32) {
        let mut guard = self.inner.write();
        let entry = guard.entry(service.clone()).or_default();
        if !entry.pids.contains(&pid) {
            entry.pids.push(pid);
        }
    }

    /// Record (or clear) the cgroup parent path for a service. The
    /// snapshotter consults this when sampling so containerised pids that
    /// aren't process-tree descendants of the registered child still get
    /// attributed correctly.
    pub fn set_cgroup_parent(&self, service: &SmolStr, parent: Option<SmolStr>) {
        let mut guard = self.inner.write();
        guard.entry(service.clone()).or_default().cgroup_parent = parent;
    }

    /// Record an observation of the service's combined footprint
    /// (`vram + rss`) and its VRAM component. Both peaks update
    /// monotonically; `vram_bytes` may be zero on a CPU-only service.
    pub fn update_peak(&self, service: &SmolStr, vram_bytes: u64, rss_bytes: u64) {
        let mut guard = self.inner.write();
        let entry = guard.entry(service.clone()).or_default();
        let total = vram_bytes.saturating_add(rss_bytes);
        if total > entry.peak_bytes {
            entry.peak_bytes = total;
        }
        if vram_bytes > entry.peak_vram_bytes {
            entry.peak_vram_bytes = vram_bytes;
        }
    }

    /// Combined `vram + rss` peak. The frontend / `/api/services`
    /// surfaces this as the operator-facing observed footprint.
    pub fn read_peak(&self, service: &SmolStr) -> u64 {
        self.inner
            .read()
            .get(service)
            .map(|s| s.peak_bytes)
            .unwrap_or(0)
    }

    /// VRAM-only peak. The dynamic-allocation balloon resolver pledges
    /// against this, never the combined `read_peak` — see the comment
    /// on `ObservedState::peak_vram_bytes` for why.
    pub fn read_peak_vram(&self, service: &SmolStr) -> u64 {
        self.inner
            .read()
            .get(service)
            .map(|s| s.peak_vram_bytes)
            .unwrap_or(0)
    }

    pub fn pids(&self, service: &SmolStr) -> Vec<u32> {
        self.inner
            .read()
            .get(service)
            .map(|s| s.pids.clone())
            .unwrap_or_default()
    }

    pub fn cgroup_parent(&self, service: &SmolStr) -> Option<SmolStr> {
        self.inner.read().get(service)?.cgroup_parent.clone()
    }

    pub fn clear(&self, service: &SmolStr) {
        self.inner.write().remove(service);
    }
}

/// Thin rename of [`crate::system::ProcFs::vm_rss`] so the snapshotter
/// can keep calling `read_vm_rss(proc, pid)` and not have to know which
/// `/proc` file actually backs it. Returns `None` when the pid has
/// exited or the status entry isn't populated yet.
pub fn read_vm_rss(proc: &dyn crate::system::ProcFs, pid: u32) -> Option<u64> {
    proc.vm_rss(pid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::InMemoryProcFs;

    #[test]
    fn read_vm_rss_goes_through_procfs() {
        let proc = InMemoryProcFs::new();
        proc.set_vm_rss(4242, 5120 * 1024);
        assert_eq!(read_vm_rss(&proc, 4242), Some(5120 * 1024));
    }

    #[test]
    fn read_vm_rss_none_when_pid_missing() {
        let proc = InMemoryProcFs::new();
        assert_eq!(read_vm_rss(&proc, 9999), None);
    }

    #[test]
    fn peak_is_monotonic() {
        let t = ObservationTable::new();
        let svc = SmolStr::new("demo");
        // Combined peak walks up: vram+rss = 100, 50, 200.
        t.update_peak(&svc, 60, 40);
        t.update_peak(&svc, 30, 20);
        t.update_peak(&svc, 120, 80);
        assert_eq!(t.read_peak(&svc), 200);
        // VRAM-only peak tracks separately (60 → 30 doesn't lower it).
        assert_eq!(t.read_peak_vram(&svc), 120);
    }

    /// Observed combined peak and VRAM-only peak don't interfere: a tick
    /// with high RSS but low VRAM doesn't lift the VRAM peak (so the
    /// dynamic pledge stays modest), and a tick with high VRAM but low
    /// RSS lifts the VRAM peak even when the combined peak doesn't move.
    #[test]
    fn vram_and_combined_peaks_track_independently() {
        let t = ObservationTable::new();
        let svc = SmolStr::new("demo");
        // Tick 1: 4 GB VRAM + 6 GB RSS. Combined 10 GB, VRAM 4 GB.
        t.update_peak(&svc, 4 * 1024 * 1024 * 1024, 6 * 1024 * 1024 * 1024);
        assert_eq!(t.read_peak(&svc), 10 * 1024 * 1024 * 1024);
        assert_eq!(t.read_peak_vram(&svc), 4 * 1024 * 1024 * 1024);
        // Tick 2: 8 GB VRAM + 1 GB RSS. Combined 9 GB (won't move), VRAM 8 GB.
        t.update_peak(&svc, 8 * 1024 * 1024 * 1024, 1024 * 1024 * 1024);
        assert_eq!(t.read_peak(&svc), 10 * 1024 * 1024 * 1024);
        assert_eq!(t.read_peak_vram(&svc), 8 * 1024 * 1024 * 1024);
    }

    #[test]
    fn clear_resets() {
        let t = ObservationTable::new();
        let svc = SmolStr::new("demo");
        t.update_peak(&svc, 50, 50);
        t.clear(&svc);
        assert_eq!(t.read_peak(&svc), 0);
        assert_eq!(t.read_peak_vram(&svc), 0);
    }
}
