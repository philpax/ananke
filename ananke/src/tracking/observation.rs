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
    peak_bytes: u64,
    pids: Vec<u32>,
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

    pub fn update_peak(&self, service: &SmolStr, bytes: u64) {
        let mut guard = self.inner.write();
        let entry = guard.entry(service.clone()).or_default();
        if bytes > entry.peak_bytes {
            entry.peak_bytes = bytes;
        }
    }

    pub fn read_peak(&self, service: &SmolStr) -> u64 {
        self.inner
            .read()
            .get(service)
            .map(|s| s.peak_bytes)
            .unwrap_or(0)
    }

    pub fn pids(&self, service: &SmolStr) -> Vec<u32> {
        self.inner
            .read()
            .get(service)
            .map(|s| s.pids.clone())
            .unwrap_or_default()
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
        t.update_peak(&svc, 100);
        t.update_peak(&svc, 50);
        t.update_peak(&svc, 200);
        assert_eq!(t.read_peak(&svc), 200);
    }

    #[test]
    fn clear_resets() {
        let t = ObservationTable::new();
        let svc = SmolStr::new("demo");
        t.update_peak(&svc, 100);
        t.clear(&svc);
        assert_eq!(t.read_peak(&svc), 0);
    }
}
