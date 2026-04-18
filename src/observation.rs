//! Per-service observed memory peaks (GPU VRAM via NVML + CPU VmRSS from /proc).

use std::collections::BTreeMap;
use std::sync::Arc;

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

/// Read `/proc/<pid>/status` and return `VmRSS` in bytes.
pub fn read_vm_rss(pid: u32) -> Option<u64> {
    let content = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb = rest
                .trim()
                .trim_end_matches("kB")
                .trim()
                .parse::<u64>()
                .ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

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
