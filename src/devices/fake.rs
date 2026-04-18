//! In-memory fake `GpuProbe` for tests.

use std::sync::Arc;

use parking_lot::Mutex;

use super::probe::{GpuInfo, GpuMemory, GpuProbe, GpuProcess};

#[derive(Debug, Clone)]
pub struct FakeGpu {
    pub info: GpuInfo,
    pub free_bytes: u64,
    pub processes: Vec<GpuProcess>,
}

#[derive(Default, Clone)]
pub struct FakeProbe {
    inner: Arc<Mutex<Vec<FakeGpu>>>,
}

impl FakeProbe {
    pub fn new(gpus: Vec<FakeGpu>) -> Self {
        Self { inner: Arc::new(Mutex::new(gpus)) }
    }

    pub fn set_free(&self, id: u32, free_bytes: u64) {
        let mut g = self.inner.lock();
        if let Some(gpu) = g.iter_mut().find(|g| g.info.id == id) {
            gpu.free_bytes = free_bytes;
        }
    }

    pub fn add_process(&self, id: u32, proc_info: GpuProcess) {
        let mut g = self.inner.lock();
        if let Some(gpu) = g.iter_mut().find(|g| g.info.id == id) {
            gpu.processes.push(proc_info);
        }
    }
}

impl GpuProbe for FakeProbe {
    fn list(&self) -> Vec<GpuInfo> {
        self.inner.lock().iter().map(|g| g.info.clone()).collect()
    }

    fn query(&self, id: u32) -> Option<GpuMemory> {
        self.inner.lock().iter().find(|g| g.info.id == id).map(|g| GpuMemory {
            total_bytes: g.info.total_bytes,
            free_bytes: g.free_bytes,
        })
    }

    fn processes(&self, id: u32) -> Vec<GpuProcess> {
        self.inner
            .lock()
            .iter()
            .find(|g| g.info.id == id)
            .map(|g| g.processes.clone())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> FakeProbe {
        FakeProbe::new(vec![
            FakeGpu {
                info: GpuInfo {
                    id: 0,
                    name: "RTX 4090".into(),
                    total_bytes: 24 * 1024 * 1024 * 1024,
                },
                free_bytes: 20 * 1024 * 1024 * 1024,
                processes: Vec::new(),
            },
            FakeGpu {
                info: GpuInfo {
                    id: 1,
                    name: "RTX 4090".into(),
                    total_bytes: 24 * 1024 * 1024 * 1024,
                },
                free_bytes: 22 * 1024 * 1024 * 1024,
                processes: Vec::new(),
            },
        ])
    }

    #[test]
    fn lists_all() {
        let p = fixture();
        assert_eq!(p.list().len(), 2);
    }

    #[test]
    fn query_returns_free_after_set() {
        let p = fixture();
        p.set_free(0, 1024);
        assert_eq!(p.query(0).unwrap().free_bytes, 1024);
    }

    #[test]
    fn processes_round_trip() {
        let p = fixture();
        p.add_process(0, GpuProcess { pid: 1234, used_bytes: 100, name: "test".into() });
        assert_eq!(p.processes(0).len(), 1);
        assert_eq!(p.processes(0)[0].pid, 1234);
    }
}
