//! NVML-backed [`GpuProbe`] impl.

use std::sync::Arc;

use nvml_wrapper::Nvml;
use parking_lot::Mutex;
use tracing::warn;

use super::probe::{GpuInfo, GpuMemory, GpuProbe, GpuProcess};

pub struct NvmlProbe {
    nvml: Arc<Nvml>,
    cache: Mutex<Vec<GpuInfo>>,
}

impl NvmlProbe {
    pub fn init() -> Result<Self, String> {
        // Unset CUDA_VISIBLE_DEVICES so NVML sees every GPU regardless of ambient env (spec §4.3).
        // SAFETY: single-threaded at init time; no other threads are reading env vars.
        unsafe { std::env::remove_var("CUDA_VISIBLE_DEVICES") };

        let nvml = Nvml::init().map_err(|e| format!("NVML init failed: {e}"))?;
        let count = nvml
            .device_count()
            .map_err(|e| format!("NVML device_count failed: {e}"))?;
        let mut infos = Vec::with_capacity(count as usize);
        for i in 0..count {
            let dev = nvml
                .device_by_index(i)
                .map_err(|e| format!("NVML device_by_index({i}) failed: {e}"))?;
            let name = dev.name().unwrap_or_else(|_| format!("GPU {i}"));
            let total = dev.memory_info().map(|m| m.total).unwrap_or(0);
            infos.push(GpuInfo {
                id: i,
                name,
                total_bytes: total,
            });
        }
        Ok(Self {
            nvml: Arc::new(nvml),
            cache: Mutex::new(infos),
        })
    }
}

impl GpuProbe for NvmlProbe {
    fn list(&self) -> Vec<GpuInfo> {
        self.cache.lock().clone()
    }

    fn query(&self, id: u32) -> Option<GpuMemory> {
        match self.nvml.device_by_index(id) {
            Ok(dev) => match dev.memory_info() {
                Ok(m) => Some(GpuMemory {
                    total_bytes: m.total,
                    free_bytes: m.free,
                }),
                Err(e) => {
                    warn!(gpu = id, error = %e, "NVML memory_info failed");
                    None
                }
            },
            Err(e) => {
                warn!(gpu = id, error = %e, "NVML device_by_index failed");
                None
            }
        }
    }

    fn processes(&self, id: u32) -> Vec<GpuProcess> {
        let Ok(dev) = self.nvml.device_by_index(id) else {
            return Vec::new();
        };
        dev.running_compute_processes()
            .map(|procs| {
                procs
                    .into_iter()
                    .map(|p| {
                        let used = match p.used_gpu_memory {
                            nvml_wrapper::enums::device::UsedGpuMemory::Used(b) => b,
                            nvml_wrapper::enums::device::UsedGpuMemory::Unavailable => 0,
                        };
                        let name = std::fs::read_to_string(format!("/proc/{}/comm", p.pid))
                            .map(|s| s.trim().to_string())
                            .unwrap_or_else(|_| format!("pid {}", p.pid));
                        GpuProcess {
                            pid: p.pid,
                            used_bytes: used,
                            name,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}
