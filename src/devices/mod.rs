//! Device types and allocation primitives.

pub mod cpu;
pub mod cuda_env;
pub mod fake;
pub mod nvml;
pub mod probe;

use std::collections::BTreeMap;

use crate::config::validate::DeviceSlot;

pub use probe::{GpuInfo, GpuMemory, GpuProbe, GpuProcess};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Device {
    pub id: DeviceId,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DeviceId {
    Cpu,
    Gpu(u32),
}

impl DeviceId {
    pub fn to_slot(self) -> DeviceSlot {
        match self {
            DeviceId::Cpu => DeviceSlot::Cpu,
            DeviceId::Gpu(n) => DeviceSlot::Gpu(n),
        }
    }

    pub fn from_slot(slot: &DeviceSlot) -> Self {
        match slot {
            DeviceSlot::Cpu => DeviceId::Cpu,
            DeviceSlot::Gpu(n) => DeviceId::Gpu(*n),
        }
    }

    pub fn as_display(self) -> String {
        match self {
            DeviceId::Cpu => "cpu".into(),
            DeviceId::Gpu(n) => format!("gpu:{n}"),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Allocation {
    pub bytes: BTreeMap<DeviceId, u64>,
}

impl Allocation {
    pub fn from_override(map: &BTreeMap<DeviceSlot, u64>) -> Self {
        let mut bytes = BTreeMap::new();
        for (slot, b) in map {
            bytes.insert(DeviceId::from_slot(slot), b * 1024 * 1024); // MB → bytes
        }
        Self { bytes }
    }

    pub fn gpu_ids(&self) -> Vec<u32> {
        self.bytes
            .keys()
            .filter_map(|d| {
                if let DeviceId::Gpu(n) = d {
                    Some(*n)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn total(&self) -> u64 {
        self.bytes.values().sum()
    }
}

#[derive(Debug, Clone, Default)]
pub struct DeviceSnapshot {
    pub gpus: Vec<GpuSnapshot>,
    pub cpu: Option<CpuSnapshot>,
    pub taken_at_ms: u64,
}

#[derive(Debug, Clone)]
pub struct GpuSnapshot {
    pub id: u32,
    pub name: String,
    pub total_bytes: u64,
    pub free_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct CpuSnapshot {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

impl DeviceSnapshot {
    pub fn free_bytes(&self, slot: &crate::config::validate::DeviceSlot) -> Option<u64> {
        use crate::config::validate::DeviceSlot;
        match slot {
            DeviceSlot::Cpu => self.cpu.as_ref().map(|c| c.available_bytes),
            DeviceSlot::Gpu(id) => self.gpus.iter().find(|g| g.id == *id).map(|g| g.free_bytes),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::validate::DeviceSlot;

    #[test]
    fn allocation_from_override_converts_mb_to_bytes() {
        let mut m = BTreeMap::new();
        m.insert(DeviceSlot::Gpu(0), 1024);
        m.insert(DeviceSlot::Cpu, 2048);
        let a = Allocation::from_override(&m);
        assert_eq!(a.bytes[&DeviceId::Gpu(0)], 1024 * 1024 * 1024);
        assert_eq!(a.bytes[&DeviceId::Cpu], 2048 * 1024 * 1024);
    }

    #[test]
    fn gpu_ids_filters_cpu() {
        let mut m = BTreeMap::new();
        m.insert(DeviceSlot::Gpu(0), 10);
        m.insert(DeviceSlot::Gpu(1), 20);
        m.insert(DeviceSlot::Cpu, 30);
        let a = Allocation::from_override(&m);
        let mut ids = a.gpu_ids();
        ids.sort();
        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn device_id_display() {
        assert_eq!(DeviceId::Cpu.as_display(), "cpu");
        assert_eq!(DeviceId::Gpu(3).as_display(), "gpu:3");
    }
}
