//! Render `CUDA_VISIBLE_DEVICES` from an `Allocation`.

use super::Allocation;

/// Return the value to set for `CUDA_VISIBLE_DEVICES` given the service's
/// allocation and policy.
///
/// - If the allocation has no GPU entries (CPU-only), returns `""` so the
///   child cannot grab a GPU.
/// - Otherwise, returns the NVML indices comma-separated in ascending order
///   (e.g. `"0,2"`).
pub fn render(allocation: &Allocation) -> String {
    let mut ids = allocation.gpu_ids();
    ids.sort();
    ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::config::validate::DeviceSlot;

    use super::*;

    fn alloc(pairs: &[(DeviceSlot, u64)]) -> Allocation {
        let mut m = BTreeMap::new();
        for (k, v) in pairs {
            m.insert(k.clone(), *v);
        }
        Allocation::from_override(&m)
    }

    #[test]
    fn cpu_only_is_empty() {
        let a = alloc(&[(DeviceSlot::Cpu, 1000)]);
        assert_eq!(render(&a), "");
    }

    #[test]
    fn single_gpu() {
        let a = alloc(&[(DeviceSlot::Gpu(1), 1000)]);
        assert_eq!(render(&a), "1");
    }

    #[test]
    fn multi_gpu_sorted() {
        let a = alloc(&[(DeviceSlot::Gpu(3), 1), (DeviceSlot::Gpu(0), 1)]);
        assert_eq!(render(&a), "0,3");
    }

    #[test]
    fn hybrid_includes_only_gpus() {
        let a = alloc(&[(DeviceSlot::Gpu(0), 1), (DeviceSlot::Cpu, 1)]);
        assert_eq!(render(&a), "0");
    }
}
