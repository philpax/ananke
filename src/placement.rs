//! Layer-aware placement across allowed devices.
//!
//! Produces an `Allocation` (per-device byte reservation) and
//! `CommandArgs` (llama.cpp CLI flags derived from the packing).

use std::collections::BTreeMap;

use smol_str::SmolStr;

use crate::allocator::AllocationTable;
use crate::config::{DeviceSlot, PlacementPolicy, ServiceConfig};
use crate::devices::{Allocation, DeviceId, DeviceSnapshot};
use crate::estimator::Estimate;

const ONE_LAYER_FUDGE_MULTIPLIER: u64 = 1;

#[derive(Debug, Clone, Default)]
pub struct CommandArgs {
    /// `-ngl N` value. `None` means do not emit the flag (caller uses
    /// `placement_override` escape hatch or cpu-only).
    pub ngl: Option<u32>,
    /// `--tensor-split A,B,...` if multiple GPUs carry layers.
    pub tensor_split: Option<Vec<u32>>,
    /// `-ot <regex>=<device>` rules, rendered verbatim from
    /// `service.raw.override_tensor`.
    pub override_tensor: Vec<String>,
}

#[derive(Debug)]
pub struct PackError {
    pub reason: String,
}

impl std::fmt::Display for PackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.reason)
    }
}

impl std::error::Error for PackError {}

#[derive(Debug)]
pub struct Packed {
    pub allocation: Allocation,
    pub args: CommandArgs,
}

/// Pack `estimate` onto allowed devices, respecting `policy`,
/// `override_tensor`, and live device capacity (`snapshot` minus any
/// already-reserved bytes from `reserved`).
pub fn pack(
    estimate: &Estimate,
    svc: &ServiceConfig,
    snapshot: &DeviceSnapshot,
    reserved: &AllocationTable,
) -> Result<Packed, PackError> {
    // Step 0: determine the allowed GPUs.
    let allowed_gpus = allowed_gpu_list(svc, snapshot);
    let allow_cpu = matches!(
        svc.placement_policy,
        PlacementPolicy::CpuOnly | PlacementPolicy::Hybrid
    );

    // Step 1: seed per-device bytes with non-layer tensors + override_tensor attributions.
    let mut per_device: BTreeMap<DeviceSlot, u64> = BTreeMap::new();

    // Token embeddings always go to CPU.
    if estimate.non_layer.token_embd_bytes > 0 {
        *per_device.entry(DeviceSlot::Cpu).or_default() += estimate.non_layer.token_embd_bytes;
    }

    // Output head: first allowed GPU if any GPU used, else CPU.
    let head_target = if let Some(first_gpu) = allowed_gpus.first() {
        DeviceSlot::Gpu(*first_gpu)
    } else {
        DeviceSlot::Cpu
    };
    if estimate.non_layer.output_head_bytes > 0 {
        *per_device.entry(head_target.clone()).or_default() += estimate.non_layer.output_head_bytes;
    }
    if estimate.non_layer.other_bytes > 0 {
        *per_device.entry(head_target.clone()).or_default() += estimate.non_layer.other_bytes;
    }

    // override_tensor already-attributed bytes (estimator filled this map).
    for (slot, bytes) in &estimate.override_tensor_bytes {
        *per_device.entry(slot.clone()).or_default() += *bytes;
    }

    // Step 2: layer walker. If per_layer_bytes is None (Mamba),
    // place all weights on the first device with room (GPU > CPU).
    let per_layer = estimate.per_layer_bytes.clone().unwrap_or_default();
    let mut layers_per_gpu: BTreeMap<u32, u32> = BTreeMap::new();
    let mut layers_on_cpu: u32 = 0;
    let mut gpu_remaining: BTreeMap<u32, u64> = BTreeMap::new();

    // Pre-reserve per-GPU headroom so the walker doesn't fill to the brim
    // and get pushed over by Steps 3-5 (KV, compute buffer, one-layer fudge).
    // Headroom per GPU: kv_share + compute_buffer_mb + one_layer_fudge.
    let n_layers_nonzero = per_layer.iter().filter(|b| **b > 0).count() as u64;
    let per_layer_avg = per_layer
        .iter()
        .sum::<u64>()
        .checked_div(n_layers_nonzero)
        .unwrap_or(0);
    let kv_total_pre = estimate
        .kv_per_token
        .saturating_mul(estimate.context as u64);
    let compute_headroom = estimate.compute_buffer_mb as u64 * 1024 * 1024;
    let gpu_headroom_each = compute_headroom
        + kv_total_pre.saturating_div(allowed_gpus.len().max(1) as u64)
        + per_layer_avg * ONE_LAYER_FUDGE_MULTIPLIER;

    for gpu in &allowed_gpus {
        let free = snapshot.free_bytes(&DeviceSlot::Gpu(*gpu)).unwrap_or(0);
        let reserved_here = sum_reserved(reserved, &DeviceSlot::Gpu(*gpu), &svc.name);
        let raw = free
            .saturating_sub(reserved_here)
            .saturating_sub(*per_device.get(&DeviceSlot::Gpu(*gpu)).unwrap_or(&0));
        gpu_remaining.insert(*gpu, raw.saturating_sub(gpu_headroom_each));
    }

    for (idx, bytes) in per_layer.iter().enumerate() {
        if *bytes == 0 {
            continue;
        }
        // Pick the GPU with the most remaining room that can hold this
        // layer — balances layers across equal-capacity GPUs rather than
        // filling GPU 0 to the brim first.
        let best_gpu = allowed_gpus
            .iter()
            .filter(|g| gpu_remaining.get(g).copied().unwrap_or(0) >= *bytes)
            .max_by_key(|g| gpu_remaining.get(g).copied().unwrap_or(0))
            .copied();
        match best_gpu {
            Some(gpu) => {
                let rem = gpu_remaining.get_mut(&gpu).unwrap();
                *rem = rem.saturating_sub(*bytes);
                *per_device.entry(DeviceSlot::Gpu(gpu)).or_default() += *bytes;
                *layers_per_gpu.entry(gpu).or_default() += 1;
            }
            None if allow_cpu => {
                *per_device.entry(DeviceSlot::Cpu).or_default() += *bytes;
                layers_on_cpu += 1;
            }
            None => {
                return Err(PackError {
                    reason: format!(
                        "layer {idx} ({bytes} bytes) does not fit on any allowed GPU"
                    ),
                });
            }
        }
    }

    // If the architecture gave no per-layer info (Mamba, or fallback for
    // unknown architectures), place the entire weights bundle into the
    // first GPU with room (or CPU). `fallback_on_gpu` marks that we
    // reserved whole-model space on a GPU without per-layer detail —
    // the ngl derivation later sets -ngl 999 so llama.cpp offloads
    // everything.
    let mut fallback_on_gpu = false;
    if per_layer.is_empty() && estimate.weights_bytes > 0 {
        let mut placed = false;
        for gpu in &allowed_gpus {
            let rem = gpu_remaining.get_mut(gpu).unwrap();
            if *rem >= estimate.weights_bytes {
                *rem = rem.saturating_sub(estimate.weights_bytes);
                *per_device.entry(DeviceSlot::Gpu(*gpu)).or_default() += estimate.weights_bytes;
                placed = true;
                fallback_on_gpu = true;
                break;
            }
        }
        if !placed && allow_cpu {
            *per_device.entry(DeviceSlot::Cpu).or_default() += estimate.weights_bytes;
        } else if !placed {
            return Err(PackError {
                reason: "weights do not fit on any allowed device".into(),
            });
        }
    }

    // Step 3: add KV bytes to GPUs proportional to layers placed, or
    // CPU for layers that spilled.
    let n_layers = per_layer.len() as u32;
    let kv_total = estimate
        .kv_per_token
        .saturating_mul(estimate.context as u64);
    if n_layers > 0 && kv_total > 0 {
        for gpu in &allowed_gpus {
            let share = layers_per_gpu.get(gpu).copied().unwrap_or(0);
            if share > 0 {
                let bytes = kv_total * share as u64 / n_layers as u64;
                *per_device.entry(DeviceSlot::Gpu(*gpu)).or_default() += bytes;
            }
        }
        if layers_on_cpu > 0 {
            let bytes = kv_total * layers_on_cpu as u64 / n_layers as u64;
            *per_device.entry(DeviceSlot::Cpu).or_default() += bytes;
        }
    }

    // Step 4: compute buffer per active backend (default 400 MB).
    let compute_bytes = estimate.compute_buffer_mb as u64 * 1024 * 1024;
    let active_slots: Vec<DeviceSlot> = per_device.keys().cloned().collect();
    for slot in &active_slots {
        *per_device.entry(slot.clone()).or_default() += compute_bytes;
    }

    // Step 5: one-layer fudge for tensor-split slop (spec §8.2.5).
    if n_layers > 0 && !per_layer.is_empty() {
        let per_layer_avg = per_layer.iter().sum::<u64>() / n_layers as u64;
        let per_layer_kv = if n_layers > 0 {
            kv_total / n_layers as u64
        } else {
            0
        };
        let fudge_each = ONE_LAYER_FUDGE_MULTIPLIER * (per_layer_avg + per_layer_kv);
        let slots: Vec<DeviceSlot> = per_device.keys().cloned().collect();
        for slot in slots {
            match slot {
                DeviceSlot::Gpu(_) => *per_device.entry(slot).or_default() += fudge_each,
                DeviceSlot::Cpu if layers_on_cpu > 0 => {
                    *per_device.entry(slot).or_default() += fudge_each
                }
                _ => {}
            }
        }
    }

    // Step 6: CommandArgs.
    let total_on_gpus: u32 = layers_per_gpu.values().sum();
    let ngl = if allowed_gpus.is_empty() {
        // cpu-only: emit -ngl 0 via the spawn code path (kept out of our args struct).
        Some(0)
    } else if fallback_on_gpu {
        // Unknown architecture / no per-layer info: emit -ngl 999 so
        // llama.cpp offloads everything we reserved for.
        Some(999)
    } else {
        Some(total_on_gpus)
    };

    let tensor_split = if allowed_gpus.len() > 1 && total_on_gpus > 0 {
        // Ratios in CUDA_VISIBLE_DEVICES-remapped order: render in the
        // same GPU-id order as the allocation iterates (ascending ids).
        Some(
            allowed_gpus
                .iter()
                .map(|g| layers_per_gpu.get(g).copied().unwrap_or(0))
                .collect(),
        )
    } else {
        None
    };

    let override_tensor = svc.raw.override_tensor.clone().unwrap_or_default();

    let allocation = Allocation {
        bytes: per_device
            .into_iter()
            .map(|(slot, bytes)| {
                let id = match slot {
                    DeviceSlot::Cpu => DeviceId::Cpu,
                    DeviceSlot::Gpu(n) => DeviceId::Gpu(n),
                };
                (id, bytes)
            })
            .collect(),
    };

    Ok(Packed {
        allocation,
        args: CommandArgs {
            ngl,
            tensor_split,
            override_tensor,
        },
    })
}

fn allowed_gpu_list(svc: &ServiceConfig, snapshot: &DeviceSnapshot) -> Vec<u32> {
    if svc.placement_policy == PlacementPolicy::CpuOnly {
        return Vec::new();
    }
    let declared_allow: Option<Vec<u32>> =
        svc.raw.devices.as_ref().and_then(|d| d.gpu_allow.clone());
    let all: Vec<u32> = snapshot.gpus.iter().map(|g| g.id).collect();
    match declared_allow {
        Some(list) if !list.is_empty() => list.into_iter().filter(|id| all.contains(id)).collect(),
        _ => all,
    }
}

fn sum_reserved(table: &AllocationTable, slot: &DeviceSlot, exclude: &SmolStr) -> u64 {
    table
        .iter()
        .filter(|(k, _)| k.as_str() != exclude.as_str())
        .filter_map(|(_, alloc)| alloc.get(slot))
        .sum::<u64>()
        * 1024
        * 1024
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::parse::RawService;
    use crate::config::validate::{AllocationMode, Filters, HealthSettings, Lifecycle, Template};
    use crate::devices::{CpuSnapshot, GpuSnapshot};
    use crate::estimator::NonLayer;
    use smol_str::SmolStr;

    fn svc(policy: PlacementPolicy, gpu_allow: Option<Vec<u32>>) -> ServiceConfig {
        let mut placement = BTreeMap::new();
        placement.insert(DeviceSlot::Gpu(0), 1000);
        let devices = gpu_allow.map(|a| crate::config::parse::RawServiceDevices {
            gpu_allow: Some(a),
            ..Default::default()
        });
        let raw = RawService {
            name: Some(SmolStr::new("demo")),
            template: Some(SmolStr::new("llama-cpp")),
            model: Some("/fake".into()),
            port: Some(0),
            devices,
            ..Default::default()
        };
        ServiceConfig {
            name: SmolStr::new("demo"),
            template: Template::LlamaCpp,
            port: 0,
            private_port: 0,
            lifecycle: Lifecycle::OnDemand,
            priority: 50,
            health: HealthSettings {
                http_path: "/".into(),
                timeout_ms: 1000,
                probe_interval_ms: 500,
            },
            placement_override: placement,
            placement_policy: policy,
            idle_timeout_ms: 60_000,
            warming_grace_ms: 100,
            drain_timeout_ms: 1000,
            extended_stream_drain_ms: 1000,
            max_request_duration_ms: 1000,
            filters: Filters::default(),
            allocation_mode: AllocationMode::None,
            command: None,
            workdir: None,
            openai_compat: true,
            raw,
        }
    }

    fn snapshot(free_gpu_gb: &[u64]) -> DeviceSnapshot {
        let gpus = free_gpu_gb
            .iter()
            .enumerate()
            .map(|(i, gb)| GpuSnapshot {
                id: i as u32,
                name: format!("GPU {i}"),
                total_bytes: 24 * 1024 * 1024 * 1024,
                free_bytes: gb * 1024 * 1024 * 1024,
            })
            .collect();
        DeviceSnapshot {
            gpus,
            cpu: Some(CpuSnapshot {
                total_bytes: 128 * 1024 * 1024 * 1024,
                available_bytes: 64 * 1024 * 1024 * 1024,
            }),
            taken_at_ms: 0,
        }
    }

    fn trivial_estimate(n_layers: u32, per_layer_mb: u64) -> Estimate {
        Estimate {
            weights_bytes: per_layer_mb * 1024 * 1024 * n_layers as u64,
            kv_per_token: 0,
            compute_buffer_mb: 400,
            per_layer_bytes: Some(vec![per_layer_mb * 1024 * 1024; n_layers as usize]),
            attention_layers: None,
            non_layer: NonLayer::default(),
            override_tensor_bytes: BTreeMap::new(),
            expert_layers: Vec::new(),
            expert_layer_cpu_bytes: BTreeMap::new(),
            context: 4096,
            architecture: SmolStr::new("qwen3"),
        }
    }

    #[test]
    fn single_gpu_fits() {
        let e = trivial_estimate(10, 100); // 10 layers × 100 MiB = 1 GiB
        let snap = snapshot(&[8]); // 8 GB free
        let alloc = AllocationTable::new();
        let packed = pack(&e, &svc(PlacementPolicy::GpuOnly, None), &snap, &alloc).unwrap();
        assert_eq!(packed.args.ngl, Some(10));
        assert!(packed.args.tensor_split.is_none());
    }

    #[test]
    fn multi_gpu_split_ratios_match_layer_counts() {
        // 20 layers, 1 GiB each; 2 GPUs with 12 GB free each.
        let e = trivial_estimate(20, 1024); // 20 GiB
        let snap = snapshot(&[12, 12]);
        let alloc = AllocationTable::new();
        let packed = pack(&e, &svc(PlacementPolicy::GpuOnly, None), &snap, &alloc).unwrap();
        let split = packed.args.tensor_split.as_ref().unwrap();
        assert_eq!(split.len(), 2);
        assert_eq!(split.iter().sum::<u32>(), packed.args.ngl.unwrap());
    }

    #[test]
    fn hybrid_spills_to_cpu() {
        let e = trivial_estimate(10, 100);
        let snap = snapshot(&[0]); // GPU full
        let alloc = AllocationTable::new();
        let packed = pack(&e, &svc(PlacementPolicy::Hybrid, None), &snap, &alloc).unwrap();
        // All layers should have spilled to CPU.
        assert!(packed.allocation.bytes.contains_key(&DeviceId::Cpu));
    }

    #[test]
    fn cpu_only_emits_ngl_zero_and_no_split() {
        let e = trivial_estimate(10, 100);
        let snap = snapshot(&[8]);
        let alloc = AllocationTable::new();
        let packed = pack(&e, &svc(PlacementPolicy::CpuOnly, None), &snap, &alloc).unwrap();
        assert_eq!(packed.args.ngl, Some(0));
        assert!(packed.args.tensor_split.is_none());
    }
}
