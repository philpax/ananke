//! Layer-aware placement across allowed devices.
//!
//! Produces an `Allocation` (per-device byte reservation) and
//! `CommandArgs` (llama.cpp CLI flags derived from the packing).

use std::collections::BTreeMap;

use smol_str::SmolStr;

use crate::{
    allocator::AllocationTable,
    config::{DeviceSlot, PlacementPolicy, ServiceConfig},
    devices::{Allocation, DeviceId, DeviceSnapshot},
    estimator::Estimate,
};

/// Number of per-layer-equivalents added to every active backend as slop
/// tolerance for tensor-split rounding (spec §8.2.5). Bumped if empirical
/// overruns show tensor_split's remainder exceeds one layer's worth.
const ONE_LAYER_FUDGE_MULTIPLIER: u64 = 1;

/// `-ngl` value meaning "offload every layer to the GPU". Used when we
/// reserved whole-model space on a GPU without per-layer detail.
const NGL_OFFLOAD_ALL: u32 = 999;

/// `-ngl` value meaning "run entirely on CPU".
const NGL_CPU_ONLY: u32 = 0;

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
    let mut packer = Packer::new(estimate, svc, snapshot, reserved);
    packer.seed_non_layer();
    packer.walk_layers()?;
    packer.place_fallback_weights()?;
    packer.add_kv_bytes();
    packer.add_compute_buffer();
    packer.add_one_layer_fudge();
    Ok(packer.finish())
}

/// Mutable bag threaded through the pack steps. Each method mutates the
/// relevant subset of these fields and is documented with the single concern
/// it owns.
struct Packer<'a> {
    estimate: &'a Estimate,
    svc: &'a ServiceConfig,
    snapshot: &'a DeviceSnapshot,
    reserved: &'a AllocationTable,

    allowed_gpus: Vec<u32>,
    allow_cpu: bool,
    per_layer: Vec<u64>,

    /// Final per-device reservation totals. Steps 1-5 accumulate into this.
    per_device: BTreeMap<DeviceSlot, u64>,
    /// Remaining capacity per GPU as the walker consumes it.
    gpu_remaining: BTreeMap<u32, u64>,
    /// Number of layers the walker placed on each GPU.
    layers_per_gpu: BTreeMap<u32, u32>,
    /// Number of layers the walker spilled to CPU.
    layers_on_cpu: u32,
    /// Set when the layer count was unknown and we reserved whole-model
    /// space on a GPU. `-ngl 999` is emitted in that case so llama.cpp
    /// offloads everything for us.
    fallback_on_gpu: bool,
}

impl<'a> Packer<'a> {
    fn new(
        estimate: &'a Estimate,
        svc: &'a ServiceConfig,
        snapshot: &'a DeviceSnapshot,
        reserved: &'a AllocationTable,
    ) -> Self {
        // Step 0: determine the allowed GPUs and CPU permissibility.
        let allowed_gpus = allowed_gpu_list(svc, snapshot);
        let allow_cpu = matches!(
            svc.placement_policy,
            PlacementPolicy::CpuOnly | PlacementPolicy::Hybrid
        );
        let per_layer = estimate.per_layer_bytes.clone().unwrap_or_default();
        Self {
            estimate,
            svc,
            snapshot,
            reserved,
            allowed_gpus,
            allow_cpu,
            per_layer,
            per_device: BTreeMap::new(),
            gpu_remaining: BTreeMap::new(),
            layers_per_gpu: BTreeMap::new(),
            layers_on_cpu: 0,
            fallback_on_gpu: false,
        }
    }

    /// Step 1: seed per-device bytes with non-layer tensors + override_tensor
    /// attributions. Token embeddings go to CPU; output head + residual
    /// "other" tensors ride with the first allowed GPU (or CPU if there is no
    /// GPU). override_tensor has its own pre-computed map from the estimator.
    fn seed_non_layer(&mut self) {
        let non_layer = &self.estimate.non_layer;

        if non_layer.token_embd_bytes > 0 {
            *self.per_device.entry(DeviceSlot::Cpu).or_default() += non_layer.token_embd_bytes;
        }

        let head_target = match self.allowed_gpus.first() {
            Some(first_gpu) => DeviceSlot::Gpu(*first_gpu),
            None => DeviceSlot::Cpu,
        };
        if non_layer.output_head_bytes > 0 {
            *self.per_device.entry(head_target.clone()).or_default() += non_layer.output_head_bytes;
        }
        if non_layer.other_bytes > 0 {
            *self.per_device.entry(head_target).or_default() += non_layer.other_bytes;
        }

        for (slot, bytes) in &self.estimate.override_tensor_bytes {
            *self.per_device.entry(slot.clone()).or_default() += *bytes;
        }
    }

    /// Step 2: first-fit layer walker. Pre-reserves per-GPU headroom for
    /// steps 3-5 so we don't fill to the brim and then overflow. Returns
    /// `PackError` if a layer's bytes don't fit on any allowed GPU and CPU
    /// spill is disabled.
    fn walk_layers(&mut self) -> Result<(), PackError> {
        self.initialise_gpu_remaining();

        for (idx, bytes) in self.per_layer.iter().copied().enumerate() {
            if bytes == 0 {
                continue;
            }
            // First-fit: walk allowed GPUs in ascending-id order; pack onto
            // the first with room (spec §8.2). Single-GPU models stay on
            // GPU 0; multi-GPU models produce the natural unequal split.
            let placed = self
                .allowed_gpus
                .iter()
                .copied()
                .find(|gpu| self.gpu_remaining.get(gpu).copied().unwrap_or(0) >= bytes);
            match placed {
                Some(gpu) => {
                    *self.gpu_remaining.entry(gpu).or_default() -= bytes;
                    *self.per_device.entry(DeviceSlot::Gpu(gpu)).or_default() += bytes;
                    *self.layers_per_gpu.entry(gpu).or_default() += 1;
                }
                None if self.allow_cpu => {
                    *self.per_device.entry(DeviceSlot::Cpu).or_default() += bytes;
                    self.layers_on_cpu += 1;
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
        Ok(())
    }

    /// Reserve the worst-case per-GPU headroom for steps 3-5: full `kv_total`
    /// (not kv_total/N) because first-fit may land most layers on one GPU.
    /// Over-reserves on GPUs that end up with fewer layers, but guarantees no
    /// overflow.
    fn initialise_gpu_remaining(&mut self) {
        let n_layers_nonzero = self.per_layer.iter().filter(|b| **b > 0).count() as u64;
        let per_layer_avg = self
            .per_layer
            .iter()
            .sum::<u64>()
            .checked_div(n_layers_nonzero)
            .unwrap_or(0);
        let kv_total_pre = self
            .estimate
            .kv_per_token
            .saturating_mul(self.estimate.context as u64);
        let compute_headroom = self.estimate.compute_buffer_mb as u64 * 1024 * 1024;
        let gpu_headroom_each =
            compute_headroom + kv_total_pre + per_layer_avg * ONE_LAYER_FUDGE_MULTIPLIER;

        for gpu in &self.allowed_gpus {
            let free = self
                .snapshot
                .free_bytes(&DeviceSlot::Gpu(*gpu))
                .unwrap_or(0);
            let reserved_here = sum_reserved(self.reserved, &DeviceSlot::Gpu(*gpu), &self.svc.name);
            let raw = free
                .saturating_sub(reserved_here)
                .saturating_sub(*self.per_device.get(&DeviceSlot::Gpu(*gpu)).unwrap_or(&0));
            self.gpu_remaining
                .insert(*gpu, raw.saturating_sub(gpu_headroom_each));
        }
    }

    /// Fallback for architectures (Mamba, unknown) that didn't supply a
    /// per-layer breakdown: place the entire weights bundle on the first GPU
    /// with room, or spill to CPU.
    fn place_fallback_weights(&mut self) -> Result<(), PackError> {
        if !self.per_layer.is_empty() || self.estimate.weights_bytes == 0 {
            return Ok(());
        }
        let bytes = self.estimate.weights_bytes;
        for gpu in self.allowed_gpus.clone() {
            let rem = self.gpu_remaining.entry(gpu).or_default();
            if *rem >= bytes {
                *rem -= bytes;
                *self.per_device.entry(DeviceSlot::Gpu(gpu)).or_default() += bytes;
                self.fallback_on_gpu = true;
                return Ok(());
            }
        }
        if self.allow_cpu {
            *self.per_device.entry(DeviceSlot::Cpu).or_default() += bytes;
            Ok(())
        } else {
            Err(PackError {
                reason: "weights do not fit on any allowed device".into(),
            })
        }
    }

    /// Step 3: add KV bytes to GPUs proportional to layers placed, or to CPU
    /// for layers that spilled.
    fn add_kv_bytes(&mut self) {
        let n_layers = self.per_layer.len() as u32;
        let kv_total = self
            .estimate
            .kv_per_token
            .saturating_mul(self.estimate.context as u64);
        if n_layers == 0 || kv_total == 0 {
            return;
        }
        for gpu in &self.allowed_gpus {
            let share = self.layers_per_gpu.get(gpu).copied().unwrap_or(0);
            if share > 0 {
                let bytes = kv_total * share as u64 / n_layers as u64;
                *self.per_device.entry(DeviceSlot::Gpu(*gpu)).or_default() += bytes;
            }
        }
        if self.layers_on_cpu > 0 {
            let bytes = kv_total * self.layers_on_cpu as u64 / n_layers as u64;
            *self.per_device.entry(DeviceSlot::Cpu).or_default() += bytes;
        }
    }

    /// Step 4: compute buffer per active backend (default 400 MB).
    fn add_compute_buffer(&mut self) {
        let compute_bytes = self.estimate.compute_buffer_mb as u64 * 1024 * 1024;
        let active_slots: Vec<DeviceSlot> = self.per_device.keys().cloned().collect();
        for slot in active_slots {
            *self.per_device.entry(slot).or_default() += compute_bytes;
        }
    }

    /// Step 5: one-layer fudge for tensor-split slop (spec §8.2.5).
    fn add_one_layer_fudge(&mut self) {
        let n_layers = self.per_layer.len() as u32;
        if n_layers == 0 || self.per_layer.is_empty() {
            return;
        }
        let kv_total = self
            .estimate
            .kv_per_token
            .saturating_mul(self.estimate.context as u64);
        let per_layer_avg = self.per_layer.iter().sum::<u64>() / n_layers as u64;
        let per_layer_kv = kv_total / n_layers as u64;
        let fudge_each = ONE_LAYER_FUDGE_MULTIPLIER * (per_layer_avg + per_layer_kv);
        let slots: Vec<DeviceSlot> = self.per_device.keys().cloned().collect();
        for slot in slots {
            match slot {
                DeviceSlot::Gpu(_) => *self.per_device.entry(slot).or_default() += fudge_each,
                DeviceSlot::Cpu if self.layers_on_cpu > 0 => {
                    *self.per_device.entry(slot).or_default() += fudge_each
                }
                _ => {}
            }
        }
    }

    /// Step 6: materialise the final `Packed` — derive -ngl, --tensor-split,
    /// -ot, and convert the per_device map into an `Allocation`.
    fn finish(self) -> Packed {
        let total_on_gpus: u32 = self.layers_per_gpu.values().sum();
        let ngl = if self.allowed_gpus.is_empty() {
            Some(NGL_CPU_ONLY)
        } else if self.fallback_on_gpu {
            Some(NGL_OFFLOAD_ALL)
        } else {
            Some(total_on_gpus)
        };

        let tensor_split = if self.allowed_gpus.len() > 1 && total_on_gpus > 0 {
            // Ratios in CUDA_VISIBLE_DEVICES-remapped order: render in the
            // same GPU-id order as the allocation iterates (ascending ids).
            Some(
                self.allowed_gpus
                    .iter()
                    .map(|g| self.layers_per_gpu.get(g).copied().unwrap_or(0))
                    .collect(),
            )
        } else {
            None
        };

        let override_tensor = self.svc.raw.override_tensor.clone().unwrap_or_default();

        let allocation = Allocation {
            bytes: self
                .per_device
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

        Packed {
            allocation,
            args: CommandArgs {
                ngl,
                tensor_split,
                override_tensor,
            },
        }
    }
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
    use smol_str::SmolStr;

    use super::*;
    use crate::{
        config::validate::test_fixtures::minimal_service,
        devices::{CpuSnapshot, GpuSnapshot},
        estimator::NonLayer,
    };

    fn svc(policy: PlacementPolicy, gpu_allow: Option<Vec<u32>>) -> ServiceConfig {
        let mut placement = BTreeMap::new();
        placement.insert(DeviceSlot::Gpu(0), 1000);
        let mut svc = minimal_service("demo");
        svc.placement_override = placement;
        svc.placement_policy = policy;
        if let Some(a) = gpu_allow {
            svc.raw.devices = Some(crate::config::parse::RawServiceDevices {
                gpu_allow: Some(a),
                ..Default::default()
            });
        }
        svc.raw.model = Some("/fake".into());
        svc
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
