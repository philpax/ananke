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
/// tolerance for tensor-split rounding. Bumped if empirical overruns
/// show tensor_split's remainder exceeds one layer's worth.
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

/// Structured packer failure modes. Each variant carries the numbers the
/// operator needs to understand the overflow — no more string-matching
/// on the message to figure out what went wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackError {
    /// A specific block-layer's bytes didn't fit on any GPU the service
    /// was allowed to use, and CPU spill was disabled.
    LayerDoesNotFit { layer_index: u32, bytes: u64 },
    /// The estimator returned no per-layer breakdown (fallback path on
    /// an unknown architecture) and the weights can't fit on any
    /// allowed device.
    WeightsDoNotFit,
    /// A GPU's proportional KV-cache share (`kv_total × layers_placed_here /
    /// total_layers`) does not fit in its remaining capacity after the
    /// layer walk. Emitted by the post-walk validation that replaced the
    /// old "reserve full kv_total on every GPU" pessimism.
    KvShareDoesNotFit {
        gpu: u32,
        layers_placed: u32,
        kv_share_bytes: u64,
        remaining_bytes: u64,
    },
}

impl std::fmt::Display for PackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LayerDoesNotFit { layer_index, bytes } => {
                write!(
                    f,
                    "layer {layer_index} ({bytes} bytes) does not fit on any allowed GPU"
                )
            }
            Self::WeightsDoNotFit => f.write_str("weights do not fit on any allowed device"),
            Self::KvShareDoesNotFit {
                gpu,
                layers_placed,
                kv_share_bytes,
                remaining_bytes,
            } => {
                write!(
                    f,
                    "gpu {gpu}: KV share for {layers_placed} layers needs {kv_share_bytes} bytes, only {remaining_bytes} bytes remain after walk"
                )
            }
        }
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
    pack_inner(estimate, svc, snapshot, reserved, false)
}

/// Pack variant that trusts the pledge book (`total - reserved`) exclusively
/// rather than taking `min(nvml_free, total - reserved)`. Intended for the
/// retry-after-eviction path, where victims have been removed from `reserved`
/// to model "if they were gone" — nvml still shows their realized usage
/// until drains actually land.
pub fn pack_optimistic(
    estimate: &Estimate,
    svc: &ServiceConfig,
    snapshot: &DeviceSnapshot,
    reserved: &AllocationTable,
) -> Result<Packed, PackError> {
    pack_inner(estimate, svc, snapshot, reserved, true)
}

fn pack_inner(
    estimate: &Estimate,
    svc: &ServiceConfig,
    snapshot: &DeviceSnapshot,
    reserved: &AllocationTable,
    optimistic_remaining: bool,
) -> Result<Packed, PackError> {
    let mut packer = Packer::new(estimate, svc, snapshot, reserved, optimistic_remaining);
    packer.seed_non_layer();
    packer.walk_layers()?;
    packer.place_fallback_weights()?;
    packer.validate_kv_shares_fit()?;
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
    /// See `pack_optimistic` — controls whether we clamp per-GPU remaining
    /// against nvml-reported free bytes or trust the pledge book only.
    optimistic_remaining: bool,
}

impl<'a> Packer<'a> {
    fn new(
        estimate: &'a Estimate,
        svc: &'a ServiceConfig,
        snapshot: &'a DeviceSnapshot,
        reserved: &'a AllocationTable,
        optimistic_remaining: bool,
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
            optimistic_remaining,
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
            // Best-fit: pick the allowed GPU with the most remaining
            // capacity that still fits this layer. First-fit (walk by
            // ascending id) fills GPU 0 until nearly full, then the
            // post-layer compute_buffer / non-layer tails push it past
            // physical capacity for models that only just fit across two
            // GPUs (nemotron-49B at Q4 on 2×24 GB). Best-fit naturally
            // balances layers so tails land inside each GPU's budget.
            let placed = self
                .allowed_gpus
                .iter()
                .copied()
                .filter(|gpu| self.gpu_remaining.get(gpu).copied().unwrap_or(0) >= bytes)
                .max_by_key(|gpu| self.gpu_remaining.get(gpu).copied().unwrap_or(0));
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
                    return Err(PackError::LayerDoesNotFit {
                        layer_index: idx as u32,
                        bytes,
                    });
                }
            }
        }
        Ok(())
    }

    /// Reserve the fixed per-GPU headroom that does not depend on how layers
    /// end up distributed: compute buffer + one-layer fudge. The KV cache is
    /// **not** reserved here — it's validated post-walk against the actual
    /// layer distribution in [`Self::validate_kv_shares_fit`]. Reserving the
    /// full `kv_total` on every GPU up front would waste `kv_total × (N-1)`
    /// of capacity and prevent large-context models (e.g. 256K-context
    /// Gemma 4 31B on 2×24 GB) from packing even when the proportional KV
    /// share fits fine on each GPU.
    fn initialise_gpu_remaining(&mut self) {
        let n_layers_nonzero = self.per_layer.iter().filter(|b| **b > 0).count() as u64;
        let per_layer_avg = self
            .per_layer
            .iter()
            .sum::<u64>()
            .checked_div(n_layers_nonzero)
            .unwrap_or(0);
        let compute_headroom = self.estimate.compute_buffer_mb as u64 * 1024 * 1024;
        let gpu_headroom_each = compute_headroom + per_layer_avg * ONE_LAYER_FUDGE_MULTIPLIER;

        for gpu in &self.allowed_gpus {
            let slot = DeviceSlot::Gpu(*gpu);
            let free = self.snapshot.free_bytes(&slot).unwrap_or(0);
            let total = self.snapshot.total_bytes(&slot).unwrap_or(free);
            let reserved_here = sum_reserved(self.reserved, &slot, &self.svc.name);
            // Two views compete here:
            //   - `min(free, total - reserved)` (conservative): respects
            //     external VRAM pressure that nvml surfaces but our pledge
            //     book can't see.
            //   - `total - reserved` (optimistic): trusts the pledge book
            //     exclusively. Needed for retry-after-eviction, where we've
            //     removed victims from `reserved` to model "if they were
            //     gone" — nvml_free would still show their realized usage
            //     until the drain actually lands.
            // `optimistic_remaining` picks the right one.
            let via_pledge = total.saturating_sub(reserved_here);
            let available = if self.optimistic_remaining {
                via_pledge
            } else {
                free.min(via_pledge)
            };
            let raw = available.saturating_sub(*self.per_device.get(&slot).unwrap_or(&0));
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
            Err(PackError::WeightsDoNotFit)
        }
    }

    /// Step 2.5: confirm that each GPU's proportional KV-cache share fits in
    /// its post-walk remaining capacity. [`Self::initialise_gpu_remaining`]
    /// no longer pre-reserves KV (that would over-reserve by `kv_total × (N-1)`
    /// and reject fittable models); instead we wait until `walk_layers` has
    /// produced the actual `layers_per_gpu` distribution and check the real
    /// per-GPU share against what's left on each GPU. Any GPU whose share
    /// doesn't fit is reported by name with the exact deficit so the
    /// operator can see which device couldn't take its proportional KV.
    fn validate_kv_shares_fit(&self) -> Result<(), PackError> {
        let n_layers = self.per_layer.iter().filter(|b| **b > 0).count() as u64;
        if n_layers == 0 {
            return Ok(());
        }
        let kv_total = self
            .estimate
            .kv_per_token
            .saturating_mul(self.estimate.context as u64);
        if kv_total == 0 {
            return Ok(());
        }
        for gpu in &self.allowed_gpus {
            let layers_here = u64::from(*self.layers_per_gpu.get(gpu).unwrap_or(&0));
            if layers_here == 0 {
                continue;
            }
            let kv_share = kv_total.saturating_mul(layers_here) / n_layers;
            let remaining = self.gpu_remaining.get(gpu).copied().unwrap_or(0);
            if remaining < kv_share {
                return Err(PackError::KvShareDoesNotFit {
                    gpu: *gpu,
                    layers_placed: layers_here as u32,
                    kv_share_bytes: kv_share,
                    remaining_bytes: remaining,
                });
            }
        }
        Ok(())
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

    /// Step 5: one-layer fudge for tensor-split slop.
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

        let override_tensor = self
            .svc
            .llama_cpp()
            .map(|lc| lc.override_tensor.clone())
            .unwrap_or_default();

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
    let all: Vec<u32> = snapshot.gpus.iter().map(|g| g.id).collect();
    if svc.gpu_allow.is_empty() {
        all
    } else {
        svc.gpu_allow
            .iter()
            .copied()
            .filter(|id| all.contains(id))
            .collect()
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
            svc.gpu_allow = a;
        }
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

    /// Regression for scenario-02 nemotron-49B overcommit: first-fit filled
    /// GPU 0 until it nearly brimmed, then post-layer compute_buffer + KV +
    /// one-layer fudge tails pushed it past physical capacity. Best-fit
    /// spreads the layers across GPUs so the tails land inside each GPU's
    /// budget. With two equal-free GPUs and two layers, the expected outcome
    /// is a 1/1 split — first-fit would produce [2, 0].
    #[test]
    fn best_fit_balances_layers_across_equal_gpus() {
        let e = trivial_estimate(2, 1024); // 2 layers, 1 GiB each
        let snap = snapshot(&[20, 20]); // 20 GB free on each of two GPUs
        let alloc = AllocationTable::new();
        let packed = pack(&e, &svc(PlacementPolicy::GpuOnly, None), &snap, &alloc).unwrap();
        let split = packed.args.tensor_split.as_ref().unwrap();
        assert_eq!(split.len(), 2);
        assert_eq!(
            split,
            &vec![1u32, 1u32],
            "layers should split 1/1 across GPUs"
        );
        assert_eq!(packed.args.ngl, Some(2));
    }

    /// When GPU 1 starts with more free capacity, best-fit should bias
    /// placements toward it until the two are even. First-fit would ignore
    /// the imbalance and stuff everything onto GPU 0.
    #[test]
    fn best_fit_prefers_gpu_with_more_free() {
        let e = trivial_estimate(4, 1024); // 4 × 1 GiB layers
        let snap = snapshot(&[10, 16]); // GPU 1 has 6 GB more free
        let alloc = AllocationTable::new();
        let packed = pack(&e, &svc(PlacementPolicy::GpuOnly, None), &snap, &alloc).unwrap();
        let split = packed.args.tensor_split.as_ref().unwrap();
        assert_eq!(split.iter().sum::<u32>(), 4);
        assert!(
            split[1] >= split[0],
            "best-fit should place at least as many layers on GPU 1 (more free) as on GPU 0; got {split:?}"
        );
    }

    /// `pack_optimistic` ignores nvml's view of free bytes and trusts the
    /// pledge book (`total - reserved`) alone. Used on the retry-after-
    /// eviction path: the victims have been drained from `reserved`, but
    /// nvml still reports their realized usage until the drain actually
    /// lands. `pack` would reject the placement here; `pack_optimistic`
    /// should succeed.
    #[test]
    fn pack_optimistic_ignores_stale_nvml_free() {
        let e = trivial_estimate(4, 1024); // 4 GiB of layers
        let snap = snapshot(&[0]); // nvml says 0 free, but total = 24 GB
        let alloc = AllocationTable::new();

        // Conservative pack: `min(0, 24-0) = 0` per GPU, no spill allowed.
        let err = pack(&e, &svc(PlacementPolicy::GpuOnly, None), &snap, &alloc);
        assert!(
            err.is_err(),
            "pack must reject placement when nvml reports 0 free and spill is off"
        );

        // Optimistic pack: trust `total - reserved = 24 GB`, layers fit.
        let packed = pack_optimistic(&e, &svc(PlacementPolicy::GpuOnly, None), &snap, &alloc)
            .expect("pack_optimistic must succeed when the pledge book allows it");
        assert_eq!(packed.args.ngl, Some(4));
    }

    /// Regression for the 256K-context Gemma 4 31B repro: the pre-fix packer
    /// reserved full `kv_total` (11 GB) on every allowed GPU as worst-case
    /// headroom, leaving only ~9 GB per GPU for layers. That rejected layer
    /// 59 by ~40 MB even though the real proportional KV share per GPU was
    /// half of kv_total and the model genuinely fit on 2×24 GB.
    ///
    /// With the two-pass check, the walker runs with only compute-buffer +
    /// fudge reserved, lays all 60 layers out roughly 30/30 across the two
    /// GPUs, and the post-walk validator confirms each GPU's proportional
    /// KV share (~5.5 GB) fits in the ~10 GB remaining after the walk.
    #[test]
    fn long_context_moe_packs_with_proportional_kv_headroom() {
        // Gemma 4 31B numbers from the live failure log: 60 layers at ~296 MB
        // avg (≈17.8 GB total), kv_per_token = 45220, 256 K context, compute
        // buffer 3792 MB.
        let per_layer_bytes: Vec<u64> = (0..60).map(|_| 296 * 1024 * 1024).collect();
        let weights_bytes: u64 = per_layer_bytes.iter().sum();
        let e = Estimate {
            weights_bytes,
            kv_per_token: 45220,
            compute_buffer_mb: 3792,
            per_layer_bytes: Some(per_layer_bytes),
            attention_layers: None,
            non_layer: NonLayer::default(),
            override_tensor_bytes: BTreeMap::new(),
            expert_layers: Vec::new(),
            expert_layer_cpu_bytes: BTreeMap::new(),
            context: 262_144,
            architecture: SmolStr::new("gemma4"),
        };
        // 2×24 GB 3090s, fully free, empty pledge book.
        let snap = snapshot(&[24, 24]);
        let alloc = AllocationTable::new();
        let packed = pack(&e, &svc(PlacementPolicy::GpuOnly, None), &snap, &alloc)
            .expect("Gemma 4 31B at 256K context must pack on 2×24 GB");
        let split = packed.args.tensor_split.as_ref().expect("two-GPU split");
        assert_eq!(split.iter().sum::<u32>(), 60);
        // Best-fit should keep the split close to even.
        let diff = (split[0] as i32 - split[1] as i32).abs();
        assert!(
            diff <= 4,
            "best-fit should distribute layers roughly evenly; got {split:?}"
        );
    }

    /// The post-walk validator must still reject layouts where a GPU's
    /// proportional KV share truly doesn't fit — e.g. when the walker piles
    /// most layers onto one GPU because another GPU has asymmetric seed
    /// content and the context is large enough that the pile-up's KV share
    /// exceeds what's left.
    #[test]
    fn post_walk_validator_rejects_overcommitted_kv_share() {
        // 60 layers × 200 MB = 12 GB weights. kv_total at 128K tokens with
        // kv_per_token = 120 KB ≈ 15 GB. If one GPU gets all 60 layers,
        // its kv_share = 15 GB; 24 GB total − 12 GB layers − compute buffer
        // (2 GB) ≈ 10 GB remaining, < 15 GB share. Must fail.
        let per_layer_bytes: Vec<u64> = (0..60).map(|_| 200 * 1024 * 1024).collect();
        let weights_bytes: u64 = per_layer_bytes.iter().sum();
        let e = Estimate {
            weights_bytes,
            kv_per_token: 120_000,
            compute_buffer_mb: 2048,
            per_layer_bytes: Some(per_layer_bytes),
            attention_layers: None,
            non_layer: NonLayer::default(),
            override_tensor_bytes: BTreeMap::new(),
            expert_layers: Vec::new(),
            expert_layer_cpu_bytes: BTreeMap::new(),
            context: 131_072,
            architecture: SmolStr::new("qwen3"),
        };
        // Single 24 GB GPU, no spill allowed — forces all 60 layers onto GPU 0.
        let snap = snapshot(&[24]);
        let alloc = AllocationTable::new();
        let err = pack(&e, &svc(PlacementPolicy::GpuOnly, None), &snap, &alloc)
            .expect_err("kv share must overflow single GPU at long context");
        assert!(
            matches!(err, PackError::KvShareDoesNotFit { gpu: 0, .. }),
            "expected KvShareDoesNotFit on gpu 0, got {err:?}"
        );
    }
}
