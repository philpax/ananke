//! The `Packer` struct: the mutable bag of placement state threaded through
//! every pack step, plus the core helpers (construction, per-GPU capacity
//! accounting, the output-head/logits attribution, and the CPU-capacity
//! gate) shared by the layer-walk, expert-aware, and sharded pack paths.

use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet},
};

use ananke_config::placement::{DeviceSlot, OffloadMode, PlacementInputs, PlacementPolicy};
use ananke_estimate::Estimate;

pub(crate) use crate::charge::Charge;
use crate::{
    AllocationTable, Corrections,
    devices::{DeviceId, DeviceSnapshot},
    entry::{ONE_LAYER_FUDGE_MULTIPLIER, PackMode},
    reserve::{allowed_gpu_list, gpu_reserve_bytes, sum_reserved},
    types::{DeviceShortfall, PackError, ShardedPlan},
};

/// Mutable bag threaded through the pack steps. Each method mutates the
/// relevant subset of these fields and is documented with the single concern
/// it owns.
pub(crate) struct Packer<'a> {
    pub(crate) estimate: &'a Estimate,
    pub(crate) placement: &'a PlacementInputs,
    pub(crate) snapshot: &'a DeviceSnapshot,
    pub(crate) reserved: &'a AllocationTable,

    pub(crate) allowed_gpus: Vec<u32>,
    pub(crate) allow_cpu: bool,
    pub(crate) per_layer: Vec<u64>,

    /// The rolling correction factors this pack runs with, one per memory
    /// pool. Every byte charged to a device is scaled by its pool's factor —
    /// see [`Self::charge`].
    pub(crate) corrections: Corrections,

    /// Final per-device reservation totals, corrected. Steps 1-5 accumulate
    /// into this.
    pub(crate) per_device: BTreeMap<DeviceSlot, u64>,
    /// Per-device totals as the estimator predicted them, before the rolling
    /// correction. Accumulated alongside `per_device` so the next rolling
    /// update can divide an observation by what was *predicted* rather than by
    /// a base the last correction already moved — otherwise the mean
    /// integrates against its own output and decays back toward 1.0.
    pub(crate) raw_per_device: BTreeMap<DeviceSlot, u64>,
    /// The [`Charge::Weights`] subset of `raw_per_device`: model tensor bytes
    /// read from the GGUF, excluding KV, compute buffers, and slop. The
    /// host-pool observation needs the GPU-resident share of this to subtract
    /// the mmap'd file pages that llama.cpp reads through on its way to VRAM.
    pub(crate) raw_weight_per_device: BTreeMap<DeviceSlot, u64>,
    /// Remaining capacity per GPU as the walker consumes it.
    pub(crate) gpu_remaining: BTreeMap<u32, u64>,
    /// Number of layers the walker placed on each GPU.
    pub(crate) layers_per_gpu: BTreeMap<u32, u32>,
    /// Number of layers the walker spilled to CPU.
    pub(crate) layers_on_cpu: u32,
    /// Set when the layer count was unknown and we reserved whole-model
    /// space on a GPU. `-ngl 999` is emitted in that case so llama.cpp
    /// offloads everything for us.
    pub(crate) fallback_on_gpu: bool,
    /// See `pack_optimistic` — controls whether we clamp per-GPU remaining
    /// against nvml-reported free bytes or trust the pledge book only.
    pub(crate) optimistic_remaining: bool,
    /// What this pack is for. Read by the steps that must not constrain a
    /// [`PackMode::Demand`] run the way they constrain a real placement.
    pub(crate) mode: PackMode,

    /// Set by [`Self::distribute_sharded`] for tensor/row split; drives the
    /// `--split-mode`/`--main-gpu`/equal `--tensor-split` emission in
    /// [`Self::finish`]. `None` on the layer-split path.
    pub(crate) sharded: Option<ShardedPlan>,

    /// MoE expert-offload policy for this service.
    pub(crate) offload_mode: OffloadMode,
    /// `true` when the two-phase expert-aware path runs: an offload mode is
    /// enabled *and* the model carries experts. Set in [`Self::new`].
    pub(crate) expert_aware: bool,
    /// Per-layer expert byte totals (sum of the layer's fused expert tensors).
    pub(crate) expert_bytes_by_layer: BTreeMap<u32, u64>,
    /// Layer → home GPU, set by the Phase-A non-expert walk. A layer absent
    /// here either has no weight or spilled wholly to CPU.
    pub(crate) layer_home: BTreeMap<u32, u32>,
    /// Layers whose whole weight (including experts) spilled to CPU in Phase A;
    /// their experts are part of that lump and skipped in Phase B.
    pub(crate) spilled_layers: BTreeSet<u32>,
    /// Total expert bytes moved to the CPU (for the placement preview).
    pub(crate) expert_offload_cpu_bytes: u64,
    /// Distinct layers with at least one expert offloaded to CPU.
    pub(crate) expert_offload_cpu_layers: BTreeSet<u32>,
    /// `--n-cpu-moe N` value set by [`Self::distribute_experts_ncmoe`]. Drives
    /// the coarse whole-layer offload emission in [`Self::finish`].
    pub(crate) n_cpu_moe: Option<u32>,
    pub(crate) expert_tensor_split: Option<Vec<u32>>,
}

impl<'a> Packer<'a> {
    pub(crate) fn new(
        estimate: &'a Estimate,
        placement: &'a PlacementInputs,
        snapshot: &'a DeviceSnapshot,
        reserved: &'a AllocationTable,
        mode: PackMode,
        corrections: Corrections,
    ) -> Self {
        // Demand shares Optimistic's capacity view: it asks what the model
        // needs, which is a property of the model and the hardware, not of who
        // currently holds a pledge.
        let optimistic_remaining = matches!(mode, PackMode::Optimistic | PackMode::Demand);
        let mut allowed_gpus = allowed_gpu_list(placement, snapshot);
        // Sort by descending pledge-book headroom (total - already committed)
        // so the GPU with the fewest active reservations is tried first. Using
        // the pledge book rather than nvml_free avoids letting driver-level
        // VRAM fluctuations (which vary by CUDA init order even on a fresh
        // boot) influence which GPU becomes the primary for a new model.
        allowed_gpus.sort_by_key(|gpu| {
            let slot = DeviceSlot::Gpu(*gpu);
            let total = snapshot.total_bytes(&slot).unwrap_or(0);
            let pledged = sum_reserved(reserved, &slot, &placement.name);
            Reverse(total.saturating_sub(pledged))
        });
        // Demand forces CPU spill on regardless of policy: the surplus of a
        // model too large for the GPUs has to land *somewhere* for the total to
        // be countable, and an unbounded host is the premise of the question.
        // A `GpuOnly` service would otherwise fail the walk and yield no figure.
        let allow_cpu = matches!(mode, PackMode::Demand)
            || matches!(
                placement.policy,
                PlacementPolicy::CpuOnly | PlacementPolicy::Hybrid
            );
        let per_layer = estimate.layout.per_layer_bytes.clone().unwrap_or_default();

        let offload_mode = placement.expert_offload;
        let expert_tensors = estimate.layout.expert_tensors.clone().unwrap_or_default();
        let expert_aware = offload_mode.is_enabled() && !expert_tensors.is_empty();
        let mut expert_bytes_by_layer: BTreeMap<u32, u64> = BTreeMap::new();
        for e in &expert_tensors {
            *expert_bytes_by_layer.entry(e.layer).or_default() += e.bytes;
        }

        Self {
            estimate,
            placement,
            snapshot,
            reserved,
            allowed_gpus,
            allow_cpu,
            per_layer,
            corrections,
            per_device: BTreeMap::new(),
            raw_per_device: BTreeMap::new(),
            raw_weight_per_device: BTreeMap::new(),
            gpu_remaining: BTreeMap::new(),
            layers_per_gpu: BTreeMap::new(),
            layers_on_cpu: 0,
            fallback_on_gpu: false,
            optimistic_remaining,
            mode,
            sharded: None,
            offload_mode,
            expert_aware,
            expert_bytes_by_layer,
            layer_home: BTreeMap::new(),
            spilled_layers: BTreeSet::new(),
            expert_offload_cpu_bytes: 0,
            expert_offload_cpu_layers: BTreeSet::new(),
            n_cpu_moe: None,
            expert_tensor_split: None,
        }
    }

    /// Step 1: seed per-device bytes with non-layer tensors + override_tensor
    /// attributions. Token embeddings are CPU-mapped; the output head — or, on
    /// a tied-embedding model, the embedding table serving as one — plus the
    /// residual "other" tensors ride with the first allowed GPU (or CPU if
    /// there is no GPU). override_tensor has its own pre-computed map from the
    /// estimator.
    pub(crate) fn seed_non_layer(&mut self) {
        let non_layer = self.estimate.layout.non_layer.clone();

        if non_layer.token_embd_bytes > 0 {
            self.charge(DeviceSlot::Cpu, non_layer.token_embd_bytes, Charge::Weights);
        }

        let head_target = match self.head_gpu() {
            Some(head) => DeviceSlot::Gpu(head),
            None => DeviceSlot::Cpu,
        };
        // A tied model's table is GPU-resident *as well as* CPU-mapped: it is
        // the output head, and the logits are computed on the device. Across
        // every no-offload mainline cell, the GPU model buffer comes to the
        // whole GGUF for a tied model, and to the GGUF *less* the table for one
        // that ships its own head. Charging it to the CPU alone under-reserves
        // every tied model by the table's whole size.
        //
        // Only `token_embd.weight`; a Gemma 4 E-variant's
        // `per_layer_token_embd.weight` stack stays on the CPU, which is why
        // `tied_head_bytes` is not the whole `token_embd_bytes` bucket.
        let gpu_head_bytes = non_layer.output_head_bytes + non_layer.tied_head_bytes;
        if gpu_head_bytes > 0 {
            self.charge(head_target.clone(), gpu_head_bytes, Charge::Weights);
        }
        if non_layer.other_bytes > 0 {
            self.charge(head_target, non_layer.other_bytes, Charge::Weights);
        }

        for (slot, bytes) in self.estimate.layout.override_tensor_bytes.clone() {
            self.charge(slot, bytes, Charge::Weights);
        }
    }

    /// The GPU that carries the output head + logits buffer at runtime: the
    /// lowest-id allowed GPU. `cuda_env` remaps the allowed cards to
    /// `CUDA_VISIBLE_DEVICES` in ascending id order, so the lowest id is CUDA
    /// visible index 0 — llama.cpp's default `main_gpu`, where the output/logits
    /// live, and the head the `--tensor-split` emission assumes. `allowed_gpus`
    /// is sorted by *descending headroom* for the layer walk, so `.first()` is
    /// the most-free card, which is the head only when VRAM is symmetric; using
    /// it as the head mis-attributes the fixed output cost in the pledge book
    /// once a resident model makes the cards asymmetric.
    pub(crate) fn head_gpu(&self) -> Option<u32> {
        self.allowed_gpus.iter().min().copied()
    }

    /// Output logits buffer bytes attributed to the head GPU only, capped at
    /// half the compute buffer. The cap guards a large-vocab MoE on an
    /// *uncalibrated* arch (whose `compute_buffer_mb` is the 400 MiB default),
    /// where `n_vocab × ubatch × 2` could exceed the whole compute buffer and
    /// otherwise zero out — or, in the tensor-split, over-bias — a secondary
    /// card's compute reservation. Shipping calibrated archs have
    /// `compute_buffer_mb` far above this term, so the cap never binds there.
    pub(crate) fn head_logits_bytes(&self) -> u64 {
        let compute = self.estimate.buffers.compute_mb as u64 * 1024 * 1024;
        self.estimate.buffers.output_bytes.min(compute / 2)
    }

    /// Reserve the fixed per-GPU headroom that does not depend on how layers
    /// end up distributed: compute buffer + one-layer fudge. The per-layer KV
    /// *of placed layers* is reserved incrementally during the walk (folded
    /// into [`crate::layer_walk`]'s `layer_cost`); the
    /// headroom reserved here must additionally cover the *fudge* layer that
    /// `add_one_layer_fudge` adds post-walk — and that fudge is
    /// `per_layer_avg + per_layer_kv`, so both terms have to be reserved
    /// here. Reserving only the weight term lets a GPU that the walker fills
    /// to the brim overshoot its capacity by one layer's KV — the live
    /// qwen3.6-27b "insufficient_capacity on gpu:0", short by ~one
    /// `per_layer_kv`. Including `per_layer_kv` makes the post-walk total land
    /// at or below `available`.
    pub(crate) fn initialise_gpu_remaining(&mut self) {
        let n_layers = self.per_layer.len() as u64;
        let per_layer_avg = self.effective_layer_avg();
        let kv_total = self
            .estimate
            .kv_per_token
            .saturating_mul(self.estimate.context as u64);
        let per_layer_kv = kv_total.checked_div(n_layers).unwrap_or(0);
        let compute_headroom = self.estimate.buffers.compute_mb as u64 * 1024 * 1024;
        let fudge = (per_layer_avg + per_layer_kv) * ONE_LAYER_FUDGE_MULTIPLIER;
        // The output logits buffer lives only on the head GPU, so every
        // secondary GPU needs that much less compute headroom — freeing the
        // room for expert weight. Must stay in lockstep with
        // [`crate::layer_walk::add_compute_buffer`]
        // (same head, same trim).
        let head_gpu = self.head_gpu();
        let logits = self.head_logits_bytes();

        for gpu in &self.allowed_gpus {
            let slot = DeviceSlot::Gpu(*gpu);
            let device_compute = if head_gpu == Some(*gpu) {
                compute_headroom.saturating_add(self.estimate.buffers.mmproj_graph_bytes)
            } else {
                compute_headroom.saturating_sub(logits)
            };
            let available = self.gpu_available(*gpu);
            let raw = available.saturating_sub(*self.per_device.get(&slot).unwrap_or(&0));
            // Pre-reserve the corrected headroom, since `add_compute_buffer`
            // and `add_one_layer_fudge` will charge the corrected amount. A raw
            // pre-reservation here against a corrected charge there lets a card
            // the walker fills to the brim overshoot by the correction's share
            // of the compute buffer.
            let headroom = self.vram_cost(device_compute + fudge);
            self.gpu_remaining
                .insert(*gpu, raw.saturating_sub(headroom));
        }
    }

    /// Available bytes on `gpu` under the active remaining-capacity view:
    /// `min(nvml_free, total - pledged)` normally, or `total - pledged`
    /// (optimistic) on the eviction-retry path. Does *not* subtract bytes
    /// this packer has already attributed to the GPU — callers do that.
    ///
    /// Two views compete: the conservative `min(free, total - reserved)`
    /// respects external VRAM pressure nvml surfaces but the pledge book
    /// can't see; the optimistic `total - reserved` trusts the pledge book
    /// alone, needed for retry-after-eviction where victims have been removed
    /// from `reserved` but nvml still shows their realized usage until the
    /// drain lands. `optimistic_remaining` picks.
    pub(crate) fn gpu_available(&self, gpu: u32) -> u64 {
        let slot = DeviceSlot::Gpu(gpu);
        let free = self.snapshot.free_bytes(&slot).unwrap_or(0);
        let total = self.snapshot.total_bytes(&slot).unwrap_or(free);
        let reserved_here = sum_reserved(self.reserved, &slot, &self.placement.name);
        let via_pledge = total.saturating_sub(reserved_here);
        let avail = if self.optimistic_remaining {
            via_pledge
        } else {
            free.min(via_pledge)
        };
        // Keep the configured headroom (global `[devices]` reserve + this
        // service's `gpu_headroom_mb`) free on the card.
        avail.saturating_sub(gpu_reserve_bytes(self.placement, gpu))
    }

    /// Per-allowed-GPU breakdown of a failed placement of `requested` bytes,
    /// in ascending GPU-id order (`allowed_gpus` is sorted by headroom, which
    /// would otherwise make the reported order depend on live device state).
    /// `available` is the walker's *remaining* capacity, not the card's raw
    /// free bytes, so it reflects what was actually left when the fit failed.
    ///
    /// Only cards that genuinely came up short are reported. Every entry in a
    /// placement-failure list should be a reason the placement failed; a card
    /// listed as needing 1.5 GiB while holding 1.9 GiB reads as a
    /// contradiction. This matters for the pooled expert-offload failure,
    /// where `requested` is each card's even share of an aggregate ask and the
    /// cards' remaining capacity can be lopsided enough that some satisfy it.
    ///
    /// Empty when the service has no eligible GPU at all — a GPU-less host
    /// (NVML unavailable), or a `gpu_allow` that names no present card. There
    /// is genuinely no device to point at in that case. Whenever at least one
    /// card is allowed the result is non-empty, since every call site has
    /// already established that the cards cannot collectively hold
    /// `n × requested`, so at least one must hold less than `requested`.
    pub(crate) fn gpu_shortfalls(&self, requested: u64) -> Vec<DeviceShortfall> {
        let mut gpus = self.allowed_gpus.clone();
        gpus.sort_unstable();
        gpus.into_iter()
            .map(|gpu| DeviceShortfall {
                device: DeviceId::Gpu(gpu),
                requested_bytes: requested,
                available_bytes: self.gpu_remaining.get(&gpu).copied().unwrap_or(0),
            })
            .filter(|s| s.available_bytes < s.requested_bytes)
            .collect()
    }

    /// The per-layer weight average used for headroom/fudge reservations. For
    /// the expert-aware path only the non-expert part of a layer is pinned to a
    /// GPU as a unit (experts are placed individually), so the slop estimate
    /// should reflect the non-expert size — using the full (expert-inflated)
    /// average would reserve enormous, mostly-wasted headroom.
    pub(crate) fn effective_layer_avg(&self) -> u64 {
        let n = self.per_layer.len() as u64;
        if n == 0 {
            return 0;
        }
        let total: u64 = self.per_layer.iter().sum();
        // Subtract expert bytes for the expert-aware path so the fudge reflects
        // only the non-expert weight that is pinned to a GPU as a unit. The
        // source has to be `expert_bytes_by_layer`, which stands for the whole
        // pack: `add_one_layer_fudge` runs *after* the expert phase, so reading
        // anything that phase consumes makes it see zero experts and reserve a
        // full expert-inflated layer per GPU, over-committing a hybrid MoE by
        // ~one layer's expert bytes and falsely failing the fit check.
        let total = if self.expert_aware {
            total.saturating_sub(self.expert_bytes_by_layer.values().sum::<u64>())
        } else {
            total
        };
        total / n
    }

    pub(crate) fn check_gpu_capacity(&self) -> Result<(), PackError> {
        for (slot, &needed) in &self.per_device {
            let DeviceSlot::Gpu(gpu) = slot else {
                continue;
            };
            let available = self.gpu_available(*gpu);
            if needed > available {
                return Err(PackError::GpuDoesNotFit {
                    gpu_index: *gpu,
                    needed,
                    available,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ananke_config::placement::{OffloadMode, PlacementPolicy};
    use smol_str::SmolStr;

    use super::*;
    use crate::{
        devices::DeviceId,
        entry::pack,
        test_support::{GIB, MIB, cpu_bytes, moe_estimate, moe_svc, snapshot, trivial_estimate},
    };

    /// The output logits buffer is materialised only on the head GPU (the
    /// first allowed), so the packer reserves the full compute buffer there
    /// but trims `output_buffer_bytes` off every secondary GPU. That freed
    /// VRAM fills with expert weight, so a nonzero `output_buffer_bytes` keeps
    /// strictly more experts on the GPUs (less spills to CPU) than the same
    /// estimate with the term zeroed — the whole point of the split.
    #[test]
    fn output_buffer_frees_secondary_gpu_for_experts() {
        // ~84 GiB of experts over 40 layers on 2×24 GiB → most spill to CPU;
        // the GPU-resident count is bounded by per-card compute headroom.
        let snap = snapshot(&[24, 24]);
        let svc = moe_svc(OffloadMode::Auto);

        let mut without = moe_estimate(40, 150, 700);
        without.buffers.compute_mb = 3000;
        without.buffers.output_bytes = 0;

        let mut with = without.clone();
        // A logits buffer worth ~two 700 MiB expert tensors on the secondary.
        with.buffers.output_bytes = 1400 * MIB;

        let cpu_without = cpu_bytes(&pack(&without, &svc, &snap, &AllocationTable::new()).unwrap());
        let cpu_with = cpu_bytes(&pack(&with, &svc, &snap, &AllocationTable::new()).unwrap());

        assert!(
            cpu_with < cpu_without,
            "trimming the head-only logits buffer off the secondary GPU must \
             keep more experts resident (cpu_with={cpu_with} cpu_without={cpu_without})"
        );
    }

    /// The output head/logits are hosted on the lowest-id GPU at runtime (CUDA
    /// visible index 0 = llama.cpp `main_gpu`), which must match the head the
    /// `--tensor-split` assumes — NOT `allowed_gpus`' most-free ordering. Under
    /// asymmetric pledge headroom (a resident model already on gpu:0) the packer
    /// sorts gpu:1 first, but the output head must still land on gpu:0 so the
    /// per-card pledge book agrees with where the runtime puts it.
    #[test]
    fn output_head_goes_to_lowest_id_gpu_under_asymmetric_headroom() {
        let mut e = trivial_estimate(1, 1024); // 1 layer, ~1 GiB
        e.layout.non_layer.output_head_bytes = 5 * GIB;
        let svc = PlacementInputs {
            policy: PlacementPolicy::GpuOnly,
            ..PlacementInputs::named("m")
        };
        let snap = snapshot(&[24, 24]);
        // A resident model pledges 15 GiB on gpu:0, so gpu:1 has more pledge
        // headroom and sorts first — `allowed_gpus.first()` is now gpu:1.
        let mut reserved = AllocationTable::new();
        let mut resident = BTreeMap::new();
        resident.insert(DeviceSlot::Gpu(0), 15 * 1024);
        reserved.insert(SmolStr::new("resident"), resident);
        let packed = pack(&e, &svc, &snap, &reserved).unwrap();
        let g0 = packed
            .allocation
            .bytes
            .get(&DeviceId::Gpu(0))
            .copied()
            .unwrap_or(0);
        let g1 = packed
            .allocation
            .bytes
            .get(&DeviceId::Gpu(1))
            .copied()
            .unwrap_or(0);
        assert!(
            g0 >= 5 * GIB,
            "output head (5 GiB) must be booked on the lowest-id gpu:0, got g0={g0}"
        );
        assert!(
            g1 < 5 * GIB,
            "the most-free gpu:1 must not carry the output head, got g1={g1}"
        );
    }
}
