//! Layer-aware placement across allowed devices.
//!
//! Produces an `Allocation` (per-device byte reservation) and
//! `CommandArgs` (llama.cpp CLI flags derived from the packing).

use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet},
};

use smol_str::SmolStr;

use crate::{
    allocator::AllocationTable,
    config::{DeviceSlot, OffloadMode, PlacementPolicy, ServiceConfig, SplitMode},
    devices::{Allocation, DeviceId, DeviceSnapshot},
    estimator::{Estimate, ExpertKind, ExpertTensor},
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
    /// `--tensor-split A,B,...`. In layer mode these are per-GPU layer
    /// counts; in a sharded (tensor/row) mode they are equal proportions
    /// (one `1` per spanned GPU).
    pub tensor_split: Option<Vec<u32>>,
    /// `-ot <regex>=<device>` rules, rendered verbatim from
    /// `service.raw.override_tensor`.
    pub override_tensor: Vec<String>,
    /// `--split-mode {row,tensor}` when the packer used a sharded
    /// (tensor-parallel) distribution. `None` keeps llama.cpp's default
    /// (`layer`), so layer-split services emit no `--split-mode` flag and
    /// their argv is unchanged.
    pub split_mode: Option<SplitMode>,
    /// `--main-gpu N` — the CUDA-visible index (after the
    /// `CUDA_VISIBLE_DEVICES` remap) that gathers intermediate results and
    /// KV in sharded modes. Always the lowest-id spanned GPU, which
    /// `cuda_env::render`'s ascending ordering places at visible index 0.
    pub main_gpu: Option<u32>,
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
    /// A sharded (tensor/row) split's equal per-GPU share didn't fit on one
    /// of the spanned GPUs. Unlike layer split there is no CPU spill — every
    /// spanned GPU must hold its shard, so a single overflow fails the pack.
    ShardDoesNotFit { gpu_index: u32, bytes: u64 },
    /// Even after offloading every eligible expert tensor to the CPU, the
    /// bytes the packer wants to keep on the host exceed the available host
    /// RAM (minus the configured `[devices.cpu] reserved_gb`).
    CpuDoesNotFit { needed: u64, available: u64 },
    /// A manual `expert_offload = N` pins every non-offloaded layer's experts
    /// to its home GPU regardless of fit, and the experts kept on `gpu_index`
    /// overflow it. Unlike `Auto`, manual mode never spills the surplus for the
    /// operator — the fix is a larger offload count.
    ManualExpertsDoNotFit { gpu_index: u32, bytes: u64 },
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
            Self::ShardDoesNotFit { gpu_index, bytes } => {
                write!(
                    f,
                    "tensor-split shard ({bytes} bytes) does not fit on gpu:{gpu_index}"
                )
            }
            Self::CpuDoesNotFit { needed, available } => {
                write!(
                    f,
                    "host RAM offload ({needed} bytes) exceeds available CPU memory ({available} bytes)"
                )
            }
            Self::ManualExpertsDoNotFit { gpu_index, bytes } => {
                write!(
                    f,
                    "manual expert_offload keeps more expert weight on gpu:{gpu_index} than it can hold (overflow at a {bytes}-byte expert tensor); raise the expert_offload count"
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
    /// Total expert-tensor bytes the packer moved onto the CPU (MoE
    /// auto/manual offload). Zero when no experts were offloaded. Surfaced in
    /// the placement preview so the UI can show "N layers · X GiB → CPU".
    pub expert_offload_bytes: u64,
    /// Number of distinct layers with at least one expert tensor offloaded to
    /// the CPU.
    pub expert_offload_layers: u32,
}

/// A tensor/row-split distribution decided by [`Packer::distribute_sharded`].
/// [`Packer::finish`] turns it into `--split-mode`, `--main-gpu`, and an equal
/// `--tensor-split`.
#[derive(Debug)]
struct ShardedPlan {
    mode: SplitMode,
    /// Spanned GPUs in ascending id order; `gpus[0]` is the main GPU.
    gpus: Vec<u32>,
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
    // Sharded (tensor/row) split distributes every layer across all spanned
    // GPUs in parallel — a fundamentally different shape from the first-fit
    // layer walk. Taken only when the service opts in, at least two GPUs are
    // available to span, and the estimator gave a per-layer breakdown to
    // halve. Otherwise fall through to the layer path: a single-GPU "tensor
    // split" is just an ordinary placement, and a fallback-arch model (no
    // per-layer detail) can't be evenly sharded.
    if packer.svc.split_mode.is_sharded()
        && packer.allowed_gpus.len() >= 2
        && !packer.per_layer.is_empty()
    {
        packer.distribute_sharded()?;
        return Ok(packer.finish());
    }
    packer.seed_non_layer();
    packer.seed_mtp_overhead();
    if packer.expert_aware {
        // Two-phase MoE placement: pin every layer's attention + KV on a GPU,
        // then distribute the experts, offloading the surplus to CPU (or a
        // secondary GPU) and recording `-ot` rules for what moved.
        packer.place_nonexpert_layers()?;
        packer.distribute_experts()?;
    } else {
        packer.walk_layers()?;
        packer.place_fallback_weights()?;
    }
    packer.add_kv_bytes();
    packer.add_compute_buffer();
    packer.add_one_layer_fudge();
    packer.check_cpu_capacity()?;
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

    /// Set by [`Self::distribute_sharded`] for tensor/row split; drives the
    /// `--split-mode`/`--main-gpu`/equal `--tensor-split` emission in
    /// [`Self::finish`]. `None` on the layer-split path.
    sharded: Option<ShardedPlan>,

    /// MoE expert-offload policy for this service.
    offload_mode: OffloadMode,
    /// Offloadable expert tensors (the estimator's menu). Empty for non-MoE.
    expert_tensors: Vec<ExpertTensor>,
    /// `true` when the two-phase expert-aware path runs: an offload mode is
    /// enabled *and* the model carries experts. Set in [`Self::new`].
    expert_aware: bool,
    /// Per-layer expert byte totals (sum of the layer's fused expert tensors).
    expert_bytes_by_layer: BTreeMap<u32, u64>,
    /// Layer → home GPU, set by the Phase-A non-expert walk. A layer absent
    /// here either has no weight or spilled wholly to CPU.
    layer_home: BTreeMap<u32, u32>,
    /// Layers whose whole weight (including experts) spilled to CPU in Phase A;
    /// their experts are part of that lump and skipped in Phase B.
    spilled_layers: BTreeSet<u32>,
    /// Expert tensors placed somewhere other than their layer's home GPU — the
    /// set that needs explicit `-ot` rules. Pairs the tensor with its target.
    expert_assignments: Vec<(ExpertTensor, DeviceSlot)>,
    /// Total expert bytes moved to the CPU (for the placement preview).
    expert_offload_cpu_bytes: u64,
    /// Distinct layers with at least one expert offloaded to CPU.
    expert_offload_cpu_layers: BTreeSet<u32>,
}

impl<'a> Packer<'a> {
    fn new(
        estimate: &'a Estimate,
        svc: &'a ServiceConfig,
        snapshot: &'a DeviceSnapshot,
        reserved: &'a AllocationTable,
        optimistic_remaining: bool,
    ) -> Self {
        let mut allowed_gpus = allowed_gpu_list(svc, snapshot);
        // Sort by descending pledge-book headroom (total - already committed)
        // so the GPU with the fewest active reservations is tried first. Using
        // the pledge book rather than nvml_free avoids letting driver-level
        // VRAM fluctuations (which vary by CUDA init order even on a fresh
        // boot) influence which GPU becomes the primary for a new model.
        allowed_gpus.sort_by_key(|gpu| {
            let slot = DeviceSlot::Gpu(*gpu);
            let total = snapshot.total_bytes(&slot).unwrap_or(0);
            let pledged = sum_reserved(reserved, &slot, &svc.name);
            Reverse(total.saturating_sub(pledged))
        });
        let allow_cpu = matches!(
            svc.placement_policy,
            PlacementPolicy::CpuOnly | PlacementPolicy::Hybrid
        );
        let per_layer = estimate.per_layer_bytes.clone().unwrap_or_default();

        let offload_mode = svc
            .llama_cpp()
            .map(|lc| lc.expert_offload)
            .unwrap_or(OffloadMode::Off);
        let expert_tensors = estimate.expert_tensors.clone().unwrap_or_default();
        let expert_aware = offload_mode.is_enabled() && !expert_tensors.is_empty();
        let mut expert_bytes_by_layer: BTreeMap<u32, u64> = BTreeMap::new();
        for e in &expert_tensors {
            *expert_bytes_by_layer.entry(e.layer).or_default() += e.bytes;
        }

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
            sharded: None,
            offload_mode,
            expert_tensors,
            expert_aware,
            expert_bytes_by_layer,
            layer_home: BTreeMap::new(),
            spilled_layers: BTreeSet::new(),
            expert_assignments: Vec::new(),
            expert_offload_cpu_bytes: 0,
            expert_offload_cpu_layers: BTreeSet::new(),
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

    /// Reserve the MTP / NextN draft-context overhead (its KV cache plus
    /// compute buffer) as a single lump on the *last* allowed GPU. At
    /// runtime llama.cpp attaches this second context to the GPU that hosts
    /// the MTP head — the model's trailing layer — which the first-fit
    /// walker places on the GPU it fills last (the least-free in the sort,
    /// i.e. the spill target that keeps the most leftover room). Pinning the
    /// lump there both matches where it physically lands and avoids piling
    /// it onto the most-free GPU that the walker is simultaneously filling
    /// to the brim (which would overflow that card by the MTP size). Seeded
    /// *before* [`Self::walk_layers`] so the walker reserves room for it.
    /// Zero when MTP is off or the model carries no MTP head.
    fn seed_mtp_overhead(&mut self) {
        if self.estimate.mtp_bytes == 0 {
            return;
        }
        let target = match self.allowed_gpus.last() {
            Some(last_gpu) => DeviceSlot::Gpu(*last_gpu),
            None => DeviceSlot::Cpu,
        };
        *self.per_device.entry(target).or_default() += self.estimate.mtp_bytes;
    }

    /// Tensor/row split: shard the whole model across every spanned GPU in
    /// parallel rather than assigning whole layers. Each GPU pledges an equal
    /// share of the layer weights, the KV cache, the output head, and the MTP
    /// draft context, plus its own compute buffer and one-layer fudge.
    /// llama.cpp's tensor-parallel modes split those tensors across the
    /// spanned devices — empirically the main GPU carries no measurable output-
    /// head or MTP premium — so modelling them as a per-GPU share rather than a
    /// lump on `--main-gpu` keeps the pledge in line with the real footprint.
    /// Only the vision projector (the residual "other" weights, which llama.cpp
    /// keeps on the main device) and any weight bytes not in the per-layer
    /// breakdown ride the main GPU. Token embeddings ride the CPU, as on the
    /// layer path.
    ///
    /// There is no CPU spill: a share that overruns a spanned GPU's capacity
    /// is a hard [`PackError::ShardDoesNotFit`], since tensor parallelism ties
    /// every GPU's share to the same proportions and can't offload the
    /// remainder.
    fn distribute_sharded(&mut self) -> Result<(), PackError> {
        let mut gpus = self.allowed_gpus.clone();
        gpus.sort_unstable();
        let main = gpus[0];
        let n = gpus.len() as u64;

        let n_layers = self.per_layer.len() as u64;
        let per_layer_sum: u64 = self.per_layer.iter().sum();
        let per_layer_avg = per_layer_sum / n_layers;
        let non_layer = &self.estimate.non_layer;
        // The vision projector ("other") stays on the main GPU; the output head
        // is sharded across all of them (see below).
        let main_only = non_layer.other_bytes;
        // `weights_bytes` covers per-layer + non-layer + mmproj/anything else;
        // the leftover (vision projector, etc.) rides the main GPU.
        let remainder = self.estimate.weights_bytes.saturating_sub(
            per_layer_sum + non_layer.output_head_bytes + main_only + non_layer.token_embd_bytes,
        );
        let kv_total = self
            .estimate
            .kv_per_token
            .saturating_mul(self.estimate.context as u64);
        let compute = self.estimate.compute_buffer_mb as u64 * 1024 * 1024;
        let fudge = ONE_LAYER_FUDGE_MULTIPLIER * (per_layer_avg + kv_total / n_layers);

        let weights_per_gpu = per_layer_sum / n;
        let kv_per_gpu = kv_total / n;
        // The output head and the MTP draft context are tensor-parallel
        // sharded, so each GPU carries an equal slice rather than the main GPU
        // owning the whole thing.
        let sharded_non_layer = (non_layer.output_head_bytes + self.estimate.mtp_bytes) / n;

        if non_layer.token_embd_bytes > 0 {
            *self.per_device.entry(DeviceSlot::Cpu).or_default() += non_layer.token_embd_bytes;
        }

        for &gpu in &gpus {
            let mut bytes = weights_per_gpu + kv_per_gpu + sharded_non_layer + compute + fudge;
            if gpu == main {
                bytes += main_only + remainder;
            }
            if bytes > self.gpu_available(gpu) {
                return Err(PackError::ShardDoesNotFit {
                    gpu_index: gpu,
                    bytes,
                });
            }
            *self.per_device.entry(DeviceSlot::Gpu(gpu)).or_default() += bytes;
        }

        self.sharded = Some(ShardedPlan {
            mode: self.svc.split_mode,
            gpus,
        });
        Ok(())
    }

    /// Step 2: first-fit layer walker. Pre-reserves per-GPU headroom for
    /// steps 3-5 so we don't fill to the brim and then overflow. Returns
    /// `PackError` if a layer's bytes don't fit on any allowed GPU and CPU
    /// spill is disabled.
    ///
    /// KV cost is folded into the per-layer fit check (`layer_bytes +
    /// kv_per_layer`) so KV headroom accumulates alongside layers rather than
    /// being validated in a separate post-walk pass. This lets large
    /// long-context models span GPUs correctly while keeping small models on
    /// the single least-busy GPU.
    fn walk_layers(&mut self) -> Result<(), PackError> {
        self.initialise_gpu_remaining();

        let n_layers = self.per_layer.len() as u64;
        let kv_total = self
            .estimate
            .kv_per_token
            .saturating_mul(self.estimate.context as u64);
        let kv_per_layer = kv_total.checked_div(n_layers).unwrap_or(0);

        for (idx, bytes) in self.per_layer.iter().copied().enumerate() {
            if bytes == 0 {
                continue;
            }
            let layer_cost = bytes.saturating_add(kv_per_layer);
            // First-fit on the sorted (most-free-first) GPU list: fills the
            // least-busy GPU before spilling to the next. Small models stay on
            // one GPU; models that genuinely span multiple GPUs still pack
            // correctly because the KV cost is folded into the fit check.
            let placed = self
                .allowed_gpus
                .iter()
                .copied()
                .find(|gpu| self.gpu_remaining.get(gpu).copied().unwrap_or(0) >= layer_cost);
            match placed {
                Some(gpu) => {
                    *self.gpu_remaining.entry(gpu).or_default() -= layer_cost;
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
    /// end up distributed: compute buffer + one-layer fudge. The per-layer KV
    /// *of placed layers* is reserved incrementally during the walk (folded
    /// into [`Self::walk_layers`]'s `layer_cost`); the headroom reserved here
    /// must additionally cover the *fudge* layer that [`Self::add_one_layer_fudge`]
    /// adds post-walk — and that fudge is `per_layer_avg + per_layer_kv`, so
    /// both terms have to be reserved here. Reserving only the weight term let
    /// a GPU that the walker fills to the brim overshoot its capacity by one
    /// layer's KV (≈ the live qwen3.6-27b "insufficient_vram on gpu:0" by
    /// ~one `per_layer_kv`); including `per_layer_kv` makes the post-walk total
    /// land at or below `available`.
    fn initialise_gpu_remaining(&mut self) {
        let n_layers = self.per_layer.len() as u64;
        let per_layer_avg = self.effective_layer_avg();
        let kv_total = self
            .estimate
            .kv_per_token
            .saturating_mul(self.estimate.context as u64);
        let per_layer_kv = kv_total.checked_div(n_layers).unwrap_or(0);
        let compute_headroom = self.estimate.compute_buffer_mb as u64 * 1024 * 1024;
        let gpu_headroom_each =
            compute_headroom + (per_layer_avg + per_layer_kv) * ONE_LAYER_FUDGE_MULTIPLIER;

        for gpu in &self.allowed_gpus {
            let slot = DeviceSlot::Gpu(*gpu);
            let available = self.gpu_available(*gpu);
            let raw = available.saturating_sub(*self.per_device.get(&slot).unwrap_or(&0));
            self.gpu_remaining
                .insert(*gpu, raw.saturating_sub(gpu_headroom_each));
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
    fn gpu_available(&self, gpu: u32) -> u64 {
        let slot = DeviceSlot::Gpu(gpu);
        let free = self.snapshot.free_bytes(&slot).unwrap_or(0);
        let total = self.snapshot.total_bytes(&slot).unwrap_or(free);
        let reserved_here = sum_reserved(self.reserved, &slot, &self.svc.name);
        let via_pledge = total.saturating_sub(reserved_here);
        let avail = if self.optimistic_remaining {
            via_pledge
        } else {
            free.min(via_pledge)
        };
        // Keep the configured headroom (global `[devices]` reserve + this
        // service's `gpu_headroom_mb`) free on the card.
        avail.saturating_sub(gpu_reserve_bytes(self.svc, gpu))
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
        let per_layer_avg = self.effective_layer_avg();
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

    /// The per-layer weight average used for headroom/fudge reservations. For
    /// the expert-aware path only the non-expert part of a layer is pinned to a
    /// GPU as a unit (experts are placed individually), so the slop estimate
    /// should reflect the non-expert size — using the full (expert-inflated)
    /// average would reserve enormous, mostly-wasted headroom.
    fn effective_layer_avg(&self) -> u64 {
        let n = self.per_layer.len() as u64;
        if n == 0 {
            return 0;
        }
        let total: u64 = self.per_layer.iter().sum();
        let total = if self.expert_aware {
            total.saturating_sub(self.expert_tensors.iter().map(|e| e.bytes).sum())
        } else {
            total
        };
        total / n
    }

    /// Phase A of the expert-aware path: place each layer's *non-expert* weight
    /// (attention, norms) plus its KV share on a GPU via first-fit, recording
    /// the layer's home GPU. A layer whose non-expert part doesn't fit any GPU
    /// spills whole (experts included) to CPU, exactly like the non-MoE hybrid
    /// path. Experts are left for [`Self::distribute_experts`].
    fn place_nonexpert_layers(&mut self) -> Result<(), PackError> {
        self.initialise_gpu_remaining();
        let n_layers = self.per_layer.len() as u64;
        let kv_total = self
            .estimate
            .kv_per_token
            .saturating_mul(self.estimate.context as u64);
        let kv_per_layer = kv_total.checked_div(n_layers).unwrap_or(0);

        for (idx, full_bytes) in self.per_layer.iter().copied().enumerate() {
            if full_bytes == 0 {
                continue;
            }
            let idx = idx as u32;
            let exp_bytes = self.expert_bytes_by_layer.get(&idx).copied().unwrap_or(0);
            let nonexp = full_bytes.saturating_sub(exp_bytes);
            let layer_cost = nonexp.saturating_add(kv_per_layer);
            let placed = self
                .allowed_gpus
                .iter()
                .copied()
                .find(|gpu| self.gpu_remaining.get(gpu).copied().unwrap_or(0) >= layer_cost);
            match placed {
                Some(gpu) => {
                    *self.gpu_remaining.entry(gpu).or_default() -= layer_cost;
                    *self.per_device.entry(DeviceSlot::Gpu(gpu)).or_default() += nonexp;
                    *self.layers_per_gpu.entry(gpu).or_default() += 1;
                    self.layer_home.insert(idx, gpu);
                }
                None if self.allow_cpu => {
                    *self.per_device.entry(DeviceSlot::Cpu).or_default() += full_bytes;
                    self.layers_on_cpu += 1;
                    self.spilled_layers.insert(idx);
                }
                None => {
                    return Err(PackError::LayerDoesNotFit {
                        layer_index: idx,
                        bytes: nonexp,
                    });
                }
            }
        }
        Ok(())
    }

    /// Phase B of the expert-aware path: place each layer's fused expert tensors.
    /// `Auto` keeps an expert on its home GPU while that GPU has room, spills to
    /// the most-free other GPU next, and finally to CPU — so the experts that
    /// end up on CPU are the trailing ones (the walk is in `(layer, kind)`
    /// order), which keeps the synthesised `-ot` rules compact. `Layers(N)`
    /// sends the experts of the N tail-most expert layers to CPU and pins the
    /// rest to their home GPU regardless of fit — so an `N` too small to relieve
    /// the card overflows it, which we reject as [`PackError::ManualExpertsDoNotFit`]
    /// rather than silently over-committing the GPU. Anything placed off its
    /// home GPU is recorded for `-ot` synthesis in [`Self::finish`].
    fn distribute_experts(&mut self) -> Result<(), PackError> {
        let manual = matches!(self.offload_mode, OffloadMode::Layers(_));
        let manual_cpu_layers: BTreeSet<u32> = match self.offload_mode {
            OffloadMode::Layers(n) => {
                let mut layers: Vec<u32> = self.expert_bytes_by_layer.keys().copied().collect();
                layers.sort_unstable();
                layers.iter().rev().take(n as usize).copied().collect()
            }
            _ => BTreeSet::new(),
        };

        for e in std::mem::take(&mut self.expert_tensors) {
            if self.spilled_layers.contains(&e.layer) {
                continue;
            }
            let home = self.layer_home.get(&e.layer).copied();
            let target = self.choose_expert_target(&e, home, &manual_cpu_layers);
            // `Auto` only ever targets a GPU that has room; `Layers(N)`
            // force-pins to the home GPU, so it is the one path that can overflow.
            if manual
                && let DeviceSlot::Gpu(g) = target
                && self.gpu_remaining.get(&g).copied().unwrap_or(0) < e.bytes
            {
                return Err(PackError::ManualExpertsDoNotFit {
                    gpu_index: g,
                    bytes: e.bytes,
                });
            }
            self.place_expert(e, target, home);
        }
        Ok(())
    }

    /// Pick where one expert tensor lands. See [`Self::distribute_experts`].
    fn choose_expert_target(
        &self,
        e: &ExpertTensor,
        home: Option<u32>,
        manual_cpu_layers: &BTreeSet<u32>,
    ) -> DeviceSlot {
        if matches!(self.offload_mode, OffloadMode::Layers(_)) {
            return if manual_cpu_layers.contains(&e.layer) {
                DeviceSlot::Cpu
            } else {
                home.map(DeviceSlot::Gpu).unwrap_or(DeviceSlot::Cpu)
            };
        }
        // Auto: home GPU first, then the most-free other GPU, then CPU.
        if let Some(g) = home
            && self.gpu_remaining.get(&g).copied().unwrap_or(0) >= e.bytes
        {
            return DeviceSlot::Gpu(g);
        }
        let mut others: Vec<u32> = self
            .allowed_gpus
            .iter()
            .copied()
            .filter(|g| Some(*g) != home)
            .collect();
        others.sort_by_key(|g| Reverse(self.gpu_remaining.get(g).copied().unwrap_or(0)));
        for g in others {
            if self.gpu_remaining.get(&g).copied().unwrap_or(0) >= e.bytes {
                return DeviceSlot::Gpu(g);
            }
        }
        DeviceSlot::Cpu
    }

    /// Commit an expert tensor to its chosen device, updating remaining GPU
    /// capacity and recording an `-ot` assignment when it lands off its home
    /// GPU (the only case llama.cpp needs told about).
    fn place_expert(&mut self, e: ExpertTensor, target: DeviceSlot, home: Option<u32>) {
        *self.per_device.entry(target.clone()).or_default() += e.bytes;
        if let DeviceSlot::Gpu(g) = target {
            let rem = self.gpu_remaining.entry(g).or_default();
            *rem = rem.saturating_sub(e.bytes);
        }
        let is_home = matches!((&target, home), (DeviceSlot::Gpu(g), Some(h)) if *g == h);
        if is_home {
            return;
        }
        if target == DeviceSlot::Cpu {
            self.expert_offload_cpu_bytes += e.bytes;
            self.expert_offload_cpu_layers.insert(e.layer);
        }
        self.expert_assignments.push((e, target));
    }

    /// Reject the pack if the bytes the packer wants to keep on the host exceed
    /// the available host RAM (minus the configured `[devices.cpu] reserved_gb`).
    /// Skipped when the snapshot carries no CPU info.
    fn check_cpu_capacity(&self) -> Result<(), PackError> {
        let needed = self.per_device.get(&DeviceSlot::Cpu).copied().unwrap_or(0);
        if needed == 0 {
            return Ok(());
        }
        let Some(free) = self.snapshot.free_bytes(&DeviceSlot::Cpu) else {
            return Ok(());
        };
        let available = free.saturating_sub(self.svc.reserves.cpu_bytes);
        if needed > available {
            return Err(PackError::CpuDoesNotFit { needed, available });
        }
        Ok(())
    }

    /// Step 6: materialise the final `Packed` — derive -ngl, --tensor-split,
    /// -ot, and convert the per_device map into an `Allocation`.
    fn finish(self) -> Packed {
        // Sharded (tensor/row) split: every layer is offloaded and divided
        // across all spanned GPUs by equal proportions, so emit `-ngl 999`,
        // a `1`-per-GPU `--tensor-split`, the `--split-mode`, and `--main-gpu`
        // (visible index 0 — the lowest-id GPU, which cuda_env renders first).
        let (ngl, tensor_split, split_mode, main_gpu) = if let Some(plan) = &self.sharded {
            (
                Some(NGL_OFFLOAD_ALL),
                Some(vec![1u32; plan.gpus.len()]),
                Some(plan.mode),
                Some(0),
            )
        } else {
            let total_on_gpus: u32 = self.layers_per_gpu.values().sum();
            let ngl = if self.allowed_gpus.is_empty() {
                Some(NGL_CPU_ONLY)
            } else if self.fallback_on_gpu {
                Some(NGL_OFFLOAD_ALL)
            } else {
                Some(total_on_gpus)
            };

            let tensor_split = if self.allowed_gpus.len() > 1 && total_on_gpus > 0 {
                // Ratios in CUDA_VISIBLE_DEVICES-remapped order: must be in
                // ascending GPU-id order to match CUDA device numbering,
                // regardless of the placement sort order.
                let mut gpus_by_id = self.allowed_gpus.clone();
                gpus_by_id.sort_unstable();
                Some(
                    gpus_by_id
                        .iter()
                        .map(|g| self.layers_per_gpu.get(g).copied().unwrap_or(0))
                        .collect(),
                )
            } else {
                None
            };
            (ngl, tensor_split, None, None)
        };

        // Operator-declared rules first, then the packer's synthesised expert
        // offload rules. The `-ot` device index for a cross-GPU move is the
        // target GPU's rank among the GPU ids in the final allocation, matching
        // `cuda_env::render`'s ascending CUDA_VISIBLE_DEVICES ordering.
        let mut override_tensor = self
            .svc
            .llama_cpp()
            .map(|lc| lc.override_tensor.clone())
            .unwrap_or_default();
        let mut gpu_ids: Vec<u32> = self
            .per_device
            .keys()
            .filter_map(|s| match s {
                DeviceSlot::Gpu(g) => Some(*g),
                DeviceSlot::Cpu => None,
            })
            .collect();
        gpu_ids.sort_unstable();
        override_tensor.extend(synth_expert_ot_rules(&self.expert_assignments, &gpu_ids));

        let expert_offload_bytes = self.expert_offload_cpu_bytes;
        let expert_offload_layers = self.expert_offload_cpu_layers.len() as u32;

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
                split_mode,
                main_gpu,
            },
            expert_offload_bytes,
            expert_offload_layers,
        }
    }
}

/// Build compact `-ot <regex>=<device>` rules from the packer's off-home expert
/// placements. Groups by target device and identical kind-set so a tail of
/// fully-offloaded layers collapses to a single
/// `blk\.(16|17|18)\.ffn_(gate|up|down)_exps\.=CPU` rule. `gpu_ids` is the
/// ascending allocation GPU-id list used to map a physical id to its CUDA
/// visible index.
fn synth_expert_ot_rules(
    assignments: &[(ExpertTensor, DeviceSlot)],
    gpu_ids: &[u32],
) -> Vec<String> {
    // device token -> (layer -> set of offloaded kinds).
    let mut by_device: BTreeMap<String, BTreeMap<u32, BTreeSet<ExpertKind>>> = BTreeMap::new();
    for (e, slot) in assignments {
        let token = match slot {
            DeviceSlot::Cpu => "CPU".to_string(),
            DeviceSlot::Gpu(g) => {
                let idx = gpu_ids.iter().position(|x| x == g).unwrap_or(0);
                format!("CUDA{idx}")
            }
        };
        by_device
            .entry(token)
            .or_default()
            .entry(e.layer)
            .or_default()
            .insert(e.kind);
    }

    let mut rules = Vec::new();
    for (token, layers) in by_device {
        // Group layers that share an identical kind-set into one rule.
        let mut by_kinds: BTreeMap<Vec<ExpertKind>, Vec<u32>> = BTreeMap::new();
        for (layer, kinds) in layers {
            by_kinds
                .entry(kinds.into_iter().collect())
                .or_default()
                .push(layer);
        }
        for (kinds, mut group) in by_kinds {
            group.sort_unstable();
            let layer_alt = group
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join("|");
            let kind_alt = kinds
                .iter()
                .map(|k| k.tensor_token())
                .collect::<Vec<_>>()
                .join("|");
            rules.push(format!(
                r"blk\.({layer_alt})\.ffn_({kind_alt})_exps\.={token}"
            ));
        }
    }
    rules
}

/// VRAM-aware GPU pick for a command-template service.
///
/// Walks `svc`'s allowed GPU list (filtered by `gpu_allow` if set) and
/// returns the GPU with the most available capacity that still satisfies
/// `min_mb`. When `prefer_mb` is `Some`, GPUs that meet that headroom
/// target take priority; only if no candidate hits `prefer_mb` do we fall
/// back to "best of those that meet `min_mb`". This lets dynamic services
/// (`min_mb` floor, `max_mb` growth ceiling) favour a GPU with room to grow.
///
/// `optimistic_remaining` mirrors [`pack_optimistic`]: when `false` the
/// availability view is `min(nvml_free, total - pledged)`; when `true` we
/// trust the pledge book exclusively (`total - pledged`). Eviction-retry
/// passes `true` because nvml hasn't yet caught up to the in-flight drains.
///
/// Returns `None` when no allowed GPU can host `min_mb` — the caller should
/// surface this as a [`PackError::WeightsDoNotFit`] so the supervisor can
/// run its eviction-retry loop.
pub fn pick_command_gpu(
    svc: &ServiceConfig,
    snapshot: &DeviceSnapshot,
    reserved: &AllocationTable,
    min_mb: u64,
    prefer_mb: Option<u64>,
    optimistic_remaining: bool,
) -> Option<u32> {
    let allowed = allowed_gpu_list(svc, snapshot);
    if allowed.is_empty() {
        return None;
    }
    let need_min_bytes = min_mb.saturating_mul(1024 * 1024);
    let prefer_bytes = prefer_mb.map(|m| m.saturating_mul(1024 * 1024));

    let mut candidates: Vec<(u32, u64)> = allowed
        .into_iter()
        .map(|gpu| {
            let slot = DeviceSlot::Gpu(gpu);
            let free = snapshot.free_bytes(&slot).unwrap_or(0);
            let total = snapshot.total_bytes(&slot).unwrap_or(free);
            let pledged = sum_reserved(reserved, &slot, &svc.name);
            let via_pledge = total.saturating_sub(pledged);
            let available = if optimistic_remaining {
                via_pledge
            } else {
                free.min(via_pledge)
            };
            let available = available.saturating_sub(gpu_reserve_bytes(svc, gpu));
            (gpu, available)
        })
        .filter(|(_, available)| *available >= need_min_bytes)
        .collect();

    if candidates.is_empty() {
        return None;
    }

    if let Some(target) = prefer_bytes
        && candidates.iter().any(|(_, a)| *a >= target)
    {
        candidates.retain(|(_, a)| *a >= target);
    }
    // Sort: most-available first, ties broken by ascending GPU id for determinism.
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    Some(candidates[0].0)
}

/// Capacity check for a command-template service that pinned the
/// reservation across multiple devices via `placement_override`.
///
/// Each `(slot, mib)` entry in the override is checked against the
/// device's available bytes using the same `min(nvml_free, total
/// pledged)` view as [`pick_command_gpu`]. CPU entries are skipped
/// (we don't model CPU capacity). Returns `Ok` when every entry fits;
/// otherwise the first slot that overflows is reported as
/// [`PackError::WeightsDoNotFit`] so the supervisor's eviction-retry
/// loop can engage.
pub fn check_command_placement_override(
    svc: &ServiceConfig,
    snapshot: &DeviceSnapshot,
    reserved: &AllocationTable,
    optimistic_remaining: bool,
) -> Result<(), PackError> {
    for (slot, mib) in &svc.placement_override {
        let DeviceSlot::Gpu(gid) = slot else { continue };
        let need_bytes = mib.saturating_mul(1024 * 1024);
        let free = snapshot.free_bytes(slot).unwrap_or(0);
        let total = snapshot.total_bytes(slot).unwrap_or(free);
        let pledged = sum_reserved(reserved, slot, &svc.name);
        let via_pledge = total.saturating_sub(pledged);
        let available = if optimistic_remaining {
            via_pledge
        } else {
            free.min(via_pledge)
        };
        let available = available.saturating_sub(gpu_reserve_bytes(svc, *gid));
        if available < need_bytes {
            return Err(PackError::WeightsDoNotFit);
        }
    }
    Ok(())
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

/// VRAM (bytes) this service keeps free on `gpu`: the global `[devices]`
/// reserve for that GPU (per-GPU override, else default) plus the service's
/// own `gpu_headroom_mb`.
fn gpu_reserve_bytes(svc: &ServiceConfig, gpu: u32) -> u64 {
    let r = &svc.reserves;
    let mb = r
        .per_gpu_mb
        .get(&gpu)
        .copied()
        .unwrap_or(r.default_gpu_mb)
        .saturating_add(svc.gpu_headroom_mb);
    mb.saturating_mul(1024 * 1024)
}

#[cfg(test)]
mod tests {
    use smol_str::SmolStr;

    use super::*;
    use crate::{
        config::{
            OffloadMode,
            validate::test_fixtures::{expect_llama_cpp, minimal_service},
        },
        devices::{CpuSnapshot, GpuSnapshot},
        estimator::{ExpertKind, ExpertTensor, NonLayer},
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
            mtp_bytes: 0,
            per_layer_bytes: Some(vec![per_layer_mb * 1024 * 1024; n_layers as usize]),
            attention_layers: None,
            non_layer: NonLayer::default(),
            override_tensor_bytes: BTreeMap::new(),
            expert_layers: Vec::new(),
            expert_tensors: None,
            context: 4096,
            architecture: SmolStr::new("qwen3"),
        }
    }

    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * 1024 * 1024;

    /// A Hybrid llama-cpp service with `expert_offload` set, for the
    /// expert-aware packer path. `placement_override` is cleared so `pack`
    /// takes the estimator path.
    fn moe_svc(offload: OffloadMode) -> ServiceConfig {
        let mut svc = minimal_service("moe");
        svc.placement_override = BTreeMap::new();
        svc.placement_policy = PlacementPolicy::Hybrid;
        expect_llama_cpp(&mut svc).expert_offload = offload;
        svc
    }

    /// A MoE estimate: every layer carries `nonexp_mb` of non-expert weight
    /// plus three fused expert tensors of `exp_mb` each. `per_layer_bytes`
    /// holds the full cost; `expert_tensors` itemises the experts (already
    /// counted in the per-layer total).
    fn moe_estimate(n_layers: u32, nonexp_mb: u64, exp_mb: u64) -> Estimate {
        let layer_total = (nonexp_mb + 3 * exp_mb) * MIB;
        let mut per_layer = Vec::new();
        let mut experts = Vec::new();
        for layer in 0..n_layers {
            per_layer.push(layer_total);
            for kind in [ExpertKind::Gate, ExpertKind::Up, ExpertKind::Down] {
                experts.push(ExpertTensor {
                    layer,
                    kind,
                    bytes: exp_mb * MIB,
                });
            }
        }
        Estimate {
            weights_bytes: layer_total * n_layers as u64,
            kv_per_token: 0,
            compute_buffer_mb: 400,
            mtp_bytes: 0,
            per_layer_bytes: Some(per_layer),
            attention_layers: None,
            non_layer: NonLayer::default(),
            override_tensor_bytes: BTreeMap::new(),
            expert_layers: (0..n_layers).collect(),
            expert_tensors: Some(experts),
            context: 4096,
            architecture: SmolStr::new("qwen3moe"),
        }
    }

    fn cpu_bytes(p: &Packed) -> u64 {
        p.allocation.bytes.get(&DeviceId::Cpu).copied().unwrap_or(0)
    }

    /// Auto offload: a model whose full layers overflow the card but whose
    /// non-expert parts fit keeps every layer on the GPU (`-ngl == n_layers`)
    /// and spills the surplus experts to CPU with a synthesised `-ot` rule.
    #[test]
    fn expert_offload_auto_spills_surplus_experts_to_cpu() {
        // 10 layers: 100 MiB non-expert + 900 MiB experts each (10 GiB total),
        // non-expert only ≈ 1 GiB. A 4 GiB card holds all attention but not all
        // experts.
        let e = moe_estimate(10, 100, 300);
        let snap = snapshot(&[4]);
        let alloc = AllocationTable::new();
        let packed = pack(&e, &moe_svc(OffloadMode::Auto), &snap, &alloc).unwrap();

        assert_eq!(packed.args.ngl, Some(10), "every layer stays on the GPU");
        assert!(cpu_bytes(&packed) > 0, "surplus experts land on the CPU");
        assert!(packed.expert_offload_bytes > 0);
        assert!(packed.expert_offload_layers > 0);
        assert!(
            packed
                .args
                .override_tensor
                .iter()
                .any(|r| r.contains("_exps") && r.ends_with("=CPU")),
            "a synthesised expert -ot rule is emitted, got {:?}",
            packed.args.override_tensor
        );
        // The GPU pledge must stay within the card.
        let gpu = packed
            .allocation
            .bytes
            .get(&DeviceId::Gpu(0))
            .copied()
            .unwrap_or(0);
        assert!(gpu <= 24 * GIB);
    }

    /// When the whole model fits, the expert-aware path offloads nothing and
    /// emits no synthesised rule — identical shape to a non-MoE fit.
    #[test]
    fn expert_offload_auto_no_offload_when_everything_fits() {
        let e = moe_estimate(10, 100, 100); // 400 MiB/layer, 4 GiB total
        let snap = snapshot(&[24]);
        let alloc = AllocationTable::new();
        let packed = pack(&e, &moe_svc(OffloadMode::Auto), &snap, &alloc).unwrap();

        assert_eq!(packed.args.ngl, Some(10));
        assert_eq!(packed.expert_offload_bytes, 0);
        assert_eq!(packed.expert_offload_layers, 0);
        assert!(cpu_bytes(&packed) == 0);
        assert!(packed.args.override_tensor.is_empty());
    }

    /// `expert_offload = N` offloads exactly the N tail-most expert layers to
    /// CPU even on a roomy card, and the synthesised rule names those layers.
    #[test]
    fn expert_offload_layers_n_offloads_tail_layers() {
        let e = moe_estimate(10, 100, 100); // fits easily
        let snap = snapshot(&[24]);
        let alloc = AllocationTable::new();
        let packed = pack(&e, &moe_svc(OffloadMode::Layers(3)), &snap, &alloc).unwrap();

        assert_eq!(packed.args.ngl, Some(10), "attention stays on GPU");
        assert_eq!(packed.expert_offload_layers, 3);
        // 3 layers × 3 experts × 100 MiB.
        assert_eq!(packed.expert_offload_bytes, 9 * 100 * MIB);
        assert_eq!(
            packed.args.override_tensor,
            vec![r"blk\.(7|8|9)\.ffn_(gate|up|down)_exps\.=CPU".to_string()],
            "offloads exactly the 3 tail expert layers"
        );
    }

    /// A manual `expert_offload = N` too small to relieve the card pins the
    /// remaining experts to their home GPU regardless of fit. That overflow is
    /// rejected with `ManualExpertsDoNotFit` rather than silently
    /// over-committing the GPU into a spawn-time OOM.
    #[test]
    fn expert_offload_manual_rejects_when_gpu_overflows() {
        // 10 layers, 100 MiB attn + 900 MiB experts each (10 GiB). Offloading
        // only the 2 tail layers leaves ~7 GiB of experts pinned to a 4 GiB card.
        let e = moe_estimate(10, 100, 300);
        let snap = snapshot(&[4]);
        let alloc = AllocationTable::new();
        let err = pack(&e, &moe_svc(OffloadMode::Layers(2)), &snap, &alloc)
            .expect_err("under-sized manual offload must not over-commit the GPU");
        assert!(
            matches!(err, PackError::ManualExpertsDoNotFit { gpu_index: 0, .. }),
            "expected ManualExpertsDoNotFit on gpu:0, got {err:?}"
        );
    }

    /// Auto offload prefers a second GPU over the CPU: when GPU 0 fills with
    /// the attention layers and some experts, the surplus experts move to GPU 1
    /// (a `=CUDA1` rule) rather than the host, so nothing lands on CPU.
    #[test]
    fn expert_offload_auto_prefers_second_gpu() {
        // 20 layers, 100 MiB attn + 900 MiB experts. GPU 0 (4 GiB) holds all
        // attention and a little expert weight; GPU 1 (24 GiB) absorbs the rest.
        let e = moe_estimate(20, 100, 300);
        let snap = snapshot(&[4, 24]);
        let alloc = AllocationTable::new();
        let packed = pack(&e, &moe_svc(OffloadMode::Auto), &snap, &alloc).unwrap();

        assert_eq!(packed.args.ngl, Some(20));
        assert_eq!(cpu_bytes(&packed), 0, "experts prefer GPU 1 over the CPU");
        assert_eq!(
            packed.expert_offload_bytes, 0,
            "CPU offload metric counts host bytes only"
        );
        assert!(
            packed
                .args
                .override_tensor
                .iter()
                .any(|r| r.contains("=CUDA1")),
            "a cross-GPU expert rule targets the second card, got {:?}",
            packed.args.override_tensor
        );
        assert!(
            packed
                .allocation
                .bytes
                .get(&DeviceId::Gpu(1))
                .copied()
                .unwrap_or(0)
                > 0
        );
    }

    /// Offloading more experts than host RAM can hold (minus the CPU reserve)
    /// is rejected with `CpuDoesNotFit` rather than silently over-committing.
    #[test]
    fn expert_offload_rejects_when_cpu_is_full() {
        let e = moe_estimate(10, 100, 900); // ~27 GiB of experts
        // 1 GiB card forces almost everything to CPU, but the host has only
        // 2 GiB available.
        let mut snap = snapshot(&[1]);
        snap.cpu = Some(CpuSnapshot {
            total_bytes: 4 * GIB,
            available_bytes: 2 * GIB,
        });
        let alloc = AllocationTable::new();
        let err = pack(&e, &moe_svc(OffloadMode::Auto), &snap, &alloc)
            .expect_err("CPU offload must not exceed host RAM");
        assert!(
            matches!(err, PackError::CpuDoesNotFit { .. }),
            "expected CpuDoesNotFit, got {err:?}"
        );
    }

    /// Per-service `gpu_headroom_mb` shrinks usable VRAM, so a model that packs
    /// with no headroom offloads strictly more once headroom is reserved.
    #[test]
    fn expert_offload_headroom_forces_more_offload() {
        let e = moe_estimate(12, 100, 300);
        let snap = snapshot(&[6]);
        let alloc = AllocationTable::new();

        let tight = pack(&e, &moe_svc(OffloadMode::Auto), &snap, &alloc).unwrap();
        let mut svc = moe_svc(OffloadMode::Auto);
        svc.gpu_headroom_mb = 2048;
        let with_headroom = pack(&e, &svc, &snap, &alloc).unwrap();

        assert!(
            with_headroom.expert_offload_bytes >= tight.expert_offload_bytes,
            "more reserved headroom must not offload less: {} < {}",
            with_headroom.expert_offload_bytes,
            tight.expert_offload_bytes
        );
        assert!(with_headroom.expert_offload_bytes > tight.expert_offload_bytes);
    }

    /// The `-ot` synthesiser collapses a tail of fully-offloaded layers into a
    /// single grouped rule with layer and kind alternations.
    #[test]
    fn synth_expert_ot_rules_groups_layers_and_kinds() {
        let assignments: Vec<(ExpertTensor, DeviceSlot)> = [18u32, 16, 17]
            .into_iter()
            .flat_map(|layer| {
                [ExpertKind::Gate, ExpertKind::Up, ExpertKind::Down]
                    .into_iter()
                    .map(move |kind| {
                        (
                            ExpertTensor {
                                layer,
                                kind,
                                bytes: MIB,
                            },
                            DeviceSlot::Cpu,
                        )
                    })
            })
            .collect();
        let rules = synth_expert_ot_rules(&assignments, &[]);
        assert_eq!(
            rules,
            vec![r"blk\.(16|17|18)\.ffn_(gate|up|down)_exps\.=CPU".to_string()]
        );
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

    /// Small models (fewer bytes than a single GPU's headroom-adjusted
    /// capacity) should land entirely on one GPU rather than being splayed
    /// across devices. With two equal-free GPUs and two layers that both fit
    /// on GPU 0, first-fit should produce [2, 0], not [1, 1].
    #[test]
    fn first_fit_packs_small_model_onto_one_gpu() {
        let e = trivial_estimate(2, 1024); // 2 layers, 1 GiB each
        let snap = snapshot(&[20, 20]); // 20 GB free on each of two GPUs
        let alloc = AllocationTable::new();
        let packed = pack(&e, &svc(PlacementPolicy::GpuOnly, None), &snap, &alloc).unwrap();
        // Both layers fit on GPU 0 — no reason to split across GPUs.
        assert_eq!(packed.args.ngl, Some(2));
        // tensor_split is emitted but GPU 1 carries zero layers.
        let split = packed.args.tensor_split.as_ref().unwrap();
        assert_eq!(split.iter().sum::<u32>(), 2);
        assert_eq!(
            split[0], 2,
            "all layers should be on GPU 0 (first in sorted order for equal capacity)"
        );
        assert_eq!(split[1], 0);
    }

    /// When GPU 0 already has reservations from other services, GPU 1 has
    /// more pledge-book headroom and is sorted first. A new model that fits
    /// on one GPU goes entirely to GPU 1, regardless of how nvml_free
    /// compares between the two devices.
    #[test]
    fn first_fit_targets_least_pledged_gpu() {
        let e = trivial_estimate(4, 1024); // 4 × 1 GiB layers
        // Free bytes differ from the pledge picture — the sort must ignore them.
        let snap = snapshot(&[10, 16]);
        // Another service has 6 GiB committed on GPU 0; GPU 1 is unencumbered.
        let mut reserved = AllocationTable::new();
        let mut other = BTreeMap::new();
        other.insert(DeviceSlot::Gpu(0), 6 * 1024u64); // MB
        reserved.insert(SmolStr::new("other"), other);
        let packed = pack(&e, &svc(PlacementPolicy::GpuOnly, None), &snap, &reserved).unwrap();
        let split = packed.args.tensor_split.as_ref().unwrap();
        assert_eq!(split.iter().sum::<u32>(), 4);
        // GPU 1 (more pledge headroom) is sorted first; all layers land there.
        assert!(
            split[1] >= split[0],
            "first-fit should place all layers on GPU 1 (less pledged); got {split:?}"
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

    /// Regression for the 256K-context Gemma 4 31B repro: folding KV cost
    /// into the per-layer fit check during the walk (rather than a post-walk
    /// validation step) must still allow this model to pack across two GPUs.
    /// The model (60 layers × 296 MB ≈ 17.8 GB weights + 11.85 GB KV) does
    /// not fit on a single 24 GB GPU once KV is included, so the walk spills
    /// part of it to the second GPU.
    #[test]
    fn long_context_moe_packs_across_two_gpus() {
        // Gemma 4 31B numbers from the live failure log: 60 layers at ~296 MB
        // avg (≈17.8 GB total), kv_per_token = 45220, 256 K context, compute
        // buffer 3792 MB.
        let per_layer_bytes: Vec<u64> = (0..60).map(|_| 296 * 1024 * 1024).collect();
        let weights_bytes: u64 = per_layer_bytes.iter().sum();
        let e = Estimate {
            weights_bytes,
            kv_per_token: 45220,
            compute_buffer_mb: 3792,
            mtp_bytes: 0,
            per_layer_bytes: Some(per_layer_bytes),
            attention_layers: None,
            non_layer: NonLayer::default(),
            override_tensor_bytes: BTreeMap::new(),
            expert_layers: Vec::new(),
            expert_tensors: None,
            context: 262_144,
            architecture: SmolStr::new("gemma4"),
        };
        // 2×24 GB 3090s, fully free, empty pledge book.
        let snap = snapshot(&[24, 24]);
        let alloc = AllocationTable::new();
        let packed = pack(&e, &svc(PlacementPolicy::GpuOnly, None), &snap, &alloc)
            .expect("Gemma 4 31B at 256K context must pack on 2×24 GB");
        let split = packed.args.tensor_split.as_ref().expect("two-GPU split");
        // Total layers must add up.
        assert_eq!(split.iter().sum::<u32>(), 60);
        // KV cost folds layers onto the first GPU until full; the remainder
        // spills to the second. Both must carry layers.
        assert!(
            split[0] > 0 && split[1] > 0,
            "both GPUs must carry layers for this model; got {split:?}"
        );
    }

    /// A model whose KV-inclusive layer cost overflows even a single GPU must
    /// be rejected. With KV folded into the per-layer walk cost, the walker
    /// detects the overflow directly (LayerDoesNotFit) rather than via a
    /// separate post-walk validation step.
    #[test]
    fn long_context_overflows_single_gpu() {
        // 60 layers × 200 MB = 12 GB weights. kv_total at 128K tokens with
        // kv_per_token = 120 KB ≈ 15 GB. Per-layer cost = 200 + 261 MB ≈
        // 461 MB. Only 47 layers fit on a 24 GB GPU once compute buffer (2 GB)
        // and fudge are reserved; layer 48 overflows.
        let per_layer_bytes: Vec<u64> = (0..60).map(|_| 200 * 1024 * 1024).collect();
        let weights_bytes: u64 = per_layer_bytes.iter().sum();
        let e = Estimate {
            weights_bytes,
            kv_per_token: 120_000,
            compute_buffer_mb: 2048,
            mtp_bytes: 0,
            per_layer_bytes: Some(per_layer_bytes),
            attention_layers: None,
            non_layer: NonLayer::default(),
            override_tensor_bytes: BTreeMap::new(),
            expert_layers: Vec::new(),
            expert_tensors: None,
            context: 131_072,
            architecture: SmolStr::new("qwen3"),
        };
        // Single 24 GB GPU, no spill allowed.
        let snap = snapshot(&[24]);
        let alloc = AllocationTable::new();
        let err = pack(&e, &svc(PlacementPolicy::GpuOnly, None), &snap, &alloc)
            .expect_err("KV-inclusive layer cost must overflow the single GPU");
        assert!(
            matches!(err, PackError::LayerDoesNotFit { .. }),
            "expected LayerDoesNotFit when KV pushes layer cost past GPU capacity, got {err:?}"
        );
    }

    /// `pick_command_gpu` should return the GPU with the most available
    /// capacity when several satisfy `min_mb`. Ties broken by ascending id.
    #[test]
    fn pick_command_gpu_prefers_most_free() {
        let s = svc(PlacementPolicy::GpuOnly, None);
        // GPU 0 has 6 GB free, GPU 1 has 18 GB, GPU 2 has 12 GB.
        let snap = snapshot(&[6, 18, 12]);
        let table = AllocationTable::new();
        let pick = pick_command_gpu(&s, &snap, &table, 2 * 1024, None, false);
        assert_eq!(pick, Some(1), "GPU 1 has the most free; should be picked");
    }

    /// When a `prefer_mb` headroom target is set (dynamic services'
    /// `max_mb`), `pick_command_gpu` should reject GPUs whose available
    /// capacity falls below that target if any other GPU does meet it.
    #[test]
    fn pick_command_gpu_honours_prefer_headroom() {
        let s = svc(PlacementPolicy::GpuOnly, None);
        // GPU 0 = 4 GB free (above min, below prefer), GPU 1 = 10 GB.
        let snap = snapshot(&[4, 10]);
        let table = AllocationTable::new();
        let pick = pick_command_gpu(
            &s,
            &snap,
            &table,
            2 * 1024,       // min_mb: 2 GB
            Some(8 * 1024), // prefer_mb: 8 GB
            false,
        );
        assert_eq!(pick, Some(1), "GPU 1 meets prefer_mb headroom");
    }

    /// When no GPU meets `prefer_mb`, fall back to "best of those that meet
    /// `min_mb`" rather than returning None — the pick is still better than
    /// no pick at all, and the dynamic balloon resolver will fast-kill the
    /// service if it actually overshoots.
    #[test]
    fn pick_command_gpu_falls_back_when_prefer_unmet() {
        let s = svc(PlacementPolicy::GpuOnly, None);
        // Both GPUs satisfy min (2 GB) but neither meets prefer (16 GB).
        let snap = snapshot(&[6, 10]);
        let table = AllocationTable::new();
        let pick = pick_command_gpu(&s, &snap, &table, 2 * 1024, Some(16 * 1024), false);
        assert_eq!(pick, Some(1), "fall back to most-free when prefer unmet");
    }

    /// A pledge from another service must be subtracted from availability,
    /// so a busy GPU 0 cedes to a free GPU 1 even when nvml currently reports
    /// the same free bytes for both.
    #[test]
    fn pick_command_gpu_subtracts_pledged_reservations() {
        let s = svc(PlacementPolicy::GpuOnly, None);
        let snap = snapshot(&[20, 20]); // both GPUs report 20 GB free
        // Another service has 19 GB pledged on GPU 0.
        let mut table = AllocationTable::new();
        let mut other = BTreeMap::new();
        other.insert(DeviceSlot::Gpu(0), 19 * 1024u64); // MB
        table.insert(SmolStr::new("other"), other);
        let pick = pick_command_gpu(&s, &snap, &table, 4 * 1024, None, false);
        assert_eq!(pick, Some(1), "GPU 1 has the pledged-aware headroom");
    }

    /// `gpu_allow` is a hard restriction. Even if a non-listed GPU has more
    /// free capacity, the pick must come from the allowed set.
    #[test]
    fn pick_command_gpu_respects_gpu_allow() {
        // GPU 0 = 24 GB, GPU 1 = 4 GB, GPU 2 = 16 GB; allow only [1, 2].
        let s = svc(PlacementPolicy::GpuOnly, Some(vec![1, 2]));
        let snap = snapshot(&[24, 4, 16]);
        let table = AllocationTable::new();
        let pick = pick_command_gpu(&s, &snap, &table, 2 * 1024, None, false);
        assert_eq!(pick, Some(2), "GPU 2 is the most-free among allowed");
    }

    /// When no GPU has enough capacity to host `min_mb`, return `None` so
    /// the supervisor can run its eviction-retry path.
    #[test]
    fn pick_command_gpu_returns_none_when_nothing_fits() {
        let s = svc(PlacementPolicy::GpuOnly, None);
        let snap = snapshot(&[1, 1]); // 1 GB free each
        let table = AllocationTable::new();
        let pick = pick_command_gpu(&s, &snap, &table, 4 * 1024, None, false);
        assert_eq!(pick, None);
    }

    /// `optimistic_remaining = true` ignores nvml's view of free bytes and
    /// trusts the pledge book alone, matching `pack_optimistic`.
    #[test]
    fn pick_command_gpu_optimistic_ignores_nvml_free() {
        let s = svc(PlacementPolicy::GpuOnly, None);
        // nvml reports 0 free on both, but total = 24 GB and the pledge book
        // says nothing is reserved.
        let snap = snapshot(&[0, 0]);
        let table = AllocationTable::new();

        // Conservative: clamps to nvml_free, nothing fits.
        let conservative = pick_command_gpu(&s, &snap, &table, 4 * 1024, None, false);
        assert_eq!(conservative, None);

        // Optimistic: pledge book has 24 GB headroom, GPU 0 wins on tiebreak.
        let optimistic = pick_command_gpu(&s, &snap, &table, 4 * 1024, None, true);
        assert_eq!(optimistic, Some(0));
    }

    /// `placement_policy = CpuOnly` collapses the allowed-GPU list to empty;
    /// the helper must return `None` so the caller routes the reservation
    /// onto Cpu instead.
    #[test]
    fn pick_command_gpu_cpu_only_returns_none() {
        let s = svc(PlacementPolicy::CpuOnly, None);
        let snap = snapshot(&[24, 24]);
        let table = AllocationTable::new();
        let pick = pick_command_gpu(&s, &snap, &table, 2 * 1024, None, false);
        assert_eq!(pick, None);
    }

    /// Build a service with an explicit per-GPU placement_override.
    /// Mirrors the multi-GPU vLLM use case: TP=2 across two devices,
    /// each pledged separately.
    fn svc_with_override(pairs: &[(u32, u64)]) -> ServiceConfig {
        let mut svc = minimal_service("vllm-demo");
        let mut placement = BTreeMap::new();
        for (id, mb) in pairs {
            placement.insert(DeviceSlot::Gpu(*id), *mb);
        }
        svc.placement_override = placement;
        svc.placement_policy = PlacementPolicy::GpuOnly;
        svc
    }

    /// Two-GPU pledge that fits on both devices: every per-slot pledge
    /// has room, so `check_command_placement_override` should accept.
    #[test]
    fn check_placement_override_accepts_multi_gpu_pledge_that_fits() {
        let s = svc_with_override(&[(0, 22 * 1024), (1, 22 * 1024)]);
        let snap = snapshot(&[24, 24]);
        let table = AllocationTable::new();
        let r = check_command_placement_override(&s, &snap, &table, false);
        assert_eq!(r, Ok(()));
    }

    /// One slot in the override exceeds that GPU's free capacity. The
    /// helper must surface `WeightsDoNotFit` so the supervisor's
    /// eviction-retry loop can engage; partial fits never silently land.
    #[test]
    fn check_placement_override_rejects_when_one_slot_overflows() {
        let s = svc_with_override(&[(0, 22 * 1024), (1, 22 * 1024)]);
        // GPU 1 only has 16 GB free; pledge of 22 GB doesn't fit.
        let snap = snapshot(&[24, 16]);
        let table = AllocationTable::new();
        let r = check_command_placement_override(&s, &snap, &table, false);
        assert_eq!(r, Err(PackError::WeightsDoNotFit));
    }

    /// Existing peer reservations on a slot have to be subtracted from
    /// available capacity. A second 22 GiB pledge on a GPU that's
    /// already pledged 10 GiB to a peer should fail.
    #[test]
    fn check_placement_override_subtracts_existing_pledges() {
        let s = svc_with_override(&[(0, 22 * 1024), (1, 22 * 1024)]);
        let snap = snapshot(&[24, 24]);
        let mut table = AllocationTable::new();
        let mut peer = BTreeMap::new();
        peer.insert(DeviceSlot::Gpu(0), 10 * 1024);
        table.insert(SmolStr::new("peer"), peer);
        let r = check_command_placement_override(&s, &snap, &table, false);
        assert_eq!(r, Err(PackError::WeightsDoNotFit));
    }

    /// Optimistic mode (eviction retry) ignores nvml-free and trusts the
    /// pledge book. A pledged-but-not-yet-drained peer should NOT block
    /// our pledge once the supervisor has removed it from the table.
    #[test]
    fn check_placement_override_optimistic_ignores_nvml_free() {
        let s = svc_with_override(&[(0, 22 * 1024), (1, 22 * 1024)]);
        // nvml shows GPU 0 nearly full (peer is still draining), but the
        // pledge book is empty — optimistic mode should accept.
        let snap = snapshot(&[2, 24]);
        let table = AllocationTable::new();
        let conservative = check_command_placement_override(&s, &snap, &table, false);
        assert_eq!(conservative, Err(PackError::WeightsDoNotFit));
        let optimistic = check_command_placement_override(&s, &snap, &table, true);
        assert_eq!(optimistic, Ok(()));
    }

    /// CPU entries in the override are ignored (we don't model CPU
    /// capacity here). A pledge that's only on CPU should accept.
    #[test]
    fn check_placement_override_ignores_cpu_slots() {
        let mut svc = minimal_service("demo");
        let mut placement = BTreeMap::new();
        placement.insert(DeviceSlot::Cpu, 8 * 1024);
        svc.placement_override = placement;
        svc.placement_policy = PlacementPolicy::CpuOnly;
        let snap = snapshot(&[]);
        let table = AllocationTable::new();
        let r = check_command_placement_override(&svc, &snap, &table, false);
        assert_eq!(r, Ok(()));
    }

    /// Regression for the live "insufficient_vram on gpu:0" failure: the MTP
    /// draft-context lump must ride the *last* GPU (the spill target the
    /// trailing MTP head lands on), not pile onto the most-free GPU the
    /// first-fit walker is already filling to the brim. A model that spans
    /// both 24 GB cards plus a 3 GiB MTP lump must pack without overflowing
    /// GPU 0.
    #[test]
    fn mtp_overhead_rides_last_gpu_without_overflowing_first() {
        // 40 layers × 700 MiB ≈ 27.3 GiB of weights — does not fit one card,
        // so the walker spills onto GPU 1.
        let per_layer_bytes: Vec<u64> = (0..40).map(|_| 700 * 1024 * 1024).collect();
        let weights_bytes: u64 = per_layer_bytes.iter().sum();
        let mtp_bytes = 3 * 1024 * 1024 * 1024;
        let e = Estimate {
            weights_bytes,
            kv_per_token: 0,
            compute_buffer_mb: 1000,
            mtp_bytes,
            per_layer_bytes: Some(per_layer_bytes),
            attention_layers: None,
            non_layer: NonLayer::default(),
            override_tensor_bytes: BTreeMap::new(),
            expert_layers: Vec::new(),
            expert_tensors: None,
            context: 4096,
            architecture: SmolStr::new("qwen35"),
        };
        let snap = snapshot(&[24, 24]);
        let alloc = AllocationTable::new();
        let packed = pack(&e, &svc(PlacementPolicy::GpuOnly, None), &snap, &alloc)
            .expect("MTP model spanning two cards must pack");
        let gpu0 = packed
            .allocation
            .bytes
            .get(&DeviceId::Gpu(0))
            .copied()
            .unwrap_or(0);
        let gpu1 = packed
            .allocation
            .bytes
            .get(&DeviceId::Gpu(1))
            .copied()
            .unwrap_or(0);
        let cap = 24u64 * 1024 * 1024 * 1024;
        assert!(
            gpu0 <= cap,
            "GPU 0 must not be over-pledged: {gpu0} > {cap}"
        );
        assert!(
            gpu1 >= mtp_bytes,
            "the MTP lump must ride the last GPU (gpu1={gpu1}, mtp={mtp_bytes})"
        );
    }

    /// Tensor split shards every layer across both GPUs in parallel: emits
    /// `-ngl 999`, equal `--tensor-split 1,1`, `--split-mode tensor`, and
    /// `--main-gpu 0`, with each GPU pledged roughly half the model rather
    /// than first-fit filling GPU 0.
    #[test]
    fn tensor_split_shards_equally_across_gpus() {
        let e = trivial_estimate(20, 1024); // 20 layers × 1 GiB = 20 GiB
        let snap = snapshot(&[24, 24]);
        let alloc = AllocationTable::new();
        let mut s = svc(PlacementPolicy::GpuOnly, None);
        s.split_mode = SplitMode::Tensor;
        let packed = pack(&e, &s, &snap, &alloc).unwrap();

        assert_eq!(packed.args.ngl, Some(999));
        assert_eq!(packed.args.split_mode, Some(SplitMode::Tensor));
        assert_eq!(packed.args.main_gpu, Some(0));
        assert_eq!(packed.args.tensor_split.as_deref(), Some(&[1u32, 1][..]));

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
        // With no non-layer tensors or MTP overhead (trivial estimate), the
        // two shards are exactly equal, and each holds ~half the 20 GiB.
        assert_eq!(g0, g1, "shards must be balanced; got {g0} vs {g1}");
        let half = 10u64 * 1024 * 1024 * 1024;
        assert!(g0 >= half, "each shard should carry ~half the weights");
    }

    /// Tensor split shards the output head and the MTP draft context across
    /// every spanned GPU; only the vision projector (the non-layer "other"
    /// bytes) rides the main GPU. Measured on Qwen 3.6 27B (`--split-mode
    /// tensor`), enabling MTP added the same VRAM to *both* cards (≈1.4 GiB
    /// each), and the main GPU's only premium was the non-sharded mmproj — so
    /// the per-GPU difference must equal exactly `other_bytes`. Regression
    /// against the original lump-on-`--main-gpu` accounting, which over-pledged
    /// the main GPU by the whole output head plus the whole MTP context.
    #[test]
    fn tensor_split_shards_output_head_and_mtp_across_gpus() {
        let gib = 1024 * 1024 * 1024u64;
        let per_layer_bytes: Vec<u64> = (0..20).map(|_| gib).collect();
        let output_head = gib; // tensor-parallel sharded
        let other = gib / 2; // mmproj/vision — main GPU only
        let token_embd = gib / 2; // CPU
        let mtp_bytes = 3 * gib; // tensor-parallel sharded
        let weights_bytes = 20 * gib + output_head + other + token_embd;
        let e = Estimate {
            weights_bytes,
            kv_per_token: 0,
            compute_buffer_mb: 400,
            mtp_bytes,
            per_layer_bytes: Some(per_layer_bytes),
            attention_layers: None,
            non_layer: NonLayer {
                output_head_bytes: output_head,
                token_embd_bytes: token_embd,
                other_bytes: other,
            },
            override_tensor_bytes: BTreeMap::new(),
            expert_layers: Vec::new(),
            expert_tensors: None,
            context: 4096,
            architecture: SmolStr::new("qwen35"),
        };
        let snap = snapshot(&[24, 24]);
        let alloc = AllocationTable::new();
        let mut s = svc(PlacementPolicy::GpuOnly, None);
        s.split_mode = SplitMode::Tensor;
        let packed = pack(&e, &s, &snap, &alloc).unwrap();

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
        // If the output head and MTP rode the main GPU, the difference would be
        // `other + output_head + mtp`. Sharded, the only premium is the mmproj.
        assert_eq!(
            g0 - g1,
            other,
            "main GPU premium must be only the mmproj (got g0={g0}, g1={g1})"
        );
        // The last GPU must carry its half of the MTP draft context — proof it
        // is sharded, not lumped on the main GPU.
        assert!(
            g1 >= mtp_bytes / 2,
            "MTP must be sharded onto the last GPU (g1={g1}, mtp/2={})",
            mtp_bytes / 2
        );
        // Token embeddings ride the CPU, as on the layer path.
        let cpu = packed
            .allocation
            .bytes
            .get(&DeviceId::Cpu)
            .copied()
            .unwrap_or(0);
        assert_eq!(cpu, token_embd, "token embeddings must ride the CPU");
    }

    /// In a sharded split every spanned GPU must hold its shard — there is no
    /// CPU spill. A GPU too small for its equal share fails the pack with
    /// `ShardDoesNotFit`, naming the offending GPU.
    #[test]
    fn tensor_split_rejects_when_a_shard_overflows() {
        let e = trivial_estimate(40, 1024); // 40 GiB → 20 GiB per shard
        let snap = snapshot(&[24, 8]); // GPU 1 can't hold a 20 GiB shard
        let alloc = AllocationTable::new();
        let mut s = svc(PlacementPolicy::GpuOnly, None);
        s.split_mode = SplitMode::Tensor;
        let err = pack(&e, &s, &snap, &alloc).unwrap_err();
        assert!(
            matches!(err, PackError::ShardDoesNotFit { gpu_index: 1, .. }),
            "expected ShardDoesNotFit on gpu:1, got {err:?}"
        );
    }

    /// With only one GPU available, tensor split is meaningless — fall back to
    /// the ordinary single-GPU placement and emit no `--split-mode`/`--main-gpu`.
    #[test]
    fn tensor_split_with_one_gpu_falls_back_to_layer() {
        let e = trivial_estimate(4, 1024);
        let snap = snapshot(&[24]); // single GPU
        let alloc = AllocationTable::new();
        let mut s = svc(PlacementPolicy::GpuOnly, None);
        s.split_mode = SplitMode::Tensor;
        let packed = pack(&e, &s, &snap, &alloc).unwrap();
        assert_eq!(packed.args.split_mode, None);
        assert_eq!(packed.args.main_gpu, None);
        assert_eq!(packed.args.ngl, Some(4), "single-GPU layer count, not 999");
    }
}
