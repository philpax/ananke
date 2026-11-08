//! The packer's public result and error types: `CommandArgs`, `PackError`,
//! `Packed`, and the internal `ShardedPlan` handed between the sharded-split
//! step and `finish`.

use ananke_config::placement::SplitMode;

use crate::devices::{Allocation, DeviceId};

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
    /// (`layer`), so a layer-split service emits no `--split-mode` flag at
    /// all.
    pub split_mode: Option<SplitMode>,
    /// `--main-gpu N` — the CUDA-visible index (after the
    /// `CUDA_VISIBLE_DEVICES` remap) that gathers intermediate results and
    /// KV in sharded modes. Always the lowest-id spanned GPU, which
    /// `cuda_env::render`'s ascending ordering places at visible index 0.
    pub main_gpu: Option<u32>,
    /// `--n-cpu-moe N` — offload the trailing `N` expert layers' experts to
    /// CPU as whole layers. Set on the coarse expert-offload path instead of
    /// synthesising per-tensor `-ot` rules: keeping whole layers together
    /// keeps the runtime's fused multi-threaded CPU MoE kernel engaged
    /// (~24× faster on ik_llama than scattered `-ot`) and stays under
    /// llama.cpp's `GGML_SCHED_MAX_SPLIT_INPUTS` graph-split limit. When
    /// set, the runtime distributes the GPU-resident experts across cards
    /// itself, so `-ngl 999` is emitted and no expert `-ot`/`--tensor-split`
    /// is synthesised. `None` on the non-expert paths.
    pub n_cpu_moe: Option<u32>,
}

impl CommandArgs {
    /// Whether the packer actually chose an offload layout for this service.
    ///
    /// A service placed from an operator-declared reservation never runs the
    /// packer: the command path produces a `Packed` for the device allocation
    /// alone and leaves every offload field unset. The argv renderer has to
    /// tell that apart from a real layout, or it reads the empty fields as
    /// decisions and drops the service's own `-ngl` and `-ot`.
    pub fn describes_offload(&self) -> bool {
        self.ngl.is_some()
    }
}

/// One device's contribution to a placement failure: how much the packer
/// needed there set against how much the device could actually offer.
///
/// `device` is the same identity `GET /api/devices` reports, so a failure can
/// be cross-referenced against the live device view rather than leaving the
/// operator to guess which card (or the host) was the binding constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceShortfall {
    pub device: DeviceId,
    /// Bytes the placement needed on this device.
    pub requested_bytes: u64,
    /// Bytes the device could offer under the active remaining-capacity view.
    pub available_bytes: u64,
}

/// Structured packer failure modes. Each variant carries the numbers the
/// operator needs to understand the overflow, so nothing has to string-match
/// on the message to figure out what went wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackError {
    /// A specific block-layer's bytes didn't fit on any GPU the service
    /// was allowed to use, and CPU spill was disabled. `shortfalls` carries
    /// one entry per allowed GPU, since the layer was rejected by all of them.
    LayerDoesNotFit {
        layer_index: u32,
        bytes: u64,
        shortfalls: Vec<DeviceShortfall>,
    },
    /// The estimator returned no per-layer breakdown (fallback path on
    /// an unknown architecture) and the weights can't fit on any
    /// allowed device.
    WeightsDoNotFit { shortfalls: Vec<DeviceShortfall> },
    GpuDoesNotFit {
        gpu_index: u32,
        needed: u64,
        available: u64,
    },
    /// A sharded (tensor/row) split's equal per-GPU share didn't fit on one
    /// of the spanned GPUs. Unlike layer split there is no CPU spill — every
    /// spanned GPU must hold its shard, so a single overflow fails the pack.
    ShardDoesNotFit {
        gpu_index: u32,
        bytes: u64,
        available: u64,
    },
    /// Even after offloading every eligible expert tensor to the CPU, the
    /// bytes the packer wants to keep on the host exceed the available host
    /// RAM (minus the configured `[devices.cpu] reserved_gb`).
    CpuDoesNotFit { needed: u64, available: u64 },
    /// A manual `expert_offload = N` pins every non-offloaded layer's experts
    /// to the GPUs regardless of fit, and the retained experts overflow the
    /// pooled GPU capacity. Unlike `Auto`, manual mode never spills the surplus
    /// for the operator — the fix is a larger offload count. The constraint is
    /// the pool rather than any one card, so `needed`/`available` are totals and
    /// `shortfalls` breaks the even per-card share down by GPU.
    ManualExpertsDoNotFit {
        needed: u64,
        available: u64,
        shortfalls: Vec<DeviceShortfall>,
    },
    /// `tensor_split_weights` count doesn't match the number of spanned GPUs.
    /// This is a configuration error, not a capacity problem — eviction
    /// cannot fix it.
    InvalidTensorSplitWeights { expected: usize, got: usize },
}

impl PackError {
    /// Per-device breakdown of why the pack failed, for the placement preview
    /// and the wire. Empty for [`Self::InvalidTensorSplitWeights`], which is a
    /// configuration error rather than a capacity one — there is no device to
    /// point at, and no amount of freed memory would fix it.
    pub fn shortfalls(&self) -> Vec<DeviceShortfall> {
        match self {
            Self::LayerDoesNotFit { shortfalls, .. }
            | Self::WeightsDoNotFit { shortfalls }
            | Self::ManualExpertsDoNotFit { shortfalls, .. } => shortfalls.clone(),
            Self::GpuDoesNotFit {
                gpu_index,
                needed: bytes,
                available,
            }
            | Self::ShardDoesNotFit {
                gpu_index,
                bytes,
                available,
            } => vec![DeviceShortfall {
                device: DeviceId::Gpu(*gpu_index),
                requested_bytes: *bytes,
                available_bytes: *available,
            }],
            Self::CpuDoesNotFit { needed, available } => vec![DeviceShortfall {
                device: DeviceId::Cpu,
                requested_bytes: *needed,
                available_bytes: *available,
            }],
            Self::InvalidTensorSplitWeights { .. } => Vec::new(),
        }
    }
}

impl std::fmt::Display for PackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LayerDoesNotFit {
                layer_index, bytes, ..
            } => {
                write!(
                    f,
                    "layer {layer_index} ({bytes} bytes) does not fit on any allowed GPU"
                )
            }
            Self::WeightsDoNotFit { .. } => f.write_str("weights do not fit on any allowed device"),
            Self::GpuDoesNotFit {
                gpu_index,
                needed,
                available,
            } => write!(
                f,
                "placement on gpu:{gpu_index}: {needed} bytes exceed available memory ({available} bytes)"
            ),
            Self::ShardDoesNotFit {
                gpu_index, bytes, ..
            } => {
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
            Self::ManualExpertsDoNotFit {
                needed, available, ..
            } => {
                write!(
                    f,
                    "manual expert_offload keeps {needed} bytes of expert weight on the GPUs but only {available} bytes are available; raise the expert_offload count"
                )
            }
            Self::InvalidTensorSplitWeights { expected, got } => {
                write!(
                    f,
                    "tensor_split_weights has {got} entries but {expected} GPUs are spanned"
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
    /// What the next rolling-correction update needs from this placement.
    pub rolling: RollingInputs,
}

/// The uncorrected per-pool bases a placement was built from, plus the
/// GPU-resident weight total, captured so the rolling correction can compare
/// like with like when the service later drains.
///
/// Every field is *raw* — as the estimator predicted it, before the
/// corrections this pack applied. A ratio taken against a corrected base would
/// measure the correction rather than the estimator.
#[derive(Debug, Clone, Copy, Default)]
pub struct RollingInputs {
    /// Sum of the `Gpu(_)` slots, uncorrected. Denominator for the VRAM pool.
    pub uncorrected_vram_bytes: u64,
    /// The `Cpu` slot, uncorrected. Denominator for the host pool.
    pub uncorrected_host_bytes: u64,
    /// Model tensor bytes placed on GPUs, uncorrected and excluding KV,
    /// compute buffers, and slop.
    ///
    /// Reported, not consumed by the correction. llama.cpp reads these through
    /// the GGUF's mmap, so at runtime they sit in the process's *file* RSS while
    /// living in VRAM — and the host pool's numerator is owned memory
    /// (`RssAnon + RssShmem`), which they never enter. Nothing subtracts this;
    /// a numerator taken from a *total* RSS peak would have to.
    ///
    /// It is the only place the GPU-resident weight total is available to a
    /// caller: the `estimate` example prints it, and the MTP tests assert a
    /// separate draft's weights reach it rather than being charged as runtime
    /// allocation.
    pub gpu_weight_bytes: u64,
    /// Model tensor bytes placed on the host, uncorrected. Gates host-pool
    /// learning: see
    /// `RollingBase::host_peak`.
    pub cpu_weight_bytes: u64,
}

/// A tensor/row-split distribution decided by
/// [`crate::sharded::distribute_sharded`].
/// [`crate::finish`] turns it into `--split-mode`,
/// `--main-gpu`, and the `--tensor-split` ratio (equal `1`s by default, or
/// the weighted integers derived from `tensor_split_weights`).
#[derive(Debug)]
pub(crate) struct ShardedPlan {
    pub(crate) mode: SplitMode,
    /// Integer tensor-split values emitted for this sharded plan. Stored here
    /// so `finish` can render the same ratio that `distribute_sharded` used
    /// for the pledge book. The length is one entry per spanned GPU, in
    /// ascending GPU-id order.
    pub(crate) tensor_split: Vec<u32>,
}
