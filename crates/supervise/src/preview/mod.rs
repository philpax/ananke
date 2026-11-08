//! Compute the launch command a service is using or would use, without
//! spawning anything.
//!
//! This mirrors the placement decisions in
//! [`crate::RunLoop::compute_reservation_map_inner`] but as a pure function:
//! it runs the estimator and the *optimistic* packer (the pledge-book planner,
//! so a service that is already running — whose own VRAM the nvml snapshot
//! already reflects — still plans against the book rather than failing on its
//! own footprint), then renders the argv via [`render_argv`]. It performs no
//! I/O beyond the read-only GGUF parse the estimator needs, and mutates
//! nothing.

mod verdict;

use std::collections::BTreeMap;

use ananke_allocator::{
    AllocationTable,
    placement::{self, CommandArgs, PackError},
};
use ananke_api::internal::fit_verdict::FitVerdict;
use ananke_config::{AllocationMode, DeviceSlot, PlacementPolicy, ServiceConfig, Template};
use ananke_devices::{Allocation, DeviceId, DeviceSnapshot};
use ananke_estimate::{self, EstimatorError};
use ananke_placement::service_inputs::placement_inputs;
use ananke_system::Fs;
use ananke_templates::SubstituteError;
use ananke_tracking::rolling::Corrections;
pub use verdict::{
    PreviewContext, preview_command_placement, preview_override_placement, preview_placement,
};

pub use crate::eviction::eviction_candidates;
use crate::spawn::{SpawnConfig, container::render_container_spec, render_argv};

/// Why a launch-command preview could not be produced.
#[derive(Debug)]
pub enum PreviewError {
    /// A llama-cpp service without a usable model path (the estimator has
    /// nothing to read).
    NoModelPath,
    /// The estimator failed to parse the GGUF.
    Estimator(EstimatorError),
    /// Placement failed against the current snapshot and pledge book.
    Pack(PackError),
    /// Argv rendering failed (a `{placeholder}` could not be substituted).
    Render(SubstituteError),
    /// Container-spec rendering failed (mount translation, CDI template, etc).
    Container(String),
}

impl std::fmt::Display for PreviewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PreviewError::NoModelPath => write!(f, "the service has no model path to estimate"),
            PreviewError::Estimator(e) => write!(f, "estimate the model: {e}"),
            PreviewError::Pack(e) => write!(f, "plan placement: {e}"),
            PreviewError::Render(e) => write!(f, "render the command line: {e}"),
            PreviewError::Container(e) => write!(f, "render the container spec: {e}"),
        }
    }
}

impl std::error::Error for PreviewError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Estimator(e) => Some(e),
            Self::Pack(e) => Some(e),
            Self::Render(e) => Some(e),
            Self::Container(_) => None,
            Self::NoModelPath => None,
        }
    }
}

/// Render the command line a service would launch with, given the current
/// config, device snapshot, and pledge book. `corrections` are the service's
/// learned per-pool estimator corrections ([`Corrections::NEUTRAL`] if none) —
/// pass them so the preview matches the placement the supervisor would
/// actually compute.
pub fn preview_command(
    svc: &ServiceConfig,
    snapshot: &DeviceSnapshot,
    table: &AllocationTable,
    fs: &dyn Fs,
    corrections: Corrections,
) -> Result<SpawnConfig, PreviewError> {
    let (alloc, cmd_args) = plan(svc, snapshot, table, fs, corrections)?;
    render_argv(svc, &alloc, cmd_args.as_ref()).map_err(PreviewError::Render)
}

/// Render the resolved container spec a service would launch with, for
/// services carrying a `[service.container]` block. The owner UUID and
/// run id are runtime values: `owner_uuid` must be the installation's stable
/// owner, and `run_id` a preview run id (the generated name is a pattern, not
/// a live name). For a non-container service this returns `None`.
pub fn preview_container_command(
    svc: &ServiceConfig,
    snapshot: &DeviceSnapshot,
    table: &AllocationTable,
    fs: &dyn Fs,
    corrections: Corrections,
    owner_uuid: &str,
    run_id: i64,
) -> Result<Option<ananke_spawn::ContainerSpec>, PreviewError> {
    if svc.container.is_none() {
        return Ok(None);
    }
    let (alloc, cmd_args) = plan(svc, snapshot, table, fs, corrections)?;
    render_container_spec(svc, &alloc, cmd_args.as_ref(), run_id, owner_uuid)
        .map(Some)
        .map_err(|e| PreviewError::Container(e.to_string()))
}

/// Resolve the allocation and (for the llama estimator path) the
/// placement-derived `CommandArgs` a service would launch with. Command
/// templates and explicit `placement_override` services carry no `CommandArgs`
/// — their argv is fully determined by the config and the chosen device.
fn plan(
    svc: &ServiceConfig,
    snapshot: &DeviceSnapshot,
    table: &AllocationTable,
    fs: &dyn Fs,
    corrections: Corrections,
) -> Result<(Allocation, Option<CommandArgs>), PreviewError> {
    if matches!(svc.template(), Template::Command) {
        let map = plan_command_map(svc, snapshot, table)?;
        return Ok((Allocation::from_override(&map), None));
    }
    if !svc.placement_override.is_empty() {
        return Ok((Allocation::from_override(&svc.placement_override), None));
    }
    let inputs = ananke_estimate::service_inputs::estimator_inputs(svc)
        .map(|i| i.with_visible_devices(snapshot.gpus.len() as u32))
        .ok_or(PreviewError::NoModelPath)?;
    let (_summary, est) =
        ananke_estimate::estimate_with_summary(fs, &inputs).map_err(PreviewError::Estimator)?;
    let packed = placement::pack_corrected(
        &est,
        &placement_inputs(svc),
        snapshot,
        table,
        corrections,
        true,
    )
    .map_err(PreviewError::Pack)?;
    Ok((packed.allocation, Some(packed.args)))
}

/// Command-template placement, mirroring
/// [`crate::RunLoop::compute_command_reservation`] in optimistic mode: honour
/// an explicit `placement_override`, else pin `CpuOnly` services to the CPU,
/// else pick the GPU with the most headroom. An empty map (no reservation)
/// renders a deterministic empty `CUDA_VISIBLE_DEVICES`.
fn plan_command_map(
    svc: &ServiceConfig,
    snapshot: &DeviceSnapshot,
    table: &AllocationTable,
) -> Result<BTreeMap<DeviceSlot, u64>, PreviewError> {
    let (min_mb, prefer_mb) = match svc.allocation_mode {
        AllocationMode::Static { reserve_mb } => (reserve_mb, Some(reserve_mb)),
        AllocationMode::Dynamic { min_mb, max_mb, .. } => (min_mb, Some(max_mb)),
        AllocationMode::None => (0, None),
    };
    let mut map = BTreeMap::new();
    if min_mb == 0 {
        return Ok(map);
    }
    if !svc.placement_override.is_empty() {
        placement::check_command_placement_override(&placement_inputs(svc), snapshot, table, true)
            .map_err(PreviewError::Pack)?;
        return Ok(svc.placement_override.clone());
    }
    let slot = if matches!(svc.placement_policy, PlacementPolicy::CpuOnly) {
        DeviceSlot::Cpu
    } else {
        match placement::pick_command_gpu(
            &placement_inputs(svc),
            snapshot,
            table,
            min_mb,
            prefer_mb,
            true,
        ) {
            Some(id) => DeviceSlot::Gpu(id),
            None if snapshot.gpus.is_empty() => DeviceSlot::Cpu,
            None => {
                return Err(PreviewError::Pack(PackError::WeightsDoNotFit {
                    shortfalls: placement::command_gpu_shortfalls(
                        &placement_inputs(svc),
                        snapshot,
                        table,
                        min_mb,
                        true,
                    ),
                }));
            }
        }
    };
    map.insert(slot, min_mb);
    Ok(map)
}

/// Per-device placement a service would take, plus whether it fits without
/// eviction. Produced by [`preview_placement`].
pub struct PlacementOutcome {
    /// Bytes the service would occupy on each device.
    pub devices: BTreeMap<DeviceId, u64>,
    /// Whether it fits now, needs room freed, or can't fit at all.
    pub verdict: FitVerdict,
    /// Total expert-tensor bytes the packer offloaded to the CPU (MoE). Zero
    /// for non-MoE services or when nothing was offloaded.
    pub expert_offload_bytes: u64,
    /// Distinct layers with at least one expert offloaded to the CPU.
    pub expert_offload_layers: u32,
}
