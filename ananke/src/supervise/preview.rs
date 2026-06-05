//! Compute the launch command a service is using or would use, without
//! spawning anything.
//!
//! This mirrors the placement decisions in
//! [`super::RunLoop::compute_reservation_map_inner`] but as a pure function:
//! it runs the estimator and the *optimistic* packer (the pledge-book planner,
//! so a service that is already running — whose own VRAM the nvml snapshot
//! already reflects — still plans against the book rather than failing on its
//! own footprint), then renders the argv via [`render_argv`]. It performs no
//! I/O beyond the read-only GGUF parse the estimator needs, and mutates
//! nothing.

use std::collections::BTreeMap;

use crate::{
    allocator::{
        AllocationTable,
        placement::{self, CommandArgs, PackError},
    },
    config::{AllocationMode, DeviceSlot, PlacementPolicy, ServiceConfig, Template},
    devices::{Allocation, DeviceSnapshot},
    estimator::{self, EstimatorError, EstimatorInputs},
    supervise::spawn::{SpawnConfig, render_argv},
    system::Fs,
    templates::SubstituteError,
};

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
}

impl std::fmt::Display for PreviewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PreviewError::NoModelPath => write!(f, "the service has no model path to estimate"),
            PreviewError::Estimator(e) => write!(f, "estimate the model: {e}"),
            PreviewError::Pack(e) => write!(f, "plan placement: {e}"),
            PreviewError::Render(e) => write!(f, "render the command line: {e}"),
        }
    }
}

impl std::error::Error for PreviewError {}

/// Render the command line a service would launch with, given the current
/// config, device snapshot, and pledge book. `rolling_mean` is the estimator
/// drift correction for this service (1.0 if none) — pass it so the preview
/// matches the placement the supervisor would actually compute.
pub fn preview_command(
    svc: &ServiceConfig,
    snapshot: &DeviceSnapshot,
    table: &AllocationTable,
    fs: &dyn Fs,
    rolling_mean: f64,
) -> Result<SpawnConfig, PreviewError> {
    let (alloc, cmd_args) = plan(svc, snapshot, table, fs, rolling_mean)?;
    render_argv(svc, &alloc, cmd_args.as_ref()).map_err(PreviewError::Render)
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
    rolling_mean: f64,
) -> Result<(Allocation, Option<CommandArgs>), PreviewError> {
    if matches!(svc.template(), Template::Command) {
        let map = plan_command_map(svc, snapshot, table)?;
        return Ok((Allocation::from_override(&map), None));
    }
    if !svc.placement_override.is_empty() {
        return Ok((Allocation::from_override(&svc.placement_override), None));
    }

    let inputs = EstimatorInputs::from_service(svc).ok_or(PreviewError::NoModelPath)?;
    let (_summary, mut est) =
        estimator::estimate_with_summary(fs, &inputs).map_err(PreviewError::Estimator)?;
    est.weights_bytes = (est.weights_bytes as f64 * rolling_mean) as u64;
    let packed =
        placement::pack_optimistic(&est, svc, snapshot, table).map_err(PreviewError::Pack)?;
    Ok((packed.allocation, Some(packed.args)))
}

/// Command-template placement, mirroring
/// [`super::RunLoop::compute_command_reservation`] in optimistic mode: honour
/// an explicit `placement_override`, else pin `CpuOnly` services to the CPU,
/// else pick the GPU with the most headroom. An empty map (no reservation)
/// renders a deterministic empty `CUDA_VISIBLE_DEVICES`.
fn plan_command_map(
    svc: &ServiceConfig,
    snapshot: &DeviceSnapshot,
    table: &AllocationTable,
) -> Result<BTreeMap<DeviceSlot, u64>, PreviewError> {
    let (min_mb, prefer_mb) = match svc.allocation_mode {
        AllocationMode::Static { vram_mb } => (vram_mb, Some(vram_mb)),
        AllocationMode::Dynamic { min_mb, max_mb, .. } => (min_mb, Some(max_mb)),
        AllocationMode::None => (0, None),
    };
    let mut map = BTreeMap::new();
    if min_mb == 0 {
        return Ok(map);
    }
    if !svc.placement_override.is_empty() {
        placement::check_command_placement_override(svc, snapshot, table, true)
            .map_err(PreviewError::Pack)?;
        return Ok(svc.placement_override.clone());
    }
    let slot = if matches!(svc.placement_policy, PlacementPolicy::CpuOnly) {
        DeviceSlot::Cpu
    } else {
        match placement::pick_command_gpu(svc, snapshot, table, min_mb, prefer_mb, true) {
            Some(id) => DeviceSlot::Gpu(id),
            None if snapshot.gpus.is_empty() => DeviceSlot::Cpu,
            None => return Err(PreviewError::Pack(PackError::WeightsDoNotFit)),
        }
    };
    map.insert(slot, min_mb);
    Ok(map)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use smol_str::SmolStr;

    use super::*;
    use crate::{
        config::validate::test_fixtures::minimal_command_service,
        devices::{CpuSnapshot, GpuSnapshot},
        system::SystemDeps,
    };

    fn two_gpu_snapshot() -> DeviceSnapshot {
        DeviceSnapshot {
            gpus: vec![
                GpuSnapshot {
                    id: 0,
                    name: "gpu0".into(),
                    total_bytes: 24 << 30,
                    free_bytes: 24 << 30,
                },
                GpuSnapshot {
                    id: 1,
                    name: "gpu1".into(),
                    total_bytes: 24 << 30,
                    free_bytes: 24 << 30,
                },
            ],
            cpu: Some(CpuSnapshot {
                total_bytes: 64 << 30,
                available_bytes: 64 << 30,
            }),
            taken_at_ms: 0,
        }
    }

    /// A command-template preview picks the GPU with headroom in the pledge
    /// book, renders the chosen device into `CUDA_VISIBLE_DEVICES`, and
    /// substitutes argv placeholders — without spawning anything.
    #[test]
    fn command_preview_picks_free_gpu_and_renders_env() {
        let mut svc = minimal_command_service(
            "comfy",
            vec!["comfyui-start".into(), "--port".into(), "{port}".into()],
        );
        svc.placement_override.clear();
        svc.placement_policy = PlacementPolicy::GpuOnly;
        svc.allocation_mode = AllocationMode::Static { vram_mb: 4096 };
        svc.private_port = 8200;

        // Peer holds 23 GiB on GPU 0 (in MB), so only GPU 1 can fit the 4 GiB
        // reservation — the optimistic planner reads the pledge book.
        let mut table = AllocationTable::new();
        let mut peer = BTreeMap::new();
        peer.insert(DeviceSlot::Gpu(0), 23_000u64);
        table.insert(SmolStr::new("peer"), peer);

        let snap = two_gpu_snapshot();
        let (deps, _fakes) = SystemDeps::fake();
        let cfg = preview_command(&svc, &snap, &table, deps.fs.as_ref(), 1.0)
            .expect("command preview must succeed");

        assert_eq!(cfg.binary, "comfyui-start");
        assert_eq!(
            cfg.env.get("CUDA_VISIBLE_DEVICES").map(String::as_str),
            Some("1"),
            "must pick GPU 1 (GPU 0 is pledged out); env={:?}",
            cfg.env
        );
        assert!(
            cfg.args.contains(&"8200".to_string()),
            "the {{port}} placeholder must be substituted; got {:?}",
            cfg.args
        );
    }
}
