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

use ananke_api::FitVerdict;

use crate::{
    allocator::{
        AllocationTable,
        placement::{self, CommandArgs, PackError},
    },
    config::{AllocationMode, DeviceSlot, PlacementPolicy, ServiceConfig, Template},
    devices::{Allocation, DeviceId, DeviceSnapshot},
    estimator::{self, Estimate, EstimatorError, EstimatorInputs},
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

/// Per-device placement a service would take, plus whether it fits without
/// eviction. Produced by [`preview_placement`].
pub struct PlacementOutcome {
    /// Bytes the service would occupy on each device.
    pub devices: BTreeMap<DeviceId, u64>,
    /// Whether it fits now, needs room freed, or can't fit at all.
    pub verdict: FitVerdict,
}

/// Compute where a llama service's VRAM would land per device and whether it
/// fits without eviction, by running the packer against the live snapshot and
/// pledge book. `est` must already have the rolling correction applied. This
/// is the estimator path only — the caller must not pass a service with a
/// manual `placement_override` (its placement is the override, not a pack).
///
/// `running` short-circuits the verdict to [`FitVerdict::Fits`]: a live
/// service is by definition placed, and the strict nvml-free check would
/// otherwise be confounded by the service's own resident VRAM (which lowers
/// reported free without being a true obstacle to its own placement).
pub fn preview_placement(
    svc: &ServiceConfig,
    est: &Estimate,
    snapshot: &DeviceSnapshot,
    table: &AllocationTable,
    running: bool,
) -> PlacementOutcome {
    // Strict honours currently-free VRAM (what the daemon checks before
    // deciding to evict); optimistic trusts the pledge book; on-empty models
    // the bare hardware capacity (could it ever fit on the allowed GPUs).
    let strict = placement::pack(est, svc, snapshot, table).ok();
    let optimistic = placement::pack_optimistic(est, svc, snapshot, table).ok();
    let on_empty = placement::pack_optimistic(est, svc, snapshot, &AllocationTable::new()).ok();

    let verdict = if running || strict.is_some() {
        FitVerdict::Fits
    } else if on_empty.is_some() {
        FitVerdict::NeedsEviction
    } else {
        FitVerdict::DoesNotFit
    };

    // Prefer the strict allocation (what it would actually get now), then the
    // pledge-book shape, then the bare-hardware shape — so a service that
    // needs eviction still shows where it would land once room is freed.
    let devices = strict
        .or(optimistic)
        .or(on_empty)
        .map(|p| p.allocation.bytes)
        .unwrap_or_default();

    PlacementOutcome { devices, verdict }
}

/// Placement for a service that declares a manual `placement_override`. The
/// per-device split is the override itself (the daemon honours it verbatim
/// rather than packing); the verdict checks each pledged GPU slot against the
/// live snapshot the same way [`preview_placement`] does — strict (currently
/// free) for `Fits`, bare-hardware capacity for `NeedsEviction`. Works for
/// both llama-cpp and command (e.g. multi-GPU vLLM) override services.
pub fn preview_override_placement(
    svc: &ServiceConfig,
    snapshot: &DeviceSnapshot,
    table: &AllocationTable,
    running: bool,
) -> PlacementOutcome {
    let devices = svc
        .placement_override
        .iter()
        .map(|(slot, mib)| {
            let id = match slot {
                DeviceSlot::Cpu => DeviceId::Cpu,
                DeviceSlot::Gpu(n) => DeviceId::Gpu(*n),
            };
            (id, mib.saturating_mul(1024 * 1024))
        })
        .collect();

    let fits_now = placement::check_command_placement_override(svc, snapshot, table, false).is_ok();
    let fits_on_empty =
        placement::check_command_placement_override(svc, snapshot, &AllocationTable::new(), true)
            .is_ok();
    let verdict = if running || fits_now {
        FitVerdict::Fits
    } else if fits_on_empty {
        FitVerdict::NeedsEviction
    } else {
        FitVerdict::DoesNotFit
    };

    PlacementOutcome { devices, verdict }
}

/// Placement for a command-template service that picks a GPU dynamically (no
/// `placement_override`): it reserves `min_mb` on the GPU with the most
/// headroom, or pins to the CPU for a cpu-only service. Mirrors
/// [`super::RunLoop::compute_command_reservation`]. Returns `None` when the
/// service reserves no VRAM (`AllocationMode::None`), so the caller renders no
/// placement at all.
pub fn preview_command_placement(
    svc: &ServiceConfig,
    snapshot: &DeviceSnapshot,
    table: &AllocationTable,
    running: bool,
) -> Option<PlacementOutcome> {
    let (min_mb, prefer_mb) = match svc.allocation_mode {
        AllocationMode::Static { vram_mb } => (vram_mb, Some(vram_mb)),
        AllocationMode::Dynamic { min_mb, max_mb, .. } => (min_mb, Some(max_mb)),
        AllocationMode::None => return None,
    };
    if min_mb == 0 {
        return None;
    }
    let bytes = min_mb.saturating_mul(1024 * 1024);

    if matches!(svc.placement_policy, PlacementPolicy::CpuOnly) {
        let mut devices = BTreeMap::new();
        devices.insert(DeviceId::Cpu, bytes);
        return Some(PlacementOutcome {
            devices,
            verdict: FitVerdict::Fits,
        });
    }

    let strict = placement::pick_command_gpu(svc, snapshot, table, min_mb, prefer_mb, false);
    let on_empty = placement::pick_command_gpu(
        svc,
        snapshot,
        &AllocationTable::new(),
        min_mb,
        prefer_mb,
        true,
    );
    let verdict = if running || strict.is_some() {
        FitVerdict::Fits
    } else if on_empty.is_some() {
        FitVerdict::NeedsEviction
    } else {
        FitVerdict::DoesNotFit
    };

    // Show the strict pick (where it would land now), else the bare-hardware
    // pick so a needs-eviction service still shows its target GPU.
    let mut devices = BTreeMap::new();
    if let Some(gpu) = strict.or(on_empty) {
        devices.insert(DeviceId::Gpu(gpu), bytes);
    }
    Some(PlacementOutcome { devices, verdict })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use smol_str::SmolStr;

    use super::*;
    use crate::{
        config::validate::test_fixtures::{minimal_command_service, minimal_service},
        devices::{CpuSnapshot, GpuSnapshot},
        estimator::NonLayer,
        system::SystemDeps,
    };

    fn two_gpu_snapshot() -> DeviceSnapshot {
        gpus_with_free(24)
    }

    /// Two 24 GiB GPUs with `free_gib` free each.
    fn gpus_with_free(free_gib: u64) -> DeviceSnapshot {
        let free = free_gib << 30;
        DeviceSnapshot {
            gpus: vec![
                GpuSnapshot {
                    id: 0,
                    name: "gpu0".into(),
                    total_bytes: 24 << 30,
                    free_bytes: free,
                },
                GpuSnapshot {
                    id: 1,
                    name: "gpu1".into(),
                    total_bytes: 24 << 30,
                    free_bytes: free,
                },
            ],
            cpu: Some(CpuSnapshot {
                total_bytes: 64 << 30,
                available_bytes: 64 << 30,
            }),
            taken_at_ms: 0,
        }
    }

    /// A GPU-only llama service (the fixture defaults to CPU-only with an
    /// override; clear both so placement actually packs onto GPUs).
    fn llama_svc() -> ServiceConfig {
        let mut s = minimal_service("m");
        s.placement_override.clear();
        s.placement_policy = PlacementPolicy::GpuOnly;
        s
    }

    /// `n_layers × per_gib` GiB of pure layer weights — no KV, MTP, or
    /// compute buffer — so the fit maths is easy to reason about.
    fn estimate_gib(n_layers: u32, per_gib: u64) -> Estimate {
        let per = per_gib << 30;
        Estimate {
            weights_bytes: per * n_layers as u64,
            kv_per_token: 0,
            compute_buffer_mb: 0,
            mtp_bytes: 0,
            per_layer_bytes: Some(vec![per; n_layers as usize]),
            attention_layers: None,
            non_layer: NonLayer::default(),
            override_tensor_bytes: BTreeMap::new(),
            expert_layers: Vec::new(),
            expert_layer_cpu_bytes: BTreeMap::new(),
            context: 4096,
            architecture: SmolStr::new("qwen3"),
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

    /// 20 GiB across two empty 24 GiB cards fits in currently-free VRAM.
    #[test]
    fn placement_fits_in_free_vram() {
        let out = preview_placement(
            &llama_svc(),
            &estimate_gib(20, 1),
            &gpus_with_free(24),
            &AllocationTable::new(),
            false,
        );
        assert_eq!(out.verdict, FitVerdict::Fits);
        assert!(!out.devices.is_empty(), "a fitting placement names devices");
    }

    /// Same 20 GiB, but the cards are nearly full (1 GiB free each): it fits
    /// within total capacity, so the daemon would reclaim/evict — and the
    /// would-be shape is still reported.
    #[test]
    fn placement_needs_eviction_when_free_is_low() {
        let out = preview_placement(
            &llama_svc(),
            &estimate_gib(20, 1),
            &gpus_with_free(1),
            &AllocationTable::new(),
            false,
        );
        assert_eq!(out.verdict, FitVerdict::NeedsEviction);
        assert!(
            !out.devices.is_empty(),
            "needs-eviction still shows where it would land"
        );
    }

    /// 60 GiB can't fit on two 24 GiB cards even with everything else gone.
    #[test]
    fn placement_does_not_fit_when_too_large() {
        let out = preview_placement(
            &llama_svc(),
            &estimate_gib(60, 1),
            &gpus_with_free(24),
            &AllocationTable::new(),
            false,
        );
        assert_eq!(out.verdict, FitVerdict::DoesNotFit);
        assert!(
            out.devices.is_empty(),
            "no valid placement names no devices"
        );
    }

    /// A running service is reported as fitting even when free VRAM is low —
    /// it is demonstrably placed, and the low free is its own resident VRAM.
    #[test]
    fn running_service_always_fits() {
        let out = preview_placement(
            &llama_svc(),
            &estimate_gib(20, 1),
            &gpus_with_free(1),
            &AllocationTable::new(),
            true,
        );
        assert_eq!(out.verdict, FitVerdict::Fits);
    }

    /// An override service reports the override as its per-device split, with a
    /// verdict checked against the live snapshot the same way the packer path
    /// is — without running the estimator at all.
    #[test]
    fn override_placement_uses_override_and_checks_fit() {
        let mut svc = minimal_service("ov");
        svc.placement_override.clear();
        svc.placement_override.insert(DeviceSlot::Gpu(1), 8000); // MB
        let table = AllocationTable::new();

        let fits = preview_override_placement(&svc, &gpus_with_free(24), &table, false);
        assert_eq!(fits.verdict, FitVerdict::Fits);
        assert_eq!(
            fits.devices.get(&DeviceId::Gpu(1)).copied(),
            Some(8000 * 1024 * 1024),
            "override MB is reported as bytes on the declared device"
        );

        // Cards nearly full: fits the hardware but not currently-free VRAM.
        let tight = preview_override_placement(&svc, &gpus_with_free(1), &table, false);
        assert_eq!(tight.verdict, FitVerdict::NeedsEviction);

        // A running override service is always reported as fitting.
        let live = preview_override_placement(&svc, &gpus_with_free(1), &table, true);
        assert_eq!(live.verdict, FitVerdict::Fits);
    }

    /// A dynamic command-template service (e.g. ComfyUI) reserves its `min_mb`
    /// on the GPU with headroom — here GPU 1, since GPU 0 is nearly full.
    #[test]
    fn command_placement_reserves_min_on_picked_gpu() {
        let mut svc = minimal_command_service("comfy", vec!["comfyui".into()]);
        svc.placement_override.clear();
        svc.placement_policy = PlacementPolicy::GpuOnly;
        svc.allocation_mode = AllocationMode::Dynamic {
            min_mb: 2048,
            max_mb: 20480,
            min_borrower_runtime_ms: 0,
        };
        // GPU 0 has 1 GiB free (can't hold 2 GiB min), GPU 1 has 24 GiB.
        let mut snap = gpus_with_free(24);
        snap.gpus[0].free_bytes = 1 << 30;

        let out = preview_command_placement(&svc, &snap, &AllocationTable::new(), false)
            .expect("a reserving command service has a placement");
        assert_eq!(out.verdict, FitVerdict::Fits);
        assert_eq!(
            out.devices.get(&DeviceId::Gpu(1)).copied(),
            Some(2048 * 1024 * 1024),
            "min_mb is reserved on the picked GPU"
        );
        assert!(!out.devices.contains_key(&DeviceId::Gpu(0)));
    }

    /// A command service that reserves no VRAM has no placement to show.
    #[test]
    fn command_placement_without_reservation_is_none() {
        let svc = minimal_command_service("ext", vec!["external".into()]);
        // The fixture's allocation_mode defaults to `None` (no reservation).
        let out =
            preview_command_placement(&svc, &gpus_with_free(24), &AllocationTable::new(), false);
        assert!(out.is_none());
    }

    /// An override larger than the card can't fit even on bare hardware.
    #[test]
    fn override_placement_too_large_does_not_fit() {
        let mut svc = minimal_service("ov");
        svc.placement_override.clear();
        svc.placement_override.insert(DeviceSlot::Gpu(0), 30_000); // 30 GB > 24 GiB
        let out =
            preview_override_placement(&svc, &gpus_with_free(24), &AllocationTable::new(), false);
        assert_eq!(out.verdict, FitVerdict::DoesNotFit);
    }
}
