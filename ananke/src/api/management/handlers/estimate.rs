//! Estimate lookup and placement-preview projection shared by the service
//! handlers.

use ananke_api::services::detail::{DevicePlacement, PlacementPreview};
use ananke_estimate::estimate_with_summary;
use smol_str::SmolStr;
use tracing::warn;

use crate::{
    config::ServiceConfig,
    daemon::app_state::AppState,
    supervise::estimate_cache::{EstimateCacheEntry, build_cache_entry},
};

/// Look up the cached `(ModelInfo, EstimateSummary)` for a service,
/// computing them on cache miss. Returns `(None, None)` for command-
/// template services and for llama-cpp services whose GGUF can't be
/// read — errors are logged on the daemon side; the frontend just
/// sees the absence.
pub(crate) fn model_estimate_entry(
    state: &AppState,
    svc_cfg: &ServiceConfig,
) -> Option<EstimateCacheEntry> {
    svc_cfg.llama_cpp()?;
    // Build the inputs once so the fingerprint we compare against is
    // identical to the one `compute_estimate_entry` would write into
    // the cache on miss.
    let inputs = crate::config::estimator_inputs(svc_cfg)
        .map(|i| i.with_visible_devices(state.snapshot.read().gpus.len() as u32))?;
    let fingerprint = inputs.config_fingerprint();
    let lc = svc_cfg.llama_cpp()?;
    let svc_name = svc_cfg.name.clone();

    if let Some(entry) = state.estimate_cache.get(
        &svc_name,
        lc.model.as_path(),
        lc.mmproj.as_deref(),
        fingerprint,
    ) {
        return Some(entry);
    }
    let entry = compute_estimate_entry(state, svc_cfg)?;
    state.estimate_cache.insert(svc_name, entry.clone());
    Some(entry)
}

/// Project a service's placement to the wire `PlacementPreview`. A manual
/// An active service shows the allocation it actually holds (its live pledge);
/// otherwise this is a what-if: a `placement_override` is honoured verbatim, a
/// command-template service picks a GPU dynamically, and the rest run the
/// estimator-path packer against the live snapshot and pledge book. Returns
/// `None` when there is nothing to show — a llama-cpp service whose GGUF
/// couldn't be read, or a command service that reserves nothing.
pub(crate) fn placement_preview(
    state: &AppState,
    svc_cfg: &ServiceConfig,
    estimate: Option<&ananke_estimate::Estimate>,
    running: bool,
) -> Option<PlacementPreview> {
    let snapshot = state.snapshot.read().clone();
    let table = state.allocations.lock().clone();
    let candidates = crate::supervise::preview::eviction_candidates(
        &svc_cfg.name,
        &state.supervisor_deps(),
        &table,
    );
    let context = crate::supervise::preview::PreviewContext {
        running,
        candidates: &candidates,
    };

    // An active service holds a real pledge — show that, not a re-computed
    // what-if (which could even differ from where it actually landed).
    let live_pledge = running
        .then(|| table.get(&svc_cfg.name).cloned())
        .flatten()
        .filter(|row| !row.is_empty());

    let outcome = if let Some(row) = live_pledge {
        let devices = row
            .into_iter()
            .map(|(slot, mb)| {
                let id = match slot {
                    crate::config::DeviceSlot::Cpu => ananke_placement::devices::DeviceId::Cpu,
                    crate::config::DeviceSlot::Gpu(n) => {
                        ananke_placement::devices::DeviceId::Gpu(n)
                    }
                };
                (id, mb.saturating_mul(1024 * 1024))
            })
            .collect();
        crate::supervise::PlacementOutcome {
            devices,
            verdict: ananke_api::internal::fit_verdict::FitVerdict::Fits,
            expert_offload_bytes: 0,
            expert_offload_layers: 0,
        }
    } else if !svc_cfg.placement_override.is_empty() {
        crate::supervise::preview_override_placement(svc_cfg, &snapshot, &table, context)
    } else if matches!(svc_cfg.template(), crate::config::Template::Command) {
        // Command-template service picking a GPU dynamically (e.g. ComfyUI):
        // `None` means it reserves nothing, so there is nothing to show.
        crate::supervise::preview_command_placement(svc_cfg, &snapshot, &table, context)?
    } else {
        let est = estimate?;
        // Match the supervisor: pack with the service's learned corrections.
        let corrections = state.rolling.get(&svc_cfg.name).corrections();
        crate::supervise::preview_placement(svc_cfg, est, &snapshot, &table, context, corrections)
    };

    // A dynamic command service can grow past its reserved floor up to its
    // configured maximum; every other service is pinned at `bytes`.
    let growth_ceiling = match svc_cfg.allocation_mode {
        crate::config::AllocationMode::Dynamic { max_mb, .. }
            if matches!(svc_cfg.template(), crate::config::Template::Command) =>
        {
            Some(max_mb.saturating_mul(1024 * 1024))
        }
        _ => None,
    };
    let expert_offload_bytes = outcome.expert_offload_bytes;
    let expert_offload_layers = outcome.expert_offload_layers;
    let devices = outcome
        .devices
        .into_iter()
        .map(|(id, bytes)| {
            let slot = match id {
                ananke_placement::devices::DeviceId::Cpu => crate::config::DeviceSlot::Cpu,
                ananke_placement::devices::DeviceId::Gpu(n) => crate::config::DeviceSlot::Gpu(n),
            };
            let total_bytes = snapshot.total_bytes(&slot).unwrap_or(0);
            let used = total_bytes.saturating_sub(snapshot.free_bytes(&slot).unwrap_or(0));
            // For a running service its own resident VRAM is already counted in
            // `used`; subtract this service's share so the bar doesn't double it.
            let used_by_others_bytes = if running {
                used.saturating_sub(bytes)
            } else {
                used
            };
            let max_bytes = growth_ceiling.map(|c| c.max(bytes)).unwrap_or(bytes);
            DevicePlacement {
                device: id.as_display(),
                bytes,
                max_bytes,
                used_by_others_bytes,
                total_bytes,
            }
        })
        .collect();
    Some(PlacementPreview {
        devices,
        verdict: outcome.verdict,
        expert_offload_bytes,
        expert_offload_layers,
    })
}

/// Run the estimator against the service's configured paths and
/// project the result through the shared `build_cache_entry`
/// constructor. Returns `None` when the GGUF can't be read or the
/// estimator refuses the architecture.
fn compute_estimate_entry(state: &AppState, svc_cfg: &ServiceConfig) -> Option<EstimateCacheEntry> {
    let lc = svc_cfg.llama_cpp()?;
    let inputs = crate::config::estimator_inputs(svc_cfg)
        .map(|i| i.with_visible_devices(state.snapshot.read().gpus.len() as u32))?;
    let config_fingerprint = inputs.config_fingerprint();
    let model_path = lc.model.clone();
    let mmproj_path = lc.mmproj.clone();

    match estimate_with_summary(state.system.fs.as_ref(), &inputs) {
        Ok((summary, estimate)) => Some(build_cache_entry(
            &summary,
            &estimate,
            model_path,
            mmproj_path,
            config_fingerprint,
        )),
        Err(e) => {
            warn!(service = %svc_cfg.name, error = %e, "model_info: estimator failed");
            None
        }
    }
}

pub(crate) fn read_current_allocation(
    state: &AppState,
    name: &SmolStr,
) -> std::collections::BTreeMap<String, u64> {
    let table = state.allocations.lock();
    let Some(row) = table.get(name) else {
        return std::collections::BTreeMap::new();
    };
    row.iter()
        .map(|(slot, mb)| {
            let key = match slot {
                crate::config::DeviceSlot::Cpu => "cpu".to_string(),
                crate::config::DeviceSlot::Gpu(n) => format!("gpu:{n}"),
            };
            (key, *mb)
        })
        .collect()
}
