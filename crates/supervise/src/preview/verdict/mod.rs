//! Placement verdicts for the estimator, override, and command-template paths.

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;

use ananke_allocator::{
    AllocationTable,
    eviction::{EvictionCandidate, plan_eviction},
    placement::{self, PackError},
};
use ananke_api::internal::fit_verdict::{DeviceShortfall, FitVerdict};
use ananke_config::{AllocationMode, DeviceSlot, PlacementPolicy, ServiceConfig};
use ananke_devices::{DeviceId, DeviceSnapshot};
use ananke_estimate::Estimate;
use ananke_placement::service_inputs::placement_inputs;
use ananke_tracking::rolling::Corrections;

use crate::preview::PlacementOutcome;

#[derive(Clone, Copy, Default)]
pub struct PreviewContext<'a> {
    pub running: bool,
    pub candidates: &'a [EvictionCandidate],
}

/// Check strict placement, then eligible victim prefixes in eviction order.
/// Running services always report `Fits`; corrections match the supervisor.
pub fn preview_placement(
    svc: &ServiceConfig,
    est: &Estimate,
    snapshot: &DeviceSnapshot,
    table: &AllocationTable,
    context: PreviewContext<'_>,
    corrections: Corrections,
) -> PlacementOutcome {
    let result = plan_eviction(
        snapshot,
        table,
        context.candidates,
        svc,
        |snap, remaining| {
            let packed = placement::pack_corrected(
                est,
                &placement_inputs(svc),
                snap,
                remaining,
                corrections,
                false,
            )?;
            let want = packed
                .allocation
                .bytes
                .iter()
                .map(|(device, bytes)| (device.to_slot(), bytes.div_ceil(1024 * 1024)))
                .collect();
            ananke_allocator::can_fit(&want, snap, remaining, Some(&svc.name)).map_err(|err| {
                PackError::WeightsDoNotFit {
                    shortfalls: vec![placement::DeviceShortfall {
                        device: DeviceId::from_slot(&err.slot),
                        requested_bytes: err.needed_bytes,
                        available_bytes: err.available_bytes,
                    }],
                }
            })?;
            Ok(packed)
        },
    );
    let verdict = verdict(&result, context.running);
    let packed = result.ok().map(|(packed, _)| packed).or_else(|| {
        placement::pack_corrected(
            est,
            &placement_inputs(svc),
            snapshot,
            table,
            corrections,
            true,
        )
        .or_else(|_| {
            placement::pack_corrected(
                est,
                &placement_inputs(svc),
                snapshot,
                &AllocationTable::new(),
                corrections,
                true,
            )
        })
        .ok()
    });
    let (devices, expert_offload_bytes, expert_offload_layers) = packed
        .map(|packed| {
            (
                packed.allocation.bytes,
                packed.expert_offload_bytes,
                packed.expert_offload_layers,
            )
        })
        .unwrap_or_default();
    PlacementOutcome {
        devices,
        verdict,
        expert_offload_bytes,
        expert_offload_layers,
    }
}

/// Honour the declared per-device split and test achievable reclamation.
pub fn preview_override_placement(
    svc: &ServiceConfig,
    snapshot: &DeviceSnapshot,
    table: &AllocationTable,
    context: PreviewContext<'_>,
) -> PlacementOutcome {
    let result = plan_eviction(
        snapshot,
        table,
        context.candidates,
        svc,
        |snap, remaining| {
            placement::check_command_placement_override(
                &placement_inputs(svc),
                snap,
                remaining,
                false,
            )?;
            ananke_allocator::can_fit(&svc.placement_override, snap, remaining, Some(&svc.name))
                .map_err(|err| PackError::WeightsDoNotFit {
                    shortfalls: vec![placement::DeviceShortfall {
                        device: DeviceId::from_slot(&err.slot),
                        requested_bytes: err.needed_bytes,
                        available_bytes: err.available_bytes,
                    }],
                })
        },
    );
    PlacementOutcome {
        devices: svc
            .placement_override
            .iter()
            .map(|(slot, mib)| (DeviceId::from_slot(slot), mib.saturating_mul(1024 * 1024)))
            .collect(),
        verdict: verdict(&result, context.running),
        expert_offload_bytes: 0,
        expert_offload_layers: 0,
    }
}

/// Reserve the command's floor on an allowed device, or return `None` without a reservation.
pub fn preview_command_placement(
    svc: &ServiceConfig,
    snapshot: &DeviceSnapshot,
    table: &AllocationTable,
    context: PreviewContext<'_>,
) -> Option<PlacementOutcome> {
    let (min_mb, prefer_mb) = match svc.allocation_mode {
        AllocationMode::Static { reserve_mb } => (reserve_mb, Some(reserve_mb)),
        AllocationMode::Dynamic { min_mb, max_mb, .. } => (min_mb, Some(max_mb)),
        AllocationMode::None => return None,
    };
    if min_mb == 0 {
        return None;
    }
    let result = plan_eviction(
        snapshot,
        table,
        context.candidates,
        svc,
        |snap, remaining| command_device(svc, snap, remaining, min_mb, prefer_mb, false),
    );
    let verdict = verdict(&result, context.running);
    let device = result.ok().map(|(device, _)| device).or_else(|| {
        command_device(
            svc,
            snapshot,
            &AllocationTable::new(),
            min_mb,
            prefer_mb,
            true,
        )
        .ok()
    });
    Some(PlacementOutcome {
        devices: device
            .into_iter()
            .map(|device| (device, min_mb.saturating_mul(1024 * 1024)))
            .collect(),
        verdict,
        expert_offload_bytes: 0,
        expert_offload_layers: 0,
    })
}

fn command_device(
    svc: &ServiceConfig,
    snapshot: &DeviceSnapshot,
    table: &AllocationTable,
    min_mb: u64,
    prefer_mb: Option<u64>,
    optimistic: bool,
) -> Result<DeviceId, PackError> {
    if matches!(svc.placement_policy, PlacementPolicy::CpuOnly) || snapshot.gpus.is_empty() {
        let want = BTreeMap::from([(DeviceSlot::Cpu, min_mb)]);
        ananke_allocator::can_fit(&want, snapshot, table, Some(&svc.name)).map_err(|err| {
            PackError::WeightsDoNotFit {
                shortfalls: vec![placement::DeviceShortfall {
                    device: DeviceId::Cpu,
                    requested_bytes: err.needed_bytes,
                    available_bytes: err.available_bytes,
                }],
            }
        })?;
        return Ok(DeviceId::Cpu);
    }
    placement::pick_command_gpu(
        &placement_inputs(svc),
        snapshot,
        table,
        min_mb,
        prefer_mb,
        optimistic,
    )
    .map(DeviceId::Gpu)
    .ok_or_else(|| PackError::WeightsDoNotFit {
        shortfalls: placement::command_gpu_shortfalls(
            &placement_inputs(svc),
            snapshot,
            table,
            min_mb,
            optimistic,
        ),
    })
}

fn verdict<T>(
    result: &Result<(T, Vec<smol_str::SmolStr>), PackError>,
    running: bool,
) -> FitVerdict {
    if running {
        return FitVerdict::Fits;
    }
    match result {
        Ok((_, victims)) if victims.is_empty() => FitVerdict::Fits,
        Ok(_) => FitVerdict::NeedsEviction,
        Err(err) => does_not_fit(err),
    }
}

fn does_not_fit(err: &PackError) -> FitVerdict {
    FitVerdict::DoesNotFit {
        shortfalls: err
            .shortfalls()
            .into_iter()
            .map(|shortfall| DeviceShortfall {
                device: shortfall.device.as_display(),
                requested_bytes: shortfall.requested_bytes,
                available_bytes: shortfall.available_bytes,
            })
            .collect(),
    }
}
