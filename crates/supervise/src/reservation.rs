//! [`RunLoop`]'s reservation-map computation: running the estimator + packer
//! (or the command-template GPU pick) to decide what a spawn would reserve,
//! without touching the allocation table.

use ananke_placement::service_inputs::placement_inputs;

use crate::{
    RunLoop,
    ensure::{MisconfiguredKind, ReservationFailure, pack_err_to_reservation_failure},
};

impl RunLoop {
    /// Determine the reservation map for an Ensure. On the llama-cpp path this
    /// runs the estimator + packer and caches `Packed` on `self` for the
    /// eventual `render_argv` call. Typed `Err` so the caller can branch on
    /// pack failures (retry with eviction) vs config / estimator failures
    /// (surface verbatim).
    pub(crate) fn compute_reservation_map(
        &mut self,
        snap: &ananke_devices::DeviceSnapshot,
        table: &ananke_allocator::AllocationTable,
    ) -> Result<std::collections::BTreeMap<ananke_config::DeviceSlot, u64>, ReservationFailure>
    {
        let current = self.current_svc();
        let svc = &current;
        // An explicit allocation mode replaces the estimator outright: the
        // operator states the reservation and the service is placed from that
        // figure alone. Every command service is here, because ananke does not
        // build its argv; a llama-cpp service is only here when its
        // architecture is one the estimator refuses, which leaves the operator
        // nothing else to place it with.
        if !matches!(svc.allocation_mode, ananke_config::AllocationMode::None) {
            return self.compute_command_reservation(svc, snap, table);
        }
        if !svc.placement_override.is_empty() {
            self.packed_for_spawn = None;
            return Ok(svc.placement_override.clone());
        }

        // Estimator + placement path.
        let inputs = ananke_estimate::service_inputs::estimator_inputs(svc)
            .map(|i| i.with_visible_devices(snap.gpus.len() as u32))
            .ok_or(ReservationFailure::Misconfigured(
                MisconfiguredKind::NoModelPath,
            ))?;
        let fingerprint = inputs.config_fingerprint();
        let (summary, est) =
            ananke_estimate::estimate_with_summary(self.deps.system.fs.as_ref(), &inputs)
                .map_err(ReservationFailure::EstimatorError)?;
        // Warm the daemon-wide estimate cache with the *base* estimate
        // (pre-rolling-correction) and the GGUF summary we just
        // parsed. The management `ServiceDetail` handler reads this
        // cache instead of re-parsing the file on every detail poll;
        // populating it here turns the first detail-page view of a
        // running service into a cache hit. The cache stores the base
        // numbers because that's what the wire `EstimateSummary`
        // documents — the rolling correction applied below is a
        // supervisor-internal placement tweak, not a user-facing
        // estimate.
        let lc = svc.llama_cpp();
        if let Some(lc) = lc {
            self.deps.estimate_cache.insert(
                svc.name.clone(),
                crate::estimate_cache::build_cache_entry(
                    &summary,
                    &est,
                    lc.model.clone(),
                    lc.mmproj.clone(),
                    fingerprint,
                ),
            );
        }
        // Hand the service's learned per-pool corrections to the packer, which
        // scales every byte it charges to a device by that pool's factor.
        // `RollingCorrection::corrections` gates each factor to a neutral 1.0
        // until enough samples accumulate, so a single noisy early observation
        // can't over-pledge a shard past a GPU's capacity.
        let corrections = self.deps.rolling.get(&svc.name).corrections();
        let packed = ananke_allocator::placement::pack_corrected(
            &est,
            &placement_inputs(svc),
            snap,
            table,
            corrections,
            false,
        )
        .map_err(pack_err_to_reservation_failure)?;
        // Convert Allocation bytes (per-DeviceId, in bytes) to the
        // BTreeMap<DeviceSlot, u64> in MB that can_fit + insert expects.
        let want_mb: std::collections::BTreeMap<ananke_config::DeviceSlot, u64> = packed
            .allocation
            .bytes
            .iter()
            .map(|(id, bytes)| {
                let slot = match id {
                    ananke_devices::DeviceId::Cpu => ananke_config::DeviceSlot::Cpu,
                    ananke_devices::DeviceId::Gpu(n) => ananke_config::DeviceSlot::Gpu(*n),
                };
                // Convert bytes → MB, rounding up so we never under-reserve.
                let mb = bytes.div_ceil(1024 * 1024);
                (slot, mb)
            })
            .collect();
        self.packed_for_spawn = Some(packed);
        Ok(want_mb)
    }

    /// Reservation-map computation for command-template services.
    ///
    /// Picks the GPU with the most available headroom (subject to
    /// `gpu_allow`) via `placement::pick_command_gpu`, falling back to
    /// `PackError::WeightsDoNotFit` when nothing fits — that variant is the
    /// trigger for the supervisor's eviction-retry path. The chosen
    /// allocation is stashed on `packed_for_spawn` so `handle_active_lifecycle`
    /// renders `CUDA_VISIBLE_DEVICES` against the actual pick instead of the
    /// always-empty `init.allocation` (which is built from
    /// `placement_override` only).
    pub(crate) fn compute_command_reservation(
        &mut self,
        svc: &ananke_config::ServiceConfig,
        snap: &ananke_devices::DeviceSnapshot,
        table: &ananke_allocator::AllocationTable,
    ) -> Result<std::collections::BTreeMap<ananke_config::DeviceSlot, u64>, ReservationFailure>
    {
        self.packed_for_spawn = None;
        let (min_mb, prefer_mb) = match svc.allocation_mode {
            ananke_config::AllocationMode::Static { reserve_mb } => (reserve_mb, Some(reserve_mb)),
            ananke_config::AllocationMode::Dynamic { min_mb, max_mb, .. } => (min_mb, Some(max_mb)),
            ananke_config::AllocationMode::None => (0, None),
        };
        let mut map = std::collections::BTreeMap::new();
        if min_mb == 0 {
            // No reservation requested. Still publish an empty Packed so the
            // spawn path renders a deterministic (empty) CUDA_VISIBLE_DEVICES.
            let alloc = ananke_devices::Allocation::from_override(&map);
            self.packed_for_spawn = Some(ananke_allocator::placement::Packed {
                allocation: alloc,
                args: ananke_allocator::placement::CommandArgs::default(),
                expert_offload_bytes: 0,
                expert_offload_layers: 0,
                // A command service's reservation is the operator's declared
                // `min_mb`, not an estimate, so there is nothing for the
                // rolling correction to learn about. Zero bases skip it.
                rolling: ananke_allocator::placement::RollingInputs::default(),
            });
            return Ok(map);
        }
        // Operator pinned the layout explicitly (e.g. multi-GPU vLLM with
        // TP=2). Honour the per-device pledge verbatim instead of trying to
        // land the whole `min_mb` on a single GPU.
        if !svc.placement_override.is_empty() {
            ananke_allocator::placement::check_command_placement_override(
                &placement_inputs(svc),
                snap,
                table,
                false,
            )
            .map_err(ReservationFailure::PackFailed)?;
            map = svc.placement_override.clone();
            let alloc = ananke_devices::Allocation::from_override(&map);
            self.packed_for_spawn = Some(ananke_allocator::placement::Packed {
                allocation: alloc,
                args: ananke_allocator::placement::CommandArgs::default(),
                expert_offload_bytes: 0,
                expert_offload_layers: 0,
                // A command service's reservation is the operator's declared
                // `min_mb`, not an estimate, so there is nothing for the
                // rolling correction to learn about. Zero bases skip it.
                rolling: ananke_allocator::placement::RollingInputs::default(),
            });
            return Ok(map);
        }
        let slot = if matches!(
            svc.placement_policy,
            ananke_config::PlacementPolicy::CpuOnly
        ) {
            ananke_config::DeviceSlot::Cpu
        } else {
            match ananke_allocator::placement::pick_command_gpu(
                &placement_inputs(svc),
                snap,
                table,
                min_mb,
                prefer_mb,
                false,
            ) {
                Some(id) => ananke_config::DeviceSlot::Gpu(id),
                None if snap.gpus.is_empty() => {
                    // No GPUs visible at all (typical in tests with a CPU-only
                    // snapshot). Fall back to CPU so the reservation lands
                    // somewhere rather than failing the pack outright.
                    ananke_config::DeviceSlot::Cpu
                }
                None => {
                    return Err(ReservationFailure::PackFailed(
                        ananke_allocator::placement::PackError::WeightsDoNotFit {
                            shortfalls: ananke_allocator::placement::command_gpu_shortfalls(
                                &placement_inputs(svc),
                                snap,
                                table,
                                min_mb,
                                false,
                            ),
                        },
                    ));
                }
            }
        };
        map.insert(slot.clone(), min_mb);
        let alloc = ananke_devices::Allocation::from_override(&map);
        self.packed_for_spawn = Some(ananke_allocator::placement::Packed {
            allocation: alloc,
            args: ananke_allocator::placement::CommandArgs::default(),
            expert_offload_bytes: 0,
            expert_offload_layers: 0,
            // See the `min_mb == 0` arm above: an operator-declared reservation
            // is not an estimate.
            rolling: ananke_allocator::placement::RollingInputs::default(),
        });
        Ok(map)
    }
}
