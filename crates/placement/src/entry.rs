//! Public entry points: `pack` and `pack_optimistic` drive the packer
//! through its steps in order and return the finished `Packed` result.

use ananke_config::placement::{DeviceSlot, PlacementInputs};
use ananke_estimate::Estimate;

use crate::{
    AllocationTable, Corrections, Packed, devices::DeviceSnapshot, packer::Packer, types::PackError,
};

/// Number of per-layer-equivalents added to every active backend as slop
/// tolerance for tensor-split rounding. Bumped if empirical overruns
/// show tensor_split's remainder exceeds one layer's worth.
pub(crate) const ONE_LAYER_FUDGE_MULTIPLIER: u64 = 1;

/// `-ngl` value meaning "offload every layer to the GPU". Used when we
/// reserved whole-model space on a GPU without per-layer detail.
pub(crate) const NGL_OFFLOAD_ALL: u32 = 999;

/// `-ngl` value meaning "run entirely on CPU".
pub(crate) const NGL_CPU_ONLY: u32 = 0;

/// What a pack is being asked to compute.
///
/// The three modes share every byte-accounting step; they differ only in how
/// much capacity the devices are treated as having. Keeping them one code path
/// is deliberate — see [`PackMode::Demand`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackMode {
    /// Place against currently-free capacity: `min(nvml_free, total - pledged)`.
    /// What the daemon checks before deciding to evict.
    Strict,
    /// Place trusting the pledge book alone (`total - pledged`). Intended for
    /// the retry-after-eviction path, where victims have been removed from
    /// `reserved` to model "if they were gone" — nvml still shows their
    /// realized usage until drains actually land.
    Optimistic,
    /// Not a placement: how much memory the model would need if the host were
    /// large enough to take it.
    ///
    /// The GPUs keep their real capacity, so the layer walk spans the same
    /// cards it really would and each spanned device is charged its real
    /// compute buffer. Only the host is treated as unbounded: CPU spill is
    /// forced on and the host-RAM gate is skipped, which is what makes the
    /// pack succeed for a model that is merely too big for this machine.
    ///
    /// Only [`Packed::allocation`] is meaningful for this mode. The
    /// `CommandArgs` it carries describe a layout that would never be spawned
    /// — `n_cpu_moe` excludes the whole-layer spills this mode permits, and
    /// `-ngl 999` is emitted alongside CPU-resident layers. Do not launch from
    /// a demand pack.
    ///
    /// This exists so the "would need" figure is produced by the packer rather
    /// than re-derived alongside it. A second implementation of the same
    /// accounting drifts silently, dropping terms like the head-vs-secondary
    /// logits trim, the CPU-side compute buffer, or the one-layer fudge.
    Demand,
}

/// Pack `estimate` onto allowed devices, respecting `policy`,
/// `override_tensor`, and live device capacity (`snapshot` minus any
/// already-reserved bytes from `reserved`).
///
/// Runs uncorrected. The daemon's spawn path uses [`pack_corrected`] instead,
/// so a service's learned per-pool corrections reach the placement; callers
/// that describe the model rather than a specific service's history (previews
/// of bare hardware, tests) want this one.
pub fn pack(
    estimate: &Estimate,
    placement: &PlacementInputs,
    snapshot: &DeviceSnapshot,
    reserved: &AllocationTable,
) -> Result<Packed, PackError> {
    pack_inner(
        estimate,
        placement,
        snapshot,
        reserved,
        PackMode::Strict,
        Corrections::NEUTRAL,
    )
}

/// [`pack`] / [`pack_optimistic`] with the service's learned rolling
/// corrections applied to every byte charged to a device.
///
/// `optimistic` selects the capacity view exactly as the two neutral entry
/// points do. The returned [`Packed::rolling`] carries the uncorrected bases
/// this pack was built from, which the supervisor captures so the next
/// observation can be divided by what was predicted rather than by a base the
/// correction already moved.
pub fn pack_corrected(
    estimate: &Estimate,
    placement: &PlacementInputs,
    snapshot: &DeviceSnapshot,
    reserved: &AllocationTable,
    corrections: Corrections,
    optimistic: bool,
) -> Result<Packed, PackError> {
    let mode = if optimistic {
        PackMode::Optimistic
    } else {
        PackMode::Strict
    };
    pack_inner(estimate, placement, snapshot, reserved, mode, corrections)
}

/// See [`PackMode::Optimistic`].
pub fn pack_optimistic(
    estimate: &Estimate,
    placement: &PlacementInputs,
    snapshot: &DeviceSnapshot,
    reserved: &AllocationTable,
) -> Result<Packed, PackError> {
    pack_inner(
        estimate,
        placement,
        snapshot,
        reserved,
        PackMode::Optimistic,
        Corrections::NEUTRAL,
    )
}

/// Total bytes the model would occupy across every device, ignoring whether
/// the host can actually hold it. See [`PackMode::Demand`].
///
/// Packs against bare hardware (an empty pledge book), so the figure describes
/// the model and the machine's shape, not the current tenancy. `Err` when even
/// an unbounded host can't produce a layout — a config error, a tensor-split
/// that doesn't fit, or no eligible device — in which case there is no honest
/// number to report and the caller should render nothing rather than a guess.
pub fn pack_demand(
    estimate: &Estimate,
    placement: &PlacementInputs,
    snapshot: &DeviceSnapshot,
    corrections: Corrections,
) -> Result<Packed, PackError> {
    pack_inner(
        estimate,
        placement,
        snapshot,
        &AllocationTable::new(),
        PackMode::Demand,
        corrections,
    )
}

fn pack_inner(
    estimate: &Estimate,
    placement: &PlacementInputs,
    snapshot: &DeviceSnapshot,
    reserved: &AllocationTable,
    mode: PackMode,
    corrections: Corrections,
) -> Result<Packed, PackError> {
    let mut packer = Packer::new(estimate, placement, snapshot, reserved, mode, corrections);
    // Sharded (tensor/row) split distributes every layer across all spanned
    // GPUs in parallel — a fundamentally different shape from the first-fit
    // layer walk. Taken only when the service opts in, at least two GPUs are
    // available to span, and the estimator gave a per-layer breakdown to
    // shard. Otherwise fall through to the layer path: a single-GPU "tensor
    // split" is just an ordinary placement, and a model with no per-layer
    // detail cannot be evenly sharded.
    if packer.placement.split_mode.is_sharded()
        && packer.allowed_gpus.len() >= 2
        && !packer.per_layer.is_empty()
    {
        packer.distribute_sharded()?;
        if mode != PackMode::Demand {
            packer.check_gpu_capacity()?;
            packer.check_cpu_capacity()?;
        }
        return Ok(packer.finish());
    }
    packer.seed_non_layer();
    packer.seed_mtp_overhead();
    if packer.expert_aware {
        // Two-phase MoE placement: pin every layer's attention + KV on a GPU,
        // then offload the trailing surplus expert *layers* to CPU as whole
        // units via `--n-cpu-moe`. Whole-layer offload (rather than per-tensor
        // `-ot`) keeps the runtime's fused multi-threaded CPU MoE kernel
        // engaged and stays under llama.cpp's graph-split limit.
        packer.initialise_gpu_remaining();
        packer.distribute_experts_ncmoe()?;
        if packer.expert_tensor_split.is_some() {
            for gpu in &packer.allowed_gpus {
                packer.per_device.entry(DeviceSlot::Gpu(*gpu)).or_default();
            }
        }
    } else {
        packer.walk_layers()?;
        packer.place_fallback_weights()?;
    }
    packer.add_kv_bytes();
    packer.add_compute_buffer();
    packer.add_one_layer_fudge();
    // Demand describes memory requirements rather than admitting a launch.
    if mode != PackMode::Demand {
        packer.check_gpu_capacity()?;
        packer.check_cpu_capacity()?;
    }
    Ok(packer.finish())
}

#[cfg(test)]
mod tests {
    use ananke_config::placement::{
        DeviceSlot, OffloadMode, PlacementInputs, PlacementPolicy, SplitMode,
    };

    use super::*;
    use crate::{
        devices::{CpuSnapshot, DeviceId},
        test_support::{GIB, MIB, cpu_bytes, moe_estimate, moe_svc, snapshot, trivial_estimate},
    };

    #[test]
    fn expert_split_retains_unused_first_and_middle_visible_devices() {
        for unused in [0, 1] {
            let estimate = moe_estimate(20, 100, 300);
            let mut snap = snapshot(&[24, 24, 24]);
            snap.gpus[unused].free_bytes = 600 * MIB;
            let packed = pack(
                &estimate,
                &moe_svc(OffloadMode::Auto),
                &snap,
                &AllocationTable::new(),
            )
            .unwrap();
            let split = packed.args.tensor_split.as_ref().unwrap();
            assert_eq!(split[unused], 0, "the constrained GPU carries no layers");
            assert_eq!(packed.allocation.gpu_ids(), vec![0, 1, 2]);
            assert_eq!(split.len(), packed.allocation.gpu_ids().len());
            assert_eq!(
                packed.allocation.bytes[&DeviceId::Gpu(unused as u32)],
                500 * MIB
            );
            for gpu in &snap.gpus {
                assert!(packed.allocation.bytes[&DeviceId::Gpu(gpu.id)] <= gpu.free_bytes);
            }
        }
    }

    #[test]
    fn fixed_head_cost_cannot_overflow_a_gpu_while_layers_fit_elsewhere() {
        let mut estimate = trivial_estimate(4, 256);
        estimate.layout.non_layer.output_head_bytes = 2 * GIB;
        let placement = PlacementInputs::named("fixed-head");
        let snap = snapshot(&[1, 24]);
        let error = pack(&estimate, &placement, &snap, &AllocationTable::new()).unwrap_err();
        let shortfalls = error.shortfalls();
        assert_eq!(shortfalls.len(), 1);
        assert_eq!(shortfalls[0].device, DeviceId::Gpu(0));
        assert_eq!(shortfalls[0].available_bytes, GIB);
        assert!(shortfalls[0].requested_bytes > 2 * GIB);
        assert!(pack_optimistic(&estimate, &placement, &snap, &AllocationTable::new()).is_ok());
    }

    #[test]
    fn runtime_costs_on_a_head_without_layers_are_capacity_checked() {
        let mut estimate = trivial_estimate(4, 256);
        estimate.layout.non_layer.output_head_bytes = 800 * MIB;
        let placement = PlacementInputs::named("runtime");
        let snap = snapshot(&[1, 24]);
        let error = pack(&estimate, &placement, &snap, &AllocationTable::new()).unwrap_err();
        assert!(matches!(
            error,
            PackError::GpuDoesNotFit { gpu_index: 0, .. }
        ));
    }

    #[test]
    fn final_gpu_gate_includes_headroom_reservations_and_corrections() {
        let mut estimate = trivial_estimate(4, 256);
        estimate.layout.non_layer.output_head_bytes = 700 * MIB;
        let mut placement = PlacementInputs::named("headroom");
        placement.gpu_headroom_mb = 512;
        let snap = snapshot(&[24, 24]);
        let mut reserved = AllocationTable::new();
        reserved.insert("peer".into(), [(DeviceSlot::Gpu(0), 22 * 1024)].into());
        assert!(pack(&estimate, &placement, &snap, &reserved).is_ok());
        assert!(
            pack_corrected(
                &estimate,
                &placement,
                &snap,
                &AllocationTable::new(),
                Corrections {
                    vram: 1.25,
                    host: 1.0
                },
                false,
            )
            .is_ok()
        );
        let error = pack_corrected(
            &estimate,
            &placement,
            &snap,
            &reserved,
            Corrections {
                vram: 1.25,
                host: 1.0,
            },
            false,
        )
        .unwrap_err();
        let shortfalls = error.shortfalls();
        assert_eq!(shortfalls[0].device, DeviceId::Gpu(0));
        assert_eq!(shortfalls[0].available_bytes, 1536 * MIB);
        assert!(shortfalls[0].requested_bytes > shortfalls[0].available_bytes);
    }

    #[test]
    fn a_sharded_pack_checks_host_capacity_after_charging_embeddings() {
        let mut estimate = trivial_estimate(4, 256);
        estimate.layout.non_layer.token_embd_bytes = 3 * GIB;
        let mut placement = PlacementInputs::named("sharded");
        placement.split_mode = SplitMode::Tensor;
        let mut snap = snapshot(&[24, 24]);
        snap.cpu = Some(CpuSnapshot {
            total_bytes: 2 * GIB,
            available_bytes: 2 * GIB,
        });
        assert!(matches!(
            pack(&estimate, &placement, &snap, &AllocationTable::new()),
            Err(PackError::CpuDoesNotFit { .. })
        ));
        assert!(pack_demand(&estimate, &placement, &snap, Corrections::NEUTRAL).is_ok());
    }

    #[test]
    fn a_vision_graph_reserves_head_capacity_before_the_layer_walk() {
        let mut estimate = trivial_estimate(4, 256);
        estimate.buffers.mmproj_graph_bytes = GIB;
        let mut placement = PlacementInputs::named("vision");
        placement.policy = PlacementPolicy::GpuOnly;
        let mut snap = snapshot(&[2]);
        snap.gpus[0].free_bytes = 2000 * MIB;
        assert!(matches!(
            pack(&estimate, &placement, &snap, &AllocationTable::new()),
            Err(PackError::LayerDoesNotFit { .. })
        ));
        placement.policy = PlacementPolicy::Hybrid;
        let packed = pack(&estimate, &placement, &snap, &AllocationTable::new()).unwrap();
        assert!(cpu_bytes(&packed) > 0);
        assert!(packed.allocation.bytes[&DeviceId::Gpu(0)] <= snap.gpus[0].free_bytes);
    }

    fn total(p: &Packed) -> u64 {
        p.allocation.bytes.values().sum()
    }

    /// The invariant the demand figure exists to hold: against a pack that
    /// shares its capacity view, `Demand` must agree to the byte. Any
    /// divergence means a step was skipped or double-counted — the class of
    /// error a second, hand-written aggregate cannot rule out, and which the
    /// logits trim, the CPU-side compute buffer, and the one-layer fudge are
    /// each easy to fall into.
    ///
    /// The comparison is against `pack_optimistic` on an empty table, *not*
    /// `pack`. Strict packing clamps to `min(free, total)`, so with a card in
    /// use it legitimately spans more devices — and therefore reserves more
    /// compute buffers — than a demand figure describing bare hardware. That
    /// is a real difference between the questions, not a bookkeeping error;
    /// asserting equality there would only hold on a fixture where
    /// `free == total`, which neutralises the very difference under test.
    #[test]
    fn demand_matches_a_real_pack_when_the_model_fits() {
        let e = trivial_estimate(20, 1024); // 20 layers × ~1 GiB.
        let svc = PlacementInputs {
            policy: PlacementPolicy::Hybrid,
            ..PlacementInputs::named("m")
        };
        // `free < total` on one card so the two modes' capacity views actually
        // differ. The stock fixture sets free == total, which collapses
        // `Strict`'s `min(free, total)` into `Demand`'s `total` and makes the
        // assertion vacuous.
        let mut snap = snapshot(&[24, 24]);
        // On gpu:0 — the card the walk actually fills. Starving gpu:1 would
        // leave the divergence untested, since nothing lands there.
        snap.gpus[0].free_bytes = 21 * GIB;

        let bare = pack_optimistic(&e, &svc, &snap, &AllocationTable::new()).expect("fits");
        let demand = pack_demand(&e, &svc, &snap, Corrections::NEUTRAL)
            .expect("demand always resolves when it fits");
        assert_eq!(
            total(&demand),
            total(&bare),
            "demand must agree byte-for-byte with a pack sharing its capacity view"
        );
    }

    /// Same invariant on the expert-aware path, where the packer's accounting
    /// is most intricate (per-layer non-expert pinning, `--n-cpu-moe` offload,
    /// the retained-expert redistribution).
    #[test]
    fn demand_matches_a_real_pack_for_a_fitting_moe() {
        let e = moe_estimate(10, 100, 300);
        let svc = moe_svc(OffloadMode::Auto);
        let snap = snapshot(&[24]);

        let bare = pack_optimistic(&e, &svc, &snap, &AllocationTable::new()).expect("fits");
        let demand = pack_demand(&e, &svc, &snap, Corrections::NEUTRAL).expect("demand resolves");
        assert_eq!(total(&demand), total(&bare));
    }

    /// The case the figure exists for: a model whose host-RAM spill exceeds
    /// the machine. The real pack fails, so there is no per-device split to
    /// sum — but demand still reports the model's size rather than nothing.
    #[test]
    fn demand_resolves_when_the_host_is_too_small_to_place() {
        let e = trivial_estimate(60, 1024); // 60 GiB over two 24 GiB cards.
        let svc = PlacementInputs {
            policy: PlacementPolicy::Hybrid,
            ..PlacementInputs::named("m")
        };
        let mut snap = snapshot(&[24, 24]);
        snap.cpu = Some(CpuSnapshot {
            total_bytes: 2 * GIB,
            available_bytes: 2 * GIB,
        });

        assert!(
            pack(&e, &svc, &snap, &AllocationTable::new()).is_err(),
            "the premise: this model cannot be placed on this host"
        );
        let demand = pack_demand(&e, &svc, &snap, Corrections::NEUTRAL)
            .expect("demand ignores the host-RAM gate");
        assert!(
            total(&demand) >= 60 * GIB,
            "demand reports the whole model, got {}",
            total(&demand)
        );
        assert!(
            cpu_bytes(&demand) > 0,
            "the surplus the GPUs can't hold is charged to the host"
        );
    }

    /// Demand forces CPU spill on so a `GpuOnly` service still yields a figure.
    /// Without it the layer walk fails and the row falls back to reporting
    /// nothing — the "0 B" hole `Demand` exists to close.
    #[test]
    fn demand_resolves_for_a_gpu_only_service_that_overflows_its_cards() {
        let e = trivial_estimate(60, 1024);
        let svc = PlacementInputs {
            policy: PlacementPolicy::GpuOnly,
            ..PlacementInputs::named("m")
        };
        let snap = snapshot(&[24, 24]);

        assert!(
            pack(&e, &svc, &snap, &AllocationTable::new()).is_err(),
            "the premise: GpuOnly cannot spill, so this cannot be placed"
        );
        let demand = pack_demand(&e, &svc, &snap, Corrections::NEUTRAL)
            .expect("demand spills to a notional host");
        assert!(total(&demand) >= 60 * GIB, "got {}", total(&demand));
    }

    /// A `GpuOnly` service exercises the other axis the fitting-model parity
    /// test cannot: `Demand` forces `allow_cpu` on, so if the two modes ever
    /// diverge on a model that fits entirely on the cards, that flag is
    /// leaking into a case where nothing should spill.
    #[test]
    fn demand_matches_a_real_pack_for_a_gpu_only_service() {
        let e = trivial_estimate(10, 1024);
        let svc = PlacementInputs {
            policy: PlacementPolicy::GpuOnly,
            ..PlacementInputs::named("m")
        };
        let mut snap = snapshot(&[24, 24]);
        snap.gpus[0].free_bytes = 12 * GIB;

        let placed = pack(&e, &svc, &snap, &AllocationTable::new()).expect("fits on the cards");
        let demand = pack_demand(&e, &svc, &snap, Corrections::NEUTRAL).expect("demand resolves");
        assert_eq!(total(&demand), total(&placed));
        assert_eq!(
            cpu_bytes(&demand),
            cpu_bytes(&placed),
            "nothing should spill"
        );
    }

    /// The MoE shape from issue #29. The expert-aware path refuses whole-layer
    /// CPU spill for a real placement (the `-ngl 999` child would OOM), so
    /// without `Demand`'s exemption from that rule it returns `Err` and the row
    /// reports nothing — the "no figure for an unplaceable model" hole, for
    /// exactly the class of model the issue was filed about.
    ///
    /// The non-expert weight must exceed the cards' *physical total*, not
    /// merely their free bytes: `Demand` shares `Optimistic`'s `total -
    /// pledged` view, so a fixture that only starves `free_bytes` never
    /// reaches the spill arm and the test cannot fail.
    #[test]
    fn demand_resolves_for_a_moe_whose_attention_overflows_the_cards() {
        // 100 layers × 600 MiB of attention ≈ 58.6 GiB against 48 GiB of card.
        let e = moe_estimate(100, 600, 300);
        let svc = moe_svc(OffloadMode::Auto);
        let snap = snapshot(&[24, 24]);

        assert!(
            pack(&e, &svc, &snap, &AllocationTable::new()).is_err(),
            "the premise: non-expert weight is GPU-only and exceeds both cards"
        );
        let demand = pack_demand(&e, &svc, &snap, Corrections::NEUTRAL)
            .expect("demand still yields a figure for a MoE");
        assert!(
            cpu_bytes(&demand) > 0,
            "the attention that cannot fit is charged to the notional host"
        );
        // Weights are counted exactly once: the spilled layers carry their own
        // experts, and `distribute_experts_ncmoe` filters them out rather than
        // charging those experts a second time.
        let weights = e.weights_bytes;
        assert!(
            total(&demand) >= weights && total(&demand) < weights + 8 * GIB,
            "expected ~weights + per-device overhead, got {} vs weights {weights}",
            total(&demand)
        );
    }
}
