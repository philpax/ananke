//! Tests for the `--n-cpu-moe` expert-offload path in
//! [`crate::experts_ncmoe`]: auto and manual offload
//! counts, the pooled-overflow rejection, and multi-GPU expert spread.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use ananke_estimate::{
    Buffers, Estimate, EstimatorInputs, ExpertKind, ExpertTensor, Fork, IkAssignment, Layout,
    NonLayer,
};
use ananke_gguf::{Architecture, GgufSummary, GgufTensor, GgufType, GgufValue};

use super::*;
use crate::{
    AllocationTable,
    devices::{CpuSnapshot, DeviceId},
    entry::{NGL_OFFLOAD_ALL, pack},
    test_support::{GIB, MIB, cpu_bytes, moe_estimate, moe_svc, snapshot},
};

#[test]
fn experts_mainline_command_replay_matches_reservation_on_asymmetric_devices() {
    let e = moe_estimate(12, 120, 250);
    let svc = moe_svc(OffloadMode::Auto);
    let snap = snapshot(&[3, 8]);
    let packed = pack(&e, &svc, &snap, &AllocationTable::new()).unwrap();
    let split = packed.args.tensor_split.as_ref().unwrap();
    let sum = split.iter().sum::<u32>() as f32;
    let boundary = split[0] as f32 / sum;
    let n_cpu = packed.args.n_cpu_moe.unwrap_or(0);
    let mut expected = [0u64; 2];
    for layer in 0..12 {
        let home = usize::from(layer as f32 / 13.0 >= boundary);
        expected[home] += 120 * MIB;
        if layer >= n_cpu {
            expected[home] += 750 * MIB;
        }
    }
    for (gpu, bytes) in expected.iter_mut().enumerate() {
        *bytes += (e.buffers.compute_mb as u64 + 120) * MIB;
        assert_eq!(packed.allocation.bytes[&DeviceId::Gpu(gpu as u32)], *bytes);
        assert!(*bytes <= snap.gpus[gpu].free_bytes);
    }
    assert!(packed.args.override_tensor.is_empty());
}

#[test]
fn experts_manual_rejects_indivisible_layer_even_with_sufficient_combined_room() {
    let e = moe_estimate(2, 100, 1500);
    let snap = snapshot(&[4, 4]);
    let error = pack(
        &e,
        &moe_svc(OffloadMode::Layers(1)),
        &snap,
        &AllocationTable::new(),
    )
    .unwrap_err();
    assert!(matches!(error, PackError::ManualExpertsDoNotFit { .. }));
    assert!(
        error
            .shortfalls()
            .iter()
            .all(|s| s.requested_bytes > s.available_bytes)
    );
    let automatic = pack(
        &e,
        &moe_svc(OffloadMode::Auto),
        &snap,
        &AllocationTable::new(),
    )
    .unwrap();
    assert_eq!(automatic.args.n_cpu_moe, Some(2));
}

#[test]
fn experts_ik_offloads_device_local_suffixes_not_one_global_suffix() {
    let e = moe_estimate(12, 100, 200);
    let mut svc = moe_svc(OffloadMode::Layers(4));
    svc.ik_llama = true;
    let snap = snapshot(&[24, 24]);
    let alloc = AllocationTable::new();
    let mut packer = Packer::new(
        &e,
        &svc,
        &snap,
        &alloc,
        PackMode::Strict,
        crate::Corrections::NEUTRAL,
    );
    packer.seed_non_layer();
    packer.initialise_gpu_remaining();
    packer.distribute_experts_ncmoe().unwrap();
    for gpu in 0..2 {
        let hosted: Vec<u32> = packer
            .layer_home
            .iter()
            .filter_map(|(&l, &home)| (home == gpu).then_some(l))
            .collect();
        assert!(
            hosted
                .iter()
                .any(|l| packer.expert_offload_cpu_layers.contains(l))
        );
        assert!(
            hosted
                .iter()
                .any(|l| !packer.expert_offload_cpu_layers.contains(l))
        );
    }
    assert_eq!(packer.n_cpu_moe, Some(4));
    assert_eq!(packer.expert_offload_cpu_layers.len(), 4);
}

#[test]
fn experts_ik_conservative_reservations_keep_gpu_experts_and_bound_unknown_ranges() {
    let e = moe_estimate(12, 100, 200);
    let mut svc = moe_svc(OffloadMode::Auto);
    svc.ik_llama = true;
    let snap = snapshot(&[6, 6]);
    let packed = pack(&e, &svc, &snap, &AllocationTable::new()).unwrap();
    assert!(packed.expert_offload_layers > 0 && packed.expert_offload_layers < 12);
    let retained_bound = (12 - packed.args.n_cpu_moe.unwrap()) as u64 * 600 * MIB;
    for gpu in 0..2 {
        let bytes = packed.allocation.bytes[&DeviceId::Gpu(gpu)];
        assert!(bytes >= 1200 * MIB + retained_bound);
        assert!(bytes <= 6 * GIB);
    }
    assert!(packed.args.override_tensor.is_empty());
}

#[test]
fn experts_ik_eighty_nonexpert_layers_fit_two_six_gib_devices() {
    let mut e = moe_estimate(80, 100, 100);
    e.layout.ik_assignment = Some(IkAssignment {
        layer_weights: e.layout.per_layer_bytes.clone().unwrap(),
        cache_lower: vec![0; 80],
        cache_upper: vec![0; 80],
        output_weights: 0,
        logits_bytes: 128 * MIB,
        scratch_upper: 2 * GIB,
    });
    let mut svc = moe_svc(OffloadMode::Auto);
    svc.ik_llama = true;
    let packed = pack(&e, &svc, &snapshot(&[6, 6]), &AllocationTable::new()).unwrap();
    assert!(packed.args.n_cpu_moe.is_some());
    assert_eq!(packed.allocation.gpu_ids(), vec![0, 1]);
    for gpu in 0..2 {
        let reserved = packed.allocation.bytes[&DeviceId::Gpu(gpu)];
        assert!((4000 * MIB..=6 * GIB).contains(&reserved));
    }
}

#[test]
fn experts_ik_production_estimator_assignment_fits_eighty_layers() {
    let mut summary = GgufSummary {
        path: "/fake/moe.gguf".into(),
        total_tensor_bytes: 0,
        tensors: BTreeMap::new(),
        metadata: BTreeMap::new(),
        block_count: Some(80),
        architecture: Architecture::Qwen3Moe,
        shards: vec![],
    };
    for (key, value) in [
        ("embedding_length", 128),
        ("attention.head_count", 4),
        ("attention.head_count_kv", 1),
        ("expert_used_count", 2),
    ] {
        summary
            .metadata
            .insert(format!("qwen3moe.{key}").into(), GgufValue::U32(value));
    }
    for layer in 0..80 {
        for suffix in [
            "attn_q.weight",
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        ] {
            let name = format!("blk.{layer}.{suffix}");
            let tensor = GgufTensor {
                name: name.clone().into(),
                dtype: GgufType::F16,
                shape: vec![128, 128],
                byte_size: 100 * MIB,
                shard_idx: 0,
                offset: 0,
            };
            summary.total_tensor_bytes += tensor.byte_size;
            summary.tensors.insert(name.into(), tensor);
        }
    }
    summary.tensors.insert(
        "output.weight".into(),
        GgufTensor {
            name: "output.weight".into(),
            dtype: GgufType::F16,
            shape: vec![128, 512],
            byte_size: 128 * 512 * 2,
            shard_idx: 0,
            offset: 0,
        },
    );
    let inputs = EstimatorInputs {
        fork: Fork::Ik { dsa: false },
        context: 1025,
        ubatch: Some(32),
        compute_buffer_mb: Some(400),
        ..EstimatorInputs::empty(Path::new("/fake/moe.gguf"))
    };
    let estimate = ananke_estimate::dispatch(&summary, &inputs).unwrap();
    assert!(estimate.layout.ik_assignment.is_some());
    let mut placement = moe_svc(OffloadMode::Auto);
    placement.ik_llama = true;
    let packed = pack(
        &estimate,
        &placement,
        &snapshot(&[6, 6]),
        &AllocationTable::new(),
    )
    .unwrap();
    assert!(packed.args.n_cpu_moe.is_some());
    for gpu in 0..2 {
        assert!(packed.allocation.bytes[&DeviceId::Gpu(gpu)] <= 6 * GIB);
    }
}

#[test]
fn experts_ik_range_bounds_cover_independent_runtime_replays() {
    let n = 16;
    let weights: Vec<_> = (0..n).map(|i| (100 + i as u64 * 13) * MIB).collect();
    let experts: Vec<_> = weights.iter().map(|&w| w / 2).collect();
    let cache_upper: Vec<_> = (0..n)
        .map(|i| if i % 3 == 0 { 70 * MIB } else { 2 * MIB })
        .collect();
    let assignment = IkAssignment {
        layer_weights: weights.clone(),
        cache_lower: vec![0; n],
        cache_upper: cache_upper.clone(),
        output_weights: 99 * MIB,
        logits_bytes: 900 * MIB,
        scratch_upper: 3 * GIB,
    };
    for budgets in [
        vec![3 * GIB, 3 * GIB],
        vec![2 * GIB, 7 * GIB],
        vec![2 * GIB, 3 * GIB, 5 * GIB],
    ] {
        for n_cpu in [0, 3, 8, 16] {
            let input = plan::Inputs {
                weights: &weights,
                experts: &experts,
                budgets: &budgets,
                kv: cache_upper.iter().sum(),
                compute: 123 * MIB,
                output: assignment.output_weights,
                ik: true,
                n_cpu,
                assignment: Some(&assignment),
            };
            let result = plan::plan(&input);
            for scratch in [0, 1, 127 * MIB, 899 * MIB, 900 * MIB, 901 * MIB, 3 * GIB] {
                for cache_case in 0..4 {
                    let cache: Vec<_> = cache_upper
                        .iter()
                        .enumerate()
                        .map(|(i, &upper)| {
                            if cache_case == 0
                                || cache_case == 2 && i < n / 2
                                || cache_case == 3 && i >= n / 2
                            {
                                upper
                            } else {
                                0
                            }
                        })
                        .collect();
                    let homes = replay_ik_homes(&assignment, &cache, scratch, &result.split);
                    let cpu = replay_ik_cpu(&homes, budgets.len(), n_cpu);
                    let mut actual = vec![assignment.output_weights; budgets.len()];
                    for (layer, &home) in homes.iter().enumerate() {
                        actual[home] += weights[layer] + cache[layer]
                            - if cpu.contains(&(layer as u32)) {
                                experts[layer]
                            } else {
                                0
                            };
                    }
                    for (id, &bytes) in actual.iter().enumerate() {
                        assert!(
                            bytes <= result.costs[id],
                            "gpu={id} scratch={scratch} cache={cache_case} ncmoe={n_cpu}: {bytes} > {}",
                            result.costs[id]
                        );
                    }
                }
            }
        }
    }
}

fn replay_ik_cpu(homes: &[usize], devices: usize, n_cpu: u32) -> BTreeSet<u32> {
    let counts: Vec<_> = (0..devices)
        .map(|id| homes.iter().filter(|&&h| h == id).count())
        .collect();
    let wanted: Vec<_> = counts
        .iter()
        .map(|&n| n as f32 / homes.len() as f32 * n_cpu as f32)
        .collect();
    let mut quotas: Vec<_> = wanted.iter().map(|&x| x.round() as usize).collect();
    while quotas.iter().sum::<usize>() > n_cpu as usize {
        let id = (0..devices)
            .filter(|&i| quotas[i] > 0)
            .max_by(|&a, &b| {
                (quotas[a] as f32 - 1.0 - wanted[a])
                    .total_cmp(&(quotas[b] as f32 - 1.0 - wanted[b]))
                    .then_with(|| b.cmp(&a))
            })
            .unwrap();
        quotas[id] -= 1;
    }
    while quotas.iter().sum::<usize>() < n_cpu as usize {
        let id = (0..devices)
            .filter(|&i| quotas[i] < counts[i])
            .min_by(|&a, &b| {
                (quotas[a] as f32 + 1.0 - wanted[a])
                    .total_cmp(&(quotas[b] as f32 + 1.0 - wanted[b]))
                    .then_with(|| a.cmp(&b))
            })
            .unwrap();
        quotas[id] += 1;
    }
    let mut cpu = BTreeSet::new();
    for layer in (0..homes.len()).rev() {
        let id = homes[layer];
        if quotas[id] > 0 {
            quotas[id] -= 1;
            cpu.insert(layer as u32);
        }
    }
    cpu
}

fn replay_ik_homes(
    assignment: &IkAssignment,
    cache: &[u64],
    scratch: u64,
    split: &[u32],
) -> Vec<usize> {
    let n = assignment.layer_weights.len();
    let mut sizes: Vec<_> = assignment
        .layer_weights
        .iter()
        .zip(cache)
        .map(|(&w, &kv)| (w + kv) as f64)
        .collect();
    sizes
        .push((assignment.output_weights + assignment.logits_bytes.saturating_sub(scratch)) as f64);
    let total = sizes.iter().sum::<f64>() + scratch as f64 * split.len() as f64;
    let mut cumulative = 0.0f32;
    let points: Vec<_> = split
        .iter()
        .map(|&s| {
            cumulative += s as f32;
            cumulative
        })
        .collect();
    let mut loaded = 0.0f32;
    let mut last = 0;
    let mut homes = vec![split.len() - 1; n + 1];
    for (id, point) in points.iter().enumerate() {
        loaded = (f64::from(loaded) + scratch as f64) as f32;
        let target = (f64::from(*point / cumulative) * total) as f32;
        while last < sizes.len() {
            let next = f64::from(loaded) + sizes[last];
            if next <= f64::from(target) {
                homes[last] = id;
                loaded = next as f32;
                last += 1;
            } else {
                if next - f64::from(target) < f64::from(target - loaded) {
                    homes[last] = id;
                    loaded = next as f32;
                    last += 1;
                }
                break;
            }
        }
    }
    homes.truncate(n);
    homes
}

#[test]
fn experts_ik_single_gpu_offloads_trailing_expert_layers() {
    let mut e = moe_estimate(6, 100, 200);
    let tensors = e.layout.expert_tensors.as_mut().unwrap();
    for t in tensors.iter_mut().filter(|t| t.layer >= 4) {
        t.bytes *= 2;
    }
    let per_layer = e.layout.per_layer_bytes.as_mut().unwrap();
    for bytes in &mut per_layer[4..] {
        *bytes += 600 * MIB;
    }
    let mut svc = moe_svc(OffloadMode::Layers(2));
    svc.ik_llama = true;
    let packed = pack(&e, &svc, &snapshot(&[24]), &AllocationTable::new()).unwrap();
    assert_eq!(packed.expert_offload_bytes, 2400 * MIB);
}

#[test]
fn expert_offload_auto_spills_surplus_experts_to_cpu() {
    // 10 layers: 100 MiB non-expert + 900 MiB experts each (10 GiB total),
    // non-expert only ≈ 1 GiB. A 4 GiB card holds all attention but not all
    // experts.
    let e = moe_estimate(10, 100, 300);
    let snap = snapshot(&[4]);
    let alloc = AllocationTable::new();
    let packed = pack(&e, &moe_svc(OffloadMode::Auto), &snap, &alloc).unwrap();

    assert_eq!(
        packed.args.ngl,
        Some(NGL_OFFLOAD_ALL),
        "-ngl 999: all layers on GPU, then -ncmoe pulls trailing experts back"
    );
    assert!(cpu_bytes(&packed) > 0, "surplus experts land on the CPU");
    assert!(packed.expert_offload_bytes > 0);
    assert!(packed.expert_offload_layers > 0);
    assert!(
        matches!(packed.args.n_cpu_moe, Some(n) if n > 0),
        "coarse whole-layer offload via --n-cpu-moe, got {:?}",
        packed.args.n_cpu_moe
    );
    assert!(
        packed.args.override_tensor.is_empty(),
        "no per-tensor expert -ot is synthesised, got {:?}",
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

    assert_eq!(packed.args.ngl, Some(NGL_OFFLOAD_ALL));
    assert_eq!(packed.expert_offload_bytes, 0);
    assert_eq!(packed.expert_offload_layers, 0);
    assert!(cpu_bytes(&packed) == 0);
    assert!(packed.args.override_tensor.is_empty());
}

/// `expert_offload = N` offloads exactly the expert layers of blocks `[0, N)`
/// to CPU even on a roomy card, via `--n-cpu-moe N` (not per-tensor `-ot`).
///
/// The *leading* blocks, matching what llama.cpp does: its load log names each
/// tensor it moves, and they run `blk.0`-`blk.39` for Qwen3.6-35B-A3B at
/// `--n-cpu-moe 40` and `blk.1`-`blk.38` for laguna at 39 (whose block 0 is
/// dense). Taking the wrong end matters whenever the quant widens later blocks'
/// experts — laguna's trailing 39 come to 43188 MiB against the leading 38's
/// 41496.
#[test]
fn expert_offload_layers_n_offloads_leading_layers() {
    let e = moe_estimate(10, 100, 100); // fits easily
    let snap = snapshot(&[24]);
    let alloc = AllocationTable::new();
    let packed = pack(&e, &moe_svc(OffloadMode::Layers(3)), &snap, &alloc).unwrap();

    assert_eq!(
        packed.args.ngl,
        Some(NGL_OFFLOAD_ALL),
        "attention stays on GPU; -ncmoe pulls the leading experts back"
    );
    assert_eq!(
        packed.args.n_cpu_moe,
        Some(3),
        "offload the experts of blocks 0..3"
    );
    assert_eq!(packed.expert_offload_layers, 3);
    // 3 layers × 3 experts × 100 MiB.
    assert_eq!(packed.expert_offload_bytes, 9 * 100 * MIB);
    assert!(
        packed.args.override_tensor.is_empty(),
        "no per-tensor -ot, got {:?}",
        packed.args.override_tensor
    );
}

/// Which end is offloaded, pinned where it is observable: give the trailing
/// blocks wider experts than the leading ones and the two directions produce
/// different byte totals.
///
/// This is laguna's shape — its quant stores later blocks' experts wider — and
/// it is what makes the direction observable: the trailing 39 come to 43188 MiB
/// against the leading 38's 41496, so offloading the wrong end leaves its cards
/// 1692 MiB short of what they hold.
#[test]
fn the_offloaded_end_is_the_leading_one_when_layer_sizes_differ() {
    // Blocks 0-4 carry 100 MiB experts, blocks 5-9 carry 300 MiB.
    let n_layers = 10u32;
    let mut per_layer = Vec::new();
    let mut experts = Vec::new();
    for layer in 0..n_layers {
        let exp_mb = if layer < 5 { 100 } else { 300 };
        per_layer.push((100 + 3 * exp_mb) * MIB);
        for kind in [ExpertKind::Gate, ExpertKind::Up, ExpertKind::Down] {
            experts.push(ExpertTensor {
                layer,
                kind,
                bytes: exp_mb * MIB,
            });
        }
    }
    let e = Estimate {
        weights_bytes: per_layer.iter().sum(),
        layout: Layout {
            per_layer_bytes: Some(per_layer),
            expert_layers: (0..n_layers).collect(),
            expert_tensors: Some(experts),
            ..Layout::default()
        },
        ..moe_estimate(1, 0, 0)
    };
    let packed = pack(
        &e,
        &moe_svc(OffloadMode::Layers(4)),
        &snapshot(&[24]),
        &AllocationTable::new(),
    )
    .unwrap();

    assert_eq!(packed.args.n_cpu_moe, Some(4));
    assert_eq!(packed.expert_offload_layers, 4);
    // Blocks 0-3: 4 layers × 3 experts × 100 MiB. The trailing four would be
    // three times that, which is what makes the assertion mean something.
    assert_eq!(
        packed.expert_offload_bytes,
        12 * 100 * MIB,
        "the leading blocks' experts, not the trailing blocks'"
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
    let PackError::ManualExpertsDoNotFit {
        needed, available, ..
    } = err
    else {
        panic!("expected ManualExpertsDoNotFit, got {err:?}");
    };
    assert!(
        needed > available,
        "the retained experts must exceed the pooled GPU capacity, \
         got needed={needed} available={available}"
    );
    // The single allowed card is named, so the operator can cross-reference
    // the overflow against `GET /api/devices`.
    let shortfalls = err.shortfalls();
    assert_eq!(
        shortfalls.iter().map(|s| s.device).collect::<Vec<_>>(),
        vec![DeviceId::Gpu(0)],
        "got {shortfalls:?}"
    );
    // Every entry has to be a genuine shortfall. This is the pooled
    // failure path, where `requested` is a per-card share of an aggregate
    // ask — without the filter a card with room shows up in the
    // placement-failure list claiming it needs less than it holds.
    assert!(
        shortfalls
            .iter()
            .all(|s| s.available_bytes < s.requested_bytes),
        "got {shortfalls:?}"
    );
}

/// The pooled-overflow report must not name a card that can take its
/// share. Two lopsided GPUs (one nearly full, one roomy) whose combined
/// capacity still can't hold the retained experts: the roomy card
/// satisfies the even share, so only the full one is a reason for the
/// failure and only it belongs in the list.
#[test]
fn manual_expert_overflow_names_only_the_cards_that_came_up_short() {
    let e = moe_estimate(10, 100, 300);
    // gpu:0 is tiny, gpu:1 is roomy; neither combination fits the pinned
    // experts, but gpu:1 individually clears the even per-card share.
    let snap = snapshot(&[1, 7]);
    let alloc = AllocationTable::new();
    let err = pack(&e, &moe_svc(OffloadMode::Layers(2)), &snap, &alloc)
        .expect_err("under-sized manual offload must not over-commit the GPUs");

    let shortfalls = err.shortfalls();
    assert!(
        !shortfalls.is_empty(),
        "a capacity failure always names at least one device"
    );
    assert!(
        shortfalls
            .iter()
            .all(|s| s.available_bytes < s.requested_bytes),
        "no card that fits its share may be listed, got {shortfalls:?}"
    );
}

/// Auto offload spreads across both GPUs before touching the CPU: a model
/// that fits in the two cards' combined VRAM but not either alone lands
/// entirely on the GPUs. Nothing is offloaded, so no `--n-cpu-moe` and no
/// `-ot` — the runtime splits the layers across both cards itself.
#[test]
fn expert_offload_auto_prefers_second_gpu() {
    // 20 layers, 100 MiB attn + 900 MiB experts = ~20 GiB. Two 12 GiB
    // cards hold it together (24 GiB) but neither alone does, so the
    // experts must split across both.
    let e = moe_estimate(20, 100, 300);
    let snap = snapshot(&[12, 12]);
    let alloc = AllocationTable::new();
    let packed = pack(&e, &moe_svc(OffloadMode::Auto), &snap, &alloc).unwrap();

    assert_eq!(packed.args.ngl, Some(NGL_OFFLOAD_ALL));
    assert_eq!(
        packed.args.n_cpu_moe, None,
        "nothing offloaded → no --n-cpu-moe"
    );
    assert_eq!(
        cpu_bytes(&packed),
        0,
        "experts prefer the GPUs over the CPU"
    );
    assert_eq!(
        packed.expert_offload_bytes, 0,
        "CPU offload metric counts host bytes only"
    );
    assert!(
        packed.args.override_tensor.is_empty(),
        "the runtime owns the cross-GPU split; no synthesised -ot, got {:?}",
        packed.args.override_tensor
    );
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
        g0 > 0 && g1 > 0,
        "both cards carry weight (g0={g0} g1={g1})"
    );
}

/// Symmetric two-GPU balance (the deepseek4 shape): tiny non-expert weight
/// plus huge experts must spread evenly across both cards. First-fit piles
/// every layer — and thus every expert's home GPU — onto gpu:0, overloading it
/// into an `insufficient_capacity` error while gpu:1 sits idle.
#[test]
fn expert_offload_auto_balances_symmetric_gpus() {
    // 40 layers, 150 MiB attn + 3×700 MiB experts: ~6 GiB attention, ~84
    // GiB experts — far past 2×24 GiB, so the surplus spills to CPU, but
    // the GPU-resident half must be balanced across both cards.
    let e = moe_estimate(40, 150, 700);
    let snap = snapshot(&[24, 24]);
    let packed = pack(
        &e,
        &moe_svc(OffloadMode::Auto),
        &snap,
        &AllocationTable::new(),
    )
    .unwrap();

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
        g0 > 0 && g1 > 0,
        "both cards must hold weight (g0={g0} g1={g1})"
    );
    assert!(
        cpu_bytes(&packed) > 0,
        "the surplus experts must spill to CPU"
    );
    // Balanced within ~one expert tensor — neither card overloaded.
    let (hi, lo) = (g0.max(g1), g0.min(g1));
    assert!(
        hi - lo <= 1024 * MIB,
        "cards must be balanced within ~1 expert (g0={g0} g1={g1})"
    );
    // And each card must fit inside its 24 GiB.
    assert!(
        g0 <= 24 * GIB && g1 <= 24 * GIB,
        "must fit 24 GiB (g0={g0} g1={g1})"
    );
}

/// Regression for the live `deepseek-v4-flash` failure: the real estimate
/// (~96 GiB weights, 9848 MiB compute buffer, 6657 B/token KV over 131072
/// context, 43 all-MoE layers) must auto-fit on two 24 GiB cards. It is the
/// shape that reports `insufficient_capacity: no fit on gpu:0` if the layer
/// balance or the one-layer fudge is wrong.
#[test]
fn deepseek4_like_auto_fits_two_24gib_cards() {
    let n_layers = 43u32;
    let nonexp = 140 * MIB; // ~6 GiB of attention across 43 layers
    let exp = 700 * MIB; // 3 × 700 MiB experts/layer → ~88 GiB experts
    let mut per_layer = Vec::new();
    let mut experts = Vec::new();
    for layer in 0..n_layers {
        per_layer.push(nonexp + 3 * exp);
        for kind in [ExpertKind::Gate, ExpertKind::Up, ExpertKind::Down] {
            experts.push(ExpertTensor {
                layer,
                kind,
                bytes: exp,
            });
        }
    }
    let e = Estimate {
        weights_bytes: (nonexp + 3 * exp) * n_layers as u64 + 414 * MIB,
        kv_per_token: 6657,
        layout: Layout {
            per_layer_bytes: Some(per_layer),
            non_layer: NonLayer {
                output_head_bytes: 414 * MIB,
                token_embd_bytes: 414 * MIB,
                tied_head_bytes: 0,
                other_bytes: 0,
            },
            expert_layers: (0..n_layers).collect(),
            expert_tensors: Some(experts),
            ..Layout::default()
        },
        buffers: Buffers {
            compute_mb: 9848,
            ..Buffers::default()
        },
        ..Estimate::empty(Architecture::DeepSeek4, 131072)
    };
    // The real box has 125 GiB RAM for the ~60 GiB of CPU-side experts;
    // widen the default snapshot's host budget to match.
    let mut snap = snapshot(&[24, 24]);
    snap.cpu = Some(CpuSnapshot {
        total_bytes: 125 * GIB,
        available_bytes: 110 * GIB,
    });
    let packed = pack(
        &e,
        &moe_svc(OffloadMode::Auto),
        &snap,
        &AllocationTable::new(),
    )
    .expect("deepseek4 auto must fit two 24 GiB cards");
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
    // Both cards used, both within capacity, and balanced.
    assert!(g0 > 0 && g1 > 0 && cpu_bytes(&packed) > 0);
    assert!(
        g0 <= 24 * GIB && g1 <= 24 * GIB,
        "g0={g0} g1={g1} must fit 24 GiB"
    );
    assert!(
        g0.abs_diff(g1) <= 1500 * MIB,
        "cards balanced: g0={g0} g1={g1}"
    );
    // Roughly the empirical G8-G10: a meaningful chunk of experts on GPU.
    assert!(
        packed.expert_offload_layers > 0,
        "some experts spill to CPU"
    );
}

/// Offloading more experts than host RAM can hold (minus the CPU reserve)
/// is rejected with `CpuDoesNotFit` rather than silently over-committing.
#[test]
fn expert_offload_rejects_when_cpu_is_full() {
    let e = moe_estimate(10, 100, 900); // ~1 GiB attn, ~27 GiB experts
    // A 24 GiB card holds the attention plus most experts; the ~4 GiB
    // expert surplus must spill, but the host has only 2 GiB free.
    let mut snap = snapshot(&[24]);
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
