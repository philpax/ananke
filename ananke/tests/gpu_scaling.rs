//! Integration tests that verify ananke's placement and CUDA-env rendering
//! scale across single-GPU, three-GPU, four-GPU, and zero-GPU configurations.
//!
//! Most existing tests target a dual-GPU layout, which is the primary
//! development setup. These tests pin down the boundary cases:
//!
//! - **One GPU.** No `tensor_split` is emitted; the lone GPU's id (which need
//!   not be `0`) must round-trip correctly into `CUDA_VISIBLE_DEVICES`.
//! - **Three or more GPUs.** `tensor_split` widens to one entry per allowed
//!   GPU, the layer counts add up to `-ngl`, and `gpu_allow` continues to
//!   restrict the candidate set.
//! - **Zero GPUs.** A snapshot with no GPUs must still produce a valid
//!   CPU-only placement under `CpuOnly`/`Hybrid`, and must reject `GpuOnly`
//!   services with the structured `PackError`. Today this also stands in for
//!   the "NVML init failed" / non-NVIDIA-host case: the only NVIDIA-coupled
//!   surface below `placement::pack` is `cuda_env::render`, which receives an
//!   `Allocation` rather than probing the driver, so it stays correct as
//!   long as the snapshot agrees that no GPUs are present.
#![cfg(feature = "test-fakes")]

mod common;

use std::collections::BTreeMap;

use ananke::{
    allocator::{
        AllocationTable,
        placement::{self, PackError, pick_command_gpu},
    },
    config::{PlacementPolicy, ServiceConfig},
    devices::{Allocation, CpuSnapshot, DeviceId, DeviceSnapshot, GpuSnapshot, cuda_env},
    estimator::{Estimate, NonLayer},
};
use smol_str::SmolStr;

/// Build a `DeviceSnapshot` from a list of `(gpu_id, total_gb, free_gb)`
/// triples plus a generously-sized CPU pool. Lets each test name the GPU
/// indices and capacities it cares about without rewriting the boilerplate.
fn snapshot(gpus: &[(u32, u64, u64)]) -> DeviceSnapshot {
    let gb = 1024u64 * 1024 * 1024;
    DeviceSnapshot {
        gpus: gpus
            .iter()
            .map(|(id, total, free)| GpuSnapshot {
                id: *id,
                name: format!("GPU {id}"),
                total_bytes: total * gb,
                free_bytes: free * gb,
            })
            .collect(),
        cpu: Some(CpuSnapshot {
            total_bytes: 64 * gb,
            available_bytes: 64 * gb,
        }),
        taken_at_ms: 0,
    }
}

/// Estimate of `n_layers` × `per_layer_mib` MiB layers, no KV, no compute
/// buffer overhead. Mirrors the `large_estimate` helper in
/// `multi_gpu_split.rs` so packer arithmetic stays predictable.
fn flat_estimate(n_layers: usize, per_layer_mib: u64) -> Estimate {
    let per_layer_bytes = per_layer_mib * 1024 * 1024;
    Estimate {
        weights_bytes: per_layer_bytes * n_layers as u64,
        kv_per_token: 0,
        compute_buffer_mb: 0,
        per_layer_bytes: Some(vec![per_layer_bytes; n_layers]),
        attention_layers: None,
        non_layer: NonLayer::default(),
        override_tensor_bytes: BTreeMap::new(),
        expert_layers: Vec::new(),
        expert_layer_cpu_bytes: BTreeMap::new(),
        context: 4096,
        architecture: SmolStr::new("qwen3"),
    }
}

/// Build a packable llama-cpp service with no `placement_override` and the
/// requested policy. Returns it ready for `placement::pack` to drive.
fn svc_for_policy(name: &str, policy: PlacementPolicy) -> ServiceConfig {
    let mut svc = common::minimal_llama_service(name, 0);
    svc.placement_override = BTreeMap::new();
    svc.placement_policy = policy;
    svc
}

// -- single GPU ---------------------------------------------------------

/// One 16 GB GPU, a 6 GB model: every layer lands on the only GPU and no
/// `tensor_split` is emitted (the flag is only useful for ≥2 GPUs).
#[test]
fn single_gpu_packs_without_tensor_split() {
    let svc = svc_for_policy("solo", PlacementPolicy::GpuOnly);
    let snap = snapshot(&[(0, 16, 16)]);
    let est = flat_estimate(10, 600);
    let packed = placement::pack(&est, &svc, &snap, &AllocationTable::new())
        .expect("model must fit on one 16 GB GPU");

    assert_eq!(
        packed.args.ngl,
        Some(10),
        "every layer should land on the single GPU"
    );
    assert!(
        packed.args.tensor_split.is_none(),
        "tensor_split must not be emitted for single-GPU placement; got {:?}",
        packed.args.tensor_split
    );
    assert!(packed.allocation.bytes.contains_key(&DeviceId::Gpu(0)));
    assert!(
        !packed.allocation.bytes.contains_key(&DeviceId::Gpu(1)),
        "no second GPU should appear in the allocation"
    );
}

/// The lone GPU is *not* index 0. The packer must still place layers on it
/// and `cuda_env::render` must emit the actual id, not assume 0. Regression
/// guard for any "first GPU is 0" shortcut.
#[test]
fn single_gpu_with_nonzero_id_renders_cuda_env() {
    let svc = svc_for_policy("solo", PlacementPolicy::GpuOnly);
    // Single GPU with id = 3 (e.g., the host has CUDA_VISIBLE_DEVICES=3 set
    // before ananke runs, so nvml only enumerates that one card).
    let snap = snapshot(&[(3, 24, 24)]);
    let est = flat_estimate(8, 600);
    let packed = placement::pack(&est, &svc, &snap, &AllocationTable::new())
        .expect("model must fit on the only GPU regardless of its id");

    assert!(packed.allocation.bytes.contains_key(&DeviceId::Gpu(3)));
    assert_eq!(
        cuda_env::render(&packed.allocation),
        "3",
        "CUDA_VISIBLE_DEVICES must follow the actual GPU id"
    );
}

/// One GPU, model larger than its free capacity, `Hybrid` policy: layers
/// that don't fit on the GPU spill to CPU. Must not panic and must still
/// route the GPU portion through `Gpu(0)`.
#[test]
fn single_gpu_overflow_spills_to_cpu_under_hybrid() {
    let svc = svc_for_policy("solo", PlacementPolicy::Hybrid);
    // 8 GB free — only ~13 of 30 × 600 MiB layers fit after the per-layer
    // fudge headroom is reserved.
    let snap = snapshot(&[(0, 16, 8)]);
    let est = flat_estimate(30, 600);
    let packed = placement::pack(&est, &svc, &snap, &AllocationTable::new())
        .expect("hybrid policy must spill rather than fail");

    assert!(packed.allocation.bytes.contains_key(&DeviceId::Gpu(0)));
    assert!(
        packed.allocation.bytes.contains_key(&DeviceId::Cpu),
        "overflow layers must be visible on CPU; got {:?}",
        packed.allocation.bytes
    );
    let ngl = packed.args.ngl.expect("-ngl must be emitted");
    assert!(
        (1..30).contains(&ngl),
        "-ngl must reflect the partial GPU offload, got {ngl}"
    );
}

/// Same overflow under `GpuOnly`: structured `LayerDoesNotFit` rather than
/// a silent fall-through.
#[test]
fn single_gpu_overflow_under_gpu_only_returns_pack_error() {
    let svc = svc_for_policy("solo", PlacementPolicy::GpuOnly);
    let snap = snapshot(&[(0, 16, 8)]);
    let est = flat_estimate(30, 600);
    let err = placement::pack(&est, &svc, &snap, &AllocationTable::new())
        .expect_err("GpuOnly with overflow must fail");
    assert!(
        matches!(err, PackError::LayerDoesNotFit { .. }),
        "expected LayerDoesNotFit, got {err:?}"
    );
}

// -- three+ GPUs --------------------------------------------------------

/// Three GPUs, each 8 GB free. A 30-layer × 600 MiB model (≈17.6 GB) does
/// not fit on any single card, so the walker spans all three. The emitted
/// `tensor_split` must therefore have three entries that sum to `-ngl`.
#[test]
fn three_gpus_split_layers_across_all_three() {
    let svc = svc_for_policy("triple", PlacementPolicy::GpuOnly);
    let snap = snapshot(&[(0, 16, 8), (1, 16, 8), (2, 16, 8)]);
    let est = flat_estimate(30, 600);
    let packed = placement::pack(&est, &svc, &snap, &AllocationTable::new())
        .expect("a 17 GB model must pack across 3 × 8 GB GPUs");

    let split = packed
        .args
        .tensor_split
        .as_ref()
        .expect("tensor_split must be emitted for ≥2 active GPUs");
    assert_eq!(
        split.len(),
        3,
        "tensor_split must have one entry per GPU; got {split:?}"
    );
    assert!(
        split.iter().all(|&n| n > 0),
        "every GPU should carry at least one layer; got {split:?}"
    );
    assert_eq!(
        split.iter().sum::<u32>(),
        packed.args.ngl.unwrap(),
        "-ngl must equal the sum of tensor_split entries"
    );
    for id in 0..3 {
        assert!(
            packed.allocation.bytes.contains_key(&DeviceId::Gpu(id)),
            "GPU {id} must hold an allocation"
        );
    }
    assert_eq!(cuda_env::render(&packed.allocation), "0,1,2");
}

/// Four GPUs, mixed free capacity. The placer should spread a model that
/// exceeds any single GPU across exactly the GPUs it needs — and the
/// `tensor_split` width matches the *allowed* GPU count, including the
/// cards that received zero layers (an existing tensor_split invariant).
#[test]
fn four_gpus_emit_tensor_split_with_one_entry_per_allowed_gpu() {
    let svc = svc_for_policy("quad", PlacementPolicy::GpuOnly);
    let snap = snapshot(&[(0, 24, 24), (1, 24, 24), (2, 24, 24), (3, 24, 24)]);
    // 60 layers × 600 MiB ≈ 35 GB — needs ≥2 GPUs, comfortably fits across 4.
    let est = flat_estimate(60, 600);
    let packed = placement::pack(&est, &svc, &snap, &AllocationTable::new())
        .expect("a 35 GB model must pack across 4 × 24 GB GPUs");

    let split = packed
        .args
        .tensor_split
        .as_ref()
        .expect("tensor_split must be emitted");
    assert_eq!(
        split.len(),
        4,
        "tensor_split must have one entry per allowed GPU even if some carry 0 layers"
    );
    assert_eq!(split.iter().sum::<u32>(), 60);

    // CUDA env must list every GPU that actually got an allocation,
    // ascending.
    let env = cuda_env::render(&packed.allocation);
    let listed: Vec<&str> = env.split(',').collect();
    assert!(!listed.is_empty(), "at least one GPU should be allocated");
    for id in &listed {
        assert!(
            "0123".contains(id),
            "unexpected GPU id {id} in CUDA_VISIBLE_DEVICES={env}"
        );
    }
}

/// `gpu_allow` is honoured at scale: with four physical GPUs but only `[1,
/// 3]` allowed, no allocation may land on GPUs 0 or 2, and `tensor_split`
/// has exactly two entries.
#[test]
fn four_gpus_with_gpu_allow_restricts_candidates() {
    let mut svc = svc_for_policy("restricted", PlacementPolicy::GpuOnly);
    svc.gpu_allow = vec![1, 3];
    let snap = snapshot(&[(0, 24, 24), (1, 24, 24), (2, 24, 24), (3, 24, 24)]);
    let est = flat_estimate(40, 600);
    let packed = placement::pack(&est, &svc, &snap, &AllocationTable::new())
        .expect("model must pack across the two allowed GPUs");

    let split = packed
        .args
        .tensor_split
        .as_ref()
        .expect("tensor_split must be emitted for two-GPU placement");
    assert_eq!(
        split.len(),
        2,
        "tensor_split must mirror the allowed-GPU count, not the physical count"
    );
    assert!(
        !packed.allocation.bytes.contains_key(&DeviceId::Gpu(0))
            && !packed.allocation.bytes.contains_key(&DeviceId::Gpu(2)),
        "disallowed GPUs must be untouched; got {:?}",
        packed.allocation.bytes
    );
    assert_eq!(cuda_env::render(&packed.allocation), "1,3");
}

/// Heterogeneous fleet: one tiny GPU + two large ones, with a peer pledge
/// that biases pledge-book headroom toward GPU 2. A small model that fits
/// on a single large GPU should land entirely on the most-headroom GPU
/// (not the tiny one), and `tensor_split` should still be three-wide with
/// only that GPU carrying layers. Confirms the first-fit-on-largest-
/// pledge-headroom invariant when GPU sizes and pledges are mismatched.
#[test]
fn three_gpus_heterogeneous_pack_small_model_on_largest_headroom() {
    let svc = svc_for_policy("heterogeneous", PlacementPolicy::GpuOnly);
    // GPU 0: small (4 GB). GPUs 1 & 2: 24 GB total, but GPU 1 already has
    // a 12 GB peer pledge, leaving GPU 2 with the most pledge-book headroom.
    let snap = snapshot(&[(0, 4, 4), (1, 24, 12), (2, 24, 22)]);
    let mut reserved = AllocationTable::new();
    let mut peer = BTreeMap::new();
    peer.insert(ananke::config::DeviceSlot::Gpu(1), 12 * 1024u64); // MB
    reserved.insert(SmolStr::new("peer"), peer);

    let est = flat_estimate(8, 600); // ≈ 4.7 GB, fits on a single large GPU.
    let packed = placement::pack(&est, &svc, &snap, &reserved)
        .expect("model must fit on the largest-headroom GPU");

    assert!(
        packed.allocation.bytes.contains_key(&DeviceId::Gpu(2)),
        "GPU 2 has the most pledge headroom and should hold the model; got {:?}",
        packed.allocation.bytes
    );
    assert!(
        !packed.allocation.bytes.contains_key(&DeviceId::Gpu(0)),
        "the tiny GPU must not be involved when a larger GPU has room"
    );
    let split = packed
        .args
        .tensor_split
        .as_ref()
        .expect("tensor_split must be emitted when ≥2 GPUs are allowed");
    assert_eq!(split.len(), 3, "one entry per allowed GPU");
    assert_eq!(split.iter().sum::<u32>(), 8);
    assert_eq!(
        split[2], 8,
        "all 8 layers should land on GPU 2; got {split:?}"
    );
}

/// `pick_command_gpu` (the path used by command-template + Dynamic
/// services) extends naturally to four GPUs: the most-free GPU wins.
#[test]
fn pick_command_gpu_scales_to_four_gpus() {
    let svc = svc_for_policy("comfy", PlacementPolicy::GpuOnly);
    // Free bytes: 4, 12, 8, 18 GB. GPU 3 is the clear winner.
    let snap = snapshot(&[(0, 24, 4), (1, 24, 12), (2, 24, 8), (3, 24, 18)]);
    let pick = pick_command_gpu(&svc, &snap, &AllocationTable::new(), 2 * 1024, None, false);
    assert_eq!(
        pick,
        Some(3),
        "pick_command_gpu must choose the most-free GPU even at >2 cards"
    );
}

// -- zero GPUs ----------------------------------------------------------

/// No GPUs in the snapshot at all (e.g. NVML init failed, host has no
/// NVIDIA driver, or operator pinned the daemon to CPU). A `CpuOnly`
/// service must still produce a packable allocation.
#[test]
fn zero_gpus_cpu_only_packs_to_cpu() {
    let mut svc = svc_for_policy("cpu-only", PlacementPolicy::CpuOnly);
    // CpuOnly services typically declare a CPU floor via placement_override;
    // packing without one is fine because the walker spills every layer.
    svc.placement_override = BTreeMap::new();
    let snap = snapshot(&[]);
    let est = flat_estimate(12, 200);
    let packed = placement::pack(&est, &svc, &snap, &AllocationTable::new())
        .expect("CpuOnly placement must succeed when no GPUs exist");

    assert_eq!(
        packed.args.ngl,
        Some(0),
        "-ngl must be 0 when no GPUs are available"
    );
    assert!(
        packed.args.tensor_split.is_none(),
        "tensor_split has no meaning with zero GPUs"
    );
    assert!(packed.allocation.bytes.contains_key(&DeviceId::Cpu));
    assert!(
        packed.allocation.gpu_ids().is_empty(),
        "no GPU ids should appear in the allocation"
    );
    assert_eq!(
        cuda_env::render(&packed.allocation),
        "",
        "CUDA_VISIBLE_DEVICES must be empty when no GPU is allocated"
    );
}

/// Same empty snapshot, but the service insists on `GpuOnly`. The packer
/// must surface a structured error rather than silently producing a CPU
/// placement — the supervisor relies on this to stay disabled until the
/// operator either fixes the driver or relaxes the policy.
#[test]
fn zero_gpus_gpu_only_returns_pack_error() {
    let svc = svc_for_policy("gpu-only", PlacementPolicy::GpuOnly);
    let snap = snapshot(&[]);
    let est = flat_estimate(12, 200);
    let err = placement::pack(&est, &svc, &snap, &AllocationTable::new())
        .expect_err("GpuOnly with no GPUs must fail");
    assert!(
        matches!(err, PackError::LayerDoesNotFit { layer_index: 0, .. }),
        "expected LayerDoesNotFit{{layer_index: 0}}, got {err:?}"
    );
}

/// `pick_command_gpu` returns `None` for the same configuration, letting
/// the command-template path either route to CPU (if policy permits) or
/// surface the eviction-retry signal.
#[test]
fn pick_command_gpu_returns_none_with_zero_gpus() {
    let svc = svc_for_policy("comfy", PlacementPolicy::GpuOnly);
    let snap = snapshot(&[]);
    let pick = pick_command_gpu(&svc, &snap, &AllocationTable::new(), 1024, None, false);
    assert_eq!(pick, None);
}

/// `cuda_env::render` for a CPU-only allocation in a no-GPU world produces
/// the empty string, which we rely on as the "do not let the child grab a
/// GPU" wire signal.
#[test]
fn cuda_env_renders_empty_for_cpu_only_allocation() {
    let mut bytes = BTreeMap::new();
    bytes.insert(DeviceId::Cpu, 8 * 1024 * 1024 * 1024);
    let alloc = Allocation { bytes };
    assert_eq!(cuda_env::render(&alloc), "");
}
