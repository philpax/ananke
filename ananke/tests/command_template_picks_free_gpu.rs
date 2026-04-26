//! Scenario: when GPUs 0+1 are pinned by a llama-cpp service that spans both
//! cards (the production Qwen 3.6 35B persistent setup), a command-template
//! ComfyUI service launching with `Dynamic { min_mb, max_mb }` must land on
//! the free GPU 2 — not the conflicted GPU 0 — and its child must inherit
//! `CUDA_VISIBLE_DEVICES=2`.
//!
//! Regression: pre-fix the daemon picked `gpu_allow.first().or(snap.gpus.first())`
//! (always GPU 0) for command-template services, and additionally fell back
//! to `init.allocation` (built from `placement_override` only — empty for
//! ComfyUI), so the child saw `CUDA_VISIBLE_DEVICES=""` and silently ran on CPU.
#![cfg(feature = "test-fakes")]

mod common;

use std::{collections::BTreeMap, time::Duration};

use ananke::{
    config::{
        AllocationMode, CommandConfig, DeviceSlot, Filters, HealthSettings, Lifecycle,
        PlacementPolicy, ServiceConfig, TemplateConfig, parse::DEFAULT_START_QUEUE_DEPTH,
    },
    devices::{CpuSnapshot, DeviceSnapshot, GpuSnapshot},
    supervise::EnsureSource,
    system::FakeProcessState,
};
use common::build_harness;
use smol_str::SmolStr;

fn comfy_like_service(name: &str, port: u16) -> ServiceConfig {
    ServiceConfig {
        name: SmolStr::new(name),
        port,
        private_port: 0,
        lifecycle: Lifecycle::OnDemand,
        priority: 50,
        health: HealthSettings {
            http_path: "/health".into(),
            timeout_ms: 5_000,
            probe_interval_ms: 200,
        },
        placement_override: BTreeMap::new(),
        placement_policy: PlacementPolicy::GpuOnly,
        gpu_allow: Vec::new(),
        idle_timeout_ms: 60_000,
        drain_timeout_ms: 1_000,
        extended_stream_drain_ms: 1_000,
        max_request_duration_ms: 5_000,
        filters: Filters::default(),
        allocation_mode: AllocationMode::Dynamic {
            min_mb: 2 * 1024,
            max_mb: 20 * 1024,
            min_borrower_runtime_ms: 60_000,
        },
        openai_compat: false,
        description: None,
        start_queue_depth: DEFAULT_START_QUEUE_DEPTH,
        extra_args: Vec::new(),
        env: BTreeMap::new(),
        tracking: ananke::config::TrackingSettings::default(),
        metadata: ananke_api::AnankeMetadata::new(),
        template_config: TemplateConfig::Command(CommandConfig {
            command: vec!["comfyui-start".into(), "--port".into(), "{port}".into()],
            workdir: None,
            shutdown_command: None,
            private_port_override: None,
        }),
    }
}

fn three_gpu_snapshot_with_two_busy() -> DeviceSnapshot {
    DeviceSnapshot {
        gpus: vec![
            // GPU 0 + GPU 1 are mostly full — modelling Qwen 3.6 35B spanning
            // both 24 GB cards. nvml says 1 GB free on each.
            GpuSnapshot {
                id: 0,
                name: "GPU 0".into(),
                total_bytes: 24 * 1024 * 1024 * 1024,
                free_bytes: 1024 * 1024 * 1024,
            },
            GpuSnapshot {
                id: 1,
                name: "GPU 1".into(),
                total_bytes: 24 * 1024 * 1024 * 1024,
                free_bytes: 1024 * 1024 * 1024,
            },
            // GPU 2 is unoccupied — ComfyUI must land here.
            GpuSnapshot {
                id: 2,
                name: "GPU 2".into(),
                total_bytes: 24 * 1024 * 1024 * 1024,
                free_bytes: 23 * 1024 * 1024 * 1024,
            },
        ],
        cpu: Some(CpuSnapshot {
            total_bytes: 64 * 1024 * 1024 * 1024,
            available_bytes: 64 * 1024 * 1024 * 1024,
        }),
        taken_at_ms: 0,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn comfyui_lands_on_free_gpu_when_others_busy() {
    let comfy = comfy_like_service("comfyui", common::free_port());
    let h = build_harness(vec![comfy]).await;

    // Snapshot reflects GPUs 0+1 saturated by a peer; only GPU 2 has room.
    *h.state.snapshot.write() = three_gpu_snapshot_with_two_busy();

    // Pre-seed the pledge book to match: an external "qwen-3.6" service holds
    // 22 GB on each of GPUs 0+1. This is how the production setup looks while
    // Qwen is loaded; ComfyUI's pick must respect the pledge book the same
    // way the placement engine does for llama-cpp services.
    {
        let mut alloc = h.state.allocations.lock();
        let mut qwen = BTreeMap::new();
        qwen.insert(DeviceSlot::Gpu(0), 22 * 1024u64); // MB
        qwen.insert(DeviceSlot::Gpu(1), 22 * 1024u64);
        alloc.insert(SmolStr::new("qwen-3.6"), qwen);
    }

    // Drive the start. Command-template + Dynamic allocation runs through
    // `compute_command_reservation`, which calls `pick_command_gpu` and
    // stashes the result on `packed_for_spawn` for the spawner.
    let handle = h
        .state
        .registry
        .get("comfyui")
        .expect("comfyui handle registered");
    let resp = handle
        .ensure(EnsureSource::UserRequest)
        .await
        .expect("ensure must respond");
    if let ananke::supervise::EnsureResponse::Waiting { mut rx } = resp {
        // Wait for the start outcome — health probe hits the harness echo
        // server, so this resolves quickly.
        let outcome = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("ensure must complete within 5 s")
            .expect("StartOutcome must be sent");
        match outcome {
            ananke::supervise::StartOutcome::Ok => {}
            ananke::supervise::StartOutcome::Err(f) => panic!("start failed: {}", f.message),
        }
    }

    // Assertion 1: the pledge book records ComfyUI on GPU 2 with min_mb.
    let alloc = h.state.allocations.lock().clone();
    let comfy_alloc = alloc
        .get(&SmolStr::new("comfyui"))
        .expect("comfyui must hold a reservation after ensure");
    assert!(
        comfy_alloc.contains_key(&DeviceSlot::Gpu(2)),
        "comfyui must reserve on GPU 2 (the only one with room); got {comfy_alloc:?}"
    );
    assert!(
        !comfy_alloc.contains_key(&DeviceSlot::Gpu(0))
            && !comfy_alloc.contains_key(&DeviceSlot::Gpu(1)),
        "comfyui must not touch GPUs 0 or 1; got {comfy_alloc:?}"
    );

    // Assertion 2: the spawned child inherited CUDA_VISIBLE_DEVICES=2 (the
    // wire-protocol the ComfyUI process actually consumes). The pre-fix bug
    // emitted `CUDA_VISIBLE_DEVICES=""` here.
    let children = h.process_spawner.children();
    let comfy_child = children
        .iter()
        .find(|c| c.binary == "comfyui-start")
        .expect("comfyui child must have been spawned");
    assert_eq!(
        comfy_child
            .env
            .get("CUDA_VISIBLE_DEVICES")
            .map(String::as_str),
        Some("2"),
        "comfyui child must inherit CUDA_VISIBLE_DEVICES=2; got env={:?}",
        comfy_child.env
    );
    assert!(
        matches!(comfy_child.state, FakeProcessState::Running),
        "comfyui child must be Running; got {:?}",
        comfy_child.state
    );

    h.cleanup().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn comfyui_prefers_gpu_with_growth_headroom() {
    // Three GPUs:
    //   - GPU 0: 4 GB free (above min, well below max — bad pick).
    //   - GPU 1: 22 GB free (above max — best pick).
    //   - GPU 2: 6 GB free (above min, below max).
    // Min = 2 GB, max = 20 GB. The pick must honour the max headroom target.
    let comfy = comfy_like_service("comfyui", common::free_port());
    let h = build_harness(vec![comfy]).await;
    *h.state.snapshot.write() = DeviceSnapshot {
        gpus: vec![
            GpuSnapshot {
                id: 0,
                name: "GPU 0".into(),
                total_bytes: 24 * 1024 * 1024 * 1024,
                free_bytes: 4 * 1024 * 1024 * 1024,
            },
            GpuSnapshot {
                id: 1,
                name: "GPU 1".into(),
                total_bytes: 24 * 1024 * 1024 * 1024,
                free_bytes: 22 * 1024 * 1024 * 1024,
            },
            GpuSnapshot {
                id: 2,
                name: "GPU 2".into(),
                total_bytes: 24 * 1024 * 1024 * 1024,
                free_bytes: 6 * 1024 * 1024 * 1024,
            },
        ],
        cpu: Some(CpuSnapshot {
            total_bytes: 64 * 1024 * 1024 * 1024,
            available_bytes: 64 * 1024 * 1024 * 1024,
        }),
        taken_at_ms: 0,
    };

    let handle = h.state.registry.get("comfyui").unwrap();
    let resp = handle.ensure(EnsureSource::UserRequest).await.unwrap();
    if let ananke::supervise::EnsureResponse::Waiting { mut rx } = resp {
        let _ = tokio::time::timeout(Duration::from_secs(5), rx.recv()).await;
    }

    let alloc = h.state.allocations.lock().clone();
    let comfy_alloc = alloc.get(&SmolStr::new("comfyui")).unwrap();
    assert!(
        comfy_alloc.contains_key(&DeviceSlot::Gpu(1)),
        "comfyui should land on GPU 1 (the only one with full max_mb headroom); got {comfy_alloc:?}"
    );

    h.cleanup().await;
}
