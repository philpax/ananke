//! Integration test: a start that follows a stop at once sees the memory the
//! stopped child freed.
//!
//! Placement counts the smaller of NVML's free figure and the pledge book's,
//! and the device snapshot is sampled on a two-second cadence. A service that
//! fills most of its only allowed GPU, stopped and started again (or
//! restarted) within one tick, used to find its own dead child still counted
//! in the snapshot: "weights do not fit on any allowed device" on an empty
//! card, with no peer to evict. Ending a run now asks the sampler for a fresh
//! sample before the next placement.
#![cfg(feature = "test-fakes")]

mod common;

use std::{sync::Arc, time::Duration};

use ananke::{
    api::management::router,
    config::{AllocationMode, CommandConfig, PlacementPolicy, ServiceConfig, TemplateConfig},
};
use ananke_api::services::StartResponse;
use ananke_placement::devices::{CpuSnapshot, DeviceSnapshot, GpuSnapshot};
use ananke_system::{FakeProcessState, FakeSpawner};
use axum::{body::to_bytes, http::StatusCode};
use common::build_harness;
use tower::util::ServiceExt;

const GIB: u64 = 1024 * 1024 * 1024;
const GPU_TOTAL: u64 = 24 * GIB;
/// What one child of the service holds on GPU 0, as NVML would see it.
const CHILD_VRAM: u64 = 14 * GIB;

/// A command service pinned to GPU 0 with a static 14 GiB reserve: two
/// copies never fit on the 24 GiB card, one always does.
fn gpu0_command_service() -> ServiceConfig {
    let mut s = common::minimal_llama_service("explorer", 0);
    s.template_config = TemplateConfig::Command(CommandConfig {
        command: vec!["/fake/explorer".into(), "--port".into(), "{port}".into()],
        workdir: None,
        shutdown_command: None,
        private_port_override: None,
        openai_proxy: None,
    });
    s.placement_override.clear();
    s.placement_policy = PlacementPolicy::GpuOnly;
    s.gpu_allow = vec![0];
    s.allocation_mode = AllocationMode::Static {
        reserve_mb: 14 * 1024,
    };
    s
}

fn snapshot_with_used(used: u64) -> DeviceSnapshot {
    DeviceSnapshot {
        gpus: vec![GpuSnapshot {
            id: 0,
            name: "GPU 0".into(),
            total_bytes: GPU_TOTAL,
            free_bytes: GPU_TOTAL - used,
        }],
        cpu: Some(CpuSnapshot {
            total_bytes: 64 * GIB,
            available_bytes: 64 * GIB,
        }),
        taken_at_ms: ananke_time::now_unix_ms_u64(),
    }
}

fn live_vram(spawner: &FakeSpawner) -> u64 {
    let live = spawner
        .children()
        .iter()
        .filter(|c| matches!(c.state, FakeProcessState::Running))
        .count() as u64;
    live * CHILD_VRAM
}

/// Stands in for the device snapshotter, minus its cadence: it samples only
/// when asked, so the snapshot stays stale unless a refresh is requested.
fn spawn_on_demand_sampler(
    snapshot: ananke_observation::SharedSnapshot,
    spawner: Arc<FakeSpawner>,
) -> tokio::task::JoinHandle<()> {
    let sampler = snapshot.attach_sampler();
    tokio::spawn(async move {
        loop {
            sampler.requested().await;
            let ticket = sampler.begin();
            *snapshot.write() = snapshot_with_used(live_vram(&spawner));
            sampler.served(ticket);
        }
    })
}

async fn post(app: &axum::Router, uri: &str) -> (StatusCode, String) {
    let req = axum::http::Request::builder()
        .method("POST")
        .uri(uri)
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

fn assert_started(op: &str, status: StatusCode, body: &str) {
    assert_eq!(status, StatusCode::ACCEPTED, "{op} failed: {body}");
    let parsed: StartResponse = serde_json::from_str(body).unwrap();
    assert!(
        matches!(parsed, StartResponse::Started { .. }),
        "unexpected {op} response: {parsed:?}"
    );
}

/// Take the sample a cadence tick would, now that a child is running.
async fn sample_now(h: &common::TestHarness) {
    assert!(h.state.snapshot.refresh(Duration::from_secs(1)).await);
    assert_eq!(
        h.state.snapshot.read().gpus[0].free_bytes,
        GPU_TOTAL - CHILD_VRAM,
        "the sampler should see the running child"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn stop_then_start_at_once_fits_on_the_freed_gpu() {
    let h = build_harness(vec![gpu0_command_service()]).await;
    *h.state.snapshot.write() = snapshot_with_used(0);
    let sampler = spawn_on_demand_sampler(h.state.snapshot.clone(), h.process_spawner.clone());
    let app = router(h.state.clone());

    let (status, body) = post(&app, "/api/services/explorer/start").await;
    assert_started("first start", status, &body);
    sample_now(&h).await;

    let (status, body) = post(&app, "/api/services/explorer/stop").await;
    assert_eq!(status, StatusCode::ACCEPTED, "stop failed: {body}");

    // No cadence tick in between: only the refresh at the end of the run can
    // have cleared the dead child from the snapshot.
    let (status, body) = post(&app, "/api/services/explorer/start").await;
    assert_started("start right after stop", status, &body);

    sampler.abort();
    h.cleanup().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn restart_of_a_service_filling_its_gpu_succeeds() {
    let h = build_harness(vec![gpu0_command_service()]).await;
    *h.state.snapshot.write() = snapshot_with_used(0);
    let sampler = spawn_on_demand_sampler(h.state.snapshot.clone(), h.process_spawner.clone());
    let app = router(h.state.clone());

    let (status, body) = post(&app, "/api/services/explorer/start").await;
    assert_started("first start", status, &body);
    sample_now(&h).await;

    let (status, body) = post(&app, "/api/services/explorer/restart").await;
    assert_started("restart", status, &body);
    assert_eq!(
        live_vram(&h.process_spawner),
        CHILD_VRAM,
        "exactly one child should be running after the restart"
    );

    sampler.abort();
    h.cleanup().await;
}
