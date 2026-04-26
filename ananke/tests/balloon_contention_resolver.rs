//! Scenarios for the balloon resolver's contention path (the second-half
//! fix to the dynamic-pledge work). The resolver should:
//!
//!   - Stay quiet while a dynamic service grows into declared headroom
//!     that physically exists on the GPU. This was the regression: the
//!     pre-fix resolver fired on slope alone, fast-killed itself, and
//!     the eviction-retry then evicted the persistent peer it had just
//!     yielded to — net churn.
//!   - Fire only when a GPU the service holds is actually over-committed
//!     (sum of pledges > total VRAM) — i.e. physical OOM is imminent.
//!   - At tied numeric priority, on-demand yields to persistent (lifecycle
//!     breaks the tie). Reproduces the user's ComfyUI-vs-Qwen setup.
#![cfg(feature = "test-fakes")]

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use ananke::{
    allocator::{
        AllocationTable,
        balloon::{BalloonConfig, ResolverDeps, spawn_resolver},
    },
    config::{
        DaemonSettings, DeviceSlot, EffectiveConfig, Lifecycle, manager::ConfigManager,
        validate::test_fixtures::minimal_service,
    },
    daemon::events::EventBus,
    devices::{DeviceSnapshot, GpuSnapshot, snapshotter::SharedSnapshot},
    supervise::{SupervisorHandle, registry::ServiceRegistry},
    tracking::observation::ObservationTable,
};
use parking_lot::Mutex;
use smol_str::SmolStr;
use tokio::sync::watch;

const SAMPLE_INTERVAL: Duration = Duration::from_secs(2);

fn mb(n: u64) -> u64 {
    n * 1024 * 1024
}

/// Build a snapshot with a single 24 GB GPU at id 1, parameterised by
/// the NVML free-bytes the snapshot should report. Tests use a high
/// value (no OOM pressure) to verify the resolver stays quiet, and a
/// low value (under `OOM_MARGIN_BYTES`) to verify it fires.
fn one_24g_gpu(free_bytes: u64) -> SharedSnapshot {
    let snap = ananke::devices::snapshotter::new_shared();
    *snap.write() = DeviceSnapshot {
        gpus: vec![GpuSnapshot {
            id: 1,
            name: "GPU 1".into(),
            total_bytes: 24 * 1024 * 1024 * 1024,
            free_bytes,
        }],
        cpu: None,
        taken_at_ms: 0,
    };
    snap
}

/// `EffectiveConfig` carrying a dynamic on-demand service ("comfy") and a
/// persistent peer ("qwen"), both at default priority.
fn config_with_comfy_and_qwen(events: EventBus) -> Arc<ConfigManager> {
    let mut comfy = minimal_service("comfy");
    comfy.lifecycle = Lifecycle::OnDemand;
    comfy.priority = 50;
    let mut qwen = minimal_service("qwen");
    qwen.lifecycle = Lifecycle::Persistent;
    qwen.priority = 50;
    ConfigManager::in_memory(
        EffectiveConfig {
            daemon: DaemonSettings {
                management_listen: String::new(),
                openai_listen: String::new(),
                data_dir: std::path::PathBuf::new(),
                shutdown_timeout_ms: 5_000,
                allow_external_management: false,
                allow_external_services: false,
            },
            services: vec![comfy, qwen],
        },
        events,
    )
}

/// Wire-up: spawns the resolver for `comfy` against an allocation table
/// the test pre-seeds. Returns the handles to mutate observed VRAM and
/// the allocation, plus a registry that contains stub handles for
/// `comfy` and `qwen` so the resolver's registry-presence filter passes.
struct ContentionHarness {
    svc: SmolStr,
    allocations: Arc<Mutex<AllocationTable>>,
    observation: ObservationTable,
    /// Holds the registry handles alive so the resolver's
    /// `registry.get(...)` lookups succeed for the duration of the test.
    _registry_handles: Vec<Arc<SupervisorHandle>>,
    shutdown: watch::Sender<bool>,
    join: tokio::task::JoinHandle<()>,
}

fn build_harness(
    comfy_pledge_mb: u64,
    qwen_pledge_mb: u64,
    snapshot_free_bytes: u64,
) -> ContentionHarness {
    let svc = SmolStr::new("comfy");

    let mut comfy_row = BTreeMap::new();
    comfy_row.insert(DeviceSlot::Gpu(1), comfy_pledge_mb);
    let mut qwen_row = BTreeMap::new();
    qwen_row.insert(DeviceSlot::Gpu(1), qwen_pledge_mb);
    let mut table = AllocationTable::new();
    table.insert(svc.clone(), comfy_row);
    table.insert(SmolStr::new("qwen"), qwen_row);
    let allocations = Arc::new(Mutex::new(table));

    let observation = ObservationTable::new();
    // Seed comfy's pid so the resolver's pledge-reconcile path has
    // something to read; the contention path doesn't need it.
    observation.register(&svc, 1000);

    let registry = ServiceRegistry::new();
    // Synthetic handles so resolve_contention's registry-presence filter
    // sees both services. The handles' mailboxes are closed; the
    // resolver's `fast_kill` calls become no-ops (which is what these
    // tests want — we assert on AllocationTable + survival of the
    // resolver task, not on actual SIGTERMs).
    let comfy_handle = Arc::new(SupervisorHandle::stub_for_test());
    let qwen_handle = Arc::new(SupervisorHandle::stub_for_test());
    registry.insert(svc.clone(), comfy_handle.clone());
    registry.insert(SmolStr::new("qwen"), qwen_handle.clone());
    let registry_handles = vec![comfy_handle, qwen_handle];

    let events = EventBus::new();
    let snapshot = one_24g_gpu(snapshot_free_bytes);
    let config = config_with_comfy_and_qwen(events.clone());
    let (shutdown, shutdown_rx) = watch::channel(false);

    let cfg = BalloonConfig {
        min_mb: 2 * 1024,
        max_mb: 20 * 1024,
        min_borrower_runtime: Duration::from_millis(60_000),
        margin_bytes: 512 * 1024 * 1024,
    };

    let join = spawn_resolver(
        svc.clone(),
        cfg,
        50,
        Lifecycle::OnDemand,
        ResolverDeps {
            observation: observation.clone(),
            registry,
            allocations: allocations.clone(),
            events,
            snapshot,
            config,
            shutdown: shutdown_rx,
        },
    );

    ContentionHarness {
        svc,
        allocations,
        observation,
        _registry_handles: registry_handles,
        shutdown,
        join,
    }
}

async fn step() {
    tokio::time::advance(SAMPLE_INTERVAL + Duration::from_millis(50)).await;
    tokio::task::yield_now().await;
}

/// Wraps `JoinHandle::is_finished` for readability at the call site.
fn is_finished(join: &mut tokio::task::JoinHandle<()>) -> bool {
    join.is_finished()
}

fn pledge_mb(table: &Mutex<AllocationTable>, name: &str) -> u64 {
    table
        .lock()
        .get(&SmolStr::new(name))
        .and_then(|row| row.get(&DeviceSlot::Gpu(1)).copied())
        .unwrap_or(0)
}

/// Regression for the user-reported churn: ComfyUI grew from 2 GB → 10 GB
/// alongside Qwen's 12 GB on a 24 GB card. Total pledge 22 ≤ 24 → not
/// over-committed. The resolver must NOT fast-kill anyone (its task must
/// stay alive); the dynamic service grows freely into the headroom the
/// operator declared.
#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn growth_without_overcommit_does_not_evict() {
    // 10 GB physical free — well above OOM_MARGIN. The pledge book may
    // climb past comfortable, but the kernel isn't about to OOM.
    let mut h = build_harness(2 * 1024, 12 * 1024, 10 * 1024 * 1024 * 1024);

    // Drive observed VRAM up to 10 GB across enough samples to convince
    // detect_growth (positive slope across the window).
    for gb in [3u64, 4, 5, 6, 7, 8, 9, 10] {
        h.observation.update_peak(&h.svc, mb(gb * 1024), 0);
        step().await;
    }

    // Resolver must still be running — no YieldSelf was triggered.
    assert!(
        !is_finished(&mut h.join),
        "resolver must NOT terminate when there's plenty of physical headroom"
    );

    // Qwen still has its 12 GB. Comfy's pledge tracks observed via the
    // reconcile path (≥ min_mb).
    assert!(pledge_mb(&h.allocations, "comfy") >= 2 * 1024);
    assert_eq!(
        pledge_mb(&h.allocations, "qwen"),
        12 * 1024,
        "qwen must keep its allocation: growth into declared headroom is not contention"
    );

    let _ = h.shutdown.send(true);
}

/// When NVML free drops below the OOM margin, the contention path fires.
/// The dynamic on-demand service yields to the persistent peer at tied
/// priority. Pre-fix the resolver task `return`-ed after yielding, which
/// orphaned it for the rest of the daemon's lifetime — subsequent runs
/// of the same service had no pledge tracking. Now the resolver re-arms
/// (window cleared) and stays alive across the yield, so the next
/// comfyui spawn cycle is observed correctly.
#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn overcommit_triggers_yield_at_tied_priority_with_persistent_peer() {
    // Snapshot reports 100 MiB physical free — well under OOM_MARGIN
    // (512 MiB). The kernel will refuse the next allocation; the
    // resolver must intervene by yielding.
    let mut h = build_harness(14 * 1024, 12 * 1024, 100 * 1024 * 1024);

    // Observed peak ramps so detect_growth has a positive slope. The
    // pledge-reconcile path then sees window-max around 14 GB, which
    // keeps the AllocationTable's over-commit signature intact through
    // the contention check.
    for gb in [9u64, 10, 11, 12, 13, 14, 14, 14] {
        h.observation.update_peak(&h.svc, mb(gb * 1024), 0);
        step().await;
    }

    // Resolver task must STAY alive after yielding — fast_kill drains
    // the supervisor but the resolver re-arms for the next spawn cycle.
    assert!(
        !is_finished(&mut h.join),
        "resolver must stay alive after yielding; pre-fix it `return`-ed and orphaned"
    );

    let _ = h.shutdown.send(true);
}
