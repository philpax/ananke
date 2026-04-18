//! Shared helpers for integration tests.

// Not every integration test binary uses every symbol from this module.
#![allow(dead_code)]

pub mod echo_server;

use std::net::TcpListener;

/// Binds an ephemeral port and returns it, releasing the listener before returning.
///
/// There is a small TOCTOU window between releasing the listener and the test
/// code binding the same port; in practice this is harmless in CI because the
/// port is chosen by the OS from the ephemeral range and is not reused immediately.
pub fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().expect("local_addr").port()
}

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use ananke::activity::ActivityTable;
use ananke::allocator::AllocationTable;
use ananke::app_state::AppState;
use ananke::config::{
    DaemonSettings, DeviceSlot, EffectiveConfig, Filters, HealthSettings, Lifecycle,
    PlacementPolicy, ServiceConfig, Template,
};
use ananke::config::parse::RawService;
use ananke::db::Database;
use ananke::db::logs::spawn as spawn_batcher;
use ananke::devices::{Allocation, CpuSnapshot, DeviceSnapshot};
use ananke::service_registry::ServiceRegistry;
use ananke::snapshotter;
use ananke::supervise::{SupervisorHandle, spawn_supervisor};
use parking_lot::Mutex;
use smol_str::SmolStr;
use tempfile::TempDir;

/// Full test harness with a running echo server and real supervisors.
pub struct TestHarness {
    pub state: AppState,
    pub echo_state: echo_server::EchoState,
    pub echo_addr: std::net::SocketAddr,
    pub echo_shutdown: tokio::sync::watch::Sender<bool>,
    pub supervisors: Vec<Arc<SupervisorHandle>>,
    pub _tmp: TempDir,
}

/// Build a `TestHarness` from a list of service configs.
///
/// Each service's `private_port` is rewritten to point at a single echo
/// server so all proxy traffic is captured without spawning a real
/// llama-server. Supervisors are registered in a `ServiceRegistry` and
/// placed in the returned `AppState`.
pub async fn build_harness(services: Vec<ServiceConfig>) -> TestHarness {
    let tmp = tempfile::tempdir().unwrap();
    let db = Database::open(&tmp.path().join("a.sqlite")).unwrap();
    let batcher = spawn_batcher(db.clone());

    let echo_state = echo_server::EchoState::default();
    let echo_port = free_port();
    let echo_addr: std::net::SocketAddr = format!("127.0.0.1:{echo_port}").parse().unwrap();
    let (echo_shutdown, echo_rx) = tokio::sync::watch::channel(false);
    tokio::spawn(echo_server::serve(echo_addr, echo_state.clone(), echo_rx));

    // Rewrite private_port on each service to point at the echo server.
    let services_rewritten: Vec<ServiceConfig> = services
        .into_iter()
        .map(|mut s| {
            s.private_port = echo_port;
            s
        })
        .collect();

    let effective = Arc::new(EffectiveConfig {
        daemon: DaemonSettings {
            management_listen: "127.0.0.1:0".into(),
            openai_listen: "127.0.0.1:0".into(),
            data_dir: tmp.path().to_path_buf(),
            shutdown_timeout_ms: 5_000,
        },
        services: services_rewritten.clone(),
    });

    let activity = ActivityTable::new();
    let allocations = Arc::new(Mutex::new(AllocationTable::new()));
    let snapshot = snapshotter::new_shared();
    // Pre-seed with ample CPU memory so the allocator does not reject services
    // that declare a CPU placement. The echo server stands in for the real
    // model binary, so actual memory is never consumed.
    *snapshot.write() = DeviceSnapshot {
        gpus: vec![],
        cpu: Some(CpuSnapshot {
            total_bytes: 64 * 1024 * 1024 * 1024,
            available_bytes: 64 * 1024 * 1024 * 1024,
        }),
        taken_at_ms: 0,
    };

    let registry = ServiceRegistry::new();
    let mut supervisors = Vec::new();
    for svc in &services_rewritten {
        let service_id = db.upsert_service(&svc.name, 0).unwrap();
        let alloc = Allocation::from_override(&svc.placement_override);
        let last_activity = activity.get_or_init(&svc.name);
        let handle = Arc::new(spawn_supervisor(
            svc.clone(),
            alloc,
            db.clone(),
            batcher.clone(),
            service_id,
            last_activity,
            snapshot.clone(),
            allocations.clone(),
        ));
        registry.insert(svc.name.clone(), handle.clone());
        supervisors.push(handle);
    }

    let state = AppState {
        config: effective,
        registry,
        allocations,
        snapshot,
        activity,
        db,
    };

    TestHarness {
        state,
        echo_state,
        echo_addr,
        echo_shutdown,
        supervisors,
        _tmp: tmp,
    }
}

/// Build a minimal on-demand `ServiceConfig` backed by a fake model path.
///
/// `placement_override` is `{Cpu: 100}` so the allocator never blocks.
/// Health probes hit `/health` with a short interval so warming completes
/// quickly against the echo server.
pub fn minimal_llama_service(name: &str, port: u16) -> ServiceConfig {
    let mut placement = BTreeMap::new();
    placement.insert(DeviceSlot::Cpu, 100);
    ServiceConfig {
        name: SmolStr::new(name),
        template: Template::LlamaCpp,
        port,
        private_port: 0,
        lifecycle: Lifecycle::OnDemand,
        priority: 50,
        health: HealthSettings {
            http_path: "/health".into(),
            timeout_ms: 5_000,
            probe_interval_ms: 200,
        },
        placement_override: placement,
        placement_policy: PlacementPolicy::CpuOnly,
        idle_timeout_ms: 60_000,
        warming_grace_ms: 100,
        drain_timeout_ms: 1_000,
        extended_stream_drain_ms: 1_000,
        max_request_duration_ms: 5_000,
        filters: Filters::default(),
        raw: RawService {
            name: Some(SmolStr::new(name)),
            template: Some(SmolStr::new("llama-cpp")),
            model: Some(PathBuf::from("/fake/model.gguf")),
            port: Some(port),
            ..Default::default()
        },
    }
}

impl TestHarness {
    /// Shut down the echo server and all supervisors.
    pub async fn cleanup(self) {
        let _ = self.echo_shutdown.send(true);
        for sup in &self.supervisors {
            sup.shutdown().await;
        }
    }
}
