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
use ananke::inflight::InflightTable;
use ananke::config::parse::RawService;
use ananke::config::{
    AllocationMode, DaemonSettings, DeviceSlot, EffectiveConfig, Filters, HealthSettings,
    Lifecycle, PlacementPolicy, ServiceConfig, Template,
};
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

    let rolling = ananke::rolling::RollingTable::new();
    let observation = ananke::observation::ObservationTable::new();
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
            rolling.clone(),
            observation.clone(),
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
        rolling,
        observation,
        db,
        inflight: InflightTable::new(),
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
        allocation_mode: AllocationMode::None,
        command: None,
        workdir: None,
        openai_compat: true,
        raw: RawService {
            name: Some(SmolStr::new(name)),
            template: Some(SmolStr::new("llama-cpp")),
            model: Some(PathBuf::from("/fake/model.gguf")),
            port: Some(port),
            ..Default::default()
        },
    }
}

/// Build a minimal on-demand `ServiceConfig` with a capped start queue.
///
/// Identical to `minimal_llama_service` except `start_queue_depth` is set to
/// `depth`, which limits how many concurrent callers may wait for the same
/// in-flight start before the supervisor returns `QueueFull`.
pub fn service_with_queue_depth(name: &str, port: u16, depth: usize) -> ServiceConfig {
    let mut s = minimal_llama_service(name, port);
    s.raw.start_queue_depth = Some(depth);
    s
}

/// Build a `TestHarness` and then overwrite the snapshot with a caller-supplied
/// one. Useful for tests that need to simulate specific device layouts (e.g.,
/// two GPUs with known free bytes) before issuing requests.
pub async fn build_harness_with_snapshot(
    services: Vec<ServiceConfig>,
    snapshot: ananke::devices::DeviceSnapshot,
) -> TestHarness {
    let h = build_harness(services).await;
    *h.state.snapshot.write() = snapshot;
    h
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

/// Synthetic GGUF builder for integration tests.
///
/// Writes a minimal but structurally valid GGUF v3 file to a path of the
/// caller's choosing. The produced file contains only the header, KV section,
/// and tensor-info table — there is no actual tensor data, which is fine
/// because the reader never mmaps the data region.
pub mod synth_gguf {
    use std::path::Path;

    pub struct Builder {
        /// Accumulated KV + tensor-info bytes (written after the fixed header).
        buf: Vec<u8>,
        n_tensors: u64,
        n_kv: u64,
    }

    impl Builder {
        pub fn new() -> Self {
            Self {
                buf: Vec::new(),
                n_tensors: 0,
                n_kv: 0,
            }
        }

        /// Append `general.architecture = name` as a string KV entry.
        pub fn arch(self, name: &str) -> Self {
            self.kv_string("general.architecture", name)
        }

        /// Append a u32 KV entry (type tag 4 in GGUF).
        pub fn kv_u32(mut self, key: &str, val: u32) -> Self {
            self.n_kv += 1;
            write_string(&mut self.buf, key);
            self.buf.extend_from_slice(&4u32.to_le_bytes());
            self.buf.extend_from_slice(&val.to_le_bytes());
            self
        }

        /// Append a u64 KV entry (type tag 10 in GGUF).
        pub fn kv_u64(mut self, key: &str, val: u64) -> Self {
            self.n_kv += 1;
            write_string(&mut self.buf, key);
            self.buf.extend_from_slice(&10u32.to_le_bytes());
            self.buf.extend_from_slice(&val.to_le_bytes());
            self
        }

        /// Append a string KV entry (type tag 8 in GGUF).
        pub fn kv_string(mut self, key: &str, val: &str) -> Self {
            self.n_kv += 1;
            write_string(&mut self.buf, key);
            self.buf.extend_from_slice(&8u32.to_le_bytes());
            write_string(&mut self.buf, val);
            self
        }

        /// Append a tensor-info entry with dtype F16 (tag 1) and `elements`
        /// elements. The byte_size seen by the reader is `elements * 2`.
        pub fn tensor_f16(mut self, name: &str, elements: u64) -> Self {
            self.n_tensors += 1;
            write_string(&mut self.buf, name);
            self.buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
            self.buf.extend_from_slice(&elements.to_le_bytes()); // dim[0]
            self.buf.extend_from_slice(&1u32.to_le_bytes()); // dtype = F16
            self.buf.extend_from_slice(&0u64.to_le_bytes()); // offset within data
            self
        }

        /// Serialise and write the complete GGUF file to `path`.
        pub fn write_to(self, path: &Path) {
            let mut out = Vec::<u8>::new();
            out.extend_from_slice(b"GGUF");
            out.extend_from_slice(&3u32.to_le_bytes()); // version
            out.extend_from_slice(&self.n_tensors.to_le_bytes());
            out.extend_from_slice(&self.n_kv.to_le_bytes());
            out.extend_from_slice(&self.buf);
            std::fs::write(path, &out).unwrap();
        }
    }

    impl Default for Builder {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Create a named temp file with `.gguf` suffix. The caller keeps the
    /// `NamedTempFile` alive for the duration of the test.
    pub fn tempfile(prefix: &str) -> tempfile::NamedTempFile {
        tempfile::Builder::new()
            .prefix(prefix)
            .suffix(".gguf")
            .tempfile()
            .unwrap()
    }

    fn write_string(v: &mut Vec<u8>, s: &str) {
        v.extend_from_slice(&(s.len() as u64).to_le_bytes());
        v.extend_from_slice(s.as_bytes());
    }
}
