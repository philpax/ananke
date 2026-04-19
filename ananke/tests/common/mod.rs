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

use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

use ananke::{
    allocator::AllocationTable,
    config::{
        AllocationMode, DaemonSettings, DeviceSlot, EffectiveConfig, Filters, HealthSettings,
        Lifecycle, LlamaCppConfig, PlacementPolicy, ServiceConfig, TemplateConfig,
        manager::ConfigManager,
        parse::{DEFAULT_START_QUEUE_DEPTH, EstimationConfig, SamplingConfig},
    },
    daemon::app_state::AppState,
    db::{Database, logs::spawn as spawn_batcher},
    devices::{Allocation, CpuSnapshot, DeviceSnapshot, snapshotter},
    supervise::{SupervisorHandle, registry::ServiceRegistry, spawn_supervisor},
    tracking::{activity::ActivityTable, inflight::InflightTable},
};
use parking_lot::Mutex;
use smol_str::SmolStr;

/// Full test harness with a running echo server and real supervisors.
pub struct TestHarness {
    pub state: AppState,
    pub echo_state: echo_server::EchoState,
    pub echo_addr: std::net::SocketAddr,
    pub echo_shutdown: tokio::sync::watch::Sender<bool>,
    pub supervisors: Vec<Arc<SupervisorHandle>>,
    /// Concrete handle to the in-memory filesystem shared by the supervisors
    /// and AppState. Tests that need the estimator to find a GGUF at a
    /// particular path can insert bytes here before issuing a request.
    pub fs: ananke::system::InMemoryFs,
    /// Concrete handle to the in-memory process spawner. Tests that want to
    /// assert "service X's child was terminated by the reconciler" inspect
    /// this directly rather than polling OS pids.
    pub process_spawner: Arc<ananke::system::FakeSpawner>,
    /// Shutdown channel for the reload reconciler task.
    pub reconciler_shutdown: tokio::sync::watch::Sender<bool>,
    /// Join handle for the reload reconciler; awaited in `cleanup`.
    pub reconciler_join: tokio::task::JoinHandle<()>,
}

/// Build a `TestHarness` from a list of service configs.
///
/// Each service's `private_port` is rewritten to point at a single echo
/// server so all proxy traffic is captured without spawning a real
/// llama-server. Supervisors are registered in a `ServiceRegistry` and
/// placed in the returned `AppState`.
pub async fn build_harness(services: Vec<ServiceConfig>) -> TestHarness {
    // Fully in-memory: DB, filesystem, and the synthetic data_dir path.
    // Nothing below this line touches the real disk.
    let db = Database::open_in_memory().await.unwrap();
    let batcher = spawn_batcher(db.clone());
    let (system, fakes) = ananke::system::SystemDeps::fake();
    let fs_concrete = fakes.fs;
    let fake_spawner = fakes.process_spawner;

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
            // Synthetic path — nothing writes to it because `fs` is in-memory.
            data_dir: std::path::PathBuf::from("/tmp/ananke-test"),
            shutdown_timeout_ms: 5_000,
            allow_external_management: false,
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

    let rolling = ananke::tracking::rolling::RollingTable::new();
    let observation = ananke::tracking::observation::ObservationTable::new();
    let registry = ServiceRegistry::new();
    let events = ananke::daemon::events::EventBus::new();

    // The test harness constructs an `EffectiveConfig` directly; wrap it
    // in an in-memory `ConfigManager` so handlers that go through
    // `state.config.effective()` see the same snapshot as the supervisors.
    // Reuses the shared `events` bus so WS subscribers observe both
    // supervisor-originated and config-originated events on one channel.
    let config_manager = ConfigManager::in_memory((*effective).clone(), events.clone());

    let deps = ananke::supervise::SupervisorDeps {
        db: db.clone(),
        batcher: batcher.clone(),
        snapshot: snapshot.clone(),
        allocations: allocations.clone(),
        rolling: rolling.clone(),
        observation: observation.clone(),
        registry: registry.clone(),
        effective: effective.clone(),
        events: events.clone(),
        system: system.clone(),
    };
    let mut supervisors = Vec::new();
    for svc in &services_rewritten {
        let service_id = db.upsert_service(&svc.name, 0).await.unwrap();
        let init = ananke::supervise::SupervisorInit {
            svc: svc.clone(),
            allocation: Allocation::from_override(&svc.placement_override),
            service_id,
            last_activity: activity.get_or_init(&svc.name),
            inflight: ananke::tracking::inflight::InflightTable::new().counter(&svc.name),
        };
        let handle = Arc::new(spawn_supervisor(init, deps.clone()));
        registry.insert(svc.name.clone(), handle.clone());
        supervisors.push(handle);
    }

    let state = AppState {
        config: config_manager,
        registry: registry.clone(),
        allocations,
        snapshot,
        activity,
        rolling,
        observation,
        db,
        inflight: InflightTable::new(),
        port_pool: Arc::new(Mutex::new(ananke::oneshot::PortPool::new(18000..19000))),
        oneshots: ananke::oneshot::OneshotRegistry::new(),
        batcher,
        events: events.clone(),
        system: system.clone(),
    };

    // Drain-on-remove reconciler: matches what daemon::run wires up, so
    // integration tests can PUT a synthetic config that drops a service and
    // see the supervisor drained + removed from the registry.
    let (reconciler_shutdown, reconciler_rx) = tokio::sync::watch::channel(false);
    let reconciler_join = ananke::supervise::reconciler::spawn(
        events,
        state.config.clone(),
        registry,
        reconciler_rx,
    );

    TestHarness {
        state,
        echo_state,
        echo_addr,
        echo_shutdown,
        supervisors,
        fs: fs_concrete,
        process_spawner: fake_spawner,
        reconciler_shutdown,
        reconciler_join,
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
        gpu_allow: Vec::new(),
        idle_timeout_ms: 60_000,
        warming_grace_ms: 100,
        drain_timeout_ms: 1_000,
        extended_stream_drain_ms: 1_000,
        max_request_duration_ms: 5_000,
        filters: Filters::default(),
        allocation_mode: AllocationMode::None,
        openai_compat: true,
        description: None,
        start_queue_depth: DEFAULT_START_QUEUE_DEPTH,
        extra_args: Vec::new(),
        env: BTreeMap::new(),
        template_config: TemplateConfig::LlamaCpp(Box::new(LlamaCppConfig {
            model: PathBuf::from("/fake/model.gguf"),
            mmproj: None,
            context: None,
            n_gpu_layers: None,
            n_cpu_moe: None,
            flash_attn: None,
            cache_type_k: None,
            cache_type_v: None,
            mmap: None,
            mlock: None,
            parallel: None,
            batch_size: None,
            ubatch_size: None,
            threads: None,
            threads_batch: None,
            jinja: None,
            chat_template_file: None,
            override_tensor: Vec::new(),
            sampling: SamplingConfig::default(),
            estimation: EstimationConfig::default(),
        })),
    }
}

/// Build a minimal on-demand `ServiceConfig` with a capped start queue.
///
/// Identical to `minimal_llama_service` except `start_queue_depth` is set to
/// `depth`, which limits how many concurrent callers may wait for the same
/// in-flight start before the supervisor returns `QueueFull`.
pub fn service_with_queue_depth(name: &str, port: u16, depth: usize) -> ServiceConfig {
    let mut s = minimal_llama_service(name, port);
    s.start_queue_depth = depth;
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
        let _ = self.reconciler_shutdown.send(true);
        self.reconciler_join.abort();
        let _ = self.reconciler_join.await;
        for sup in &self.supervisors {
            sup.shutdown().await;
        }
    }

    /// Same as `cleanup` but named `shutdown` for tests that prefer that terminology.
    pub async fn shutdown(self) {
        self.cleanup().await;
    }

    /// Bind an ephemeral TCP listener, serve the management router on it, and
    /// return the bound address. Useful for WebSocket tests that need a real
    /// TCP connection rather than an in-process `oneshot` call.
    pub async fn spawn_management_server(&self) -> std::net::SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let router = ananke::api::management::register(axum::Router::new(), self.state.clone());
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });
        addr
    }
}

/// Build a `http://…` URL for a path served by the management server at `addr`.
pub fn management_url(addr: std::net::SocketAddr, path: &str) -> String {
    format!("http://{addr}{path}")
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

        /// Serialise the complete GGUF header into a byte buffer.
        pub fn build(self) -> Vec<u8> {
            let mut out = Vec::<u8>::new();
            out.extend_from_slice(b"GGUF");
            out.extend_from_slice(&3u32.to_le_bytes()); // version
            out.extend_from_slice(&self.n_tensors.to_le_bytes());
            out.extend_from_slice(&self.n_kv.to_le_bytes());
            out.extend_from_slice(&self.buf);
            out
        }

        /// Construct a single-entry [`InMemoryFs`] containing the built
        /// GGUF bytes at `path`. Convenience for tests that want to pass a
        /// concrete path to `estimate_from_path` or `gguf::read` without
        /// touching disk.
        pub fn into_in_memory_fs(self, path: &Path) -> ananke::system::InMemoryFs {
            ananke::system::InMemoryFs::new().with(path.to_path_buf(), self.build())
        }
    }

    impl Default for Builder {
        fn default() -> Self {
            Self::new()
        }
    }

    fn write_string(v: &mut Vec<u8>, s: &str) {
        v.extend_from_slice(&(s.len() as u64).to_le_bytes());
        v.extend_from_slice(s.as_bytes());
    }
}
