//! Shared application state passed to every Axum handler via `State(...)`.

use std::sync::Arc;

use parking_lot::Mutex;
use smol_str::SmolStr;

use crate::activity::ActivityTable;
use crate::allocator::AllocationTable;
use crate::config::EffectiveConfig;
use crate::db::Database;
use crate::db::logs::BatcherHandle;
use crate::inflight::InflightTable;
use crate::observation::ObservationTable;
use crate::oneshot::{OneshotId, OneshotRecord, OneshotRegistry, PortPool};
use crate::rolling::RollingTable;
use crate::service_registry::ServiceRegistry;
use crate::snapshotter::SharedSnapshot;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<EffectiveConfig>,
    pub registry: ServiceRegistry,
    pub allocations: Arc<Mutex<AllocationTable>>,
    pub snapshot: SharedSnapshot,
    pub activity: ActivityTable,
    pub rolling: RollingTable,
    pub observation: ObservationTable,
    pub db: Database,
    pub inflight: InflightTable,
    pub port_pool: Arc<Mutex<PortPool>>,
    pub oneshots: OneshotRegistry,
    pub batcher: BatcherHandle,
}

impl AppState {
    /// Spawn a oneshot service from a validated request.
    pub async fn spawn_oneshot(
        &self,
        id: OneshotId,
        req: crate::oneshot::handlers::OneshotRequest,
        port: u16,
        ttl_ms: u64,
    ) -> Result<(), String> {
        use std::collections::BTreeMap;
        use std::path::PathBuf;

        use crate::config::parse::RawService;
        use crate::config::validate::{
            AllocationMode, Filters, HealthSettings, Lifecycle, PlacementPolicy, ServiceConfig,
            Template,
        };
        use crate::devices::Allocation;

        // Resolve template.
        let template = match req.template.as_str() {
            "llama-cpp" => Template::LlamaCpp,
            "command" => Template::Command,
            other => return Err(format!("unknown template `{other}`")),
        };

        // Resolve allocation mode.
        let allocation_mode = match (template, req.allocation.mode.as_deref()) {
            (Template::Command, Some("static")) => {
                let gb = req
                    .allocation
                    .vram_gb
                    .ok_or("allocation.mode=static requires vram_gb")?;
                AllocationMode::Static {
                    vram_mb: (gb * 1024.0) as u64,
                }
            }
            (Template::Command, Some("dynamic")) => {
                let min = req
                    .allocation
                    .min_vram_gb
                    .ok_or("allocation.mode=dynamic requires min_vram_gb")?;
                let max = req
                    .allocation
                    .max_vram_gb
                    .ok_or("allocation.mode=dynamic requires max_vram_gb")?;
                AllocationMode::Dynamic {
                    min_mb: (min * 1024.0) as u64,
                    max_mb: (max * 1024.0) as u64,
                    min_borrower_runtime_ms: 60_000,
                }
            }
            (Template::LlamaCpp, _) => AllocationMode::None,
            (Template::Command, Some(other)) => {
                return Err(format!("unknown allocation.mode `{other}`"));
            }
            (Template::Command, None) => {
                return Err("command template requires allocation.mode".into());
            }
        };

        // Build the private port from the allocated port offset — oneshots
        // use the same port for public and private since they are directly
        // spawned without a wrapping proxy.
        let private_port = port;

        // Construct the ServiceConfig directly, bypassing validate()'s
        // port-uniqueness checks which would conflict with the running config.
        let svc = ServiceConfig {
            name: id.clone(),
            template,
            port,
            private_port,
            lifecycle: Lifecycle::OnDemand,
            priority: req.priority.unwrap_or(50),
            health: HealthSettings {
                http_path: "/v1/models".into(),
                timeout_ms: 180_000,
                probe_interval_ms: 5_000,
            },
            placement_override: BTreeMap::new(),
            placement_policy: PlacementPolicy::GpuOnly,
            filters: Filters::default(),
            idle_timeout_ms: ttl_ms + 60_000,
            warming_grace_ms: 60_000,
            drain_timeout_ms: 30_000,
            extended_stream_drain_ms: 30_000,
            max_request_duration_ms: 300_000,
            allocation_mode,
            command: req.command,
            workdir: req.workdir.map(PathBuf::from),
            openai_compat: false,
            raw: RawService {
                name: Some(id.clone()),
                template: Some(SmolStr::new(&req.template)),
                port: Some(port),
                ..Default::default()
            },
        };

        // Upsert DB record.
        let now_ms = chrono_like_now_ms();
        let service_id = self
            .db
            .upsert_service(&svc.name, now_ms)
            .await
            .map_err(|e| e.to_string())?;
        let _ = self.db.with_conn(|c| {
            c.execute(
                "INSERT OR IGNORE INTO oneshots(id, service_id, submitted_at, ttl_ms) \
                 VALUES (?1, ?2, ?3, ?4)",
                (id.as_str(), service_id, now_ms, ttl_ms as i64),
            )
        });

        // Spawn supervisor.
        let allocation = Allocation::from_override(&svc.placement_override);
        let last_activity = self.activity.get_or_init(&svc.name);
        let inflight_counter = self.inflight.counter(&svc.name);
        let handle = Arc::new(crate::supervise::spawn_supervisor(
            svc.clone(),
            allocation,
            self.db.clone(),
            self.batcher.clone(),
            service_id,
            last_activity,
            self.snapshot.clone(),
            self.allocations.clone(),
            self.rolling.clone(),
            self.observation.clone(),
            inflight_counter,
            self.registry.clone(),
            self.config.clone(),
        ));
        self.registry.insert(svc.name.clone(), handle.clone());

        // Register in the oneshot map.
        let record = OneshotRecord {
            id: id.clone(),
            service_name: svc.name.clone(),
            port,
            ttl_ms,
            started_at_ms: now_ms as u64,
        };
        self.oneshots.insert(record);

        // Kick-start the supervisor.
        let handle2 = handle.clone();
        tokio::spawn(async move {
            let _ = handle2.ensure().await;
        });

        // Spawn the TTL watcher. We use a dedicated watch channel; the watcher
        // exits when the TTL fires. The sender is moved into the spawned task so
        // it lives as long as the oneshot is pending. The watcher task owns sh_tx
        // and drops it only when it exits, keeping the receiver alive throughout.
        let (sh_tx, sh_rx) = tokio::sync::watch::channel(false);
        let id_for_task = id.clone();
        let svc_name = svc.name.clone();
        let ttl_dur = std::time::Duration::from_millis(ttl_ms);
        let registry = self.registry.clone();
        let oneshots = self.oneshots.clone();
        let db = self.db.clone();
        let port_pool = self.port_pool.clone();
        tokio::spawn(async move {
            // Hold sh_tx so the receiver stays open; drop it when we exit.
            let _sh_tx = sh_tx;
            let _ = crate::oneshot::ttl::spawn_watcher(
                id_for_task,
                svc_name,
                ttl_dur,
                registry,
                oneshots,
                db,
                port_pool,
                port,
                sh_rx,
            )
            .await;
        });

        Ok(())
    }
}

fn chrono_like_now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}
