//! Top-level daemon orchestration: wires config, DB, devices, supervisors,
//! proxies, signals, and retention together.
//!
//! Linux-coupled via `/proc` orphan reconciliation and the `signals` submodule.

pub mod app_state;
pub mod events;
pub mod signals;

use std::{net::SocketAddr, path::PathBuf, sync::Arc, time::Duration};

use parking_lot::Mutex;
use tokio::{net::TcpListener, sync::watch};
use tracing::{info, warn};

use crate::{
    allocator::AllocationTable,
    config::{Migration, manager::ConfigManager},
    daemon::{
        app_state::AppState,
        signals::{ShutdownKind, await_shutdown},
    },
    db::{Database, logs::spawn as spawn_batcher, retention},
    devices::{GpuProbe, cpu, nvml::NvmlProbe, snapshotter},
    errors::ExpectedError,
    oneshot::{OneshotRegistry, PortPool},
    supervise::{SupervisorHandle, orphans::reconcile, registry::ServiceRegistry},
    tracking::{activity::ActivityTable, inflight::InflightTable},
};

pub async fn run() -> Result<(), ExpectedError> {
    init_tracing();

    let cli_config = parse_cli_config_arg();
    let config_path = crate::config::resolve_from_env(cli_config.as_deref())?;
    info!(config_path = %config_path.display(), "resolved config path");

    let events = crate::daemon::events::EventBus::new();
    let config = ConfigManager::open(config_path.clone(), events.clone()).await?;
    let migrations = config.take_boot_migrations();
    let effective = config.effective().clone();
    let db = Database::open(&effective.daemon.data_dir.join("ananke.sqlite")).await?;
    apply_migrations(&db, &migrations).await;

    let probe: Option<Arc<dyn GpuProbe>> = match NvmlProbe::init() {
        Ok(p) => {
            for g in p.list() {
                info!(gpu = g.id, name = %g.name, total_bytes = g.total_bytes, "detected GPU");
            }
            Some(Arc::new(p) as Arc<dyn GpuProbe>)
        }
        Err(e) => {
            warn!(error = %e, "NVML init failed; falling back to CPU-only");
            None
        }
    };
    let system = crate::system::SystemDeps::local();
    if let Ok(m) = cpu::read(system.proc.as_ref()) {
        info!(
            total = m.total_bytes,
            avail = m.available_bytes,
            "CPU memory"
        );
    }

    for disposition in reconcile(system.proc.as_ref(), &db).await {
        info!(?disposition, "orphan reconcile");
    }

    let batcher = spawn_batcher(db.clone());
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let rolling = crate::tracking::rolling::RollingTable::with_events(events.clone());
    let observation = crate::tracking::observation::ObservationTable::new();
    let registry = ServiceRegistry::new();

    let shared_snapshot = snapshotter::new_shared();
    let snapshotter_join = snapshotter::spawn(
        shared_snapshot.clone(),
        probe,
        observation.clone(),
        registry.clone(),
        system.proc.clone(),
        shutdown_rx.clone(),
    );

    let activity = ActivityTable::new();
    let allocations = Arc::new(Mutex::new(AllocationTable::new()));
    let inflight = InflightTable::new();

    // Persistent services start in priority-desc + name-asc order;
    // on_demand services are registered but remain idle.
    let mut ordered = effective.services.clone();
    ordered.sort_by(|a, b| {
        b.priority
            .cmp(&a.priority)
            .then_with(|| a.name.cmp(&b.name))
    });

    let port_pool = Arc::new(Mutex::new(PortPool::new(18000..19000)));
    let oneshots = OneshotRegistry::new();

    // Build AppState early — it holds only Arc-backed handles, and the
    // provisioner wants to pull `ProvisioningDeps` out of it. Supervisors
    // get registered into `app_state.registry` as `provision_service`
    // inserts them, so post-boot the state is complete.
    let app_state = AppState {
        config: config.clone(),
        registry: registry.clone(),
        allocations: allocations.clone(),
        snapshot: shared_snapshot.clone(),
        activity: activity.clone(),
        rolling: rolling.clone(),
        observation: observation.clone(),
        db: db.clone(),
        inflight: inflight.clone(),
        port_pool,
        oneshots,
        batcher: batcher.clone(),
        events: events.clone(),
        system: system.clone(),
    };

    let provisioning_deps =
        crate::supervise::provision::ProvisioningDeps::from_state(&app_state, shutdown_rx.clone());

    let mut supervisors: Vec<Arc<SupervisorHandle>> = Vec::new();
    let mut proxy_tasks = Vec::new();
    let mut balloon_tasks = Vec::new();
    for svc in ordered {
        let provisioned =
            crate::supervise::provision::provision_service(svc, &provisioning_deps).await?;
        supervisors.push(provisioned.handle);
        proxy_tasks.push(provisioned.proxy_task);
        if let Some(balloon) = provisioned.balloon_task {
            balloon_tasks.push(balloon);
        }
    }

    // Drain-on-remove + spawn-on-add reconciler. Threads the same
    // `ProvisioningDeps` that the boot loop used so a reload-added
    // service is spawned through exactly the same path.
    let reconciler_task = crate::supervise::reconciler::spawn(
        events.clone(),
        config.clone(),
        registry.clone(),
        Some(provisioning_deps.clone()),
        shutdown_rx.clone(),
    );

    // OpenAI listener.
    let openai_listen: SocketAddr =
        effective
            .daemon
            .openai_listen
            .parse()
            .map_err(|e: std::net::AddrParseError| {
                ExpectedError::bind_failed(effective.daemon.openai_listen.clone(), e.to_string())
            })?;
    let openai_router = crate::api::openai::router(app_state.clone());
    let openai_listener = TcpListener::bind(openai_listen)
        .await
        .map_err(|e| ExpectedError::bind_failed(openai_listen.to_string(), e.to_string()))?;
    let openai_shutdown = shutdown_rx.clone();
    let openai_server = tokio::spawn(async move {
        let _ = axum::serve(openai_listener, openai_router)
            .with_graceful_shutdown(wait_shutdown(openai_shutdown))
            .await;
    });
    info!(%openai_listen, "openai listener bound");

    // Management listener.
    let mgmt_listen: SocketAddr =
        effective
            .daemon
            .management_listen
            .parse()
            .map_err(|e: std::net::AddrParseError| {
                ExpectedError::bind_failed(
                    effective.daemon.management_listen.clone(),
                    e.to_string(),
                )
            })?;
    let mgmt_router = crate::api::management::router(app_state.clone());
    let mgmt_listener = TcpListener::bind(mgmt_listen)
        .await
        .map_err(|e| ExpectedError::bind_failed(mgmt_listen.to_string(), e.to_string()))?;
    let mgmt_shutdown = shutdown_rx.clone();
    let mgmt_server = tokio::spawn(async move {
        let _ = axum::serve(mgmt_listener, mgmt_router)
            .with_graceful_shutdown(wait_shutdown(mgmt_shutdown))
            .await;
    });
    if !mgmt_listen.ip().is_loopback() {
        warn!(
            bind = %mgmt_listen,
            "management API reachable from the network — no authentication enabled; \
             trust your network perimeter (e.g. Tailscale) or terminate TLS + auth at a reverse proxy"
        );
    }
    info!(%mgmt_listen, "management listener bound");
    if effective.daemon.allow_external_services {
        warn!(
            "per-service reverse proxies reachable from the network — no authentication enabled; \
             trust your network perimeter (e.g. Tailscale) or terminate TLS + auth at a reverse proxy"
        );
    }

    let retention_task = tokio::spawn(retention::run_loop(db.clone(), shutdown_rx.clone()));
    let persistent_watcher_task = tokio::spawn(crate::supervise::persistent_watcher::run_loop(
        app_state.clone(),
        shutdown_rx.clone(),
    ));

    let shutdown_kind = await_shutdown().await;
    info!(?shutdown_kind, "shutdown initiated");
    let _ = shutdown_tx.send(true);

    let drain_bound = match shutdown_kind {
        ShutdownKind::Graceful => Duration::from_millis(effective.daemon.shutdown_timeout_ms),
        ShutdownKind::Emergency => crate::daemon::signals::grace_for(shutdown_kind),
    };
    let _ = tokio::time::timeout(drain_bound, async {
        for sup in &supervisors {
            sup.shutdown().await;
        }
    })
    .await;

    for t in proxy_tasks {
        t.abort();
        let _ = t.await;
    }
    openai_server.abort();
    let _ = openai_server.await;
    mgmt_server.abort();
    let _ = mgmt_server.await;
    snapshotter_join.abort();
    let _ = snapshotter_join.await;
    retention_task.abort();
    let _ = retention_task.await;
    persistent_watcher_task.abort();
    let _ = persistent_watcher_task.await;
    for t in balloon_tasks {
        t.abort();
        let _ = t.await;
    }
    reconciler_task.abort();
    let _ = reconciler_task.await;

    batcher.flush().await;
    Ok(())
}

async fn wait_shutdown(mut rx: watch::Receiver<bool>) {
    while rx.changed().await.is_ok() {
        if *rx.borrow() {
            return;
        }
    }
}

fn parse_cli_config_arg() -> Option<PathBuf> {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--config" {
            return args.next().map(PathBuf::from);
        }
        if let Some(rest) = a.strip_prefix("--config=") {
            return Some(PathBuf::from(rest));
        }
    }
    None
}

async fn apply_migrations(db: &Database, migs: &[Migration]) {
    let now = crate::tracking::now_unix_ms();
    for m in migs {
        if let Err(e) = db.reparent(&m.old_name, &m.new_name, now).await {
            warn!(old = %m.old_name, new = %m.new_name, error = %e, "migrate_from failed");
        } else {
            info!(old = %m.old_name, new = %m.new_name, "migrated service");
        }
    }
}

fn init_tracing() {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(true)
        .with_writer(std::io::stderr)
        .init();
}
