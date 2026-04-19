//! Top-level daemon orchestration: wires config, DB, devices, supervisors,
//! proxies, signals, and retention together.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use tokio::net::TcpListener;
use tokio::sync::watch;
use tracing::{error, info, warn};

use crate::activity::ActivityTable;
use crate::allocator::AllocationTable;
use crate::app_state::AppState;
use crate::config::{AllocationMode, Lifecycle, Migration, load_config};
use crate::db::Database;
use crate::db::logs::spawn as spawn_batcher;
use crate::devices::{Allocation, GpuProbe, cpu, nvml::NvmlProbe};
use crate::errors::ExpectedError;
use crate::inflight::InflightTable;
use crate::oneshot::{OneshotRegistry, PortPool};
use crate::proxy;
use crate::retention;
use crate::service_registry::ServiceRegistry;
use crate::signals::{ShutdownKind, await_shutdown};
use crate::snapshotter;
use crate::supervise::{
    EnsureResponse, StartFailureKind, StartOutcome, SupervisorHandle, orphans::reconcile,
    spawn_supervisor,
};

pub async fn run() -> Result<(), ExpectedError> {
    init_tracing();

    let cli_config = parse_cli_config_arg();
    let config_path = crate::config::resolve_from_env(cli_config.as_deref())?;
    info!(config_path = %config_path.display(), "resolved config path");

    let (effective, migrations) = load_config(&config_path)?;
    let effective = Arc::new(effective);
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
    if let Ok(m) = cpu::read() {
        info!(
            total = m.total_bytes,
            avail = m.available_bytes,
            "CPU memory"
        );
    }

    let procfs = PathBuf::from("/proc");
    for disposition in reconcile(&db, &procfs) {
        info!(?disposition, "orphan reconcile");
    }

    let batcher = spawn_batcher(db.clone());
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let rolling = crate::rolling::RollingTable::new();
    let observation = crate::observation::ObservationTable::new();
    let registry = ServiceRegistry::new();

    let shared_snapshot = snapshotter::new_shared();
    let snapshotter_join = snapshotter::spawn(
        shared_snapshot.clone(),
        probe,
        observation.clone(),
        registry.clone(),
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

    let mut supervisors: Vec<Arc<SupervisorHandle>> = Vec::new();
    let mut proxy_tasks = Vec::new();
    for svc in ordered {
        let service_id = db.upsert_service(&svc.name, now_ms()).await?;
        let allocation = Allocation::from_override(&svc.placement_override);
        let last_activity = activity.get_or_init(&svc.name);
        let inflight_counter = inflight.counter(&svc.name);
        let handle = Arc::new(spawn_supervisor(
            svc.clone(),
            allocation,
            db.clone(),
            batcher.clone(),
            service_id,
            last_activity,
            shared_snapshot.clone(),
            allocations.clone(),
            rolling.clone(),
            observation.clone(),
            inflight_counter,
            registry.clone(),
            effective.clone(),
        ));
        registry.insert(svc.name.clone(), handle.clone());

        // Persistent services kick-start via an implicit Ensure so they begin
        // transitioning out of Idle immediately without blocking the startup loop.
        if matches!(svc.lifecycle, Lifecycle::Persistent) {
            let handle2 = handle.clone();
            tokio::spawn(async move {
                let _ = handle2.ensure().await;
            });
        }

        let listen: SocketAddr =
            format!("127.0.0.1:{}", svc.port)
                .parse()
                .map_err(|e: std::net::AddrParseError| {
                    ExpectedError::bind_failed(format!("127.0.0.1:{}", svc.port), e.to_string())
                })?;
        let shutdown_rx2 = shutdown_rx.clone();
        let upstream = svc.private_port;
        let name = svc.name.clone();
        let activity_for_proxy = activity.clone();
        let inflight_counter = inflight.counter(&svc.name);
        let handle_for_proxy = handle.clone();
        let max_request_duration_ms = svc.max_request_duration_ms;
        proxy_tasks.push(tokio::spawn(async move {
            let name_ping = name.clone();
            let before_request = std::sync::Arc::new(move || {
                let name_inner = name_ping.clone();
                let activity_inner = activity_for_proxy.clone();
                let handle_inner = handle_for_proxy.clone();
                Box::pin(async move {
                    activity_inner.ping(&name_inner);
                    match handle_inner.ensure().await {
                        Some(EnsureResponse::AlreadyRunning) => None,
                        Some(EnsureResponse::Waiting { mut rx }) => {
                            let timeout = Duration::from_millis(max_request_duration_ms);
                            match tokio::time::timeout(timeout, rx.recv()).await {
                                Ok(Ok(StartOutcome::Ok)) => None,
                                Ok(Ok(StartOutcome::Err(f))) => Some(match f.kind {
                                    StartFailureKind::NoFit | StartFailureKind::Oom => {
                                        proxy::error_response("insufficient_vram", &f.message)
                                    }
                                    StartFailureKind::Disabled => {
                                        proxy::error_response("service_disabled", &f.message)
                                    }
                                    StartFailureKind::HealthTimeout
                                    | StartFailureKind::LaunchFailed => {
                                        proxy::error_response("start_failed", &f.message)
                                    }
                                }),
                                Ok(Err(e)) => Some(proxy::error_response(
                                    "start_failed",
                                    &format!("start broadcast closed: {e}"),
                                )),
                                Err(_) => {
                                    Some(proxy::error_response("start_failed", "start timed out"))
                                }
                            }
                        }
                        Some(EnsureResponse::QueueFull) => Some(proxy::error_response(
                            "start_queue_full",
                            "start queue full",
                        )),
                        Some(EnsureResponse::Unavailable { reason }) => {
                            if reason.starts_with("no fit") {
                                Some(proxy::error_response("insufficient_vram", &reason))
                            } else {
                                Some(proxy::error_response("service_disabled", &reason))
                            }
                        }
                        None => Some(proxy::error_response(
                            "start_failed",
                            "supervisor unreachable",
                        )),
                    }
                }) as futures::future::BoxFuture<'static, _>
            })
                as std::sync::Arc<dyn Fn() -> futures::future::BoxFuture<'static, _> + Send + Sync>;
            if let Err(e) = proxy::serve_with_activity(
                listen,
                upstream,
                shutdown_rx2,
                before_request,
                inflight_counter,
            )
            .await
            {
                error!(service = %name, error = %e, "proxy failed");
            }
        }));
        supervisors.push(handle);
    }

    // Spawn balloon resolvers for dynamic services. Each resolver monitors
    // observed VRAM usage and fast-kills the lower-priority side when
    // growth pressure builds under contention.
    let mut balloon_tasks = Vec::new();
    for svc in &effective.services {
        if let AllocationMode::Dynamic {
            min_mb,
            max_mb,
            min_borrower_runtime_ms,
        } = svc.allocation_mode
        {
            let cfg = crate::balloon::BalloonConfig {
                min_mb,
                max_mb,
                min_borrower_runtime: Duration::from_millis(min_borrower_runtime_ms),
                margin_bytes: 512 * 1024 * 1024,
            };
            let join = crate::balloon::spawn_resolver(
                svc.name.clone(),
                cfg,
                svc.priority,
                observation.clone(),
                registry.clone(),
                allocations.clone(),
                shutdown_rx.clone(),
            );
            balloon_tasks.push(join);
        }
    }

    let port_pool = Arc::new(Mutex::new(PortPool::new(18000..19000)));
    let oneshots = OneshotRegistry::new();

    // Build AppState for the routers.
    let app_state = AppState {
        config: effective.clone(),
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
    };

    // OpenAI listener.
    let openai_listen: SocketAddr =
        effective
            .daemon
            .openai_listen
            .parse()
            .map_err(|e: std::net::AddrParseError| {
                ExpectedError::bind_failed(effective.daemon.openai_listen.clone(), e.to_string())
            })?;
    let openai_router = crate::openai_api::router(app_state.clone());
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
    let mgmt_router = crate::management_api::router(app_state.clone());
    let mgmt_listener = TcpListener::bind(mgmt_listen)
        .await
        .map_err(|e| ExpectedError::bind_failed(mgmt_listen.to_string(), e.to_string()))?;
    let mgmt_shutdown = shutdown_rx.clone();
    let mgmt_server = tokio::spawn(async move {
        let _ = axum::serve(mgmt_listener, mgmt_router)
            .with_graceful_shutdown(wait_shutdown(mgmt_shutdown))
            .await;
    });
    info!(%mgmt_listen, "management listener bound");

    let retention_task = tokio::spawn(retention::run_loop(db.clone(), shutdown_rx.clone()));

    let shutdown_kind = await_shutdown().await;
    info!(?shutdown_kind, "shutdown initiated");
    let _ = shutdown_tx.send(true);

    let drain_bound = match shutdown_kind {
        ShutdownKind::Graceful => Duration::from_millis(effective.daemon.shutdown_timeout_ms),
        ShutdownKind::Emergency => Duration::from_secs(5),
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
    for t in balloon_tasks {
        t.abort();
        let _ = t.await;
    }

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

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

async fn apply_migrations(db: &Database, migs: &[Migration]) {
    let now = now_ms();
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
