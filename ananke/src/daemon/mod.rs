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
use tracing::{error, info, warn};

use crate::{
    allocator::AllocationTable,
    api::proxy,
    config::{AllocationMode, Lifecycle, Migration, manager::ConfigManager},
    daemon::{
        app_state::AppState,
        signals::{ShutdownKind, await_shutdown},
    },
    db::{Database, logs::spawn as spawn_batcher, retention},
    devices::{Allocation, GpuProbe, cpu, nvml::NvmlProbe, snapshotter},
    errors::ExpectedError,
    oneshot::{OneshotRegistry, PortPool},
    supervise::{
        SupervisorHandle, orphans::reconcile, registry::ServiceRegistry, spawn_supervisor,
    },
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
    if let Ok(m) = cpu::read() {
        info!(
            total = m.total_bytes,
            avail = m.available_bytes,
            "CPU memory"
        );
    }

    let procfs = PathBuf::from("/proc");
    for disposition in reconcile(&crate::system::LocalFs, &db, &procfs).await {
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

    let system = crate::system::SystemDeps::local();

    let supervisor_deps = crate::supervise::SupervisorDeps {
        db: db.clone(),
        batcher: batcher.clone(),
        snapshot: shared_snapshot.clone(),
        allocations: allocations.clone(),
        rolling: rolling.clone(),
        observation: observation.clone(),
        registry: registry.clone(),
        effective: effective.clone(),
        events: events.clone(),
        system: system.clone(),
    };

    let mut supervisors: Vec<Arc<SupervisorHandle>> = Vec::new();
    let mut proxy_tasks = Vec::new();
    for svc in ordered {
        let service_id = db
            .upsert_service(&svc.name, crate::tracking::now_unix_ms())
            .await?;
        let init = crate::supervise::SupervisorInit {
            svc: svc.clone(),
            allocation: Allocation::from_override(&svc.placement_override),
            service_id,
            last_activity: activity.get_or_init(&svc.name),
            inflight: inflight.counter(&svc.name),
        };
        let handle = Arc::new(spawn_supervisor(init, supervisor_deps.clone()));
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
        let inflight_counter = inflight.counter(&svc.name);
        let before_request = make_proxy_before_request(
            svc.name.clone(),
            handle.clone(),
            activity.clone(),
            Duration::from_millis(svc.max_request_duration_ms),
        );
        proxy_tasks.push(tokio::spawn(async move {
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
            // 512 MiB headroom above `min_mb` before balloon triggers growth detection.
            const BALLOON_MARGIN_BYTES: u64 = 512 * 1024 * 1024;
            let cfg = crate::allocator::balloon::BalloonConfig {
                min_mb,
                max_mb,
                min_borrower_runtime: Duration::from_millis(min_borrower_runtime_ms),
                margin_bytes: BALLOON_MARGIN_BYTES,
            };
            let join = crate::allocator::balloon::spawn_resolver(
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

    // Drain-on-remove reconciler: watches ConfigReloaded and shuts down any
    // supervisor whose service name has disappeared from the new effective
    // config. Without this, a reload that removes a service would leave the
    // supervisor alive and its child continuing to consume VRAM.
    let reconciler_task = crate::supervise::reconciler::spawn(
        events.clone(),
        config.clone(),
        registry.clone(),
        shutdown_rx.clone(),
    );

    let port_pool = Arc::new(Mutex::new(PortPool::new(18000..19000)));
    let oneshots = OneshotRegistry::new();

    // Build AppState for the routers.
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

    let retention_task = tokio::spawn(retention::run_loop(db.clone(), shutdown_rx.clone()));

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

/// Build the `before_request` hook passed to `proxy::serve_with_activity`:
/// each request pings the activity table, ensures the supervisor is running,
/// and — on failure — short-circuits with a proxy error body.
fn make_proxy_before_request(
    name: smol_str::SmolStr,
    handle: Arc<crate::supervise::SupervisorHandle>,
    activity: ActivityTable,
    max_request_duration: Duration,
) -> Arc<
    dyn Fn() -> futures::future::BoxFuture<'static, Option<crate::api::proxy::ProxyError>>
        + Send
        + Sync,
> {
    Arc::new(move || {
        let name = name.clone();
        let handle = handle.clone();
        let activity = activity.clone();
        Box::pin(async move {
            activity.ping(&name);
            match crate::supervise::await_ensure(&handle, max_request_duration).await {
                crate::supervise::EnsureOutcome::Ready { .. } => None,
                crate::supervise::EnsureOutcome::Failed(f) => {
                    Some(ensure_failure_to_proxy_error(f))
                }
            }
        }) as futures::future::BoxFuture<'static, _>
    })
}

fn ensure_failure_to_proxy_error(
    f: crate::supervise::EnsureFailure,
) -> crate::api::proxy::ProxyError {
    use crate::supervise::EnsureFailure;
    match f {
        EnsureFailure::InsufficientVram(msg) => proxy::error_response("insufficient_vram", &msg),
        EnsureFailure::ServiceDisabled(msg) => proxy::error_response("service_disabled", &msg),
        EnsureFailure::StartQueueFull => {
            proxy::error_response("start_queue_full", "start queue full")
        }
        EnsureFailure::StartFailed(msg) => proxy::error_response("start_failed", &msg),
    }
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
