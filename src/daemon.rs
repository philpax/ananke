//! Top-level daemon orchestration: wires config, DB, devices, supervisors,
//! proxies, signals, and retention together.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use tokio::sync::watch;
use tracing::{error, info, warn};

use crate::config::{Migration, load_config};
use crate::db::Database;
use crate::db::logs::spawn as spawn_batcher;
use crate::devices::{Allocation, GpuProbe, cpu, nvml::NvmlProbe};
use crate::errors::ExpectedError;
use crate::proxy;
use crate::retention;
use crate::signals::{ShutdownKind, await_shutdown, grace_for};
use crate::supervise::{SupervisorHandle, orphans::reconcile, spawn_supervisor};

pub async fn run() -> Result<(), ExpectedError> {
    init_tracing();

    let cli_config = parse_cli_config_arg();
    let config_path = crate::config::resolve_from_env(cli_config.as_deref())?;
    info!(config_path = %config_path.display(), "resolved config path");

    let (effective, migrations) = load_config(&config_path)?;
    let db = Database::open(&effective.daemon.data_dir.join("ananke.sqlite"))?;
    apply_migrations(&db, &migrations);

    let _probe: Option<NvmlProbe> = match NvmlProbe::init() {
        Ok(p) => {
            for g in p.list() {
                info!(gpu = g.id, name = %g.name, total_bytes = g.total_bytes, "detected GPU");
            }
            Some(p)
        }
        Err(e) => {
            warn!(error = %e, "NVML init failed; falling back to CPU-only");
            None
        }
    };
    let cpu_mem = cpu::read().ok();
    if let Some(m) = cpu_mem {
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

    // Order persistent services by priority DESC, name ASC (spec §9.4).
    let mut ordered = effective.services.clone();
    ordered.sort_by(|a, b| {
        b.priority
            .cmp(&a.priority)
            .then_with(|| a.name.cmp(&b.name))
    });

    let mut supervisors: Vec<SupervisorHandle> = Vec::new();
    let mut proxy_tasks = Vec::new();
    for svc in ordered {
        let service_id = db.upsert_service(&svc.name, now_ms())?;
        let allocation = Allocation::from_override(&svc.placement_override);
        let handle = spawn_supervisor(
            svc.clone(),
            allocation,
            db.clone(),
            batcher.clone(),
            service_id,
        );
        let listen: SocketAddr =
            format!("127.0.0.1:{}", svc.port)
                .parse()
                .map_err(|e: std::net::AddrParseError| {
                    ExpectedError::bind_failed(format!("127.0.0.1:{}", svc.port), e.to_string())
                })?;
        let shutdown_rx2 = shutdown_rx.clone();
        let upstream = svc.private_port;
        let name = svc.name.clone();
        proxy_tasks.push(tokio::spawn(async move {
            if let Err(e) = proxy::serve(listen, upstream, shutdown_rx2).await {
                error!(service = %name, error = %e, "proxy failed");
            }
        }));
        supervisors.push(handle);
    }

    let retention_task = tokio::spawn(retention::run_loop(db.clone(), shutdown_rx.clone()));

    let shutdown_kind = await_shutdown().await;
    info!(?shutdown_kind, "shutdown initiated");
    let _ = shutdown_tx.send(true);

    let drain_bound = match shutdown_kind {
        ShutdownKind::Graceful => Duration::from_millis(effective.daemon.shutdown_timeout_ms),
        ShutdownKind::Emergency => Duration::from_secs(5),
    };
    let _ = tokio::time::timeout(drain_bound, async {
        for sup in supervisors {
            sup.shutdown().await;
        }
    })
    .await;

    for t in proxy_tasks {
        t.abort();
        let _ = t.await;
    }
    retention_task.abort();
    let _ = retention_task.await;

    batcher.flush().await;
    let _ = grace_for(shutdown_kind); // reserved for future per-signal grace tuning
    Ok(())
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

fn apply_migrations(db: &Database, migs: &[Migration]) {
    let now = now_ms();
    for m in migs {
        if let Err(e) = db.reparent(&m.old_name, &m.new_name, now) {
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
