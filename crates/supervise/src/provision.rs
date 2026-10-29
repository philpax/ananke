//! Provision a service: spawn the supervisor, bind the proxy listener,
//! and (for dynamic services) spawn the balloon resolver. Shared by the
//! daemon's boot loop and the reload reconciler so both paths use one
//! code path and stay in sync.

use std::{net::SocketAddr, sync::Arc, time::Duration};

use ananke_allocator::AllocationTable;
use ananke_config::{AllocationMode, Lifecycle, ServiceConfig};
use ananke_db::Database;
use ananke_devices::Allocation;
use ananke_errors::ExpectedError;
use ananke_observation::ObservationTable;
use ananke_proxy::{self as proxy, ApiErrorCode};
use ananke_tracking::{activity::ActivityTable, inflight::InflightTable};
use futures::future::BoxFuture;
use parking_lot::Mutex;
use tokio::{sync::watch, task::JoinHandle};
use tracing::{error, info, warn};

use crate::{
    ServiceIdentity, SupervisorDeps, SupervisorHandle, SupervisorInit, await_ensure,
    spawn_supervisor,
};

/// Everything non-service-specific a provision step needs. Cloned cheaply
/// — every field is `Arc`-backed or a thin handle.
#[derive(Clone)]
pub struct ProvisioningDeps {
    pub db: Database,
    pub activity: ActivityTable,
    pub inflight: InflightTable,
    pub observation: ObservationTable,
    pub allocations: Arc<Mutex<AllocationTable>>,
    pub supervisor_deps: SupervisorDeps,
    pub shutdown_rx: watch::Receiver<bool>,
    /// Mints per-request proxy token-usage recorders. The daemon supplies
    /// the concrete recorder (OpenAI metrics + db handle); supervise only
    /// threads it into the per-service proxy.
    pub metrics_factory: ananke_proxy::RecorderFactory,
}

/// Tasks owned by a provisioned service. The daemon keeps these in a pool
/// and aborts them on shutdown; the reconciler, for now, fires-and-forgets
/// (teardown on service-remove lives in `reconciler::handle_reload`, and
/// the per-proxy shutdown channel is a follow-up).
pub struct ProvisionedService {
    pub handle: Arc<SupervisorHandle>,
    pub proxy_task: JoinHandle<()>,
    pub balloon_task: Option<JoinHandle<()>>,
}

/// Provision a single service. Spawns its supervisor, binds the proxy
/// listener on `127.0.0.1:{svc.port}`, spawns the balloon resolver if the
/// service is dynamic, and — for `persistent` services — fires an
/// implicit `ensure()` so it begins transitioning out of Idle immediately.
/// On-demand services stay idle until their first request, matching
/// boot-time behaviour.
pub async fn provision_service(
    svc: ServiceConfig,
    deps: &ProvisioningDeps,
) -> Result<ProvisionedService, ExpectedError> {
    let service_id = deps
        .db
        .upsert_service(&svc.name, ananke_tracking::now_unix_ms())
        .await?;
    // Restore an operator disable from an earlier run. The supervisor starts
    // in that state rather than being sent a disable, so no request,
    // persistent ensure or watcher tick can start it first, and restoring
    // needs no database write that could fail.
    let disabled = deps.db.service_disabled_at(service_id).await?.is_some();
    if disabled {
        info!(service = %svc.name, "restoring operator disable from the database");
    }
    let init = SupervisorInit {
        identity: ServiceIdentity::from_service(&svc),
        allocation: Allocation::from_override(&svc.placement_override),
        service_id,
        last_activity: deps.activity.get_or_init(&svc.name),
        inflight: deps.inflight.counter(&svc.name),
        disabled,
    };
    let handle = Arc::new(spawn_supervisor(
        init,
        svc.clone(),
        deps.supervisor_deps.clone(),
    ));
    deps.supervisor_deps
        .registry
        .insert(svc.name.clone(), handle.clone());

    if !disabled && matches!(svc.lifecycle, Lifecycle::Persistent) {
        let h = handle.clone();
        tokio::spawn(async move {
            let _ = h.ensure(crate::EnsureSource::BackgroundWatcher).await;
        });
    }

    // `daemon.allow_external_services = true` opts every per-service
    // reverse proxy onto 0.0.0.0. Default is loopback-only; the per-
    // service endpoints are unauthenticated so matching the
    // `allow_external_management` posture is the right default.
    let bind_host = if deps
        .supervisor_deps
        .config
        .effective()
        .daemon
        .allow_external_services
    {
        "0.0.0.0"
    } else {
        "127.0.0.1"
    };
    let listen_str = format!("{bind_host}:{}", svc.port);
    let listen: SocketAddr = listen_str.parse().map_err(|e: std::net::AddrParseError| {
        ExpectedError::bind_failed(listen_str.clone(), e.to_string())
    })?;
    let shutdown_rx = deps.shutdown_rx.clone();
    let upstream = svc.private_port;
    let name_for_error = svc.name.clone();
    let inflight_counter = deps.inflight.counter(&svc.name);
    let before_request = make_proxy_before_request(
        svc.name.clone(),
        handle.clone(),
        deps.activity.clone(),
        Duration::from_millis(svc.max_request_duration_ms),
    );
    let activity_ping: Arc<dyn Fn() + Send + Sync> = {
        let activity = deps.activity.clone();
        let name = svc.name.clone();
        Arc::new(move || activity.ping(&name))
    };
    // Record per-request token metrics for direct hits on the per-service
    // port, matching what the OpenAI multiplexer records for its own traffic.
    // `service_id` is stable (resolved once above via `upsert_service`); the
    // run_id changes on each (re)load, so it is read from the supervisor's
    // mirror cell at request time via the closure. The factory captures the
    // db handle (daemon-side) so the proxy's erased recorder can write it.
    let metrics = {
        let run_handle = handle.clone();
        ananke_proxy::ProxyMetrics::new(
            deps.metrics_factory.clone(),
            service_id,
            svc.name.clone(),
            Arc::new(move || run_handle.peek().run_id),
        )
    };
    let proxy_task = tokio::spawn(async move {
        if let Err(e) = proxy::serve_with_activity(
            listen,
            upstream,
            shutdown_rx,
            before_request,
            inflight_counter,
            activity_ping,
            Some(metrics),
        )
        .await
        {
            error!(service = %name_for_error, error = %e, "proxy failed");
        }
    });

    let balloon_task = match svc.allocation_mode {
        AllocationMode::Dynamic {
            min_mb,
            max_mb,
            min_borrower_runtime_ms,
        } => {
            // 512 MiB headroom above `min_mb` before balloon triggers growth detection.
            const BALLOON_MARGIN_BYTES: u64 = 512 * 1024 * 1024;
            let cfg = ananke_allocator::balloon::BalloonConfig {
                min_mb,
                max_mb,
                min_borrower_runtime: Duration::from_millis(min_borrower_runtime_ms),
                margin_bytes: BALLOON_MARGIN_BYTES,
            };
            Some(ananke_allocator::balloon::spawn_resolver(
                svc.name.clone(),
                cfg,
                svc.priority,
                svc.lifecycle,
                ananke_allocator::balloon::ResolverDeps {
                    observation: deps.observation.clone(),
                    registry: deps.supervisor_deps.registry.clone(),
                    allocations: deps.allocations.clone(),
                    events: deps.supervisor_deps.events.clone(),
                    snapshot: deps.supervisor_deps.snapshot.clone(),
                    config: deps.supervisor_deps.config.clone(),
                    shutdown: deps.shutdown_rx.clone(),
                },
            ))
        }
        _ => None,
    };

    Ok(ProvisionedService {
        handle,
        proxy_task,
        balloon_task,
    })
}

fn make_proxy_before_request(
    name: smol_str::SmolStr,
    handle: Arc<SupervisorHandle>,
    activity: ActivityTable,
    max_request_duration: Duration,
) -> Arc<dyn Fn() -> BoxFuture<'static, Option<proxy::ProxyError>> + Send + Sync> {
    Arc::new(move || {
        let name = name.clone();
        let handle = handle.clone();
        let activity = activity.clone();
        Box::pin(async move {
            activity.ping(&name);
            match await_ensure(&handle, max_request_duration).await {
                crate::EnsureOutcome::Ready { .. } => None,
                crate::EnsureOutcome::Failed(f) => {
                    warn!(service = %name, failure = ?f, "per-service proxy rejected request");
                    Some(ensure_failure_to_proxy_error(&name, f))
                }
            }
        }) as BoxFuture<'static, _>
    })
}

fn ensure_failure_to_proxy_error(
    service_name: &smol_str::SmolStr,
    f: crate::EnsureFailure,
) -> proxy::ProxyError {
    proxy::error_response(ensure_failure_to_api_code(service_name, f))
}

/// Lift an [`EnsureFailure`] into the unified [`ApiErrorCode`] enum.
/// Shared between the proxy data plane's `error_response`, the OpenAI
/// surface's handler match arms, and the management API's lifecycle
/// endpoints so the mapping is single-source.
pub fn ensure_failure_to_api_code(
    service_name: &smol_str::SmolStr,
    f: crate::EnsureFailure,
) -> ApiErrorCode {
    use crate::EnsureFailure;
    match f {
        EnsureFailure::InsufficientCapacity(reason) => ApiErrorCode::InsufficientCapacity {
            name: service_name.clone(),
            reason,
        },
        EnsureFailure::ServiceDisabled(reason) => ApiErrorCode::ServiceDisabled {
            name: service_name.clone(),
            reason,
        },
        EnsureFailure::StartQueueFull => ApiErrorCode::StartQueueFull {
            name: service_name.clone(),
        },
        EnsureFailure::StartFailed(reason) => ApiErrorCode::StartFailed {
            name: service_name.clone(),
            reason,
        },
        EnsureFailure::Blocked { busy_peers } => ApiErrorCode::ServiceBlocked {
            name: service_name.clone(),
            busy_peers,
        },
    }
}
