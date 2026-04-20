//! Provision a service: spawn the supervisor, bind the proxy listener,
//! and (for dynamic services) spawn the balloon resolver. Shared by the
//! daemon's boot loop and the reload reconciler so both paths use one
//! code path and stay in sync.

use std::{net::SocketAddr, sync::Arc, time::Duration};

use futures::future::BoxFuture;
use parking_lot::Mutex;
use tokio::{sync::watch, task::JoinHandle};
use tracing::error;

use crate::{
    allocator::AllocationTable,
    api::{openai::errors::ProxyErrorCode, proxy},
    config::{AllocationMode, Lifecycle, ServiceConfig},
    db::Database,
    devices::Allocation,
    errors::ExpectedError,
    supervise::{
        ServiceIdentity, SupervisorDeps, SupervisorHandle, SupervisorInit, await_ensure,
        spawn_supervisor,
    },
    tracking::{activity::ActivityTable, inflight::InflightTable, observation::ObservationTable},
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
}

impl ProvisioningDeps {
    /// Assemble a `ProvisioningDeps` from a daemon [`AppState`] and the
    /// daemon-wide shutdown channel. Every field already lives on
    /// `AppState`; this constructor keeps the call sites in
    /// `daemon::run` and the test harness from enumerating them by hand.
    pub fn from_state(
        state: &crate::daemon::app_state::AppState,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Self {
        Self {
            db: state.db.clone(),
            activity: state.activity.clone(),
            inflight: state.inflight.clone(),
            observation: state.observation.clone(),
            allocations: state.allocations.clone(),
            supervisor_deps: state.supervisor_deps(),
            shutdown_rx,
        }
    }
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
        .upsert_service(&svc.name, crate::tracking::now_unix_ms())
        .await?;
    let init = SupervisorInit {
        identity: ServiceIdentity::from_service(&svc),
        allocation: Allocation::from_override(&svc.placement_override),
        service_id,
        last_activity: deps.activity.get_or_init(&svc.name),
        inflight: deps.inflight.counter(&svc.name),
    };
    let handle = Arc::new(spawn_supervisor(
        init,
        svc.clone(),
        deps.supervisor_deps.clone(),
    ));
    deps.supervisor_deps
        .registry
        .insert(svc.name.clone(), handle.clone());

    if matches!(svc.lifecycle, Lifecycle::Persistent) {
        let h = handle.clone();
        tokio::spawn(async move {
            let _ = h.ensure().await;
        });
    }

    let listen: SocketAddr =
        format!("127.0.0.1:{}", svc.port)
            .parse()
            .map_err(|e: std::net::AddrParseError| {
                ExpectedError::bind_failed(format!("127.0.0.1:{}", svc.port), e.to_string())
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
    let proxy_task = tokio::spawn(async move {
        if let Err(e) = proxy::serve_with_activity(
            listen,
            upstream,
            shutdown_rx,
            before_request,
            inflight_counter,
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
            let cfg = crate::allocator::balloon::BalloonConfig {
                min_mb,
                max_mb,
                min_borrower_runtime: Duration::from_millis(min_borrower_runtime_ms),
                margin_bytes: BALLOON_MARGIN_BYTES,
            };
            Some(crate::allocator::balloon::spawn_resolver(
                svc.name.clone(),
                cfg,
                svc.priority,
                deps.observation.clone(),
                deps.supervisor_deps.registry.clone(),
                deps.allocations.clone(),
                deps.shutdown_rx.clone(),
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
                crate::supervise::EnsureOutcome::Ready { .. } => None,
                crate::supervise::EnsureOutcome::Failed(f) => {
                    Some(ensure_failure_to_proxy_error(f))
                }
            }
        }) as BoxFuture<'static, _>
    })
}

fn ensure_failure_to_proxy_error(f: crate::supervise::EnsureFailure) -> proxy::ProxyError {
    use crate::supervise::EnsureFailure;
    match f {
        EnsureFailure::InsufficientVram(msg) => {
            proxy::error_response(ProxyErrorCode::InsufficientVram, &msg)
        }
        EnsureFailure::ServiceDisabled(msg) => {
            proxy::error_response(ProxyErrorCode::ServiceDisabled, &msg)
        }
        EnsureFailure::StartQueueFull => {
            proxy::error_response(ProxyErrorCode::StartQueueFull, "start queue full")
        }
        EnsureFailure::StartFailed(msg) => proxy::error_response(ProxyErrorCode::StartFailed, &msg),
    }
}
