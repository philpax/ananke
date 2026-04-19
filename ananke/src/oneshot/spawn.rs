//! Oneshot orchestration: convert a validated request into a running
//! supervisor plus a TTL watcher. Lives here rather than on `AppState` so
//! `AppState` can remain a passive aggregate struct.

use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

use smol_str::SmolStr;

use crate::{
    config::{
        parse::RawService,
        validate::{
            AllocationMode, DEFAULT_DRAIN_TIMEOUT_MS, DEFAULT_EXTENDED_STREAM_DRAIN_MS,
            DEFAULT_HEALTH_PROBE_INTERVAL_MS, DEFAULT_HEALTH_TIMEOUT_MS,
            DEFAULT_MAX_REQUEST_DURATION_MS, DEFAULT_MIN_BORROWER_RUNTIME_MS,
            DEFAULT_SERVICE_PRIORITY, DEFAULT_WARMING_GRACE_MS, Filters, HealthSettings, Lifecycle,
            PlacementPolicy, ServiceConfig, Template,
        },
    },
    daemon::app_state::AppState,
    devices::Allocation,
    oneshot::{OneshotId, OneshotRecord, handlers::OneshotRequest},
};

/// Spawn a oneshot service from a validated request: synthesise a
/// `ServiceConfig`, upsert the DB rows, launch a supervisor, and arm the
/// TTL watcher. Any error is surfaced to the HTTP handler, which is
/// responsible for releasing the port it allocated beforehand.
pub async fn spawn_oneshot(
    state: &AppState,
    id: OneshotId,
    req: OneshotRequest,
    port: u16,
    ttl_ms: u64,
) -> Result<(), String> {
    let template = match req.template.as_str() {
        "llama-cpp" => Template::LlamaCpp,
        "command" => Template::Command,
        other => return Err(format!("unknown template `{other}`")),
    };

    let allocation_mode = AllocationMode::from_parts(
        template,
        req.allocation.mode.as_deref(),
        req.allocation.vram_gb,
        req.allocation.min_vram_gb,
        req.allocation.max_vram_gb,
        DEFAULT_MIN_BORROWER_RUNTIME_MS,
    )?;

    // Oneshots use the same port for public and private since they are
    // directly spawned without a wrapping proxy.
    let private_port = port;

    // Construct the ServiceConfig directly, bypassing validate()'s
    // port-uniqueness checks which would conflict with the running config.
    let svc = ServiceConfig {
        name: id.clone(),
        template,
        port,
        private_port,
        lifecycle: Lifecycle::OnDemand,
        priority: req.priority.unwrap_or(DEFAULT_SERVICE_PRIORITY),
        health: HealthSettings {
            http_path: "/v1/models".into(),
            timeout_ms: DEFAULT_HEALTH_TIMEOUT_MS,
            probe_interval_ms: DEFAULT_HEALTH_PROBE_INTERVAL_MS,
        },
        placement_override: BTreeMap::new(),
        placement_policy: PlacementPolicy::GpuOnly,
        filters: Filters::default(),
        idle_timeout_ms: ttl_ms + DEFAULT_WARMING_GRACE_MS,
        warming_grace_ms: DEFAULT_WARMING_GRACE_MS,
        drain_timeout_ms: DEFAULT_DRAIN_TIMEOUT_MS,
        extended_stream_drain_ms: DEFAULT_EXTENDED_STREAM_DRAIN_MS,
        max_request_duration_ms: DEFAULT_MAX_REQUEST_DURATION_MS,
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

    let now_ms = crate::tracking::now_unix_ms();
    let service_id = state
        .db
        .upsert_service(&svc.name, now_ms)
        .await
        .map_err(|e| e.to_string())?;
    {
        use crate::db::models::Oneshot;
        let mut handle = state.db.handle();
        let _ = toasty::create!(Oneshot {
            id: id.to_string(),
            service_id,
            submitted_at: now_ms,
            ttl_ms: ttl_ms as i64,
        })
        .exec(&mut handle)
        .await;
    }

    let init = crate::supervise::SupervisorInit {
        svc: svc.clone(),
        allocation: Allocation::from_override(&svc.placement_override),
        service_id,
        last_activity: state.activity.get_or_init(&svc.name),
        inflight: state.inflight.counter(&svc.name),
    };
    let handle = Arc::new(crate::supervise::spawn_supervisor(
        init,
        state.supervisor_deps(),
    ));
    state.registry.insert(svc.name.clone(), handle.clone());

    let record = OneshotRecord {
        id: id.clone(),
        service_name: svc.name.clone(),
        port,
        ttl_ms,
        started_at_ms: now_ms as u64,
    };
    state.oneshots.insert(record);

    let handle2 = handle.clone();
    tokio::spawn(async move {
        let _ = handle2.ensure().await;
    });

    // The TTL watcher uses a dedicated watch channel. The sender is moved
    // into the spawned task so it lives as long as the oneshot is pending;
    // dropping it when the watcher exits closes the receiver cleanly.
    let (sh_tx, sh_rx) = tokio::sync::watch::channel(false);
    let watcher_cfg = crate::oneshot::ttl::WatcherConfig {
        id: id.clone(),
        service_name: svc.name.clone(),
        ttl: std::time::Duration::from_millis(ttl_ms),
        port,
        registry: state.registry.clone(),
        oneshots: state.oneshots.clone(),
        db: state.db.clone(),
        port_pool: state.port_pool.clone(),
        shutdown: sh_rx,
    };
    tokio::spawn(async move {
        let _sh_tx = sh_tx;
        let _ = crate::oneshot::ttl::spawn_watcher(watcher_cfg).await;
    });

    Ok(())
}
