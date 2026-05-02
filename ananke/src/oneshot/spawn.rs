//! Oneshot orchestration: convert a validated request into a running
//! supervisor plus a TTL watcher. Lives here rather than on `AppState` so
//! `AppState` can remain a passive aggregate struct.

use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

use ananke_api::oneshot::OneshotRequest;

use crate::{
    config::{
        parse::DEFAULT_START_QUEUE_DEPTH,
        validate::{
            AllocationMode, CommandConfig, DEFAULT_DRAIN_TIMEOUT_MS,
            DEFAULT_EXTENDED_STREAM_DRAIN_MS, DEFAULT_HEALTH_PROBE_INTERVAL_MS,
            DEFAULT_HEALTH_TIMEOUT_MS, DEFAULT_MAX_REQUEST_DURATION_MS,
            DEFAULT_MIN_BORROWER_RUNTIME_MS, DEFAULT_SERVICE_PRIORITY, Filters, HealthSettings,
            Lifecycle, PlacementPolicy, ServiceConfig, Template, TemplateConfig,
        },
    },
    daemon::app_state::AppState,
    devices::Allocation,
    oneshot::{OneshotId, OneshotRecord},
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

    // OneshotRequest doesn't carry a model path, so llama-cpp oneshots are
    // structurally ill-defined at the request level. Reject them explicitly
    // rather than constructing a ServiceConfig with bogus fields.
    let template_config = match template {
        Template::LlamaCpp => {
            return Err(
                "llama-cpp oneshots are not supported (request has no model path)".to_string(),
            );
        }
        Template::Command => {
            let command = req
                .command
                .ok_or_else(|| "command template requires `command`".to_string())?;
            if command.is_empty() {
                return Err("command is empty".to_string());
            }
            TemplateConfig::Command(CommandConfig {
                command,
                workdir: req.workdir.map(PathBuf::from),
                shutdown_command: None,
                private_port_override: None,
                openai_proxy: None,
            })
        }
    };

    // Oneshots use the same port for public and private since they are
    // directly spawned without a wrapping proxy.
    let private_port = port;

    // Construct the ServiceConfig directly, bypassing validate()'s
    // port-uniqueness checks which would conflict with the running config.
    let svc = ServiceConfig {
        name: id.clone(),
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
        gpu_allow: Vec::new(),
        filters: Filters::default(),
        idle_timeout_ms: ttl_ms,
        drain_timeout_ms: DEFAULT_DRAIN_TIMEOUT_MS,
        extended_stream_drain_ms: DEFAULT_EXTENDED_STREAM_DRAIN_MS,
        max_request_duration_ms: DEFAULT_MAX_REQUEST_DURATION_MS,
        allocation_mode,
        openai_compat: false,
        description: None,
        start_queue_depth: DEFAULT_START_QUEUE_DEPTH,
        extra_args: Vec::new(),
        env: BTreeMap::new(),
        tracking: crate::config::TrackingSettings::default(),
        metadata: ananke_api::AnankeMetadata::new(),
        template_config,
    };

    let now_ms = crate::tracking::now_unix_ms();
    let service_id = state
        .db
        .upsert_service(&svc.name, now_ms)
        .await
        .map_err(|e| e.to_string())?;
    let _ = state
        .db
        .insert_oneshot(id.as_ref(), service_id, now_ms, ttl_ms as i64)
        .await;

    let init = crate::supervise::SupervisorInit {
        identity: crate::supervise::ServiceIdentity::from_service(&svc),
        allocation: Allocation::from_override(&svc.placement_override),
        service_id,
        last_activity: state.activity.get_or_init(&svc.name),
        inflight: state.inflight.counter(&svc.name),
    };
    let handle = Arc::new(crate::supervise::spawn_supervisor(
        init,
        svc.clone(),
        state.supervisor_deps(),
    ));
    state.registry.insert(svc.name.clone(), handle.clone());

    let record = OneshotRecord {
        id: id.clone(),
        service_name: svc.name.clone(),
        port,
        ttl_ms,
        started_at_ms: now_ms as u64,
        ended_at_ms: None,
        exit_code: None,
    };
    state.oneshots.insert(record);

    let handle2 = handle.clone();
    tokio::spawn(async move {
        let _ = handle2
            .ensure(crate::supervise::EnsureSource::UserRequest)
            .await;
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
