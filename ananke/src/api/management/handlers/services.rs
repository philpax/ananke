//! Service list, detail, and launch-command-preview handlers.

use ananke_api::{
    internal::{fit_verdict::FitVerdict, log_line::LogLine},
    services::{
        command::{
            EnvVar, LaunchCommand, LaunchCommandResponse, LaunchCommandSource, LaunchContainer,
            LaunchMount,
        },
        detail::{PlacementPreview, RestartEvent, ServiceDetail},
        list::{DeviceFootprint, ServiceSummary, ServicesResponse},
    },
    shared::errors::ApiError,
};
use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};

use crate::{
    api::{
        errors::ApiErrorCode,
        management::handlers::{model_estimate_entry, placement_preview, read_current_allocation},
    },
    config::{ServiceConfig, placement_inputs},
    daemon::app_state::AppState,
    supervise::estimate_cache::EstimateCacheEntry,
};

#[utoipa::path(
    summary = "List all services",get, path = "/api/services", responses((status = 200, body = ServicesResponse)))]
pub async fn list_services(State(state): State<AppState>) -> Response {
    let mut services = Vec::new();
    let eff = state.config.effective();
    for svc_cfg in eff.services.iter() {
        let handle = state.registry.get(&svc_cfg.name);
        // `peek()` reads the supervisor's lock-free mirror directly — no
        // mailbox round-trip. `list_services` stays responsive even while
        // a supervisor is mid-drain or mid-spawn, and `run_id` / `pid`
        // are always populated for any service with a live child.
        let peek = handle.as_ref().map(|h| h.peek());
        let state_name = peek
            .as_ref()
            .map(|p| p.state.name().to_string())
            .unwrap_or_else(|| "unknown".into());
        let running = peek.as_ref().and_then(|p| p.pid).is_some();

        // Compute the fit verdict so the frontend can flag services
        // that can't start under current device conditions. Running
        // services short-circuit to `Fits` via the live pledge; the
        // estimate cache is usually warm after the first detail view
        // or service start.
        let entry = model_estimate_entry(&state, svc_cfg);
        let placement = placement_preview(
            &state,
            svc_cfg,
            entry.as_ref().map(|e| &e.estimate_full),
            running,
        );
        let fit_verdict = placement.as_ref().map(|p| p.verdict.clone());
        // One computation, two fields: the total is the breakdown's sum, so a row
        // showing both cannot show two figures that disagree.
        let footprint_devices = summary_footprint(&state, svc_cfg, placement.as_ref(), &entry);
        let footprint_bytes = (!footprint_devices.is_empty())
            .then(|| footprint_devices.iter().map(|d| d.bytes).sum());
        let last_used_ms = state.activity.last_ms(&svc_cfg.name);

        services.push(ServiceSummary {
            name: svc_cfg.name.to_string(),
            state: state_name,
            lifecycle: svc_cfg.lifecycle.as_str().to_string(),
            priority: svc_cfg.priority,
            port: svc_cfg.port,
            openai_compat: svc_cfg.openai_compat,
            web_ui_url: svc_cfg
                .metadata
                .get("web_ui_url")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            run_id: peek.as_ref().and_then(|p| p.run_id),
            pid: peek.as_ref().and_then(|p| p.pid),
            inflight_count: state.inflight.current(&svc_cfg.name),
            // Placeholder: elastic borrower tracking is deferred to a later phase.
            elastic_borrower: None,
            // Config-only check, no GGUF read — safe to ship in the
            // list view that the frontend polls every 2 s.
            has_mmproj: svc_cfg.llama_cpp().map(|lc| lc.mmproj.is_some()),
            modality: svc_cfg.modality,
            ananke_metadata: svc_cfg.metadata.clone(),
            fit_verdict,
            footprint_bytes,
            footprint_devices,
            last_used_ms,
        });
    }
    // Extract the port from the OpenAI bind address (e.g. "127.0.0.1:7070").
    let openai_api_port = eff
        .daemon
        .openai_listen
        .rsplit(':')
        .next()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or_default();
    let out = ServicesResponse {
        services,
        openai_api_port,
    };
    (StatusCode::OK, Json(out)).into_response()
}

#[utoipa::path(
    summary = "Get service detail",
    get,
    path = "/api/services/{name}",
    responses(
        (status = 200, body = ServiceDetail),
        (status = 404, body = ApiError, description = "service_not_found")
    )
)]
pub async fn service_detail(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    let eff = state.config.effective();
    let Some(svc_cfg) = eff.services.iter().find(|s| s.name == name) else {
        return ApiErrorCode::ServiceNotFound {
            name: name.as_str().into(),
        }
        .into_response();
    };
    let handle = state.registry.get(&svc_cfg.name);
    let snap = handle.as_ref().map(|h| h.peek());
    let placement_override: std::collections::BTreeMap<String, u64> = svc_cfg
        .placement_override
        .iter()
        .map(|(k, v)| {
            let key = match k {
                crate::config::DeviceSlot::Cpu => "cpu".to_string(),
                crate::config::DeviceSlot::Gpu(n) => format!("gpu:{n}"),
            };
            (key, *v)
        })
        .collect();

    let recent_logs: Vec<LogLine> = {
        let svc_id_opt = state.db.resolve_service_id(&name).await.ok().flatten();
        match svc_id_opt {
            Some(svc_id) => {
                let mut rows = state
                    .db
                    .fetch_service_logs(svc_id)
                    .await
                    .unwrap_or_default();
                // Sort newest first by (timestamp_ms DESC, seq DESC) and
                // truncate to 200. The index on (service_id, run_id,
                // timestamp_ms) keeps the candidate set cheap to fetch.
                rows.sort_by(|a, b| {
                    b.timestamp_ms
                        .cmp(&a.timestamp_ms)
                        .then_with(|| b.seq.cmp(&a.seq))
                });
                rows.truncate(200);
                rows.into_iter()
                    .map(|r| LogLine {
                        timestamp_ms: r.timestamp_ms,
                        stream: r.stream,
                        line: r.line,
                        run_id: r.run_id,
                        seq: r.seq,
                    })
                    .collect()
            }
            None => Vec::new(),
        }
    };

    let recent_restarts: Vec<RestartEvent> = {
        let svc_id_opt = state.db.resolve_service_id(&name).await.ok().flatten();
        match svc_id_opt {
            Some(svc_id) => state
                .db
                .recent_service_restarts(svc_id, 10)
                .await
                .unwrap_or_default()
                .into_iter()
                .map(|r| RestartEvent {
                    at_ms: r.at_ms,
                    trigger: r.trigger,
                    detail: r.detail,
                    run_id: r.run_id,
                })
                .collect(),
            None => Vec::new(),
        }
    };

    let rc = state.rolling.get(&svc_cfg.name);
    let observed_peak_bytes = state.observation.read_peak(&svc_cfg.name);

    let entry = model_estimate_entry(&state, svc_cfg);
    let model_info = entry.as_ref().map(|e| e.model_info.clone());
    let estimate = entry.as_ref().map(|e| e.estimate.clone());
    let running = snap.as_ref().and_then(|s| s.pid).is_some();
    let placement_preview = placement_preview(
        &state,
        svc_cfg,
        entry.as_ref().map(|e| &e.estimate_full),
        running,
    );
    let current_allocation = read_current_allocation(&state, &svc_cfg.name);

    let detail = ServiceDetail {
        name: svc_cfg.name.to_string(),
        state: snap
            .as_ref()
            .map(|s| s.state.name().to_string())
            .unwrap_or_else(|| "unknown".into()),
        lifecycle: format!("{:?}", svc_cfg.lifecycle).to_lowercase(),
        priority: svc_cfg.priority,
        port: svc_cfg.port,
        private_port: svc_cfg.private_port,
        template: svc_cfg.template().as_str().to_string(),
        placement_override,
        idle_timeout_ms: svc_cfg.idle_timeout_ms,
        run_id: snap.as_ref().and_then(|s| s.run_id),
        pid: snap.as_ref().and_then(|s| s.pid),
        recent_logs,
        // Cast from the internal f64/u32 representation to the shared DTO's f32/u64.
        rolling_mean: if rc.vram.samples == 0 {
            None
        } else {
            Some(rc.vram.mean as f32)
        },
        rolling_samples: rc.vram.samples.into(),
        rolling_mean_host: if rc.host.samples == 0 {
            None
        } else {
            Some(rc.host.mean as f32)
        },
        rolling_samples_host: rc.host.samples.into(),
        observed_peak_bytes,
        // Placeholder: elastic borrower tracking is deferred to a later phase.
        elastic_borrower: None,
        model_info,
        estimate,
        placement_preview,
        current_allocation,
        modality: svc_cfg.modality,
        ananke_metadata: svc_cfg.metadata.clone(),
        last_used_ms: state.activity.last_ms(&svc_cfg.name),
        recent_restarts,
        runtime: runtime_info(svc_cfg),
        serving: serving_config(svc_cfg),
        container: container_detail(&state, svc_cfg, &name).await,
    };
    (StatusCode::OK, Json(detail)).into_response()
}

#[utoipa::path(
    summary = "Get launch command preview",
    get,
    path = "/api/services/{name}/command",
    params(("name" = String, Path, description = "Service name")),
    responses(
        (status = 200, body = LaunchCommandResponse),
        (status = 404, body = ApiError, description = "service_not_found"),
        (status = 422, body = ApiError, description = "preview_failed")
    )
)]
pub async fn service_command(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    let eff = state.config.effective();
    let Some(svc_cfg) = eff.services.iter().find(|s| s.name == name) else {
        return ApiErrorCode::ServiceNotFound {
            name: name.as_str().into(),
        }
        .into_response();
    };

    // A run is identified by its `run_id`, not by a host PID: a container
    // workload often has no host PID at all, and keying off one would label
    // a live container's command as a not-yet-started preview.
    let running = state
        .registry
        .get(&svc_cfg.name)
        .map(|h| h.peek())
        .and_then(|s| s.run_id)
        .is_some();
    let source = if running {
        LaunchCommandSource::Running
    } else {
        LaunchCommandSource::Preview
    };

    let snapshot = state.snapshot.read().clone();
    let table = state.allocations.lock().clone();
    let corrections = state.rolling.get(&svc_cfg.name).corrections();
    let fs = state.system.fs.as_ref();

    // On-empty: what the command would be if no other services held
    // pledges. This should succeed whenever the model fits on the
    // hardware at all. A containerized service renders a container preview
    // instead of a process SpawnConfig.
    let on_empty = match render_preview(
        svc_cfg,
        &snapshot,
        &ananke_allocator::AllocationTable::new(),
        fs,
        corrections,
        source,
    ) {
        Ok(cmd) => cmd,
        Err(e) => {
            return ApiErrorCode::PreviewFailed {
                name: svc_cfg.name.clone(),
                reason: e.to_string(),
            }
            .into_response();
        }
    };

    // Active: what the command would be under the current device state
    // and pledge book. Gracefully `None` when the service can't fit
    // alongside currently running services.
    let active = render_preview(svc_cfg, &snapshot, &table, fs, corrections, source).ok();

    let response = LaunchCommandResponse { on_empty, active };
    (StatusCode::OK, Json(response)).into_response()
}

/// Render either a process or container launch preview for a service.
fn render_preview(
    svc_cfg: &ananke_config::validate::ServiceConfig,
    snapshot: &ananke_devices::DeviceSnapshot,
    table: &ananke_allocator::AllocationTable,
    fs: &dyn ananke_system::Fs,
    corrections: ananke_tracking::rolling::Corrections,
    source: LaunchCommandSource,
) -> Result<LaunchCommand, ananke_supervise::PreviewError> {
    if svc_cfg.container.is_some() {
        // A preview is illustrative, so it is built against a placeholder
        // owner rather than this installation's. Stamping the real one
        // would make a copy-pasted `create_argv` produce a container the
        // owner-scoped sweep then deletes as an unrecorded leak.
        let spec = crate::supervise::preview_container_command(
            svc_cfg,
            snapshot,
            table,
            fs,
            corrections,
            PREVIEW_OWNER_UUID,
            0,
        )?
        .ok_or(ananke_supervise::PreviewError::NoModelPath)?;
        Ok(LaunchCommand {
            source,
            argv: spec.command.clone(),
            env: spec
                .env
                .iter()
                .map(|(k, v)| EnvVar {
                    key: k.clone(),
                    value: v.clone(),
                })
                .collect(),
            env_inherit: false,
            container: Some(render_launch_container(&spec, &svc_cfg.name)),
        })
    } else {
        let spawn_cfg =
            crate::supervise::preview_command(svc_cfg, snapshot, table, fs, corrections)?;
        Ok(render_launch_command(spawn_cfg, source))
    }
}

fn render_launch_command(
    spawn_cfg: crate::supervise::SpawnConfig,
    source: LaunchCommandSource,
) -> LaunchCommand {
    let mut argv = Vec::with_capacity(spawn_cfg.args.len() + 1);
    argv.push(spawn_cfg.binary);
    argv.extend(spawn_cfg.args);
    let env_inherit = spawn_cfg.env_inherit;
    let env = spawn_cfg
        .env
        .into_iter()
        .map(|(key, value)| EnvVar { key, value })
        .collect();
    LaunchCommand {
        source,
        argv,
        env,
        env_inherit,
        container: None,
    }
}

/// Owner label used in launch previews. Deliberately not a real
/// installation identity: a preview is for reading and for copy-pasting,
/// and neither should produce something ananke will later reap.
const PREVIEW_OWNER_UUID: &str = "00000000-0000-0000-0000-000000000000";

/// Convert a resolved [`ContainerSpec`] into the wire `LaunchContainer`
/// shape, exposing the runtime create command and in-container argv without
/// resolving passthrough secret values.
fn render_launch_container(
    spec: &ananke_spawn::ContainerSpec,
    service_name: &str,
) -> LaunchContainer {
    let publication = match (spec.host_port, spec.container_port) {
        (Some(hp), Some(cp)) => Some(format!("127.0.0.1:{hp}:{cp}")),
        _ => None,
    };
    let create_argv = ananke_system::container::render_create_argv(
        spec.runtime_executable
            .as_deref()
            .unwrap_or_else(|| spec.runtime.executable()),
        spec,
    );
    LaunchContainer {
        runtime: spec.runtime.as_str().to_string(),
        image: spec.image.clone(),
        name_pattern: ananke_supervise::spawn::container::container_name_pattern(service_name),
        argv: spec.command.clone(),
        env: spec
            .env
            .iter()
            .map(|(k, v)| EnvVar {
                key: k.clone(),
                value: v.clone(),
            })
            .collect(),
        env_passthrough: spec.env_passthrough.clone(),
        mounts: spec
            .mounts
            .iter()
            .map(|m| LaunchMount {
                source: m.source.clone(),
                target: m.target.clone(),
                read_only: m.read_only,
            })
            .collect(),
        network: match spec.network {
            ananke_spawn::ContainerNetwork::Bridge => "bridge".to_string(),
            ananke_spawn::ContainerNetwork::Host => "host".to_string(),
        },
        publication,
        ipc: match spec.ipc {
            ananke_spawn::ContainerIpc::Private => "private".to_string(),
            ananke_spawn::ContainerIpc::Host => "host".to_string(),
        },
        gpu_devices: spec.gpu_devices.clone(),
        create_argv,
    }
}

/// Container identity for the detail view: runtime, image, network mode,
/// and the live name/ID pulled from the running row when present.
async fn container_detail(
    state: &AppState,
    svc_cfg: &ServiceConfig,
    name: &str,
) -> Option<ananke_api::services::detail::ContainerDetail> {
    let container = svc_cfg.container.as_ref().cloned()?;
    let service_id = state.db.resolve_service_id(name).await.ok().flatten();
    let live = match service_id {
        Some(sid) => state.db.latest_running(sid).await.ok().flatten(),
        None => None,
    };
    let (container_id, container_name) = match live {
        Some(r) => (r.container_id, r.container_name),
        None => (None, None),
    };
    Some(ananke_api::services::detail::ContainerDetail {
        runtime: container.runtime.executable().to_string(),
        image: container.image,
        network: match container.network {
            ananke_config::validate::ContainerNetwork::Bridge => "bridge".to_string(),
            ananke_config::validate::ContainerNetwork::Host => "host".to_string(),
        },
        container_id,
        container_name,
    })
}

/// Which llama-server implementation serves a llama-cpp service, plus
/// the fork-specific parameters when it is not mainline. Computed for the
/// detail view so the frontend can render a runtime card without a spawn.
fn runtime_info(svc_cfg: &ServiceConfig) -> Option<ananke_api::services::detail::RuntimeInfo> {
    use ananke_api::services::detail::{IkParams, RuntimeInfo};
    let lc = svc_cfg.llama_cpp()?;
    let ik = match lc.runtime.ik() {
        None => {
            return Some(RuntimeInfo {
                kind: "llama-cpp".into(),
                ik: None,
            });
        }
        Some(ik) => ik,
    };
    Some(RuntimeInfo {
        kind: "ik-llama".into(),
        ik: Some(IkParams {
            mla: ik.mla,
            dsa: ik.dsa,
            attn_max_batch: ik.attn_max_batch,
            runtime_repack: ik.runtime_repack,
        }),
    })
}

/// Curated serving knobs for the detail view, including the derived
/// per-slot context (`context / parallel` under a statically-split KV
/// pool — the number that actually bounds a request, and one no single
/// config key states).
fn serving_config(svc_cfg: &ServiceConfig) -> Option<ananke_api::services::detail::ServingConfig> {
    use crate::config::OffloadMode;
    let lc = svc_cfg.llama_cpp()?;
    let parallel = lc.parallel.unwrap_or(1);
    let kv_unified = lc.kv_unified.unwrap_or(false);
    let effective_context_per_slot = lc
        .context
        .and_then(|c| (parallel > 1 && !kv_unified).then_some(c / parallel.max(1)));
    Some(ananke_api::services::detail::ServingConfig {
        binary: lc.binary.to_string_lossy().into_owned(),
        cache_type_k: lc.cache_type_k.as_deref().unwrap_or("f16").to_string(),
        cache_type_v: lc.cache_type_v.as_deref().unwrap_or("f16").to_string(),
        flash_attn: lc.flash_attn.unwrap_or(false),
        parallel,
        kv_unified,
        effective_context_per_slot,
        spec_type: lc.spec_type.as_deref().map(str::to_string),
        draft_model: lc
            .draft_model
            .as_deref()
            .and_then(|p| p.file_name().map(|f| f.to_string_lossy().into_owned())),
        expert_offload: match lc.expert_offload {
            OffloadMode::Off => "off".to_string(),
            OffloadMode::Auto => "auto".to_string(),
            OffloadMode::Layers(n) => format!("{n} layers"),
        },
        batch_size: lc.batch_size,
        ubatch_size: lc.ubatch_size,
        threads: lc.threads,
        threads_batch: lc.threads_batch,
        numa: lc.numa.map(|n| n.as_flag().to_string()),
        mmap: lc.mmap.unwrap_or(true),
        mlock: lc.mlock.unwrap_or(false),
    })
}

/// The list view's footprint figure for one service.
///
/// A successful placement is summed per device — that is what the service
/// would actually reserve. A failed one has no devices to sum, and reporting
/// the empty sum as `0` conflated three unrelated situations: a legitimately
/// CPU-only service, one that hasn't been estimated yet, and a real model that
/// simply couldn't be placed. Fall back to the estimator's aggregate demand so
/// the row still conveys the model's scale; `fit_verdict` tells the reader it
/// is a requirement rather than a reservation.
fn summary_footprint(
    state: &AppState,
    svc_cfg: &ServiceConfig,
    placement: Option<&PlacementPreview>,
    entry: &Option<EstimateCacheEntry>,
) -> Vec<DeviceFootprint> {
    let Some(placement) = placement else {
        return Vec::new();
    };
    if !placement.devices.is_empty() {
        return placement
            .devices
            .iter()
            .map(|d| DeviceFootprint {
                device: d.device.clone(),
                bytes: d.bytes,
            })
            .collect();
    }
    // Only a `does_not_fit` verdict gets the demand fallback. The frontend
    // keys its "needed" qualifier on that verdict, so reporting a requirement
    // under any other one renders it as memory the service is holding.
    if !matches!(placement.verdict, FitVerdict::DoesNotFit { .. }) {
        return Vec::new();
    }
    let Some(entry) = entry.as_ref() else {
        return Vec::new();
    };
    let est = &entry.estimate_full;
    let snapshot = state.snapshot.read();
    // Apply the same rolling corrections `placement_preview` applies, so both
    // figures describe the same model.
    let corrections = state.rolling.get(&svc_cfg.name).corrections();
    // Run the packer itself rather than re-deriving its arithmetic. Every term
    // — the head-vs-secondary logits trim, the CPU-side compute buffer, the
    // one-layer fudge, MTP, expert offload — falls out of the same code that
    // computes a real placement, so the two cannot disagree.
    let Ok(packed) =
        ananke_placement::pack_demand(est, &placement_inputs(svc_cfg), &snapshot, corrections)
    else {
        return Vec::new();
    };
    packed
        .allocation
        .bytes
        .iter()
        .map(|(&device, &bytes)| DeviceFootprint {
            device: device.as_display(),
            bytes,
        })
        .collect()
}
