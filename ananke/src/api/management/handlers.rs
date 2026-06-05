//! Read-only management endpoints.

use ananke_api::{
    DevicePlacement, DeviceReservation, DeviceSummary, EnvVar, LaunchCommand, LaunchCommandSource,
    LogLine, PlacementPreview, ServiceDetail, ServiceSummary, ServicesResponse,
};
use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{Router, get},
};
use smol_str::SmolStr;
use tracing::warn;

use crate::{
    config::ServiceConfig,
    daemon::{app_state::AppState, estimate_cache::CacheEntry},
    estimator::{EstimatorInputs, estimate_with_summary},
};

pub fn register(router: Router, state: AppState) -> Router {
    // Build the typed router against AppState, collapse to Router<()> via
    // with_state, then merge into the caller's router.
    let mgmt: Router = Router::new()
        .route("/api/services", get(list_services))
        .route("/api/services/:name", get(service_detail))
        .route("/api/services/:name/command", get(service_command))
        .route("/api/devices", get(list_devices))
        .with_state(state);
    router.merge(mgmt)
}

#[utoipa::path(get, path = "/api/services", responses((status = 200, body = ServicesResponse)))]
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
        services.push(ServiceSummary {
            name: svc_cfg.name.to_string(),
            state: state_name,
            lifecycle: svc_cfg.lifecycle.as_str().to_string(),
            priority: svc_cfg.priority,
            port: svc_cfg.port,
            run_id: peek.as_ref().and_then(|p| p.run_id),
            pid: peek.as_ref().and_then(|p| p.pid),
            // Placeholder: elastic borrower tracking is deferred to a later phase.
            elastic_borrower: None,
            // Config-only check, no GGUF read — safe to ship in the
            // list view that the frontend polls every 2 s.
            has_mmproj: svc_cfg.llama_cpp().map(|lc| lc.mmproj.is_some()),
            modality: svc_cfg.modality,
            ananke_metadata: svc_cfg.metadata.clone(),
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
    get,
    path = "/api/services/{name}",
    responses((status = 200, body = ServiceDetail), (status = 404))
)]
pub async fn service_detail(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    let eff = state.config.effective();
    let Some(svc_cfg) = eff.services.iter().find(|s| s.name == name) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "not found"})),
        )
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
        rolling_mean: if rc.sample_count == 0 {
            None
        } else {
            Some(rc.rolling_mean as f32)
        },
        rolling_samples: rc.sample_count.into(),
        observed_peak_bytes,
        // Placeholder: elastic borrower tracking is deferred to a later phase.
        elastic_borrower: None,
        model_info,
        estimate,
        placement_preview,
        current_allocation,
        modality: svc_cfg.modality,
        ananke_metadata: svc_cfg.metadata.clone(),
    };
    (StatusCode::OK, Json(detail)).into_response()
}

#[utoipa::path(
    get,
    path = "/api/services/{name}/command",
    params(("name" = String, Path, description = "Service name")),
    responses(
        (status = 200, body = LaunchCommand),
        (status = 404),
        (status = 422, description = "The command could not be computed (e.g. placement does not fit)")
    )
)]
pub async fn service_command(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    let eff = state.config.effective();
    let Some(svc_cfg) = eff.services.iter().find(|s| s.name == name) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "not found"})),
        )
            .into_response();
    };

    // A live pid means the service is running; otherwise the rendered command
    // is what it would launch with on the next start.
    let running = state
        .registry
        .get(&svc_cfg.name)
        .map(|h| h.peek())
        .and_then(|s| s.pid)
        .is_some();
    let source = if running {
        LaunchCommandSource::Running
    } else {
        LaunchCommandSource::Preview
    };

    let snapshot = state.snapshot.read().clone();
    let table = state.allocations.lock().clone();
    let rolling_mean = state.rolling.get(&svc_cfg.name).rolling_mean;

    let spawn_cfg = match crate::supervise::preview_command(
        svc_cfg,
        &snapshot,
        &table,
        state.system.fs.as_ref(),
        rolling_mean,
    ) {
        Ok(cfg) => cfg,
        Err(e) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response();
        }
    };

    let mut argv = Vec::with_capacity(spawn_cfg.args.len() + 1);
    argv.push(spawn_cfg.binary);
    argv.extend(spawn_cfg.args);
    let env = spawn_cfg
        .env
        .into_iter()
        .map(|(key, value)| EnvVar { key, value })
        .collect();

    let command = LaunchCommand { source, argv, env };
    (StatusCode::OK, Json(command)).into_response()
}

/// Look up the cached `(ModelInfo, EstimateSummary)` for a service,
/// computing them on cache miss. Returns `(None, None)` for command-
/// template services and for llama-cpp services whose GGUF can't be
/// read — errors are logged on the daemon side; the frontend just
/// sees the absence.
fn model_estimate_entry(state: &AppState, svc_cfg: &ServiceConfig) -> Option<CacheEntry> {
    svc_cfg.llama_cpp()?;
    // Build the inputs once so the fingerprint we compare against is
    // identical to the one `compute_estimate_entry` would write into
    // the cache on miss.
    let inputs = EstimatorInputs::from_service(svc_cfg)?;
    let fingerprint = inputs.config_fingerprint();
    let lc = svc_cfg.llama_cpp()?;
    let svc_name = svc_cfg.name.clone();

    if let Some(entry) = state.estimate_cache.get(
        &svc_name,
        lc.model.as_path(),
        lc.mmproj.as_deref(),
        fingerprint,
    ) {
        return Some(entry);
    }
    let entry = compute_estimate_entry(state, svc_cfg)?;
    state.estimate_cache.insert(svc_name, entry.clone());
    Some(entry)
}

/// Project a service's placement to the wire `PlacementPreview`. A manual
/// An active service shows the allocation it actually holds (its live pledge);
/// otherwise this is a what-if: a `placement_override` is honoured verbatim, a
/// command-template service picks a GPU dynamically, and the rest run the
/// estimator-path packer against the live snapshot and pledge book. Returns
/// `None` when there is nothing to show — a llama-cpp service whose GGUF
/// couldn't be read, or a command service that reserves no VRAM.
fn placement_preview(
    state: &AppState,
    svc_cfg: &ServiceConfig,
    estimate: Option<&crate::estimator::Estimate>,
    running: bool,
) -> Option<PlacementPreview> {
    let snapshot = state.snapshot.read().clone();
    let table = state.allocations.lock().clone();

    // An active service holds a real pledge — show that, not a re-computed
    // what-if (which could even differ from where it actually landed).
    let live_pledge = running
        .then(|| table.get(&svc_cfg.name).cloned())
        .flatten()
        .filter(|row| !row.is_empty());

    let outcome = if let Some(row) = live_pledge {
        let devices = row
            .into_iter()
            .map(|(slot, mb)| {
                let id = match slot {
                    crate::config::DeviceSlot::Cpu => crate::devices::DeviceId::Cpu,
                    crate::config::DeviceSlot::Gpu(n) => crate::devices::DeviceId::Gpu(n),
                };
                (id, mb.saturating_mul(1024 * 1024))
            })
            .collect();
        crate::supervise::PlacementOutcome {
            devices,
            verdict: ananke_api::FitVerdict::Fits,
        }
    } else if !svc_cfg.placement_override.is_empty() {
        crate::supervise::preview_override_placement(svc_cfg, &snapshot, &table, running)
    } else if matches!(svc_cfg.template(), crate::config::Template::Command) {
        // Command-template service picking a GPU dynamically (e.g. ComfyUI):
        // `None` means it reserves no VRAM, so there is nothing to show.
        crate::supervise::preview_command_placement(svc_cfg, &snapshot, &table, running)?
    } else {
        let mut est = estimate?.clone();
        // Match the supervisor: apply the rolling drift correction before packing.
        est.weights_bytes =
            (est.weights_bytes as f64 * state.rolling.get(&svc_cfg.name).rolling_mean) as u64;
        crate::supervise::preview_placement(svc_cfg, &est, &snapshot, &table, running)
    };

    // A dynamic command service can grow past its reserved floor up to its
    // configured maximum; every other service is pinned at `bytes`.
    let growth_ceiling = match svc_cfg.allocation_mode {
        crate::config::AllocationMode::Dynamic { max_mb, .. }
            if matches!(svc_cfg.template(), crate::config::Template::Command) =>
        {
            Some(max_mb.saturating_mul(1024 * 1024))
        }
        _ => None,
    };
    let devices = outcome
        .devices
        .into_iter()
        .map(|(id, bytes)| {
            let slot = match id {
                crate::devices::DeviceId::Cpu => crate::config::DeviceSlot::Cpu,
                crate::devices::DeviceId::Gpu(n) => crate::config::DeviceSlot::Gpu(n),
            };
            let total_bytes = snapshot.total_bytes(&slot).unwrap_or(0);
            let used = total_bytes.saturating_sub(snapshot.free_bytes(&slot).unwrap_or(0));
            // For a running service its own resident VRAM is already counted in
            // `used`; subtract this service's share so the bar doesn't double it.
            let used_by_others_bytes = if running {
                used.saturating_sub(bytes)
            } else {
                used
            };
            let max_bytes = growth_ceiling.map(|c| c.max(bytes)).unwrap_or(bytes);
            DevicePlacement {
                device: id.as_display(),
                bytes,
                max_bytes,
                used_by_others_bytes,
                total_bytes,
            }
        })
        .collect();
    Some(PlacementPreview {
        devices,
        verdict: outcome.verdict,
    })
}

/// Run the estimator against the service's configured paths and
/// project the result through the shared `CacheEntry::build`
/// constructor. Returns `None` when the GGUF can't be read or the
/// estimator refuses the architecture.
fn compute_estimate_entry(state: &AppState, svc_cfg: &ServiceConfig) -> Option<CacheEntry> {
    let lc = svc_cfg.llama_cpp()?;
    let inputs = EstimatorInputs::from_service(svc_cfg)?;
    let config_fingerprint = inputs.config_fingerprint();
    let model_path = lc.model.clone();
    let mmproj_path = lc.mmproj.clone();

    match estimate_with_summary(state.system.fs.as_ref(), &inputs) {
        Ok((summary, estimate)) => Some(CacheEntry::build(
            &summary,
            &estimate,
            model_path,
            mmproj_path,
            config_fingerprint,
        )),
        Err(e) => {
            warn!(service = %svc_cfg.name, error = %e, "model_info: estimator failed");
            None
        }
    }
}

/// Snapshot the service's current per-device pledge from the
/// allocation table. Empty when the service isn't running.
fn read_current_allocation(
    state: &AppState,
    name: &SmolStr,
) -> std::collections::BTreeMap<String, u64> {
    let table = state.allocations.lock();
    let Some(row) = table.get(name) else {
        return std::collections::BTreeMap::new();
    };
    row.iter()
        .map(|(slot, mb)| {
            let key = match slot {
                crate::config::DeviceSlot::Cpu => "cpu".to_string(),
                crate::config::DeviceSlot::Gpu(n) => format!("gpu:{n}"),
            };
            (key, *mb)
        })
        .collect()
}

#[utoipa::path(
    get,
    path = "/api/devices",
    responses((status = 200, body = Vec<DeviceSummary>))
)]
pub async fn list_devices(State(state): State<AppState>) -> Response {
    let snap = state.snapshot.read().clone();
    let alloc = state.allocations.lock().clone();

    let mut out = Vec::new();

    for g in &snap.gpus {
        let slot = crate::config::DeviceSlot::Gpu(g.id);
        let reservations: Vec<DeviceReservation> = alloc
            .iter()
            .filter_map(|(svc, a)| {
                a.get(&slot).map(|mb| DeviceReservation {
                    service: svc.to_string(),
                    bytes: mb * 1024 * 1024,
                    // Placeholder: elastic tracking is deferred to a later phase.
                    elastic: false,
                })
            })
            .collect();
        out.push(DeviceSummary {
            id: format!("gpu:{}", g.id),
            name: g.name.clone(),
            total_bytes: g.total_bytes,
            free_bytes: g.free_bytes,
            reservations,
        });
    }

    if let Some(c) = &snap.cpu {
        let reservations: Vec<DeviceReservation> = alloc
            .iter()
            .filter_map(|(svc, a)| {
                a.get(&crate::config::DeviceSlot::Cpu)
                    .map(|mb| DeviceReservation {
                        service: svc.to_string(),
                        bytes: mb * 1024 * 1024,
                        // Placeholder: elastic tracking is deferred to a later phase.
                        elastic: false,
                    })
            })
            .collect();
        out.push(DeviceSummary {
            id: "cpu".into(),
            name: "CPU".into(),
            total_bytes: c.total_bytes,
            free_bytes: c.available_bytes,
            reservations,
        });
    }

    (StatusCode::OK, Json(out)).into_response()
}
