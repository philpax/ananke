//! Read-only management endpoints.

use ananke_api::{DeviceReservation, DeviceSummary, LogLine, ServiceDetail, ServiceSummary};
use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{Router, get},
};

use crate::daemon::app_state::AppState;

pub fn register(router: Router, state: AppState) -> Router {
    // Build the typed router against AppState, collapse to Router<()> via
    // with_state, then merge into the caller's router.
    let mgmt: Router = Router::new()
        .route("/api/services", get(list_services))
        .route("/api/services/:name", get(service_detail))
        .route("/api/devices", get(list_devices))
        .with_state(state);
    router.merge(mgmt)
}

#[utoipa::path(get, path = "/api/services", responses((status = 200, body = Vec<ServiceSummary>)))]
pub async fn list_services(State(state): State<AppState>) -> Response {
    let mut out = Vec::new();
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
        out.push(ServiceSummary {
            name: svc_cfg.name.to_string(),
            state: state_name,
            lifecycle: svc_cfg.lifecycle.as_str().to_string(),
            priority: svc_cfg.priority,
            port: svc_cfg.port,
            run_id: peek.as_ref().and_then(|p| p.run_id),
            pid: peek.as_ref().and_then(|p| p.pid),
            // Placeholder: elastic borrower tracking is deferred to a later phase.
            elastic_borrower: None,
            ananke_metadata: svc_cfg.metadata.clone(),
        });
    }
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
        ananke_metadata: svc_cfg.metadata.clone(),
    };
    (StatusCode::OK, Json(detail)).into_response()
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
