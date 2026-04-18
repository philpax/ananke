//! Read-only management endpoints.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{Router, get};

use crate::app_state::AppState;
use crate::management_api::types::{
    DeviceReservation, DeviceSummary, LogLine, ServiceDetail, ServiceSummary,
};
use crate::state::ServiceState;

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
    for svc_cfg in state.config.services.iter() {
        let handle = state.registry.get(&svc_cfg.name);
        let snap = match &handle {
            Some(h) => h.snapshot().await,
            None => None,
        };
        out.push(ServiceSummary {
            name: svc_cfg.name.to_string(),
            state: snap
                .as_ref()
                .map(|s| state_name(&s.state))
                .unwrap_or_else(|| "unknown".into()),
            lifecycle: format!("{:?}", svc_cfg.lifecycle).to_lowercase(),
            priority: svc_cfg.priority,
            port: svc_cfg.port,
            run_id: snap.as_ref().and_then(|s| s.run_id),
            pid: snap.as_ref().and_then(|s| s.pid),
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
    let Some(svc_cfg) = state.config.services.iter().find(|s| s.name == name) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "not found"})),
        )
            .into_response();
    };
    let handle = state.registry.get(&svc_cfg.name);
    let snap = match &handle {
        Some(h) => h.snapshot().await,
        None => None,
    };
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

    let svc_id_opt: Option<i64> = state
        .db
        .with_conn(|c| {
            c.query_row(
                "SELECT service_id FROM services WHERE name = ?1",
                [&name],
                |r| r.get(0),
            )
        })
        .ok();

    let recent_logs: Vec<LogLine> = match svc_id_opt {
        Some(svc_id) => state
            .db
            .with_conn(|c| {
                let mut stmt = c.prepare(
                    "SELECT timestamp_ms, stream, line FROM service_logs
                     WHERE service_id = ?1 ORDER BY timestamp_ms DESC, seq DESC LIMIT 200",
                )?;
                let rows = stmt.query_map([svc_id], |r| {
                    Ok(LogLine {
                        timestamp_ms: r.get(0)?,
                        stream: r.get(1)?,
                        line: r.get(2)?,
                    })
                })?;
                rows.collect::<Result<Vec<_>, _>>()
            })
            .unwrap_or_default(),
        None => Vec::new(),
    };

    let detail = ServiceDetail {
        name: svc_cfg.name.to_string(),
        state: snap
            .as_ref()
            .map(|s| state_name(&s.state))
            .unwrap_or_else(|| "unknown".into()),
        lifecycle: format!("{:?}", svc_cfg.lifecycle).to_lowercase(),
        priority: svc_cfg.priority,
        port: svc_cfg.port,
        private_port: svc_cfg.private_port,
        template: format!("{:?}", svc_cfg.template).to_lowercase(),
        placement_override,
        idle_timeout_ms: svc_cfg.idle_timeout_ms,
        run_id: snap.as_ref().and_then(|s| s.run_id),
        pid: snap.as_ref().and_then(|s| s.pid),
        recent_logs,
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

fn state_name(s: &ServiceState) -> String {
    match s {
        ServiceState::Idle => "idle",
        ServiceState::Starting => "starting",
        ServiceState::Warming => "warming",
        ServiceState::Running => "running",
        ServiceState::Draining => "draining",
        ServiceState::Stopped => "stopped",
        ServiceState::Evicted => "evicted",
        ServiceState::Failed { .. } => "failed",
        ServiceState::Disabled { .. } => "disabled",
    }
    .to_string()
}
