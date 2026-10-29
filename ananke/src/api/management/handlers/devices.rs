//! Device inventory endpoint.

use ananke_api::devices::list::{DeviceReservation, DeviceSummary};
use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};

use crate::{daemon::app_state::AppState, supervise::manual_reservation::manual_reservation_id};

#[utoipa::path(
    summary = "List devices and reservations",
    get,
    path = "/api/devices",
    responses((status = 200, body = Vec<DeviceSummary>))
)]
pub async fn list_devices(State(state): State<AppState>) -> Response {
    (StatusCode::OK, Json(devices_snapshot(&state))).into_response()
}

/// The `GET /api/devices` body; the live `devices` resource shares it.
pub(crate) fn devices_snapshot(state: &AppState) -> Vec<DeviceSummary> {
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
                    reservation: manual_reservation_id(svc),
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
                        reservation: manual_reservation_id(svc),
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

    out
}
