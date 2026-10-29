//! Manual reservations: operator-placed holds on device memory.
//!
//! A reservation is stored in the database, so it survives daemon and host
//! restarts until an operator removes it, and mirrored into the shared
//! allocation table under a `reservation:<id>` key. Every placement check
//! sums that row like a service pledge, so ananke never places a service
//! into it, while processes outside ananke may use the memory. No
//! supervisor owns the row, so nothing evicts it. A reservation outranks
//! every service: placing one evicts whatever is in the way, busy and
//! persistent services included; see [`crate::supervise::manual_reservation`].

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use ananke_api::{
    events::Event,
    reservations::{
        CreateReservationRequest, CreateReservationResponse, ManualReservation,
        ReservationsResponse, ServiceReservation,
    },
};
use ananke_config::DeviceSlot;
use parking_lot::Mutex;
use smol_str::SmolStr;
use tracing::{info, warn};

use crate::{
    api::errors::ApiErrorCode,
    daemon::app_state::AppState,
    supervise::{
        drain::DrainReason,
        eviction_candidates,
        manual_reservation::{
            ReservationPlanFailure, manual_reservation_id, manual_reservation_key, plan_evictions,
        },
    },
};

/// Longest a placement waits for the services it evicted to stop before
/// answering anyway. Their pledges are already outweighed by the
/// reservation's row, so placement elsewhere stays correct either way.
const EVICTION_WAIT: Duration = Duration::from_secs(120);

/// How often a placement re-checks whether its victims have stopped.
const EVICTION_POLL: Duration = Duration::from_millis(100);

/// Longest label accepted, in bytes.
const MAX_LABEL_BYTES: usize = 200;

/// The daemon's manual reservations, keyed by id, plus the lock that
/// serialises placing and removing them.
#[derive(Clone, Default)]
pub struct ReservationStore {
    entries: Arc<Mutex<BTreeMap<i64, ManualReservation>>>,
    changes: Arc<tokio::sync::Mutex<()>>,
}

impl ReservationStore {
    /// The stored reservations, oldest first.
    pub fn list(&self) -> Vec<ManualReservation> {
        self.entries.lock().values().cloned().collect()
    }
}

/// Load every stored reservation into the store and the allocation table.
/// Called at boot before any service is provisioned, so services start
/// around the reservations rather than the reverse.
pub async fn load(state: &AppState) -> Result<(), ananke_errors::ExpectedError> {
    let rows = state.db.list_manual_reservations().await?;
    let mut entries = state.reservations.entries.lock();
    let mut table = state.allocations.lock();
    for row in rows {
        let mut want = BTreeMap::new();
        let mut devices = BTreeMap::new();
        for (key, mb) in &row.devices {
            let Some(slot) = parse_device(key) else {
                warn!(id = row.reservation_id, device = %key, "skipping reserved device ananke does not know");
                continue;
            };
            want.insert(slot, *mb as u64);
            devices.insert(key.clone(), *mb as u64 * 1024 * 1024);
        }
        table.insert(manual_reservation_key(row.reservation_id), want);
        let reservation = ManualReservation {
            id: row.reservation_id,
            label: row.label,
            created_at_ms: row.created_at_ms,
            devices,
        };
        info!(id = reservation.id, label = ?reservation.label, devices = ?reservation.devices, "manual reservation restored");
        entries.insert(reservation.id, reservation);
    }
    Ok(())
}

/// The `GET /api/reservations` body; the live `reservations` resource
/// shares it.
pub fn reservations_snapshot(state: &AppState) -> ReservationsResponse {
    let table = state.allocations.lock().clone();
    let config = state.config.effective();
    let services = table
        .iter()
        .filter(|(key, _)| manual_reservation_id(key).is_none())
        .filter(|(_, row)| row.values().any(|mb| *mb > 0))
        .filter_map(|(name, row)| {
            let handle = state.registry.get(name)?;
            config.services.iter().find(|s| &s.name == name)?;
            Some(ServiceReservation {
                service: name.to_string(),
                state: handle.peek_state().kind(),
                inflight_count: state.inflight.current(name),
                devices: device_bytes(row),
            })
        })
        .collect();
    ReservationsResponse {
        manual: state.reservations.list(),
        services,
    }
}

/// Place a manual reservation, evicting whatever services are in the way,
/// and wait for them to stop.
pub async fn create(
    state: &AppState,
    req: CreateReservationRequest,
) -> Result<CreateReservationResponse, ApiErrorCode> {
    let label = match req.label.map(|l| l.trim().to_string()) {
        Some(l) if l.len() > MAX_LABEL_BYTES => {
            return Err(invalid(format!(
                "label is longer than {MAX_LABEL_BYTES} bytes"
            )));
        }
        Some(l) if l.is_empty() => None,
        other => other,
    };
    let want = requested_mb(state, &req.devices)?;

    let _serial = state.reservations.changes.lock().await;
    let snapshot = state.snapshot.read().clone();
    let table = state.allocations.lock().clone();
    let deps = state.supervisor_deps();
    let candidates = eviction_candidates(&deps, None);
    let now = tokio::time::Instant::now();
    let victims = plan_evictions(&want, &snapshot, &table, &candidates, |name| {
        state
            .activity
            .last(&SmolStr::new(name))
            .map(|t| now.saturating_duration_since(t).as_millis())
    })
    .map_err(
        |ReservationPlanFailure::NoFit { reason }| ApiErrorCode::ReservationNoFit { reason },
    )?;

    for victim in &victims {
        if let Some(handle) = state.registry.get(victim) {
            info!(service = %victim, "evicting for manual reservation");
            handle.begin_drain(DrainReason::Eviction).await;
        }
    }

    let created_at_ms = ananke_tracking::now_unix_ms();
    let stored_mb: BTreeMap<String, i64> = want
        .iter()
        .map(|(slot, mb)| (ananke_supervise::slot_to_key(slot), *mb as i64))
        .collect();
    let id = state
        .db
        .insert_manual_reservation(label.as_deref(), created_at_ms, &stored_mb)
        .await
        .map_err(|e| ApiErrorCode::PersistFailed {
            reason: e.to_string(),
        })?;
    // The row goes in before the victims finish stopping, so nothing else
    // can take the memory they free.
    state
        .allocations
        .lock()
        .insert(manual_reservation_key(id), want.clone());
    let reservation = ManualReservation {
        id,
        label,
        created_at_ms,
        devices: device_bytes(&want),
    };
    state
        .reservations
        .entries
        .lock()
        .insert(id, reservation.clone());
    info!(id, label = ?reservation.label, devices = ?reservation.devices, evicted = ?victims, "manual reservation placed");
    state.events.publish(Event::ReservationPlaced {
        id,
        label: reservation.label.clone(),
        devices: reservation.devices.clone(),
        evicted: victims.clone(),
        at_ms: created_at_ms,
    });

    if req.wait.unwrap_or(true) {
        wait_for_release(state, &victims).await;
    }
    Ok(CreateReservationResponse {
        reservation,
        evicted: victims.iter().map(ToString::to_string).collect(),
    })
}

/// Remove a manual reservation, releasing its memory to placement.
pub async fn remove(state: &AppState, id: i64) -> Result<(), ApiErrorCode> {
    let _serial = state.reservations.changes.lock().await;
    let Some(reservation) = state.reservations.entries.lock().get(&id).cloned() else {
        return Err(ApiErrorCode::ReservationNotFound { id });
    };
    state
        .db
        .delete_manual_reservation(id)
        .await
        .map_err(|e| ApiErrorCode::PersistFailed {
            reason: e.to_string(),
        })?;
    state.allocations.lock().remove(&manual_reservation_key(id));
    state.reservations.entries.lock().remove(&id);
    info!(id, label = ?reservation.label, "manual reservation removed");
    state.events.publish(Event::ReservationRemoved {
        id,
        label: reservation.label,
        at_ms: ananke_tracking::now_unix_ms(),
    });
    Ok(())
}

/// Wait until every victim's pledge has left the allocation table, which
/// happens when its drain completes, or until [`EVICTION_WAIT`] passes.
async fn wait_for_release(state: &AppState, victims: &[SmolStr]) {
    if victims.is_empty() {
        return;
    }
    let deadline = tokio::time::Instant::now() + EVICTION_WAIT;
    loop {
        let pending: Vec<SmolStr> = {
            let table = state.allocations.lock();
            victims
                .iter()
                .filter(|v| {
                    table
                        .get(*v)
                        .is_some_and(|row| row.values().any(|mb| *mb > 0))
                })
                .cloned()
                .collect()
        };
        if pending.is_empty() {
            return;
        }
        if tokio::time::Instant::now() >= deadline {
            warn!(
                ?pending,
                "evicted services still hold memory; answering anyway"
            );
            return;
        }
        tokio::time::sleep(EVICTION_POLL).await;
    }
}

/// Validate the requested devices against the current snapshot and convert
/// bytes to whole MiB, rounding up.
fn requested_mb(
    state: &AppState,
    devices: &BTreeMap<String, u64>,
) -> Result<BTreeMap<DeviceSlot, u64>, ApiErrorCode> {
    if devices.is_empty() {
        return Err(invalid("a reservation needs at least one device".into()));
    }
    let snapshot = state.snapshot.read().clone();
    let mut want = BTreeMap::new();
    for (key, bytes) in devices {
        let slot = parse_device(key)
            .ok_or_else(|| invalid(format!("unknown device `{key}`; use `cpu` or `gpu:N`")))?;
        let Some(total) = snapshot.total_bytes(&slot) else {
            return Err(invalid(format!("device `{key}` is not present")));
        };
        if *bytes == 0 {
            return Err(invalid(format!(
                "device `{key}`: the amount must be positive"
            )));
        }
        if *bytes > total {
            return Err(invalid(format!(
                "device `{key}`: {bytes} bytes is more than its {total} total"
            )));
        }
        want.insert(slot, bytes.div_ceil(1024 * 1024));
    }
    Ok(want)
}

/// Parse a device id (`"cpu"` or `"gpu:N"`).
fn parse_device(key: &str) -> Option<DeviceSlot> {
    if key == "cpu" {
        return Some(DeviceSlot::Cpu);
    }
    key.strip_prefix("gpu:")?.parse().ok().map(DeviceSlot::Gpu)
}

fn device_bytes(row: &BTreeMap<DeviceSlot, u64>) -> BTreeMap<String, u64> {
    row.iter()
        .map(|(slot, mb)| (ananke_supervise::slot_to_key(slot), mb * 1024 * 1024))
        .collect()
}

fn invalid(reason: String) -> ApiErrorCode {
    ApiErrorCode::InvalidRequest { reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_ids_parse() {
        assert_eq!(parse_device("cpu"), Some(DeviceSlot::Cpu));
        assert_eq!(parse_device("gpu:1"), Some(DeviceSlot::Gpu(1)));
        assert_eq!(parse_device("gpu:x"), None);
        assert_eq!(parse_device("GPU0"), None);
    }
}
