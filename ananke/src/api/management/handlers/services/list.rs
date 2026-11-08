//! `GET /api/services` and the live `services` resource.

use ananke_api::{
    internal::fit_verdict::FitVerdict,
    services::{
        detail::PlacementPreview,
        list::{DeviceFootprint, ServiceSummary, ServicesResponse},
    },
    shared::{DisableReason, ServiceStateKind},
};
use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};

use crate::{
    api::management::handlers::{model_estimate_entry, placement_preview},
    config::{ServiceConfig, placement_inputs},
    daemon::app_state::AppState,
    supervise::{estimate_cache::EstimateCacheEntry, state::ServiceState},
};

#[utoipa::path(
    summary = "List all services",get, path = "/api/services", responses((status = 200, body = ServicesResponse)))]
pub async fn list_services(State(state): State<AppState>) -> Response {
    (StatusCode::OK, Json(services_snapshot(&state))).into_response()
}

/// The `GET /api/services` body; the live `services` resource shares it.
pub(crate) fn services_snapshot(state: &AppState) -> ServicesResponse {
    let mut services = Vec::new();
    let eff = state.config.effective();
    for svc_cfg in eff.services.iter() {
        let handle = state.registry.get(&svc_cfg.name);
        // `peek()` reads the supervisor's lock-free mirror directly — no
        // mailbox round-trip. `list_services` stays responsive even while
        // a supervisor is mid-drain or mid-spawn, and `run_id` / `pid`
        // are always populated for any service with a live child.
        let peek = handle.as_ref().map(|h| h.peek());
        let (state_kind, disabled_reason) = wire_state(peek.as_ref().map(|p| &p.state));
        let running = peek.as_ref().and_then(|p| p.pid).is_some();

        // Compute the fit verdict so the frontend can flag services
        // that can't start under current device conditions. Running
        // services short-circuit to `Fits` via the live pledge; the
        // estimate cache is usually warm after the first detail view
        // or service start.
        let entry = model_estimate_entry(state, svc_cfg);
        let placement = placement_preview(
            state,
            svc_cfg,
            entry.as_ref().map(|e| &e.estimate_full),
            running,
        );
        let fit_verdict = placement.as_ref().map(|p| p.verdict.clone());
        // One computation, two fields: the total is the breakdown's sum, so a row
        // showing both cannot show two figures that disagree.
        let footprint_devices = summary_footprint(state, svc_cfg, placement.as_ref(), &entry);
        let footprint_bytes = (!footprint_devices.is_empty())
            .then(|| footprint_devices.iter().map(|d| d.bytes).sum());
        let last_used_ms = state.activity.last_ms(&svc_cfg.name);

        services.push(ServiceSummary {
            name: svc_cfg.name.to_string(),
            description: svc_cfg.description.clone(),
            openai_compat: svc_cfg.openai_compat,
            state: state_kind,
            disabled_reason,
            lifecycle: svc_cfg.lifecycle,
            priority: svc_cfg.priority,
            port: svc_cfg.port,
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
    ServicesResponse {
        services,
        openai_api_port,
    }
}

/// A supervisor's state as the wire reports it; `unknown` when the service
/// has no supervisor yet.
pub(crate) fn wire_state(
    state: Option<&ServiceState>,
) -> (ServiceStateKind, Option<DisableReason>) {
    match state {
        Some(state) => (state.kind(), state.disabled_reason()),
        None => (ServiceStateKind::Unknown, None),
    }
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
