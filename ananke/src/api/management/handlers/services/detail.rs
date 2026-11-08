//! `GET /api/services/{name}` and the live `service` resource.

use std::collections::BTreeMap;

use ananke_api::{
    services::detail::{ContainerDetail, RestartEvent, ServiceDetail},
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
        management::handlers::{
            model_estimate_entry, placement_preview, read_current_allocation,
            services::{
                config_view::{command_detail, placement_config, runtime_info, serving_config},
                list::wire_state,
            },
        },
    },
    config::{DeviceSlot, ServiceConfig},
    daemon::app_state::AppState,
};

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
    match service_detail_snapshot(&state, &name).await {
        Ok(detail) => (StatusCode::OK, Json(detail)).into_response(),
        Err(e) => e.into_response(),
    }
}

/// The `GET /api/services/{name}` body; the live `service` resource shares it.
pub(crate) async fn service_detail_snapshot(
    state: &AppState,
    name: &str,
) -> Result<ServiceDetail, ApiErrorCode> {
    let eff = state.config.effective();
    let Some(svc_cfg) = eff.services.iter().find(|s| s.name == name) else {
        return Err(ApiErrorCode::ServiceNotFound { name: name.into() });
    };
    let handle = state.registry.get(&svc_cfg.name);
    let snap = handle.as_ref().map(|h| h.peek());
    let placement_override: BTreeMap<String, u64> = svc_cfg
        .placement_override
        .iter()
        .map(|(k, v)| {
            let key = match k {
                DeviceSlot::Cpu => "cpu".to_string(),
                DeviceSlot::Gpu(n) => format!("gpu:{n}"),
            };
            (key, *v)
        })
        .collect();

    let recent_restarts: Vec<RestartEvent> = {
        let svc_id_opt = state.db.resolve_service_id(name).await.ok().flatten();
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

    let entry = model_estimate_entry(state, svc_cfg);
    let model_info = entry.as_ref().map(|e| e.model_info.clone());
    let estimate = entry.as_ref().map(|e| e.estimate.clone());
    let running = snap.as_ref().and_then(|s| s.pid).is_some();
    let placement_preview = placement_preview(
        state,
        svc_cfg,
        entry.as_ref().map(|e| &e.estimate_full),
        running,
    );
    let current_allocation = read_current_allocation(state, &svc_cfg.name);

    let (state_kind, disabled_reason) = wire_state(snap.as_ref().map(|s| &s.state));
    let detail = ServiceDetail {
        name: svc_cfg.name.to_string(),
        state: state_kind,
        disabled_reason,
        lifecycle: svc_cfg.lifecycle,
        priority: svc_cfg.priority,
        port: svc_cfg.port,
        private_port: svc_cfg.private_port,
        template: svc_cfg.template().as_str().to_string(),
        description: svc_cfg.description.clone(),
        openai_compat: svc_cfg.openai_compat,
        placement: placement_config(svc_cfg),
        command: command_detail(svc_cfg),
        placement_override,
        idle_timeout_ms: svc_cfg.idle_timeout_ms,
        run_id: snap.as_ref().and_then(|s| s.run_id),
        pid: snap.as_ref().and_then(|s| s.pid),
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
        container: container_detail(state, svc_cfg, name).await,
    };
    Ok(detail)
}

/// Container identity for the detail view: runtime, image, network mode,
/// and the live name/ID pulled from the running row when present.
async fn container_detail(
    state: &AppState,
    svc_cfg: &ServiceConfig,
    name: &str,
) -> Option<ContainerDetail> {
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
    Some(ContainerDetail {
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
