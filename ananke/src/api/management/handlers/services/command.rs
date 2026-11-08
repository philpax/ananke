//! `GET /api/services/{name}/command`: the launch command preview.

use ananke_api::{
    services::command::{
        EnvVar, LaunchCommand, LaunchCommandResponse, LaunchCommandSource, LaunchContainer,
        LaunchMount,
    },
    shared::errors::ApiError,
};
use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};

use crate::{api::errors::ApiErrorCode, daemon::app_state::AppState};

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
