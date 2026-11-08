//! Launch orchestration: render the workload spec, spawn a native process or
//! create+start a container, persist the ownership row at the correct point
//! in each sequence, and return a running [`ManagedWorkload`] plus the
//! persisted identity.
//!
//! For containers the persistence ordering is mandatory and crash-safe:
//! a durable launch intent is written before create, the container ID is
//! attached after create, the running row is committed *before* start, and
//! only then is `start` invoked. A container can never be started before its
//! ownership row is persisted.

use ananke_allocator::placement::CommandArgs;
use ananke_config::validate::ServiceConfig;
use ananke_db::{Database, models::RunningService};
use ananke_devices::Allocation;
use ananke_system::SystemDeps;
use tracing::{info, warn};

use crate::{
    spawn::{container::render_container_spec, render_argv},
    workload::ManagedWorkload,
};

/// A fully launched workload: the running handle plus the persisted identity
/// the supervisor mirrors into its shared state.
pub struct LaunchedWorkload {
    pub workload: ManagedWorkload,
    pub run_id: i64,
    pub host_pid: Option<i32>,
    pub workload_kind: &'static str,
    pub command_line: String,
    /// Container ID for a container workload; `None` for native processes.
    pub container_id: Option<String>,
    /// Container name for a container workload; `None` for native processes.
    pub container_name: Option<String>,
}

/// Error from the launch orchestration path.
#[derive(Debug)]
pub enum LaunchError {
    Render(String),
    Create(String),
    Persist(String),
    RunId(String),
    Spawn(String),
    Start(String),
}

impl std::fmt::Display for LaunchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LaunchError::Render(e) => write!(f, "render workload: {e}"),
            LaunchError::Create(e) => write!(f, "create container: {e}"),
            LaunchError::Persist(e) => write!(f, "persist ownership: {e}"),
            LaunchError::RunId(e) => write!(f, "allocate run id: {e}"),
            LaunchError::Spawn(e) => write!(f, "spawn process: {e}"),
            LaunchError::Start(e) => write!(f, "start container: {e}"),
        }
    }
}

impl std::error::Error for LaunchError {}

/// Launch the workload for `svc` under `alloc`, returning the running handle
/// and the persisted identity. `service_id` is the DB service row id used to
/// write the `running_services` row.
/// `cmd_args` carries the placement engine's derived llama-server flags
/// (`-ngl`, `--tensor-split`, `--split-mode`, `--main-gpu`, `--n-cpu-moe`).
/// Passing `None` here silently renders the single-GPU defaults instead,
/// which is a different command from the one the packer planned and the one
/// the preview shows.
pub async fn launch_workload(
    svc: &ServiceConfig,
    alloc: &Allocation,
    cmd_args: Option<&CommandArgs>,
    service_id: i64,
    system: &SystemDeps,
    db: &Database,
) -> Result<LaunchedWorkload, LaunchError> {
    let previous = db
        .max_run_id(service_id)
        .await
        .map_err(|e| LaunchError::RunId(e.to_string()))?;
    let run_id = next_run_id(ananke_tracking::now_unix_ms(), previous);
    let allocation_json = allocation_json(svc, alloc);

    if svc.container.is_some() {
        launch_container(
            svc,
            alloc,
            cmd_args,
            service_id,
            run_id,
            allocation_json,
            system,
            db,
        )
        .await
    } else {
        launch_process(
            svc,
            alloc,
            cmd_args,
            service_id,
            run_id,
            allocation_json,
            system,
            db,
        )
        .await
    }
}

/// Pick the next run id: the launch time in ms, clamped to exceed every id the
/// service has used. Log positions order by `(run_id, seq)`, so a later run must
/// always get a larger id, even when the wall clock steps backwards.
fn next_run_id(now_ms: i64, previous: Option<i64>) -> i64 {
    match previous {
        Some(prev) => now_ms.max(prev.saturating_add(1)),
        None => now_ms,
    }
}

fn allocation_json(svc: &ServiceConfig, alloc: &Allocation) -> String {
    // Match the existing starting.rs serialisation: the init allocation bytes
    // keyed by display name. The caller normally passes the init allocation;
    // here we re-derive from the alloc argument for self-containment.
    let _ = svc;
    serde_json::to_string(
        &alloc
            .bytes
            .iter()
            .map(|(k, v)| (k.as_display(), *v))
            .collect::<std::collections::BTreeMap<_, _>>(),
    )
    .unwrap_or_default()
}

#[expect(
    clippy::too_many_arguments,
    reason = "one launch sequence, not a struct's worth of state"
)]
async fn launch_process(
    svc: &ServiceConfig,
    alloc: &Allocation,
    cmd_args: Option<&CommandArgs>,
    service_id: i64,
    run_id: i64,
    allocation_json: String,
    system: &SystemDeps,
    db: &Database,
) -> Result<LaunchedWorkload, LaunchError> {
    let spawn_cfg =
        render_argv(svc, alloc, cmd_args).map_err(|e| LaunchError::Render(e.to_string()))?;
    let cmdline = format!("{} {}", spawn_cfg.binary, spawn_cfg.args.join(" "));
    info!(service = %svc.name, binary = %spawn_cfg.binary, "spawning child");

    let child = system
        .process_spawner
        .spawn(&spawn_cfg)
        .await
        .map_err(|e| LaunchError::Spawn(e.to_string()))?;
    let pid = child.id().unwrap_or(0) as i32;

    let row = RunningService {
        service_id,
        run_id,
        pid: Some(pid as i64),
        spawned_at: ananke_tracking::now_unix_ms(),
        command_line: cmdline.clone(),
        allocation: allocation_json,
        state: "starting".to_string(),
        workload_kind: Some("process".to_string()),
        runtime: None,
        container_name: None,
        container_id: None,
        runtime_executable: None,
    };
    // Native-process row insert is best-effort, matching the existing path,
    // but we still surface persistence failures for parity with containers.
    if let Err(e) = db.insert_running(&row).await {
        warn!(error = %e, "running_services insert failed (native)");
    }

    Ok(LaunchedWorkload {
        workload: ManagedWorkload::Process(child),
        run_id,
        host_pid: Some(pid),
        workload_kind: "process",
        command_line: cmdline,
        container_id: None,
        container_name: None,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "one launch sequence, not a struct's worth of state"
)]
async fn launch_container(
    svc: &ServiceConfig,
    alloc: &Allocation,
    cmd_args: Option<&CommandArgs>,
    service_id: i64,
    run_id: i64,
    allocation_json: String,
    system: &SystemDeps,
    db: &Database,
) -> Result<LaunchedWorkload, LaunchError> {
    // 1. Durable owner + spec. The owner UUID already exists (created once
    //    during reconciliation/bootstrap), but ensure it here so every launch
    //    is self-contained if the bootstrap path was skipped.
    let owner = db
        .ensure_owner_uuid()
        .await
        .map_err(|e| LaunchError::Persist(e.to_string()))?;
    let spec = render_container_spec(svc, alloc, cmd_args, run_id, &owner)
        .map_err(|e| LaunchError::Render(e.to_string()))?;

    // 2. Durable launch intent *before* create.
    let labels_json = serde_json::to_string(&spec.labels).unwrap_or_default();
    let spec_json = serde_json::to_string(&spec).unwrap_or_default();
    let intent = db
        .insert_launch_intent(&ananke_db::models::ContainerLaunchIntent {
            intent_id: 0,
            service_id,
            run_id,
            owner_uuid: owner.clone(),
            workload_kind: "container".to_string(),
            runtime: spec.runtime.as_str().to_string(),
            runtime_executable: spec
                .runtime_executable
                .clone()
                .unwrap_or_else(|| spec.runtime.executable().to_string()),
            container_name: spec.name.clone(),
            labels_json,
            spec_json,
            container_id: None,
            state: "intent".to_string(),
            created_at: ananke_tracking::now_unix_ms(),
        })
        .await
        .map_err(|e| LaunchError::Persist(e.to_string()))?;

    // 3. Create the container.
    let prepared = match system.container_engine.create(&spec).await {
        Ok(p) => p,
        Err(e) => {
            // Nothing was created, so the intent describes an object that
            // does not exist. Dropping it here keeps a service that fails
            // to start in a loop from accruing one row per attempt, each
            // carrying a full spec blob for reconciliation to walk.
            if let Err(e) = db.delete_launch_intent(intent).await {
                warn!(error = %e, "clearing the intent after a failed create");
            }
            return Err(LaunchError::Create(e.to_string()));
        }
    };

    // 4. Attach the container ID to the intent (crash-safe reconciliation
    //    uses name + owner-label verification if this update is lost).
    if let Err(e) = db.attach_container_id(intent, &prepared.id).await {
        warn!(error = %e, "attach container id to intent failed; reconciliation will fall back to name+owner");
    }

    // 5. Persist the running row *before* start. Mandatory: if this fails,
    //    remove the prepared container and do not start it.
    let command_line = spec.command.join(" ");
    let row = RunningService {
        service_id,
        run_id,
        pid: None,
        spawned_at: ananke_tracking::now_unix_ms(),
        command_line: command_line.clone(),
        allocation: allocation_json,
        state: "starting".to_string(),
        workload_kind: Some("container".to_string()),
        runtime: Some(spec.runtime.as_str().to_string()),
        container_name: Some(prepared.name.clone()),
        container_id: Some(prepared.id.clone()),
        runtime_executable: Some(prepared.runtime_executable.clone()),
    };
    if let Err(e) = db.insert_running(&row).await {
        // Compensating removal: remove the prepared container, then surface.
        let remove_err = prepared.remove().await;
        if let Err(re) = remove_err {
            warn!(error = %re, "compensating container removal failed; leaving intent as reconciliation block");
            let _ = db.mark_intent_blocked(intent).await;
        }
        return Err(LaunchError::Persist(e.to_string()));
    }

    // 6. Start the container (ownership row is now committed).
    let running = match prepared.start().await {
        Ok(r) => r,
        Err(e) => {
            // The container exists but never ran. Remove it, and only clear
            // the records once that is confirmed — a failed removal is
            // exactly the case reconciliation needs the evidence for.
            if prepared.remove().await.is_ok() {
                let _ = db.delete_running(service_id, run_id).await;
                if let Err(e) = db.delete_launch_intent(intent).await {
                    warn!(error = %e, "clearing the intent after a failed start");
                }
            } else {
                warn!(container = %prepared.id, "start failed and removal failed; left for reconciliation");
                let _ = db.mark_intent_blocked(intent).await;
            }
            return Err(LaunchError::Start(e.to_string()));
        }
    };

    // The running row is now the authority for this container's identity,
    // so the intent has done its job. Leaving it would accumulate a row per
    // start and make every one look like an unresolved crash to the
    // reconciler.
    if let Err(e) = db.delete_launch_intent(intent).await {
        warn!(error = %e, "deleting the launch intent after start failed; reconciliation will drop it");
    }

    let host_pid = running.host_pid().map(|p| p as i32);

    info!(
        service = %svc.name,
        container_id = %prepared.id,
        container_name = %prepared.name,
        "started container"
    );

    Ok(LaunchedWorkload {
        workload: ManagedWorkload::Container(running),
        run_id,
        host_pid,
        workload_kind: "container",
        command_line,
        container_id: Some(prepared.id),
        container_name: Some(prepared.name),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_run_takes_the_clock() {
        assert_eq!(next_run_id(1_761_000_000_000, None), 1_761_000_000_000);
    }

    #[test]
    fn run_ids_increase_when_the_clock_goes_backwards() {
        let first = next_run_id(1_761_000_000_000, None);
        let second = next_run_id(1_760_000_000_000, Some(first));
        let third = next_run_id(1_760_000_000_000, Some(second));
        assert!(first < second && second < third);
    }

    #[test]
    fn ids_from_the_old_masked_scheme_stay_below_new_ones() {
        // Rows written before run ids stopped being masked to 31 bits.
        let old = 0x7FFF_FFFE;
        assert_eq!(next_run_id(1_761_000_000_000, Some(old)), 1_761_000_000_000);
    }
}
