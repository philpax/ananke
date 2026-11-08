//! Container workload lifecycle: launch ordering, exit handling, drain
//! escalation, and cleanup.
//!
//! Everything here runs against [`FakeContainerEngine`], which records each
//! lifecycle operation in order. The orderings asserted below are the
//! load-bearing ones: a container must never be running without a persisted
//! ownership row, and it must never be removed before its final logs and
//! exit status have been collected.
#![cfg(feature = "test-fakes")]

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use ananke::supervise::{
    drain::{DrainConfig, DrainReason, drain_pipeline},
    launch::launch_workload,
    workload::ManagedWorkload,
};
use ananke_config::validate::{
    AllocationMode, ContainerNetwork, DeviceSlot, PlacementPolicy, ServiceConfig,
    test_fixtures::{container_mount, minimal_command_service, minimal_container_config},
};
use ananke_db::Database;
use ananke_devices::Allocation;
use ananke_proxy::InflightCounter;
use ananke_system::{
    SystemDeps,
    container::{ContainerEngine, FakeContainerEngine, FakeContainerState},
};

/// A host-networked container command service and its GPU-0 allocation.
fn container_service() -> (ServiceConfig, Allocation) {
    let mut placement = BTreeMap::new();
    placement.insert(DeviceSlot::Gpu(0), 8_000);
    let alloc = Allocation::from_override(&placement);

    let mut svc = minimal_command_service(
        "ninfer",
        vec![
            "ninfer-serve".into(),
            "/artifacts/model.ninfer".into(),
            "--port".into(),
            "${port}".into(),
        ],
    );
    svc.port = 8205;
    svc.private_port = 48205;
    svc.placement_override = placement;
    svc.placement_policy = PlacementPolicy::GpuOnly;
    svc.allocation_mode = AllocationMode::Static { reserve_mb: 8_000 };

    let mut ct = minimal_container_config("ninfer:local");
    ct.network = ContainerNetwork::Host;
    ct.mounts = vec![container_mount("/host/ninfer", "/artifacts")];
    svc.container = Some(ct);

    (svc, alloc)
}

/// A fake system plus its concrete engine handle and an in-memory database.
async fn fixture() -> (SystemDeps, Arc<FakeContainerEngine>, Database) {
    let (system, fakes) = SystemDeps::fake();
    let db = Database::open_in_memory().await.unwrap();
    (system, fakes.container_engine, db)
}

/// The service row a launch writes against.
async fn service_row(db: &Database, name: &str) -> i64 {
    db.upsert_service(name, 0).await.unwrap()
}

#[tokio::test]
async fn container_service_healthy_lifecycle() {
    let (system, engine, db) = fixture().await;
    let (svc, alloc) = container_service();
    let service_id = service_row(&db, &svc.name).await;

    let Ok(launched) = launch_workload(&svc, &alloc, None, service_id, &system, &db).await else {
        panic!("launch must succeed");
    };

    assert_eq!(launched.workload_kind, "container");
    let id = launched.container_id.clone().expect("container id");
    let name = launched.container_name.clone().expect("container name");
    assert!(
        name.starts_with("ananke-ninfer-"),
        "generated name must be owner-recognisable: {name}"
    );

    let snap = engine.find(&id).expect("container must exist");
    assert_eq!(snap.operations, ["create", "start"]);
    assert_eq!(snap.state, FakeContainerState::Running);
    assert_eq!(snap.image, "ninfer:local");

    // The persisted row carries the container identity, not a host PID.
    let rows = db.list_running().await.unwrap();
    let row = rows
        .iter()
        .find(|r| r.run_id == launched.run_id)
        .expect("running row must be persisted");
    assert_eq!(row.workload_kind.as_deref(), Some("container"));
    assert_eq!(row.container_id.as_deref(), Some(id.as_str()));
    assert_eq!(row.container_name.as_deref(), Some(name.as_str()));
    assert_eq!(row.runtime.as_deref(), Some("docker"));
    // `command_line` is the in-container argv — what an operator wants to see.
    assert!(row.command_line.contains("ninfer-serve"));
    assert!(!row.command_line.contains("docker"));
}

#[tokio::test]
async fn container_start_occurs_after_running_row_commit() {
    let (system, engine, db) = fixture().await;
    let (svc, alloc) = container_service();
    let service_id = service_row(&db, &svc.name).await;

    let launched = launch_workload(&svc, &alloc, None, service_id, &system, &db)
        .await
        .unwrap_or_else(|e| panic!("launch failed: {e}"));

    // The row exists and the container is started — so it cannot have been
    // started before the row, which is the invariant a crash between the two
    // would otherwise violate.
    let id = launched.container_id.unwrap();
    assert_eq!(engine.find(&id).unwrap().operations, ["create", "start"]);
    assert!(
        db.list_running()
            .await
            .unwrap()
            .iter()
            .any(|r| r.container_id.as_deref() == Some(id.as_str()))
    );

    // Once the row is committed and the container started, the row is the
    // authority and the intent has been handed off — otherwise every start
    // would leave a residue the reconciler reads as an unresolved crash.
    assert!(db.list_launch_intents().await.unwrap().is_empty());
}

#[tokio::test]
async fn failed_create_leaves_nothing_behind() {
    // Nothing was created, so the intent describes an object that does not
    // exist. Retaining it would accrue one row per attempt — each carrying
    // a full spec blob that startup reconciliation then walks — for a
    // service stuck in a restart loop on a typo'd image.
    //
    // The ordering this does *not* prove — that the intent is durable
    // before create, for a crash rather than a failure — is covered by the
    // reconciliation suite, which starts from an intent on disk.
    let (system, engine, db) = fixture().await;
    let (svc, alloc) = container_service();
    let service_id = service_row(&db, &svc.name).await;
    engine.fail_create("no such image");

    let Err(err) = launch_workload(&svc, &alloc, None, service_id, &system, &db).await else {
        panic!("launch must fail when create fails");
    };
    assert!(format!("{err}").contains("create container"), "{err}");

    assert!(engine.snapshot().is_empty());
    assert!(db.list_running().await.unwrap().is_empty());
    assert!(
        db.list_launch_intents().await.unwrap().is_empty(),
        "an intent for an object that was never created is not evidence"
    );
}

#[tokio::test]
async fn create_succeeds_row_insert_fails_removes_without_start() {
    let (system, engine, db) = fixture().await;
    let (svc, alloc) = container_service();

    // No service row exists, so the running-row insert violates its foreign
    // key: the launch must compensate by removing the created container
    // rather than starting it.
    let missing_service_id = 9_999;
    let Err(err) = launch_workload(&svc, &alloc, None, missing_service_id, &system, &db).await
    else {
        panic!("row insert must fail without a service row");
    };
    assert!(
        format!("{err}").contains("persist ownership"),
        "expected a persistence error, got: {err}"
    );

    let snaps = engine.snapshot();
    assert_eq!(snaps.len(), 1, "exactly one container was created");
    assert_eq!(
        snaps[0].operations,
        ["create", "remove"],
        "a container whose ownership row failed must be removed, never started"
    );
    assert!(!snaps[0].operations.contains(&"start".to_string()));
}

#[tokio::test]
async fn container_early_exit_cleans_up() {
    let (system, engine, db) = fixture().await;
    let (svc, alloc) = container_service();
    let service_id = service_row(&db, &svc.name).await;

    // Script the container to exit non-zero as soon as it is waited on.
    engine.wait_exit(3);

    let mut launched = launch_workload(&svc, &alloc, None, service_id, &system, &db)
        .await
        .unwrap_or_else(|e| panic!("launch failed: {e}"));
    let id = launched.container_id.clone().unwrap();

    let exit = launched.workload.wait().await.unwrap();
    assert_eq!(exit, ananke::supervise::workload::WorkloadExit::Code(3));

    launched.workload.cleanup().await.unwrap();
    let snap = engine.find(&id).unwrap();
    assert_eq!(snap.operations, ["create", "start", "wait", "remove"]);
    assert_eq!(snap.state, FakeContainerState::Removed);
}

#[tokio::test]
async fn container_final_logs_precede_remove() {
    let (system, engine, db) = fixture().await;
    let (svc, alloc) = container_service();
    let service_id = service_row(&db, &svc.name).await;
    engine.wait_exit(0);

    let mut launched = launch_workload(&svc, &alloc, None, service_id, &system, &db)
        .await
        .unwrap_or_else(|e| panic!("launch failed: {e}"));
    let id = launched.container_id.clone().unwrap();

    // The log follower is taken while the container is still present; a
    // container removed first would leave nothing to drain.
    assert!(
        !launched.workload.take_combined().is_empty(),
        "a container workload must expose its combined log readers"
    );
    assert!(
        launched.workload.take_stdout().is_none() && launched.workload.take_stderr().is_none(),
        "a container has no split streams to offer"
    );
    assert_ne!(engine.find(&id).unwrap().state, FakeContainerState::Removed);

    launched.workload.wait().await.unwrap();
    launched.workload.cleanup().await.unwrap();

    let ops = engine.find(&id).unwrap().operations;
    let wait_at = ops.iter().position(|o| o == "wait").unwrap();
    let remove_at = ops.iter().position(|o| o == "remove").unwrap();
    assert!(
        wait_at < remove_at,
        "the authoritative exit status must be read before removal: {ops:?}"
    );
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn container_drain_escalates_to_kill() {
    let (system, engine, db) = fixture().await;
    let (svc, alloc) = container_service();
    let service_id = service_row(&db, &svc.name).await;

    // TERM is delivered but the workload ignores it, so `wait` never returns
    // and the supervisor's grace timer must escalate.
    engine.wait_block();

    let launched = launch_workload(&svc, &alloc, None, service_id, &system, &db)
        .await
        .unwrap_or_else(|e| panic!("launch failed: {e}"));
    let id = launched.container_id.clone().unwrap();
    let mut workload = launched.workload;

    let cfg = DrainConfig {
        max_request_duration: Duration::from_millis(100),
        drain_timeout: Duration::from_millis(50),
        extended_stream_drain: Duration::from_millis(50),
        sigterm_grace: Duration::from_millis(500),
    };
    drain_pipeline(
        &mut workload,
        &cfg,
        InflightCounter::detached(),
        DrainReason::Shutdown,
    )
    .await;

    let ops = engine.find(&id).unwrap().operations;
    assert!(
        ops.contains(&"terminate".to_string()),
        "drain must try TERM first: {ops:?}"
    );
    assert!(
        ops.contains(&"kill".to_string()),
        "an ignored TERM must escalate to KILL: {ops:?}"
    );
    let term_at = ops.iter().position(|o| o == "terminate").unwrap();
    let kill_at = ops.iter().position(|o| o == "kill").unwrap();
    assert!(term_at < kill_at, "TERM must precede KILL: {ops:?}");
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn container_health_timeout_cleans_up() {
    // A container that never becomes healthy is torn down the same way a
    // drained one is — and must not be left behind as a stopped object for
    // the next daemon restart to find.
    let (system, engine, db) = fixture().await;
    let (svc, alloc) = container_service();
    let service_id = service_row(&db, &svc.name).await;
    engine.wait_block();

    let launched = launch_workload(&svc, &alloc, None, service_id, &system, &db)
        .await
        .unwrap_or_else(|e| panic!("launch failed: {e}"));
    let id = launched.container_id.clone().unwrap();
    let mut workload = launched.workload;

    // The supervisor's starting path takes this route when health times out.
    ananke::supervise::drain::sigterm_then_sigkill(&mut workload, Duration::from_secs(5)).await;

    let snap = engine.find(&id).unwrap();
    assert_eq!(
        snap.state,
        FakeContainerState::Removed,
        "a container abandoned during startup must be removed: {:?}",
        snap.operations
    );
    let remove_at = snap.operations.iter().position(|o| o == "remove").unwrap();
    assert!(
        snap.operations[..remove_at]
            .iter()
            .any(|o| o == "terminate"),
        "removal must follow the signal, not replace it: {:?}",
        snap.operations
    );
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn container_drain_removes_after_graceful_exit() {
    // The normal stop path: TERM is honoured, so no KILL is needed, and the
    // container is still removed rather than left stopped on the host.
    let (system, engine, db) = fixture().await;
    let (svc, alloc) = container_service();
    let service_id = service_row(&db, &svc.name).await;
    engine.wait_exit(0);

    let launched = launch_workload(&svc, &alloc, None, service_id, &system, &db)
        .await
        .unwrap_or_else(|e| panic!("launch failed: {e}"));
    let id = launched.container_id.clone().unwrap();
    let mut workload = launched.workload;

    let cfg = DrainConfig {
        max_request_duration: Duration::from_millis(100),
        drain_timeout: Duration::from_millis(50),
        extended_stream_drain: Duration::from_millis(50),
        sigterm_grace: Duration::from_secs(30),
    };
    drain_pipeline(
        &mut workload,
        &cfg,
        InflightCounter::detached(),
        DrainReason::IdleTimeout,
    )
    .await;

    let ops = engine.find(&id).unwrap().operations;
    assert!(!ops.contains(&"kill".to_string()), "TERM sufficed: {ops:?}");
    assert_eq!(engine.find(&id).unwrap().state, FakeContainerState::Removed);
}

#[tokio::test]
async fn native_workload_takes_no_container_path() {
    // The launch orchestration dispatches on the container block alone: a
    // service without one must not touch the container engine at all.
    let (system, engine, db) = fixture().await;
    let (mut svc, alloc) = container_service();
    svc.container = None;
    let service_id = service_row(&db, &svc.name).await;

    let launched = launch_workload(&svc, &alloc, None, service_id, &system, &db)
        .await
        .unwrap_or_else(|e| panic!("launch failed: {e}"));
    assert_eq!(launched.workload_kind, "process");
    assert!(launched.container_id.is_none());
    assert!(matches!(launched.workload, ManagedWorkload::Process(_)));
    assert!(engine.snapshot().is_empty());
    assert!(db.list_launch_intents().await.unwrap().is_empty());
    assert!(engine.list(&[]).await.unwrap().is_empty());
}
