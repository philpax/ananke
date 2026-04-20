//! Integration test: orphan recovery cleans stale rows for non-existent PIDs.

mod common;

use ananke::{
    db::{Database, models::RunningService},
    supervise::{OrphanDisposition, reconcile},
    system::InMemoryProcFs,
};

#[tokio::test]
async fn cleans_row_for_dead_pid() {
    let db = Database::open_in_memory().await.expect("open db");

    // Register a service so the foreign key is satisfied.
    let service_id = db
        .upsert_service("test-svc", 0)
        .await
        .expect("upsert_service");

    // Insert a running_services row pointing at PID 99999. The empty
    // InMemoryProcFs reports that pid as exited, so reconcile must
    // clean the row.
    let mut handle = db.handle();
    toasty::create!(RunningService {
        service_id,
        run_id: 1,
        pid: 99999,
        spawned_at: 0,
        command_line: "fake-server".to_string(),
        allocation: "{}".to_string(),
        state: "running".to_string(),
    })
    .exec(&mut handle)
    .await
    .expect("insert row");

    let proc = InMemoryProcFs::new();
    let dispositions = reconcile(&proc, &db).await;

    assert_eq!(dispositions.len(), 1);
    assert!(
        matches!(
            dispositions[0],
            OrphanDisposition::Cleaned { pid: 99999, .. }
        ),
        "expected Cleaned, got {:?}",
        dispositions[0]
    );

    let remaining: Vec<RunningService> = RunningService::all()
        .exec(&mut handle)
        .await
        .expect("query running_services");
    assert!(remaining.is_empty(), "stale row should have been deleted");
}
