//! Integration test: orphan recovery cleans stale rows for non-existent PIDs.

mod common;

use ananke::{
    db::{Database, models::RunningService},
    supervise::{OrphanDisposition, reconcile},
};
use tempfile::tempdir;

#[tokio::test]
async fn cleans_row_for_dead_pid() {
    let tmp = tempdir().expect("tempdir");
    let db = Database::open(&tmp.path().join("ananke.sqlite"))
        .await
        .expect("open db");

    // Register a service so the foreign key is satisfied.
    let service_id = db
        .upsert_service("test-svc", 0)
        .await
        .expect("upsert_service");

    // Insert a running_services row pointing at PID 99999, which is extremely
    // unlikely to exist and, even if it did, /proc/99999 won't exist under our
    // fake procfs root.
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

    // Use a tempdir as the procfs root — /proc/99999 does not exist there.
    let fake_proc = tmp.path().join("proc");
    std::fs::create_dir_all(&fake_proc).expect("create fake proc");

    let dispositions = reconcile(&db, &fake_proc).await;

    assert_eq!(dispositions.len(), 1);
    assert!(
        matches!(
            dispositions[0],
            OrphanDisposition::Cleaned { pid: 99999, .. }
        ),
        "expected Cleaned, got {:?}",
        dispositions[0]
    );

    // Confirm the row was actually removed.
    let remaining: Vec<RunningService> = RunningService::all()
        .exec(&mut handle)
        .await
        .expect("query running_services");
    assert!(remaining.is_empty(), "stale row should have been deleted");
}
