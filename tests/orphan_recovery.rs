//! Integration test: orphan recovery cleans stale rows for non-existent PIDs.

mod common;

use tempfile::tempdir;

use ananke::db::Database;
use ananke::supervise::{OrphanDisposition, reconcile};

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
    db.with_conn(|c| {
        c.execute(
            "INSERT INTO running_services(service_id, run_id, pid, spawned_at, command_line, allocation, state) VALUES (?1, 1, 99999, 0, 'fake-server', '{}', 'running')",
            [service_id],
        )
    })
    .expect("insert row");

    // Use a tempdir as the procfs root — /proc/99999 does not exist there.
    let fake_proc = tmp.path().join("proc");
    std::fs::create_dir_all(&fake_proc).expect("create fake proc");

    let dispositions = reconcile(&db, &fake_proc);

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
    let count: i64 = db
        .with_conn(|c| c.query_row("SELECT COUNT(*) FROM running_services", [], |r| r.get(0)))
        .expect("count rows");
    assert_eq!(count, 0, "stale row should have been deleted");
}
