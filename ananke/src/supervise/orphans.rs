//! Linux-only: startup orphan recovery per spec §9.3. Reads
//! `/proc/{pid}/cmdline` (through [`crate::system::ProcFs`] — tests can
//! substitute [`crate::system::InMemoryProcFs`] with preloaded cmdlines)
//! to decide whether a previously-recorded child is still alive and
//! still ours.

use tracing::{info, warn};

use crate::{
    db::{Database, models::RunningService},
    system::ProcFs,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrphanDisposition {
    Adopted {
        pid: i32,
        service_id: i64,
        run_id: i64,
    },
    Cleaned {
        pid: i32,
        service_id: i64,
        run_id: i64,
    },
    Ignored {
        pid: i32,
        reason: String,
    },
}

/// Runs orphan recovery against the `running_services` table. Returns a
/// decision list suitable for logging and test assertion. The `proc`
/// argument is the [`ProcFs`] reader; tests pass an
/// [`crate::system::InMemoryProcFs`] with preloaded cmdlines rather
/// than staging a synthetic `/proc` on disk.
pub async fn reconcile(proc: &dyn ProcFs, db: &Database) -> Vec<OrphanDisposition> {
    let mut handle = db.handle();
    let rows: Vec<RunningService> = RunningService::all()
        .exec(&mut handle)
        .await
        .unwrap_or_default();

    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let service_id = row.service_id;
        let run_id = row.run_id;
        let pid = row.pid as i32;
        let recorded_cmdline = row.command_line.clone();
        match proc.cmdline(pid) {
            Some(live_cmdline) => {
                if live_cmdline == recorded_cmdline {
                    info!(pid, service_id, run_id, "adopted orphan");
                    out.push(OrphanDisposition::Adopted {
                        pid,
                        service_id,
                        run_id,
                    });
                } else {
                    warn!(
                        pid,
                        service_id,
                        run_id,
                        recorded = %recorded_cmdline,
                        live = %live_cmdline,
                        "unrelated process at recorded pid; cleaning row"
                    );
                    cleanup_row(&mut handle, row).await;
                    out.push(OrphanDisposition::Cleaned {
                        pid,
                        service_id,
                        run_id,
                    });
                }
            }
            None => {
                info!(pid, service_id, run_id, "dead child; cleaning row");
                cleanup_row(&mut handle, row).await;
                out.push(OrphanDisposition::Cleaned {
                    pid,
                    service_id,
                    run_id,
                });
            }
        }
    }
    out
}

async fn cleanup_row(db: &mut toasty::Db, row: RunningService) {
    if let Err(e) = row.delete().exec(db).await {
        warn!(error = %e, "delete running_services row failed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::InMemoryProcFs;

    async fn insert_row(db: &Database, service_id: i64, run_id: i64, pid: i32, cmdline: &str) {
        let mut handle = db.handle();
        toasty::create!(RunningService {
            service_id,
            run_id,
            pid: pid as i64,
            spawned_at: 0,
            command_line: cmdline.to_string(),
            allocation: "{}".to_string(),
            state: "running".to_string(),
        })
        .exec(&mut handle)
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn adopts_matching_cmdline() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let proc = InMemoryProcFs::new();
        insert_row(&db, svc, 1, 1234, "llama-server -m x").await;
        proc.set_cmdline(1234, "llama-server -m x");
        let out = reconcile(&proc, &db).await;
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0], OrphanDisposition::Adopted { .. }));
    }

    #[tokio::test]
    async fn cleans_missing_pid() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let proc = InMemoryProcFs::new();
        insert_row(&db, svc, 1, 9999, "llama-server -m x").await;
        let out = reconcile(&proc, &db).await;
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0], OrphanDisposition::Cleaned { .. }));
        let mut handle = db.handle();
        let rows: Vec<RunningService> = RunningService::all().exec(&mut handle).await.unwrap();
        assert!(rows.is_empty());
    }

    #[tokio::test]
    async fn cleans_mismatched_cmdline() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let proc = InMemoryProcFs::new();
        insert_row(&db, svc, 1, 4242, "llama-server -m x").await;
        proc.set_cmdline(4242, "firefox");
        let out = reconcile(&proc, &db).await;
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0], OrphanDisposition::Cleaned { .. }));
    }
}
