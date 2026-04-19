//! Startup orphan recovery per spec §9.3.

use std::path::Path;

use tracing::{info, warn};

use crate::db::{Database, models::RunningService};

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
/// decision list suitable for logging and test assertion.
///
/// `procfs_root` defaults to "/proc"; tests override it via a temp directory.
pub async fn reconcile(db: &Database, procfs_root: &Path) -> Vec<OrphanDisposition> {
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
        let proc_dir = procfs_root.join(pid.to_string());
        let cmdline_path = proc_dir.join("cmdline");
        match std::fs::read(&cmdline_path) {
            Ok(raw) => {
                let live_cmdline = null_sep_to_space(&raw);
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
            Err(_) => {
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

fn null_sep_to_space(bytes: &[u8]) -> String {
    let trimmed: Vec<u8> = bytes
        .iter()
        .copied()
        .map(|b| if b == 0 { b' ' } else { b })
        .collect();
    String::from_utf8_lossy(&trimmed).trim().to_string()
}

async fn cleanup_row(db: &mut toasty::Db, row: RunningService) {
    if let Err(e) = row.delete().exec(db).await {
        warn!(error = %e, "delete running_services row failed");
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

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

    fn write_cmdline(procfs: &Path, pid: i32, cmdline: &str) {
        let dir = procfs.join(pid.to_string());
        std::fs::create_dir_all(&dir).unwrap();
        // /proc/PID/cmdline separates args with NUL.
        let nul_sep = cmdline.replace(' ', "\0");
        std::fs::write(dir.join("cmdline"), nul_sep).unwrap();
    }

    #[tokio::test]
    async fn adopts_matching_cmdline() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("a.sqlite")).await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let procfs = tmp.path().join("proc");
        insert_row(&db, svc, 1, 1234, "llama-server -m x").await;
        write_cmdline(&procfs, 1234, "llama-server -m x");
        let out = reconcile(&db, &procfs).await;
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0], OrphanDisposition::Adopted { .. }));
    }

    #[tokio::test]
    async fn cleans_missing_pid() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("a.sqlite")).await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let procfs = tmp.path().join("proc");
        std::fs::create_dir_all(&procfs).unwrap();
        insert_row(&db, svc, 1, 9999, "llama-server -m x").await;
        let out = reconcile(&db, &procfs).await;
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0], OrphanDisposition::Cleaned { .. }));
        // Row should be gone.
        let mut handle = db.handle();
        let rows: Vec<RunningService> = RunningService::all().exec(&mut handle).await.unwrap();
        assert!(rows.is_empty());
    }

    #[tokio::test]
    async fn cleans_mismatched_cmdline() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("a.sqlite")).await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let procfs = tmp.path().join("proc");
        insert_row(&db, svc, 1, 4242, "llama-server -m x").await;
        write_cmdline(&procfs, 4242, "firefox");
        let out = reconcile(&db, &procfs).await;
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0], OrphanDisposition::Cleaned { .. }));
    }
}
