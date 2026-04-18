//! Startup orphan recovery per spec §9.3.

use std::path::Path;

use tracing::{info, warn};

use crate::db::Database;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrphanDisposition {
    Adopted { pid: i32, service_id: i64, run_id: i64 },
    Cleaned { pid: i32, service_id: i64, run_id: i64 },
    Ignored { pid: i32, reason: String },
}

/// Runs orphan recovery against the `running_services` table. Returns a
/// decision list suitable for logging and test assertion.
///
/// `procfs_root` defaults to "/proc"; tests override it via a temp directory.
pub fn reconcile(db: &Database, procfs_root: &Path) -> Vec<OrphanDisposition> {
    let rows: Vec<(i64, i64, i64, String)> = db
        .with_conn(|c| {
            let mut stmt = c
                .prepare(
                    "SELECT service_id, run_id, pid, command_line FROM running_services",
                )
                .unwrap();
            let rows = stmt
                .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
                .unwrap();
            Ok(rows.collect::<Result<Vec<_>, _>>().unwrap())
        })
        .unwrap_or_default();

    let mut out = Vec::with_capacity(rows.len());
    for (service_id, run_id, pid_i64, recorded_cmdline) in rows {
        let pid = pid_i64 as i32;
        let proc_dir = procfs_root.join(pid.to_string());
        let cmdline_path = proc_dir.join("cmdline");
        match std::fs::read(&cmdline_path) {
            Ok(raw) => {
                let live_cmdline = null_sep_to_space(&raw);
                if live_cmdline == recorded_cmdline {
                    info!(pid, service_id, run_id, "adopted orphan");
                    out.push(OrphanDisposition::Adopted { pid, service_id, run_id });
                } else {
                    warn!(
                        pid,
                        service_id,
                        run_id,
                        recorded = %recorded_cmdline,
                        live = %live_cmdline,
                        "unrelated process at recorded pid; cleaning row"
                    );
                    cleanup_row(db, service_id, run_id);
                    out.push(OrphanDisposition::Cleaned { pid, service_id, run_id });
                }
            }
            Err(_) => {
                info!(pid, service_id, run_id, "dead child; cleaning row");
                cleanup_row(db, service_id, run_id);
                out.push(OrphanDisposition::Cleaned { pid, service_id, run_id });
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

fn cleanup_row(db: &Database, service_id: i64, run_id: i64) {
    let _ = db.with_conn(|c| {
        c.execute(
            "DELETE FROM running_services WHERE service_id = ?1 AND run_id = ?2",
            (service_id, run_id),
        )
    });
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    fn insert_row(db: &Database, service_id: i64, run_id: i64, pid: i32, cmdline: &str) {
        db.with_conn(|c| {
            c.execute(
                "INSERT INTO running_services(service_id, run_id, pid, spawned_at, command_line, allocation, state) VALUES (?1, ?2, ?3, 0, ?4, '{}', 'running')",
                (service_id, run_id, pid, cmdline),
            )
        })
        .unwrap();
    }

    fn write_cmdline(procfs: &Path, pid: i32, cmdline: &str) {
        let dir = procfs.join(pid.to_string());
        std::fs::create_dir_all(&dir).unwrap();
        // /proc/PID/cmdline separates args with NUL.
        let nul_sep = cmdline.replace(' ', "\0");
        std::fs::write(dir.join("cmdline"), nul_sep).unwrap();
    }

    #[test]
    fn adopts_matching_cmdline() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("a.sqlite")).unwrap();
        let svc = db.upsert_service("demo", 0).unwrap();
        let procfs = tmp.path().join("proc");
        insert_row(&db, svc, 1, 1234, "llama-server -m x");
        write_cmdline(&procfs, 1234, "llama-server -m x");
        let out = reconcile(&db, &procfs);
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0], OrphanDisposition::Adopted { .. }));
    }

    #[test]
    fn cleans_missing_pid() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("a.sqlite")).unwrap();
        let svc = db.upsert_service("demo", 0).unwrap();
        let procfs = tmp.path().join("proc");
        std::fs::create_dir_all(&procfs).unwrap();
        insert_row(&db, svc, 1, 9999, "llama-server -m x");
        let out = reconcile(&db, &procfs);
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0], OrphanDisposition::Cleaned { .. }));
        // Row should be gone.
        let rows: Vec<(i64, i64)> = db
            .with_conn(|c| {
                let mut s = c
                    .prepare("SELECT service_id, run_id FROM running_services")
                    .unwrap();
                Ok(s.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))
                    .unwrap()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap())
            })
            .unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn cleans_mismatched_cmdline() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("a.sqlite")).unwrap();
        let svc = db.upsert_service("demo", 0).unwrap();
        let procfs = tmp.path().join("proc");
        insert_row(&db, svc, 1, 4242, "llama-server -m x");
        write_cmdline(&procfs, 4242, "firefox");
        let out = reconcile(&db, &procfs);
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0], OrphanDisposition::Cleaned { .. }));
    }
}
