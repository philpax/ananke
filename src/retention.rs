//! Log retention and SQLite incremental vacuum.

use std::time::Duration;

use tokio::sync::watch;
use tracing::{info, warn};

use crate::db::Database;

/// Per-service log retention: 7 days or 50,000 lines, whichever tighter
/// (spec §12). Runs once when called; call from a daily scheduled task.
pub fn trim_logs_once(db: &Database, now_ms: i64) -> rusqlite::Result<u64> {
    let seven_days_ago = now_ms - 7 * 24 * 60 * 60 * 1000;
    let per_service = 50_000i64;

    let mut deleted = 0u64;
    db.with_conn_mut(|conn| {
        let tx = conn.transaction()?;
        // 1. Drop rows older than seven days.
        let n = tx.execute(
            "DELETE FROM service_logs WHERE timestamp_ms < ?1",
            [seven_days_ago],
        )?;
        deleted += n as u64;

        // 2. Per-service row count cap.
        let service_ids: Vec<i64> = {
            let mut stmt =
                tx.prepare("SELECT service_id FROM services WHERE deleted_at IS NULL")?;
            stmt.query_map([], |row| row.get(0))?
                .collect::<Result<_, _>>()?
        };
        for sid in service_ids {
            let total: i64 = tx.query_row(
                "SELECT COUNT(*) FROM service_logs WHERE service_id = ?1",
                [sid],
                |r| r.get(0),
            )?;
            if total > per_service {
                let excess = total - per_service;
                let n = tx.execute(
                    "DELETE FROM service_logs WHERE rowid IN (SELECT rowid FROM service_logs WHERE service_id = ?1 ORDER BY timestamp_ms ASC LIMIT ?2)",
                    (sid, excess),
                )?;
                deleted += n as u64;
            }
        }

        tx.commit()
    })?;
    Ok(deleted)
}

pub fn incremental_vacuum(db: &Database, pages: u64) -> rusqlite::Result<()> {
    db.with_conn(|c| c.execute_batch(&format!("PRAGMA incremental_vacuum({pages})")))
}

pub async fn run_loop(db: Database, mut shutdown: watch::Receiver<bool>) {
    let trim_interval = Duration::from_secs(60 * 60);
    let vacuum_interval = Duration::from_secs(60 * 60);
    let mut trim_tick = tokio::time::interval(trim_interval);
    let mut vacuum_tick = tokio::time::interval(vacuum_interval);
    trim_tick.tick().await;
    vacuum_tick.tick().await;

    loop {
        tokio::select! {
            _ = shutdown.changed() => { if *shutdown.borrow() { return; } }
            _ = trim_tick.tick() => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as i64;
                match trim_logs_once(&db, now) {
                    Ok(n) if n > 0 => info!(deleted = n, "log retention trim"),
                    Ok(_) => {}
                    Err(e) => warn!(error = %e, "log retention trim failed"),
                }
            }
            _ = vacuum_tick.tick() => {
                if let Err(e) = incremental_vacuum(&db, 1000) {
                    warn!(error = %e, "incremental_vacuum failed");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn trims_old_rows_and_excess_per_service() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("a.sqlite")).await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let now = 10_000_000_000i64;
        let eight_days_ago = now - 8 * 24 * 60 * 60 * 1000;

        // Insert one old row.
        db.with_conn(|c| {
            c.execute(
                "INSERT INTO service_logs(service_id, run_id, timestamp_ms, seq, stream, line) VALUES (?1, 1, ?2, 1, 'stdout', 'old')",
                (svc, eight_days_ago),
            )
        })
        .unwrap();

        // Insert 50,010 recent rows to exceed the cap.
        db.with_conn(|c| {
            let mut stmt = c.prepare("INSERT INTO service_logs(service_id, run_id, timestamp_ms, seq, stream, line) VALUES (?1, 2, ?2, ?3, 'stdout', 'x')").unwrap();
            for i in 0..50_010i64 {
                stmt.execute((svc, now - i, i + 2)).unwrap();
            }
            Ok(())
        })
        .unwrap();

        let deleted = trim_logs_once(&db, now).unwrap();
        assert!(deleted >= 11); // 1 old + 10 excess

        let remaining: i64 = db
            .with_conn(|c| {
                c.query_row(
                    "SELECT COUNT(*) FROM service_logs WHERE service_id = ?1",
                    [svc],
                    |r| r.get(0),
                )
            })
            .unwrap();
        assert_eq!(remaining, 50_000);
    }
}
