//! Log retention and SQLite incremental vacuum.

use std::time::Duration;

use tokio::sync::watch;
use tracing::{info, warn};

use crate::db::{
    Database,
    models::{Service, ServiceLog},
};

/// Drop log rows older than this before the cap check (spec §12).
const RETENTION_WINDOW: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// Per-service row cap; oldest rows beyond this are deleted (spec §12).
const MAX_ROWS_PER_SERVICE: usize = 50_000;

/// Cadence for both the retention trim and the incremental vacuum.
const MAINTENANCE_INTERVAL: Duration = Duration::from_secs(60 * 60);

/// Pages reclaimed per incremental vacuum pass. At the default 4 KiB page
/// size this releases up to ~4 MiB per tick.
const INCREMENTAL_VACUUM_PAGES: u64 = 1000;

/// Per-service log retention: 7 days or 50,000 lines, whichever tighter
/// (spec §12). Runs once when called; call from a daily scheduled task.
pub async fn trim_logs_once(db: &Database, now_ms: i64) -> Result<u64, toasty::Error> {
    let cutoff_ms = now_ms - RETENTION_WINDOW.as_millis() as i64;

    let mut handle = db.handle();
    let mut deleted: u64 = 0;

    // 1. Drop rows older than the retention window in one pass.
    let old: Vec<ServiceLog> =
        ServiceLog::filter(ServiceLog::fields().timestamp_ms().lt(cutoff_ms))
            .exec(&mut handle)
            .await?;
    for row in old {
        row.delete().exec(&mut handle).await?;
        deleted += 1;
    }

    // 2. Per-service row count cap. For each live service, fetch its logs
    //    ordered by timestamp ASC; if total exceeds the cap, delete the
    //    oldest `total - cap` rows. Select-then-delete avoids toasty's
    //    subquery DSL; retention runs hourly so the extra round-trips
    //    are affordable.
    let services: Vec<Service> = Service::filter(Service::fields().deleted_at().is_none())
        .exec(&mut handle)
        .await?;
    for svc in services {
        let mut rows: Vec<ServiceLog> =
            ServiceLog::filter(ServiceLog::fields().service_id().eq(svc.service_id as i64))
                .exec(&mut handle)
                .await?;
        if rows.len() > MAX_ROWS_PER_SERVICE {
            rows.sort_by_key(|r| r.timestamp_ms);
            let excess = rows.len() - MAX_ROWS_PER_SERVICE;
            for row in rows.into_iter().take(excess) {
                row.delete().exec(&mut handle).await?;
                deleted += 1;
            }
        }
    }

    Ok(deleted)
}

pub fn incremental_vacuum(db: &Database, pages: u64) -> rusqlite::Result<()> {
    crate::db::pragma::incremental_vacuum(db.path(), pages)
}

pub async fn run_loop(db: Database, mut shutdown: watch::Receiver<bool>) {
    let mut trim_tick = tokio::time::interval(MAINTENANCE_INTERVAL);
    let mut vacuum_tick = tokio::time::interval(MAINTENANCE_INTERVAL);
    trim_tick.tick().await;
    vacuum_tick.tick().await;

    loop {
        tokio::select! {
            _ = shutdown.changed() => { if *shutdown.borrow() { return; } }
            _ = trim_tick.tick() => {
                match trim_logs_once(&db, crate::tracking::now_unix_ms()).await {
                    Ok(n) if n > 0 => info!(deleted = n, "log retention trim"),
                    Ok(_) => {}
                    Err(e) => warn!(error = %e, "log retention trim failed"),
                }
            }
            _ = vacuum_tick.tick() => {
                if let Err(e) = incremental_vacuum(&db, INCREMENTAL_VACUUM_PAGES) {
                    warn!(error = %e, "incremental_vacuum failed");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn trims_old_rows_and_excess_per_service() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let now = 10_000_000_000i64;
        let eight_days_ago = now - 8 * 24 * 60 * 60 * 1000;

        let mut handle = db.handle();

        // One old row (older than 7 days).
        toasty::create!(ServiceLog {
            service_id: svc,
            run_id: 1,
            seq: 1,
            timestamp_ms: eight_days_ago,
            stream: "stdout".to_string(),
            line: "old".to_string(),
        })
        .exec(&mut handle)
        .await
        .unwrap();

        // 50,010 recent rows. Use a single transaction so the insert loop
        // doesn't burn ~3 round-trips per row.
        let mut tx = handle.transaction().await.unwrap();
        for i in 0..50_010i64 {
            toasty::create!(ServiceLog {
                service_id: svc,
                run_id: 2,
                seq: i + 2,
                timestamp_ms: now - i,
                stream: "stdout".to_string(),
                line: "x".to_string(),
            })
            .exec(&mut tx)
            .await
            .unwrap();
        }
        tx.commit().await.unwrap();

        let deleted = trim_logs_once(&db, now).await.unwrap();
        assert!(deleted >= 11); // 1 old + at least 10 excess

        let remaining: Vec<ServiceLog> =
            ServiceLog::filter(ServiceLog::fields().service_id().eq(svc))
                .exec(&mut handle)
                .await
                .unwrap();
        assert_eq!(remaining.len(), 50_000);
    }
}
