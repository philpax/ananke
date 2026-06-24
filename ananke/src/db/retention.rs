//! Log retention and SQLite incremental vacuum.

use std::time::Duration;

use tokio::sync::watch;
use tracing::{info, warn};

use crate::{db::Database, errors::ExpectedError};

/// Drop log rows older than this before the cap check.
const RETENTION_WINDOW: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// Per-service row cap; oldest rows beyond this are deleted.
const MAX_ROWS_PER_SERVICE: usize = 50_000;

/// Retention for request_metrics rows (7 days, same as logs).
const METRICS_RETENTION: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// Retention for device_samples rows (24 hours).
const DEVICE_SAMPLES_RETENTION: Duration = Duration::from_secs(24 * 60 * 60);

/// Cadence for both the retention trim and the incremental vacuum.
const MAINTENANCE_INTERVAL: Duration = Duration::from_secs(60 * 60);

/// Pages reclaimed per incremental vacuum pass. At the default 4 KiB page
/// size this releases up to ~4 MiB per tick.
const INCREMENTAL_VACUUM_PAGES: u64 = 1000;

/// Per-service log retention: 7 days or 50,000 lines, whichever tighter.
/// Also prunes request_metrics (7 days) and device_samples (24h).
/// Runs once when called; call from a daily scheduled task.
pub async fn trim_logs_once(db: &Database, now_ms: i64) -> Result<u64, ExpectedError> {
    let cutoff_ms = now_ms - RETENTION_WINDOW.as_millis() as i64;

    // 1. Drop log rows older than the retention window in one pass.
    let mut deleted = db.delete_logs_older_than(cutoff_ms).await?;

    // 2. Per-service row cap. For each live service, trim anything beyond
    //    the newest `MAX_ROWS_PER_SERVICE` rows.
    let services = db.list_live_services().await?;
    for svc in services {
        deleted += db
            .trim_logs_to_cap(svc.service_id, MAX_ROWS_PER_SERVICE)
            .await?;
    }

    // 3. Prune request_metrics older than the metrics retention window.
    let metrics_cutoff = now_ms - METRICS_RETENTION.as_millis() as i64;
    deleted += db.prune_request_metrics(metrics_cutoff).await?;

    // 4. Prune device_samples older than 24h.
    let samples_cutoff = now_ms - DEVICE_SAMPLES_RETENTION.as_millis() as i64;
    deleted += db.prune_device_samples(samples_cutoff).await?;

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
    use crate::db::models::ServiceLog;

    #[tokio::test]
    async fn trims_old_rows_and_excess_per_service() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let now = 10_000_000_000i64;
        let eight_days_ago = now - 8 * 24 * 60 * 60 * 1000;

        // One old row (older than 7 days).
        db.insert_log_batch(&[ServiceLog {
            service_id: svc,
            run_id: 1,
            timestamp_ms: eight_days_ago,
            seq: 1,
            stream: "stdout".to_string(),
            line: "old".to_string(),
        }])
        .await
        .unwrap();

        // 50,010 recent rows in one batch transaction.
        let recent: Vec<ServiceLog> = (0..50_010i64)
            .map(|i| ServiceLog {
                service_id: svc,
                run_id: 2,
                timestamp_ms: now - i,
                seq: i + 2,
                stream: "stdout".to_string(),
                line: "x".to_string(),
            })
            .collect();
        db.insert_log_batch(&recent).await.unwrap();

        let deleted = trim_logs_once(&db, now).await.unwrap();
        assert!(deleted >= 11); // 1 old + at least 10 excess

        let remaining = db.fetch_service_logs(svc).await.unwrap();
        assert_eq!(remaining.len(), 50_000);
    }

    #[tokio::test]
    async fn trims_old_request_metrics() {
        use crate::db::models::RequestMetric;
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let now = 10_000_000_000i64;
        let eight_days_ago = now - 8 * 24 * 60 * 60 * 1000;

        // One old metric (older than 7-day retention).
        db.insert_request_metric(&RequestMetric {
            metric_id: 0,
            service_id: svc,
            run_id: Some(1),
            timestamp_ms: eight_days_ago,
            endpoint: "/v1/chat/completions".into(),
            model: "demo".into(),
            prompt_tokens: Some(10),
            completion_tokens: Some(5),
            duration_ms: Some(100),
            ttft_ms: None,
            status_code: 200,
        })
        .await
        .unwrap();

        // One recent metric.
        db.insert_request_metric(&RequestMetric {
            metric_id: 0,
            service_id: svc,
            run_id: Some(1),
            timestamp_ms: now,
            endpoint: "/v1/chat/completions".into(),
            model: "demo".into(),
            prompt_tokens: Some(20),
            completion_tokens: Some(10),
            duration_ms: Some(200),
            ttft_ms: None,
            status_code: 200,
        })
        .await
        .unwrap();

        let deleted = trim_logs_once(&db, now).await.unwrap();
        assert!(deleted >= 1, "should have deleted at least 1 old metric");

        let remaining = db
            .query_request_metrics(Some(svc), 0, now + 1, 60_000)
            .await
            .unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].request_count, 1);
    }

    #[tokio::test]
    async fn trims_old_device_samples() {
        let db = Database::open_in_memory().await.unwrap();
        let now = 10_000_000_000i64;
        let two_days_ago = now - 2 * 24 * 60 * 60 * 1000;

        // One old sample (older than 24h retention).
        db.insert_device_sample("gpu:0", two_days_ago, 1000, 500)
            .await
            .unwrap();

        // One recent sample.
        db.insert_device_sample("gpu:0", now, 1000, 400)
            .await
            .unwrap();

        let deleted = trim_logs_once(&db, now).await.unwrap();
        assert!(deleted >= 1, "should have deleted at least 1 old sample");

        let remaining = db.query_device_samples(None, 0, now + 1).await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].timestamp_ms, now);
    }
}
