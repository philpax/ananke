//! GPU device VRAM samples (`device_samples` table): periodic insert
//! from the snapshotter, time-range query for charts, and retention
//! pruning.

use ananke_errors::ExpectedError;
use rusqlite::params;

use crate::{Database, models::DeviceSample};

impl Database {
    /// Insert a device sample row. Called periodically by the snapshotter.
    pub async fn insert_device_sample(
        &self,
        device: &str,
        timestamp_ms: i64,
        total_bytes: i64,
        free_bytes: i64,
    ) -> Result<(), ExpectedError> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO device_samples (device, timestamp_ms, total_bytes, free_bytes, used_bytes)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                device,
                timestamp_ms,
                total_bytes,
                free_bytes,
                total_bytes - free_bytes
            ],
        )
        .map_err(|e| self.db_err(e))?;
        let _ = self.changed.send(crate::DatabaseChange::Samples);
        Ok(())
    }

    /// Query device samples for a time range.
    pub async fn query_device_samples(
        &self,
        device: Option<&str>,
        since_ms: i64,
        until_ms: i64,
    ) -> Result<Vec<DeviceSample>, ExpectedError> {
        let conn = self.conn.lock();
        let sql = "SELECT sample_id, device, timestamp_ms, total_bytes, free_bytes, used_bytes
             FROM device_samples
             WHERE timestamp_ms >= ?1 AND timestamp_ms <= ?2
               AND (?3 IS NULL OR device = ?3)
             ORDER BY timestamp_ms";
        let mut stmt = conn.prepare(sql).map_err(|e| self.db_err(e))?;
        let rows = stmt
            .query_map(params![since_ms, until_ms, device], |row| {
                Ok(DeviceSample {
                    sample_id: row.get(0)?,
                    device: row.get(1)?,
                    timestamp_ms: row.get(2)?,
                    total_bytes: row.get(3)?,
                    free_bytes: row.get(4)?,
                    used_bytes: row.get(5)?,
                })
            })
            .map_err(|e| self.db_err(e))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| self.db_err(e))
    }

    /// Prune device samples older than `cutoff_ms`.
    pub async fn prune_device_samples(&self, cutoff_ms: i64) -> Result<u64, ExpectedError> {
        let conn = self.conn.lock();
        let n = conn
            .execute(
                "DELETE FROM device_samples WHERE timestamp_ms < ?1",
                params![cutoff_ms],
            )
            .map_err(|e| self.db_err(e))?;
        Ok(n as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn device_samples_insert_and_query() {
        let db = Database::open_in_memory().await.unwrap();

        db.insert_device_sample("gpu:0", 1000, 24_000_000_000, 20_000_000_000)
            .await
            .unwrap();
        db.insert_device_sample("gpu:0", 2000, 24_000_000_000, 18_000_000_000)
            .await
            .unwrap();
        db.insert_device_sample("cpu", 1000, 64_000_000_000, 32_000_000_000)
            .await
            .unwrap();

        // Query all devices.
        let all = db.query_device_samples(None, 0, 10_000).await.unwrap();
        assert_eq!(all.len(), 3);

        // Query only gpu:0.
        let gpu0 = db
            .query_device_samples(Some("gpu:0"), 0, 10_000)
            .await
            .unwrap();
        assert_eq!(gpu0.len(), 2);
        assert_eq!(gpu0[0].device, "gpu:0");
        assert_eq!(gpu0[0].used_bytes, 4_000_000_000);

        // Time range filter.
        let recent = db.query_device_samples(None, 1500, 10_000).await.unwrap();
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].device, "gpu:0");
    }

    #[tokio::test]
    async fn device_samples_prune_old() {
        let db = Database::open_in_memory().await.unwrap();
        db.insert_device_sample("gpu:0", 100, 1000, 500)
            .await
            .unwrap();
        db.insert_device_sample("gpu:0", 2000, 1000, 500)
            .await
            .unwrap();

        let deleted = db.prune_device_samples(1000).await.unwrap();
        assert_eq!(deleted, 1);

        let remaining = db.query_device_samples(None, 0, 10_000).await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].timestamp_ms, 2000);
    }
}
