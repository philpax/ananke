//! SQLite-backed database. One process-wide [`Arc<Mutex<Connection>>`]
//! per [`Database`]; every query takes the short critical section and
//! returns. Schema is applied on open via the versioned migration chain
//! in [`migrations`], so re-opening an already-provisioned file applies
//! only pending migrations (empty set when up to date).

pub mod logs;
pub mod migrations;
pub mod models;
pub mod pragma;
pub mod retention;

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use parking_lot::Mutex;
use rusqlite::{Connection, OptionalExtension, params};

use crate::{
    db::models::{RunningService, Service, ServiceLog},
    errors::ExpectedError,
};

/// Cloneable database handle. All queries go through the shared
/// `Connection` behind a `parking_lot::Mutex`. Lock durations stay in
/// the microsecond range because SQLite on local disk is fast and
/// nothing holds the lock across `.await` points.
#[derive(Clone)]
pub struct Database {
    conn: Arc<Mutex<Connection>>,
    path: PathBuf,
}

impl Database {
    pub async fn open(path: &Path) -> Result<Self, ExpectedError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                ExpectedError::database_open_failed(path.to_path_buf(), e.to_string())
            })?;
        }
        let mut conn = Connection::open(path).map_err(|e| {
            ExpectedError::database_open_failed(path.to_path_buf(), e.to_string())
        })?;
        migrations::apply_pending(&mut conn, crate::tracking::now_unix_ms())
            .map_err(|e| ExpectedError::database_open_failed(path.to_path_buf(), e.to_string()))?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            path: path.to_path_buf(),
        })
    }

    /// Open a `:memory:` database with the same schema applied. Used by
    /// tests that want a full DB surface without touching disk.
    pub async fn open_in_memory() -> Result<Self, ExpectedError> {
        let mut conn = Connection::open_in_memory().map_err(|e| {
            ExpectedError::database_open_failed(PathBuf::from(":memory:"), e.to_string())
        })?;
        migrations::apply_pending(&mut conn, crate::tracking::now_unix_ms()).map_err(|e| {
            ExpectedError::database_open_failed(PathBuf::from(":memory:"), e.to_string())
        })?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            path: PathBuf::from(":memory:"),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Insert a service by name if absent, or un-tombstone it if the row
    /// exists with `deleted_at` set. Returns the `service_id` either way.
    pub async fn upsert_service(&self, name: &str, now_ms: i64) -> Result<i64, ExpectedError> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO services(name, created_at, deleted_at) VALUES (?1, ?2, NULL)
             ON CONFLICT(name) DO UPDATE SET deleted_at = NULL",
            params![name, now_ms],
        )
        .map_err(|e| self.db_err(e))?;
        conn.query_row(
            "SELECT service_id FROM services WHERE name = ?1",
            [name],
            |row| row.get::<_, i64>(0),
        )
        .map_err(|e| self.db_err(e))
    }

    /// Reparent a `service_id` from `old_name` to `new_name`.
    ///
    /// - If `old_name` has a live row and `new_name` does not: rename.
    /// - If both exist: tombstone `old_name` (the new service keeps its own id).
    /// - If `old_name` doesn't exist: no-op (warning is issued upstream).
    pub async fn reparent(
        &self,
        old_name: &str,
        new_name: &str,
        now_ms: i64,
    ) -> Result<(), ExpectedError> {
        let conn = self.conn.lock();
        let old_exists: Option<i64> = conn
            .query_row(
                "SELECT service_id FROM services WHERE name = ?1",
                [old_name],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| self.db_err(e))?;
        let new_exists: Option<i64> = conn
            .query_row(
                "SELECT service_id FROM services WHERE name = ?1",
                [new_name],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| self.db_err(e))?;
        match (old_exists, new_exists) {
            (Some(_), None) => {
                conn.execute(
                    "UPDATE services SET name = ?1 WHERE name = ?2",
                    params![new_name, old_name],
                )
                .map_err(|e| self.db_err(e))?;
            }
            (Some(_), Some(_)) => {
                conn.execute(
                    "UPDATE services SET deleted_at = ?1 WHERE name = ?2",
                    params![now_ms, old_name],
                )
                .map_err(|e| self.db_err(e))?;
            }
            (None, _) => {}
        }
        Ok(())
    }

    /// Resolve a service name to its id, or `None` if no row exists (the
    /// row is matched even when tombstoned — the caller decides what to
    /// do with a soft-deleted hit).
    pub async fn resolve_service_id(&self, name: &str) -> Result<Option<i64>, ExpectedError> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT service_id FROM services WHERE name = ?1",
            [name],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(|e| self.db_err(e))
    }

    /// All services without a `deleted_at`. Used by retention to iterate
    /// live services and enforce the per-service log cap.
    pub async fn list_live_services(&self) -> Result<Vec<Service>, ExpectedError> {
        let conn = self.conn.lock();
        let sql = format!(
            "SELECT {} FROM services WHERE deleted_at IS NULL",
            Service::COLUMNS
        );
        let mut stmt = conn.prepare(&sql).map_err(|e| self.db_err(e))?;
        let rows = stmt
            .query_map([], Service::from_row)
            .map_err(|e| self.db_err(e))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| self.db_err(e))
    }

    /// Fetch all log rows for a service (used by the paginated logs
    /// endpoint, which then filters in-memory — the retention cap keeps
    /// this bounded at 50k rows per service).
    pub async fn fetch_service_logs(
        &self,
        service_id: i64,
    ) -> Result<Vec<ServiceLog>, ExpectedError> {
        let conn = self.conn.lock();
        let sql = format!(
            "SELECT {} FROM service_logs WHERE service_id = ?1",
            ServiceLog::COLUMNS
        );
        let mut stmt = conn.prepare(&sql).map_err(|e| self.db_err(e))?;
        let rows = stmt
            .query_map([service_id], ServiceLog::from_row)
            .map_err(|e| self.db_err(e))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| self.db_err(e))
    }

    /// Insert (or upsert) a running_services row for a freshly-spawned
    /// supervisor child.
    pub async fn insert_running(&self, row: &RunningService) -> Result<(), ExpectedError> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT OR REPLACE INTO running_services
                 (service_id, run_id, pid, spawned_at, command_line, allocation, state)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                row.service_id,
                row.run_id,
                row.pid,
                row.spawned_at,
                row.command_line,
                row.allocation,
                row.state
            ],
        )
        .map_err(|e| self.db_err(e))?;
        Ok(())
    }

    /// Delete a running_services row on drain / orphan cleanup.
    pub async fn delete_running(
        &self,
        service_id: i64,
        run_id: i64,
    ) -> Result<(), ExpectedError> {
        let conn = self.conn.lock();
        conn.execute(
            "DELETE FROM running_services WHERE service_id = ?1 AND run_id = ?2",
            params![service_id, run_id],
        )
        .map_err(|e| self.db_err(e))?;
        Ok(())
    }

    /// Every running_services row, used by orphan recovery at startup.
    pub async fn list_running(&self) -> Result<Vec<RunningService>, ExpectedError> {
        let conn = self.conn.lock();
        let sql = format!("SELECT {} FROM running_services", RunningService::COLUMNS);
        let mut stmt = conn.prepare(&sql).map_err(|e| self.db_err(e))?;
        let rows = stmt
            .query_map([], RunningService::from_row)
            .map_err(|e| self.db_err(e))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| self.db_err(e))
    }

    /// Record a oneshot's submission. Idempotent on the id — repeated
    /// calls with the same id are a no-op.
    pub async fn insert_oneshot(
        &self,
        id: &str,
        service_id: i64,
        submitted_at: i64,
        ttl_ms: i64,
    ) -> Result<(), ExpectedError> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT OR IGNORE INTO oneshots(id, service_id, submitted_at, ttl_ms)
             VALUES (?1, ?2, ?3, ?4)",
            params![id, service_id, submitted_at, ttl_ms],
        )
        .map_err(|e| self.db_err(e))?;
        Ok(())
    }

    /// Stamp a oneshot as finished. The row stays in place so status
    /// polls keep seeing the terminal state until it falls out of
    /// retention.
    pub async fn mark_oneshot_ended(
        &self,
        id: &str,
        ended_at_ms: i64,
    ) -> Result<(), ExpectedError> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE oneshots SET ended_at = ?1 WHERE id = ?2",
            params![ended_at_ms, id],
        )
        .map_err(|e| self.db_err(e))?;
        Ok(())
    }

    /// Insert a batch of service_logs rows inside a single transaction.
    pub async fn insert_log_batch(&self, rows: &[ServiceLog]) -> Result<(), ExpectedError> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(|e| self.db_err(e))?;
        {
            let mut stmt = tx
                .prepare(
                    "INSERT INTO service_logs
                         (service_id, run_id, timestamp_ms, seq, stream, line)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                )
                .map_err(|e| self.db_err(e))?;
            for row in rows {
                stmt.execute(params![
                    row.service_id,
                    row.run_id,
                    row.timestamp_ms,
                    row.seq,
                    row.stream,
                    row.line
                ])
                .map_err(|e| self.db_err(e))?;
            }
        }
        tx.commit().map_err(|e| self.db_err(e))?;
        Ok(())
    }

    /// Delete log rows older than `cutoff_ms`. Returns rows removed.
    pub async fn delete_logs_older_than(&self, cutoff_ms: i64) -> Result<u64, ExpectedError> {
        let conn = self.conn.lock();
        let n = conn
            .execute(
                "DELETE FROM service_logs WHERE timestamp_ms < ?1",
                params![cutoff_ms],
            )
            .map_err(|e| self.db_err(e))?;
        Ok(n as u64)
    }

    /// Keep the most-recent `cap` log rows for a service, delete the
    /// rest. Returns rows removed.
    pub async fn trim_logs_to_cap(
        &self,
        service_id: i64,
        cap: usize,
    ) -> Result<u64, ExpectedError> {
        let conn = self.conn.lock();
        // SQLite lacks `DELETE ... LIMIT` in the default build. Instead,
        // pick the oldest row we want to keep (row `cap` in the DESC
        // ordering, 0-indexed) and delete anything strictly older.
        let cutoff: Option<(i64, i64)> = conn
            .query_row(
                "SELECT timestamp_ms, seq FROM service_logs
                 WHERE service_id = ?1
                 ORDER BY timestamp_ms DESC, seq DESC
                 LIMIT 1 OFFSET ?2",
                params![service_id, cap as i64],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
            )
            .optional()
            .map_err(|e| self.db_err(e))?;
        let Some((ts, seq)) = cutoff else {
            return Ok(0);
        };
        let n = conn
            .execute(
                "DELETE FROM service_logs
                 WHERE service_id = ?1
                   AND (timestamp_ms < ?2
                        OR (timestamp_ms = ?2 AND seq <= ?3))",
                params![service_id, ts, seq],
            )
            .map_err(|e| self.db_err(e))?;
        Ok(n as u64)
    }

    fn db_err(&self, e: rusqlite::Error) -> ExpectedError {
        ExpectedError::database_open_failed(self.path.clone(), e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn upsert_service_is_idempotent() {
        let db = Database::open_in_memory().await.unwrap();
        let id1 = db.upsert_service("demo", 1000).await.unwrap();
        let id2 = db.upsert_service("demo", 2000).await.unwrap();
        assert_eq!(id1, id2);
    }

    #[tokio::test]
    async fn reparent_renames_when_only_old_exists() {
        let db = Database::open_in_memory().await.unwrap();
        let original = db.upsert_service("old-name", 1000).await.unwrap();
        db.reparent("old-name", "new-name", 2000).await.unwrap();

        let new_id = db.upsert_service("new-name", 3000).await.unwrap();
        assert_eq!(new_id, original);
    }

    #[tokio::test]
    async fn reparent_tombstones_when_both_exist() {
        let db = Database::open_in_memory().await.unwrap();
        let _old = db.upsert_service("old-name", 1000).await.unwrap();
        let new_before = db.upsert_service("new-name", 1000).await.unwrap();
        db.reparent("old-name", "new-name", 2000).await.unwrap();

        let new_after = db.upsert_service("new-name", 3000).await.unwrap();
        assert_eq!(new_after, new_before);
        let live = db.list_live_services().await.unwrap();
        let live_names: Vec<_> = live.iter().map(|s| s.name.as_str()).collect();
        assert!(live_names.contains(&"new-name"));
        assert!(!live_names.contains(&"old-name"));
    }
}
