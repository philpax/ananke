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
    db::models::{DeviceSample, RequestMetric, RunningService, Service, ServiceLog},
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
        let mut conn = Connection::open(path)
            .map_err(|e| ExpectedError::database_open_failed(path.to_path_buf(), e.to_string()))?;
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

    /// Count requests and errors for one run within a recent window, for the
    /// auto-restart error-rate watchdog. Scoped to a single `run_id` so a
    /// prior (already-restarted) run's errors never count against the current
    /// process; scoped to `timestamp_ms >= since_ms` so a service that was
    /// healthy for hours before wedging is caught on the recent window rather
    /// than diluted by historical success. `min_error_status` selects the
    /// error class: 500 for server-only, 400 for client-and-server. Returns
    /// `(total, errors)`.
    pub async fn error_rate_since(
        &self,
        service_id: i64,
        run_id: i64,
        since_ms: i64,
        min_error_status: u16,
    ) -> Result<(u64, u64), ExpectedError> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT COUNT(*),
                    SUM(CASE WHEN status_code >= ?4 THEN 1 ELSE 0 END)
             FROM request_metrics
             WHERE service_id = ?1 AND run_id = ?2 AND timestamp_ms >= ?3",
            params![service_id, run_id, since_ms, min_error_status as i64],
            |row| {
                let total: i64 = row.get(0)?;
                // SUM over zero rows is NULL, hence the Option.
                let errors: i64 = row.get::<_, Option<i64>>(1)?.unwrap_or(0);
                Ok((total.max(0) as u64, errors.max(0) as u64))
            },
        )
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

    /// Load `(name, last_request_ms)` pairs for every service that has
    /// at least one row in `request_metrics`. Used on boot to seed
    /// `ActivityTable::wall_ms` so `last_used_ms` survives restarts
    /// without a dedicated persistence column.
    pub async fn load_last_request_times(&self) -> Result<Vec<(String, i64)>, ExpectedError> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT s.name, MAX(rm.timestamp_ms) AS last_used
                 FROM services s
                 JOIN request_metrics rm ON s.service_id = rm.service_id
                 WHERE s.deleted_at IS NULL
                 GROUP BY s.name",
            )
            .map_err(|e| self.db_err(e))?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
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
    pub async fn delete_running(&self, service_id: i64, run_id: i64) -> Result<(), ExpectedError> {
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

    /// Insert a single request metric row. Called by the proxy after the
    /// response stream completes.
    pub async fn insert_request_metric(&self, row: &RequestMetric) -> Result<(), ExpectedError> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO request_metrics
                 (service_id, run_id, timestamp_ms, endpoint, model,
                  prompt_tokens, completion_tokens, prompt_eval_tokens,
                  duration_ms, ttft_ms, prompt_ms, predicted_ms, status_code)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                row.service_id,
                row.run_id,
                row.timestamp_ms,
                row.endpoint,
                row.model,
                row.prompt_tokens,
                row.completion_tokens,
                row.prompt_eval_tokens,
                row.duration_ms,
                row.ttft_ms,
                row.prompt_ms,
                row.predicted_ms,
                row.status_code,
            ],
        )
        .map_err(|e| self.db_err(e))?;
        Ok(())
    }

    /// Query aggregated request metrics for the JSON `/api/metrics` endpoint.
    /// Returns pre-bucketed time-series data — the frontend doesn't aggregate.
    /// Each bucket is scoped to a single service so the frontend can distinguish
    /// per-service contributions when no service filter is given.
    pub async fn query_request_metrics(
        &self,
        service_id: Option<i64>,
        since_ms: i64,
        until_ms: i64,
        bucket_ms: i64,
    ) -> Result<Vec<MetricBucket>, ExpectedError> {
        let conn = self.conn.lock();
        // LEFT JOIN services so that orphaned metrics (service deleted but
        // rows remain) still appear with service = NULL. Grouping by
        // service_id in addition to the bucket ensures per-service breakdown
        // when no filter is given.
        // The input/output TPS split is sourced tier-by-tier per row:
        //   - output interval = engine `predicted_ms` if present, else the
        //     proxy-observed decode window (`duration_ms - ttft_ms`) when the
        //     response streamed, else null (no boundary → no split);
        //   - input interval = engine `prompt_ms` if present, else `ttft_ms`.
        // The input numerator uses the engine's evaluated prompt-token count
        // `prompt_eval_tokens` when present, falling back to the billed
        // `prompt_tokens`. The billed count includes tokens served from the KV
        // cache, so dividing it by the cache-aware `prompt_ms` would wildly
        // overstate prefill throughput.
        // Effective TPS is completion tokens over wall-clock `duration_ms`:
        // end-to-end generation throughput (prefill, TTFT, and queue wait all
        // count against it), always computable whenever a request has a
        // duration. It is the tier-3 (non-streaming, no engine timings) fall
        // back where no decode window exists to derive `output_tps`. It counts
        // only generated tokens — a prompt-only request (e.g. embeddings)
        // contributes zero and drops out rather than spiking the line.
        let out_interval = "COALESCE(rm.predicted_ms, CASE WHEN rm.ttft_ms IS NOT NULL \
             AND rm.duration_ms IS NOT NULL THEN rm.duration_ms - rm.ttft_ms END)";
        let in_interval = "COALESCE(rm.prompt_ms, rm.ttft_ms)";
        let sql = format!(
            "SELECT
                (rm.timestamp_ms / ?1) * ?1 AS bucket,
                s.name AS service_name,
                COUNT(*) AS request_count,
                COALESCE(SUM(rm.prompt_tokens), 0) AS prompt_tokens,
                COALESCE(SUM(rm.completion_tokens), 0) AS completion_tokens,
                AVG(rm.duration_ms) AS avg_duration_ms,
                SUM(CASE WHEN rm.status_code >= 400 THEN 1 ELSE 0 END) AS error_count,
                AVG(rm.ttft_ms) AS avg_ttft_ms,
                COALESCE(SUM(CASE WHEN {out} IS NOT NULL
                                  THEN rm.completion_tokens ELSE 0 END), 0) AS output_tokens,
                COALESCE(SUM({out}), 0) AS output_ms,
                COALESCE(SUM(CASE WHEN {inp} IS NOT NULL
                                  THEN COALESCE(rm.prompt_eval_tokens, rm.prompt_tokens) ELSE 0 END), 0) AS input_tokens,
                COALESCE(SUM({inp}), 0) AS input_ms,
                COALESCE(SUM(CASE WHEN rm.duration_ms IS NOT NULL
                                  THEN COALESCE(rm.completion_tokens, 0)
                                  ELSE 0 END), 0) AS effective_tokens,
                COALESCE(SUM(CASE WHEN rm.duration_ms IS NOT NULL
                                  THEN rm.duration_ms ELSE 0 END), 0) AS effective_ms
             FROM request_metrics rm
             LEFT JOIN services s ON s.service_id = rm.service_id
             WHERE rm.timestamp_ms >= ?2 AND rm.timestamp_ms <= ?3
               AND (?4 IS NULL OR rm.service_id = ?4)
             GROUP BY bucket, rm.service_id
             ORDER BY bucket, service_name",
            out = out_interval,
            inp = in_interval,
        );
        let mut stmt = conn.prepare(&sql).map_err(|e| self.db_err(e))?;
        let rows = stmt
            .query_map(params![bucket_ms, since_ms, until_ms, service_id], |row| {
                // A rate needs both a positive interval and tokens actually
                // produced. Zero tokens over a non-zero interval is not
                // "0 tok/s" — it is the absence of throughput (e.g. every
                // request in the bucket stalled or errored before emitting a
                // token), which must read as null so the chart breaks the line
                // rather than pinning it to the floor. Without the token guard
                // the effective tier (completion / wall-clock duration) reports
                // 0 for a wedged run, while the decode/prefill tiers correctly
                // go null — an inconsistency across the three lines.
                let tps = |tokens: i64, ms: i64| {
                    (ms > 0 && tokens > 0).then(|| tokens as f64 / (ms as f64 / 1000.0))
                };
                let output_tokens: i64 = row.get(8)?;
                let output_ms: i64 = row.get(9)?;
                let input_tokens: i64 = row.get(10)?;
                let input_ms: i64 = row.get(11)?;
                let effective_tokens: i64 = row.get(12)?;
                let effective_ms: i64 = row.get(13)?;
                Ok(MetricBucket {
                    service: row.get::<_, Option<String>>(1)?,
                    bucket_start: row.get(0)?,
                    request_count: row.get(2)?,
                    prompt_tokens: row.get(3)?,
                    completion_tokens: row.get(4)?,
                    avg_duration_ms: row.get::<_, Option<f64>>(5)?,
                    error_count: row.get(6)?,
                    avg_ttft_ms: row.get::<_, Option<f64>>(7)?,
                    output_tps: tps(output_tokens, output_ms),
                    input_tps: tps(input_tokens, input_ms),
                    effective_tps: tps(effective_tokens, effective_ms),
                })
            })
            .map_err(|e| self.db_err(e))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| self.db_err(e))
    }

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

    /// Prune request metrics older than `cutoff_ms`.
    pub async fn prune_request_metrics(&self, cutoff_ms: i64) -> Result<u64, ExpectedError> {
        let conn = self.conn.lock();
        let n = conn
            .execute(
                "DELETE FROM request_metrics WHERE timestamp_ms < ?1",
                params![cutoff_ms],
            )
            .map_err(|e| self.db_err(e))?;
        Ok(n as u64)
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

/// One time bucket of aggregated request metrics, scoped to a single service.
pub struct MetricBucket {
    /// Service name, or `None` if the service was deleted but metric rows remain.
    pub service: Option<String>,
    pub bucket_start: i64,
    pub request_count: i64,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub avg_duration_ms: Option<f64>,
    pub error_count: i64,
    /// Average time-to-first-token in milliseconds (streaming requests only).
    pub avg_ttft_ms: Option<f64>,
    /// Output tokens per second during decode: completion tokens divided
    /// by total decode time. `None` if no timed requests in the bucket.
    pub output_tps: Option<f64>,
    /// Input tokens per second during prompt processing: prompt tokens
    /// divided by total TTFT. `None` if no timed requests in the bucket.
    pub input_tps: Option<f64>,
    /// End-to-end effective generation throughput: completion tokens divided
    /// by total wall-clock duration (so prefill, TTFT, and queue wait all
    /// count against it). Always available whenever the bucket has any request
    /// with a recorded duration, including non-streaming requests with no
    /// engine timings where no decode window exists to derive `output_tps`.
    /// This is *not* a decode rate — it is always ≤ `output_tps`. A bucket
    /// that generated no tokens (e.g. only embeddings or stalled requests) is
    /// `None`, not zero.
    pub effective_tps: Option<f64>,
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

    #[tokio::test]
    async fn request_metrics_insert_and_query() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 1000).await.unwrap();

        // Insert two requests 5 minutes apart, one success + one error.
        db.insert_request_metric(&RequestMetric {
            metric_id: 0,
            prompt_eval_tokens: None,
            service_id: svc,
            run_id: Some(1),
            timestamp_ms: 10_000,
            endpoint: "/v1/chat/completions".into(),
            model: "demo".into(),
            prompt_tokens: Some(100),
            completion_tokens: Some(50),
            duration_ms: Some(1200),
            ttft_ms: Some(200),
            prompt_ms: None,
            predicted_ms: None,
            status_code: 200,
        })
        .await
        .unwrap();
        db.insert_request_metric(&RequestMetric {
            metric_id: 0,
            prompt_eval_tokens: None,
            service_id: svc,
            run_id: Some(1),
            timestamp_ms: 10_000 + 5 * 60_000,
            endpoint: "/v1/chat/completions".into(),
            model: "demo".into(),
            prompt_tokens: Some(200),
            completion_tokens: Some(80),
            duration_ms: Some(800),
            ttft_ms: None,
            prompt_ms: None,
            predicted_ms: None,
            status_code: 500,
        })
        .await
        .unwrap();

        // Query with a 10-minute bucket — both should land in the same bucket.
        let buckets = db
            .query_request_metrics(Some(svc), 0, 20 * 60_000, 10 * 60_000)
            .await
            .unwrap();
        assert_eq!(buckets.len(), 1);
        let b = &buckets[0];
        assert_eq!(b.service.as_deref(), Some("demo"));
        assert_eq!(b.request_count, 2);
        assert_eq!(b.prompt_tokens, 300);
        assert_eq!(b.completion_tokens, 130);
        assert!((b.avg_duration_ms.unwrap() - 1000.0).abs() < 0.1);
        assert_eq!(b.error_count, 1);
    }

    #[tokio::test]
    async fn error_rate_since_scopes_to_run_and_window() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let insert = |run_id: i64, timestamp_ms: i64, status_code: i64| {
            let db = &db;
            async move {
                db.insert_request_metric(&RequestMetric {
                    metric_id: 0,
                    prompt_eval_tokens: None,
                    service_id: svc,
                    run_id: Some(run_id),
                    timestamp_ms,
                    endpoint: "/v1/chat/completions".into(),
                    model: "demo".into(),
                    prompt_tokens: None,
                    completion_tokens: None,
                    duration_ms: None,
                    ttft_ms: None,
                    prompt_ms: None,
                    predicted_ms: None,
                    status_code,
                })
                .await
                .unwrap();
            }
        };

        // Current run 2: 3 x 500 + 1 x 404 + 1 x 200 inside the window.
        insert(2, 100_000, 500).await;
        insert(2, 101_000, 500).await;
        insert(2, 102_000, 500).await;
        insert(2, 103_000, 404).await;
        insert(2, 104_000, 200).await;
        // A previous run's errors must not count.
        insert(1, 100_500, 500).await;
        insert(1, 100_600, 500).await;
        // An in-window 500 from run 2 but before the `since` cutoff.
        insert(2, 50_000, 500).await;

        // Server-only (>=500) from 90s: 3 errors of 5 total in-window rows.
        let (total, errors) = db.error_rate_since(svc, 2, 90_000, 500).await.unwrap();
        assert_eq!((total, errors), (5, 3));

        // Client-and-server (>=400): the 404 now counts too → 4 errors.
        let (total, errors) = db.error_rate_since(svc, 2, 90_000, 400).await.unwrap();
        assert_eq!((total, errors), (5, 4));

        // No rows in window → zero total, zero errors (SUM over empty is NULL).
        let (total, errors) = db.error_rate_since(svc, 2, 200_000, 500).await.unwrap();
        assert_eq!((total, errors), (0, 0));
    }

    /// The input/output split is sourced tier-by-tier: engine timings
    /// (tier 1), proxy TTFT for streaming (tier 2), or neither (tier 3,
    /// which yields only effective TPS). Each tier lives in its own bucket.
    #[tokio::test]
    async fn request_metrics_tps_tiers() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let base = |timestamp_ms| RequestMetric {
            metric_id: 0,
            prompt_eval_tokens: None,
            service_id: svc,
            run_id: Some(1),
            timestamp_ms,
            endpoint: "/v1/chat/completions".into(),
            model: "demo".into(),
            prompt_tokens: None,
            completion_tokens: None,
            duration_ms: None,
            ttft_ms: None,
            prompt_ms: None,
            predicted_ms: None,
            status_code: 200,
        };

        // Tier 1 (engine timings) at t=0: input 100 tok / 1000 ms = 100 tps,
        // output 50 tok / 500 ms = 100 tps, effective 50 tok / 2000 ms = 25.
        db.insert_request_metric(&RequestMetric {
            prompt_tokens: Some(100),
            completion_tokens: Some(50),
            duration_ms: Some(2000),
            prompt_ms: Some(1000),
            predicted_ms: Some(500),
            ..base(0)
        })
        .await
        .unwrap();

        // Tier 2 (streaming TTFT) at t=60s: input 100 tok / 200 ms = 500 tps,
        // output 80 tok / (1000 - 200) ms = 100 tps, effective 80 / 1000 = 80.
        db.insert_request_metric(&RequestMetric {
            prompt_tokens: Some(100),
            completion_tokens: Some(80),
            duration_ms: Some(1000),
            ttft_ms: Some(200),
            ..base(60_000)
        })
        .await
        .unwrap();

        // Tier 3 (non-streaming, no timings) at t=120s: no split, effective
        // 100 tok / 1000 ms = 100 tps.
        db.insert_request_metric(&RequestMetric {
            prompt_tokens: Some(200),
            completion_tokens: Some(100),
            duration_ms: Some(1000),
            ..base(120_000)
        })
        .await
        .unwrap();

        let buckets = db
            .query_request_metrics(Some(svc), 0, 200_000, 60_000)
            .await
            .unwrap();
        assert_eq!(buckets.len(), 3);
        let close = |a: Option<f64>, b: f64| (a.unwrap() - b).abs() < 0.01;

        assert!(close(buckets[0].input_tps, 100.0));
        assert!(close(buckets[0].output_tps, 100.0));
        assert!(close(buckets[0].effective_tps, 25.0));

        assert!(close(buckets[1].input_tps, 500.0));
        assert!(close(buckets[1].output_tps, 100.0));
        assert!(close(buckets[1].effective_tps, 80.0));

        assert_eq!(buckets[2].input_tps, None);
        assert_eq!(buckets[2].output_tps, None);
        assert!(close(buckets[2].effective_tps, 100.0));
    }

    /// A bucket whose requests ran (non-null `duration_ms`) but produced no
    /// tokens — the signature of a wedged run whose cancelled requests each
    /// held the connection for their full duration — must report null for
    /// every TPS tier, not a floor-pinned effective figure of `0 tok/s`.
    /// Otherwise the effective line stays drawn across a stall while the
    /// decode/prefill lines correctly break.
    #[tokio::test]
    async fn zero_token_bucket_reports_null_tps() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let base = |timestamp_ms| RequestMetric {
            metric_id: 0,
            prompt_eval_tokens: None,
            service_id: svc,
            run_id: Some(1),
            timestamp_ms,
            endpoint: "/v1/chat/completions".into(),
            model: "demo".into(),
            prompt_tokens: None,
            completion_tokens: None,
            duration_ms: None,
            ttft_ms: None,
            prompt_ms: None,
            predicted_ms: None,
            status_code: 200,
        };

        // Two cancelled requests: each ran ~300s and emitted nothing.
        for i in 0..2 {
            db.insert_request_metric(&RequestMetric {
                duration_ms: Some(300_000),
                ..base(i)
            })
            .await
            .unwrap();
        }

        let buckets = db
            .query_request_metrics(Some(svc), 0, 60_000, 60_000)
            .await
            .unwrap();
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].request_count, 2);
        assert_eq!(buckets[0].effective_tps, None, "effective must be null");
        assert_eq!(buckets[0].input_tps, None);
        assert_eq!(buckets[0].output_tps, None);
    }

    /// Prompt caching: `prompt_tokens` is the full billed prompt (8000) but
    /// only `prompt_eval_tokens` (100) were actually evaluated. Input TPS must
    /// use the evaluated count, not the billed one — else 8000 / 200 ms =
    /// 40 000 tok/s instead of the true 100 / 200 ms = 500. Effective TPS
    /// sidesteps this entirely: it counts only completion tokens.
    #[tokio::test]
    async fn tps_uses_evaluated_prompt_tokens_not_billed() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        db.insert_request_metric(&RequestMetric {
            metric_id: 0,
            prompt_eval_tokens: Some(100),
            service_id: svc,
            run_id: Some(1),
            timestamp_ms: 0,
            endpoint: "/v1/chat/completions".into(),
            model: "demo".into(),
            prompt_tokens: Some(8000),
            completion_tokens: Some(50),
            duration_ms: Some(1000),
            ttft_ms: None,
            prompt_ms: Some(200),
            predicted_ms: Some(500),
            status_code: 200,
        })
        .await
        .unwrap();

        let b = &db
            .query_request_metrics(Some(svc), 0, 60_000, 60_000)
            .await
            .unwrap()[0];
        let close = |a: Option<f64>, b: f64| (a.unwrap() - b).abs() < 0.01;
        // input = 100 evaluated / 0.2 s = 500 (not 8000 / 0.2 = 40 000).
        assert!(close(b.input_tps, 500.0), "input_tps = {:?}", b.input_tps);
        // effective = completion only: 50 / 1 s = 50 (prompt tokens, cached or
        // billed, never enter the effective throughput).
        assert!(
            close(b.effective_tps, 50.0),
            "effective_tps = {:?}",
            b.effective_tps
        );
        // output is unaffected by prompt caching: 50 / 0.5 s = 100.
        assert!(
            close(b.output_tps, 100.0),
            "output_tps = {:?}",
            b.output_tps
        );
        // The displayed prompt-token total stays the billed count.
        assert_eq!(b.prompt_tokens, 8000);
    }

    #[tokio::test]
    async fn request_metrics_filter_by_service() {
        let db = Database::open_in_memory().await.unwrap();
        let svc_a = db.upsert_service("alpha", 1000).await.unwrap();
        let svc_b = db.upsert_service("beta", 2000).await.unwrap();

        for svc_id in [svc_a, svc_b] {
            db.insert_request_metric(&RequestMetric {
                metric_id: 0,
                prompt_eval_tokens: None,
                service_id: svc_id,
                run_id: Some(1),
                timestamp_ms: 1000,
                endpoint: "/v1/chat/completions".into(),
                model: "test".into(),
                prompt_tokens: Some(10),
                completion_tokens: Some(5),
                duration_ms: None,
                ttft_ms: None,
                prompt_ms: None,
                predicted_ms: None,
                status_code: 200,
            })
            .await
            .unwrap();
        }

        // Query all services (None) — should get one bucket per service.
        let all = db
            .query_request_metrics(None, 0, 10_000, 60_000)
            .await
            .unwrap();
        assert_eq!(all.len(), 2);
        // Buckets are ordered by (bucket, service_name) — alpha before beta.
        assert_eq!(all[0].service.as_deref(), Some("alpha"));
        assert_eq!(all[0].request_count, 1);
        assert_eq!(all[1].service.as_deref(), Some("beta"));
        assert_eq!(all[1].request_count, 1);

        // Query only svc_a.
        let filtered = db
            .query_request_metrics(Some(svc_a), 0, 10_000, 60_000)
            .await
            .unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].service.as_deref(), Some("alpha"));
        assert_eq!(filtered[0].request_count, 1);
    }

    #[tokio::test]
    async fn request_metrics_prune_old() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 1000).await.unwrap();
        db.insert_request_metric(&RequestMetric {
            metric_id: 0,
            prompt_eval_tokens: None,
            service_id: svc,
            run_id: Some(1),
            timestamp_ms: 100,
            endpoint: "/v1/chat/completions".into(),
            model: "demo".into(),
            prompt_tokens: None,
            completion_tokens: None,
            duration_ms: None,
            ttft_ms: None,
            prompt_ms: None,
            predicted_ms: None,
            status_code: 200,
        })
        .await
        .unwrap();
        db.insert_request_metric(&RequestMetric {
            metric_id: 0,
            prompt_eval_tokens: None,
            service_id: svc,
            run_id: Some(1),
            timestamp_ms: 2000,
            endpoint: "/v1/chat/completions".into(),
            model: "demo".into(),
            prompt_tokens: None,
            completion_tokens: None,
            duration_ms: None,
            ttft_ms: None,
            prompt_ms: None,
            predicted_ms: None,
            status_code: 200,
        })
        .await
        .unwrap();

        let deleted = db.prune_request_metrics(1000).await.unwrap();
        assert_eq!(deleted, 1);

        let remaining = db
            .query_request_metrics(Some(svc), 0, 10_000, 60_000)
            .await
            .unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].request_count, 1);
    }

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
