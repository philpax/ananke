//! Query and lifecycle-management side of persisted service logs
//! (`service_logs` table): fetch for the paginated logs endpoint and for
//! log followers, batched insert, and retention pruning. The batching writer
//! that calls `Database::insert_log_batch` lives in the sibling `logs` module.

use ananke_api::internal::log_line::{LogLine, LogPosition, LogStream};
use ananke_errors::ExpectedError;
use rusqlite::{OptionalExtension, params, params_from_iter, types::Value};

use crate::{Database, models::ServiceLog};

/// Which of a service's lines a follower wants.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LogFilter {
    pub run: Option<i64>,
    pub stream: Option<LogStream>,
}

impl LogFilter {
    pub fn matches(&self, line: &LogLine) -> bool {
        self.run.is_none_or(|run| line.run_id == run)
            && self.stream.is_none_or(|s| line.stream == s.as_str())
    }
}

/// The newest matching lines after a position, oldest first.
#[derive(Debug, Default)]
pub struct LogsAfter {
    pub rows: Vec<LogLine>,
    /// Matching lines after the position that the cap left out. They are the
    /// oldest ones, immediately before `rows[0]`.
    pub skipped: u64,
}

impl Database {
    /// The newest `cap` lines of a service strictly after `after` that match
    /// `filter`. Runs under the connection lock, so it sees whole flushes.
    pub async fn fetch_logs_after(
        &self,
        service_id: i64,
        after: Option<LogPosition>,
        filter: &LogFilter,
        cap: usize,
    ) -> Result<LogsAfter, ExpectedError> {
        let (clause, args) = after_clause(service_id, after, filter);
        let conn = self.conn.lock();
        let sql = format!(
            "SELECT {} FROM service_logs WHERE {clause} \
             ORDER BY run_id DESC, seq DESC LIMIT {}",
            ServiceLog::COLUMNS,
            cap.saturating_add(1)
        );
        let mut stmt = conn.prepare(&sql).map_err(|e| self.db_err(e))?;
        let mut rows = stmt
            .query_map(params_from_iter(args.iter()), ServiceLog::from_row)
            .map_err(|e| self.db_err(e))?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| self.db_err(e))?;
        let skipped = if rows.len() > cap {
            rows.truncate(cap);
            let total: i64 = conn
                .query_row(
                    &format!("SELECT COUNT(*) FROM service_logs WHERE {clause}"),
                    params_from_iter(args.iter()),
                    |row| row.get(0),
                )
                .map_err(|e| self.db_err(e))?;
            u64::try_from(total).unwrap_or(0).saturating_sub(cap as u64)
        } else {
            0
        };
        rows.reverse();
        Ok(LogsAfter {
            rows: rows.into_iter().map(ServiceLog::into_line).collect(),
            skipped,
        })
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
}

/// The `WHERE` clause and its arguments for [`Database::fetch_logs_after`].
/// The position is a row-value comparison so SQLite can seek the primary key
/// `(service_id, run_id, seq)` rather than scan the service's whole log.
fn after_clause(
    service_id: i64,
    after: Option<LogPosition>,
    filter: &LogFilter,
) -> (String, Vec<Value>) {
    let mut clause = String::from("service_id = ?");
    let mut args = vec![Value::Integer(service_id)];
    if let Some(after) = after {
        clause.push_str(" AND (run_id, seq) > (?, ?)");
        args.extend([Value::Integer(after.run_id), Value::Integer(after.seq)]);
    }
    if let Some(run) = filter.run {
        clause.push_str(" AND run_id = ?");
        args.push(Value::Integer(run));
    }
    if let Some(stream) = filter.stream {
        clause.push_str(" AND stream = ?");
        args.push(Value::Text(stream.as_str().to_string()));
    }
    (clause, args)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(service_id: i64, run_id: i64, seq: i64, stream: &str) -> ServiceLog {
        ServiceLog {
            service_id,
            run_id,
            timestamp_ms: seq,
            seq,
            stream: stream.into(),
            line: format!("{run_id}:{seq}"),
        }
    }

    fn positions(page: &LogsAfter) -> Vec<(i64, i64)> {
        page.rows.iter().map(|l| (l.run_id, l.seq)).collect()
    }

    async fn seeded() -> (Database, i64) {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let other = db.upsert_service("other", 0).await.unwrap();
        db.insert_log_batch(&[
            row(svc, 1, 1, "stdout"),
            row(svc, 1, 2, "stderr"),
            row(svc, 1, 3, "stdout"),
            row(svc, 2, 1, "stderr"),
            row(svc, 2, 2, "stdout"),
            row(other, 3, 1, "stdout"),
        ])
        .await
        .unwrap();
        (db, svc)
    }

    #[tokio::test]
    async fn fetch_logs_after_is_oldest_first_and_strictly_after() {
        let (db, svc) = seeded().await;
        let all = db
            .fetch_logs_after(svc, None, &LogFilter::default(), 100)
            .await
            .unwrap();
        assert_eq!(positions(&all), [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]);
        assert_eq!(all.skipped, 0);

        let after = Some(LogPosition { run_id: 1, seq: 2 });
        let rest = db
            .fetch_logs_after(svc, after, &LogFilter::default(), 100)
            .await
            .unwrap();
        assert_eq!(positions(&rest), [(1, 3), (2, 1), (2, 2)]);
    }

    #[tokio::test]
    async fn fetch_logs_after_keeps_the_newest_and_counts_the_rest() {
        let (db, svc) = seeded().await;
        let page = db
            .fetch_logs_after(svc, None, &LogFilter::default(), 2)
            .await
            .unwrap();
        assert_eq!(positions(&page), [(2, 1), (2, 2)]);
        assert_eq!(page.skipped, 3);
    }

    #[tokio::test]
    async fn fetch_logs_after_applies_filters() {
        let (db, svc) = seeded().await;
        let stdout = LogFilter {
            run: None,
            stream: Some(LogStream::Stdout),
        };
        let page = db.fetch_logs_after(svc, None, &stdout, 1).await.unwrap();
        assert_eq!(positions(&page), [(2, 2)]);
        assert_eq!(page.skipped, 2);

        let run_one_stderr = LogFilter {
            run: Some(1),
            stream: Some(LogStream::Stderr),
        };
        let page = db
            .fetch_logs_after(svc, None, &run_one_stderr, 100)
            .await
            .unwrap();
        assert_eq!(positions(&page), [(1, 2)]);
        assert!(page.rows.iter().all(|l| run_one_stderr.matches(l)));
    }
}
