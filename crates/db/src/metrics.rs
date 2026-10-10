//! Write and retention side of per-request metrics
//! (`request_metrics` table): insert on request completion, activity-
//! table seeding on boot, and retention pruning. The time-series query
//! side lives in the sibling `metrics_query` module.

use ananke_errors::ExpectedError;
use rusqlite::params;

use crate::{Database, models::RequestMetric};

impl Database {
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

    /// Insert a single request metric row. Called by the proxy after the
    /// response stream completes.
    pub async fn insert_request_metric(&self, row: &RequestMetric) -> Result<(), ExpectedError> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO request_metrics
                 (service_id, run_id, timestamp_ms, endpoint, model,
                  prompt_tokens, completion_tokens, prompt_eval_tokens,
                  duration_ms, ttft_ms, prompt_ms, predicted_ms,
                  draft_tokens, draft_tokens_accepted, status_code)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
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
                row.draft_tokens,
                row.draft_tokens_accepted,
                row.status_code,
            ],
        )
        .map_err(|e| self.db_err(e))?;
        let _ = self.changed.send(crate::DatabaseChange::Metrics);
        Ok(())
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
            draft_tokens: None,
            draft_tokens_accepted: None,
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
            draft_tokens: None,
            draft_tokens_accepted: None,
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
}
