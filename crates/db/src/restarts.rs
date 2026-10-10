//! Auto-restart watchdog persistence: per-run error/acceptance queries
//! feeding the watchdogs, plus the restart-firing history and its
//! monotonic per-trigger counters.

use ananke_errors::ExpectedError;
use rusqlite::params;

use crate::{Database, models::ServiceRestart};

/// Per-service cap on persisted auto-restart firings. Enforced at insert
/// time by [`Database::insert_service_restart`].
const SERVICE_RESTART_CAP: u32 = 50;

/// Result of [`Database::spec_acceptance_since`]: draft-acceptance figures
/// for the recent window plus the run's lifetime accepted total.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecAcceptance {
    /// Drafting requests whose completion falls inside the window.
    pub window_drafting: u64,
    /// Draft tokens proposed across the window's drafting requests. The
    /// spec-collapse floor counts tokens rather than requests: long
    /// generations arrive slowly but draft thousands of tokens each, so a
    /// request floor would starve on exactly the traffic a wedge produces.
    pub window_drafted: u64,
    /// Draft tokens accepted across the window's drafting requests.
    pub window_accepted: u64,
    /// Draft tokens accepted across the whole run.
    pub run_accepted: u64,
}

impl Database {
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

    /// Draft-acceptance figures for one run, for the spec_collapse
    /// auto-restart watchdog. A request "drafts" when the engine reported a
    /// positive `draft_tokens`; rows without draft columns (no speculative
    /// decoding, non-llama engines, pre-migration rows) never count. Scoped
    /// to `run_id` for the same reason as [`Self::error_rate_since`].
    ///
    /// The recency window is keyed on request *completion* time
    /// (`timestamp_ms + duration_ms`) rather than the start-stamped
    /// `timestamp_ms`: rows only appear at completion, and the failure this
    /// watchdog detects produces long garbage generations, so a start-keyed
    /// window would exclude exactly the requests that evidence the wedge.
    ///
    /// `run_accepted` sums acceptance over the whole run so the caller can
    /// require a collapse (prior acceptance, then none) rather than firing
    /// on a workload that never accepts.
    pub async fn spec_acceptance_since(
        &self,
        service_id: i64,
        run_id: i64,
        since_ms: i64,
    ) -> Result<SpecAcceptance, ExpectedError> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT SUM(CASE WHEN end_ms >= ?3 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN end_ms >= ?3 THEN drafted ELSE 0 END),
                    SUM(CASE WHEN end_ms >= ?3 THEN accepted ELSE 0 END),
                    SUM(accepted)
             FROM (SELECT timestamp_ms + COALESCE(duration_ms, 0) AS end_ms,
                          draft_tokens AS drafted,
                          COALESCE(draft_tokens_accepted, 0) AS accepted
                   FROM request_metrics
                   WHERE service_id = ?1 AND run_id = ?2 AND draft_tokens > 0)",
            params![service_id, run_id, since_ms],
            |row| {
                // SUM over zero rows is NULL, hence the Options.
                let get = |i: usize| {
                    Ok::<_, rusqlite::Error>(
                        row.get::<_, Option<i64>>(i)?.unwrap_or(0).max(0) as u64
                    )
                };
                Ok(SpecAcceptance {
                    window_drafting: get(0)?,
                    window_drafted: get(1)?,
                    window_accepted: get(2)?,
                    run_accepted: get(3)?,
                })
            },
        )
        .map_err(|e| self.db_err(e))
    }

    /// Persist one auto-restart watchdog firing and prune the per-service
    /// history to the newest [`SERVICE_RESTART_CAP`] rows. The cap-at-insert
    /// keeps the table bounded without a background sweeper; restarts are
    /// rare enough (the flap cap disables a service after a handful) that
    /// the extra DELETE per insert is negligible.
    ///
    /// The monotonic `service_restart_counts` tally is bumped in the same
    /// breath, because the capped history cannot serve a counter: eviction
    /// is shared across triggers, so a per-trigger count read off it can
    /// fall. See migration `0007_service_restart_counts`.
    pub async fn insert_service_restart(&self, row: &ServiceRestart) -> Result<(), ExpectedError> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO service_restarts
                 (service_id, run_id, at_ms, trigger_name, detail)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                row.service_id,
                row.run_id,
                row.at_ms,
                row.trigger,
                row.detail,
            ],
        )
        .map_err(|e| self.db_err(e))?;
        conn.execute(
            "INSERT INTO service_restart_counts (service_id, trigger_name, count)
             VALUES (?1, ?2, 1)
             ON CONFLICT(service_id, trigger_name)
             DO UPDATE SET count = count + 1",
            params![row.service_id, row.trigger],
        )
        .map_err(|e| self.db_err(e))?;
        conn.execute(
            "DELETE FROM service_restarts
             WHERE service_id = ?1
               AND restart_id NOT IN (
                   SELECT restart_id FROM service_restarts
                   WHERE service_id = ?1
                   ORDER BY at_ms DESC, restart_id DESC
                   LIMIT ?2
               )",
            params![row.service_id, SERVICE_RESTART_CAP],
        )
        .map_err(|e| self.db_err(e))?;
        let _ = self.changed.send(crate::DatabaseChange::Restarts);
        Ok(())
    }

    /// Per-trigger counts of auto-restart firings for a service, for the
    /// Prometheus `ananke_auto_restarts_total` counter. Read from the
    /// monotonic `service_restart_counts` tally rather than the capped
    /// `service_restarts` history, so the value never decreases — a falling
    /// counter reads as a reset to Prometheus and would fabricate a restart
    /// spike out of a history eviction.
    pub async fn count_service_restarts_by_trigger(
        &self,
        service_id: i64,
    ) -> Result<Vec<(String, u64)>, ExpectedError> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT trigger_name, count
                 FROM service_restart_counts
                 WHERE service_id = ?1",
            )
            .map_err(|e| self.db_err(e))?;
        let rows = stmt
            .query_map([service_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?.max(0) as u64,
                ))
            })
            .map_err(|e| self.db_err(e))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| self.db_err(e))
    }

    /// Auto-restart firings within a time window across services, oldest
    /// first, for the `/api/restarts` chart endpoint. `service_id` narrows
    /// to one service. Returns `(service_name, firing)` pairs; firings of
    /// tombstoned services are included as long as their rows survive the
    /// per-service cap.
    pub async fn query_service_restarts(
        &self,
        service_id: Option<i64>,
        since_ms: i64,
        until_ms: i64,
    ) -> Result<Vec<(String, ServiceRestart)>, ExpectedError> {
        let conn = self.conn.lock();
        let sql = format!(
            "SELECT s.name, {cols}
             FROM service_restarts r
             JOIN services s ON s.service_id = r.service_id
             WHERE r.at_ms >= ?1 AND r.at_ms <= ?2
               AND (?3 IS NULL OR r.service_id = ?3)
             ORDER BY r.at_ms ASC, r.restart_id ASC",
            cols = ServiceRestart::COLUMNS
                .split(", ")
                .map(|c| format!("r.{c}"))
                .collect::<Vec<_>>()
                .join(", "),
        );
        let mut stmt = conn.prepare(&sql).map_err(|e| self.db_err(e))?;
        let rows = stmt
            .query_map(params![since_ms, until_ms, service_id], |row| {
                let name: String = row.get(0)?;
                // ServiceRestart::from_row reads columns 0.., but the name
                // occupies column 0 here — shift by re-reading explicitly.
                Ok((
                    name,
                    ServiceRestart {
                        restart_id: row.get(1)?,
                        service_id: row.get(2)?,
                        run_id: row.get(3)?,
                        at_ms: row.get(4)?,
                        trigger: row.get(5)?,
                        detail: row.get(6)?,
                    },
                ))
            })
            .map_err(|e| self.db_err(e))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| self.db_err(e))
    }

    /// The newest auto-restart firings for a service, most recent first.
    /// Ordered by wall-clock time rather than insertion order, so the
    /// history reads chronologically even if two inserts ever land out of
    /// order; `restart_id` only breaks ties within a millisecond.
    pub async fn recent_service_restarts(
        &self,
        service_id: i64,
        limit: u32,
    ) -> Result<Vec<ServiceRestart>, ExpectedError> {
        let conn = self.conn.lock();
        let sql = format!(
            "SELECT {} FROM service_restarts
             WHERE service_id = ?1
             ORDER BY at_ms DESC, restart_id DESC
             LIMIT ?2",
            ServiceRestart::COLUMNS
        );
        let mut stmt = conn.prepare(&sql).map_err(|e| self.db_err(e))?;
        let rows = stmt
            .query_map(params![service_id, limit], ServiceRestart::from_row)
            .map_err(|e| self.db_err(e))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| self.db_err(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::RequestMetric;

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
                    draft_tokens: None,
                    draft_tokens_accepted: None,
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

    /// The Prometheus counter must never fall. The stored history is capped
    /// per service across all triggers, so a service that restarts
    /// periodically evicts its watchdog rows; the tally the counter reads
    /// has to survive that, or a dashboard shows a phantom restart spike
    /// where an eviction happened.
    #[tokio::test]
    async fn restart_counts_are_monotonic_across_history_eviction() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let insert = |trigger: &'static str, at_ms: i64| {
            let db = &db;
            async move {
                db.insert_service_restart(&ServiceRestart {
                    restart_id: 0,
                    service_id: svc,
                    run_id: Some(1),
                    at_ms,
                    trigger: trigger.to_string(),
                    detail: "d".into(),
                })
                .await
                .unwrap();
            }
        };

        insert("spec_collapse", 1_000).await;
        // Enough periodic firings to evict every row of the older trigger.
        for i in 0..SERVICE_RESTART_CAP as i64 {
            insert("periodic", 2_000 + i).await;
        }

        // The history has forgotten the spec_collapse firing entirely...
        let recent = db
            .recent_service_restarts(svc, SERVICE_RESTART_CAP)
            .await
            .unwrap();
        assert_eq!(recent.len(), SERVICE_RESTART_CAP as usize);
        assert!(recent.iter().all(|r| r.trigger == "periodic"));

        // ...but the counter still reports it, and does not go backwards.
        let counts: std::collections::HashMap<String, u64> = db
            .count_service_restarts_by_trigger(svc)
            .await
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(counts.get("spec_collapse"), Some(&1));
        assert_eq!(counts.get("periodic"), Some(&(SERVICE_RESTART_CAP as u64)));
    }

    #[tokio::test]
    async fn spec_acceptance_since_scopes_and_counts_drafting_rows() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let insert = |run_id: i64, timestamp_ms: i64, draft: Option<i64>, accepted: Option<i64>| {
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
                    draft_tokens: draft,
                    draft_tokens_accepted: accepted,
                    status_code: 200,
                })
                .await
                .unwrap();
            }
        };

        // Current run 2, in-window: the incident shape — every drafting
        // request rejected wholesale — plus a non-drafting row that must not
        // count either way.
        insert(2, 100_000, Some(59), Some(0)).await;
        insert(2, 101_000, Some(11), Some(0)).await;
        insert(2, 102_000, None, None).await;
        // A previous run's healthy acceptance must not leak into run 2.
        insert(1, 100_500, Some(100), Some(80)).await;
        // A healthy row before the `since` cutoff: outside the window, but
        // still counted in the run's lifetime acceptance.
        insert(2, 50_000, Some(20), Some(18)).await;

        let a = db.spec_acceptance_since(svc, 2, 90_000).await.unwrap();
        assert_eq!(
            (
                a.window_drafting,
                a.window_drafted,
                a.window_accepted,
                a.run_accepted
            ),
            (2, 70, 0, 18)
        );

        // One accepted token inside the window shows up in both sums.
        insert(2, 103_000, Some(14), Some(12)).await;
        let a = db.spec_acceptance_since(svc, 2, 90_000).await.unwrap();
        assert_eq!(
            (
                a.window_drafting,
                a.window_drafted,
                a.window_accepted,
                a.run_accepted
            ),
            (3, 84, 12, 30)
        );

        // Empty window → zeros (SUM over empty is NULL); the run total stays.
        let a = db.spec_acceptance_since(svc, 2, 200_000).await.unwrap();
        assert_eq!(
            (
                a.window_drafting,
                a.window_drafted,
                a.window_accepted,
                a.run_accepted
            ),
            (0, 0, 0, 30)
        );
    }

    /// The window keys on completion time (`timestamp_ms + duration_ms`),
    /// not the start-stamped `timestamp_ms`: a long generation that started
    /// before the window but finished inside it must count.
    #[tokio::test]
    async fn spec_acceptance_windows_on_completion_time() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        db.insert_request_metric(&RequestMetric {
            metric_id: 0,
            prompt_eval_tokens: None,
            service_id: svc,
            run_id: Some(1),
            // Started at 50s, ran for 5 minutes: completion at 350s.
            timestamp_ms: 50_000,
            endpoint: "/v1/chat/completions".into(),
            model: "demo".into(),
            prompt_tokens: None,
            completion_tokens: None,
            duration_ms: Some(300_000),
            ttft_ms: None,
            prompt_ms: None,
            predicted_ms: None,
            draft_tokens: Some(59),
            draft_tokens_accepted: Some(0),
            status_code: 200,
        })
        .await
        .unwrap();

        // A start-keyed window from 300s would miss the row; the
        // completion-keyed window counts it.
        let a = db.spec_acceptance_since(svc, 1, 300_000).await.unwrap();
        assert_eq!(a.window_drafting, 1);

        // A window opening after the completion excludes it.
        let a = db.spec_acceptance_since(svc, 1, 400_000).await.unwrap();
        assert_eq!(a.window_drafting, 0);
    }
}
