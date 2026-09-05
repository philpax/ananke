//! Service lifecycle rows: create, rename/tombstone, and resolve
//! `services` table entries.

use ananke_errors::ExpectedError;
use rusqlite::{OptionalExtension, params};

use crate::{Database, models::Service};

impl Database {
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

    /// All services, including tombstones, for retention's per-service cap.
    pub async fn list_services_for_retention(&self) -> Result<Vec<Service>, ExpectedError> {
        let conn = self.conn.lock();
        let sql = format!("SELECT {} FROM services", Service::COLUMNS);
        let mut stmt = conn.prepare(&sql).map_err(|e| self.db_err(e))?;
        let rows = stmt
            .query_map([], Service::from_row)
            .map_err(|e| self.db_err(e))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| self.db_err(e))
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
        let live = db.list_services_for_retention().await.unwrap();
        let live: Vec<_> = live
            .into_iter()
            .filter(|s| s.deleted_at.is_none())
            .collect();
        let live_names: Vec<_> = live.iter().map(|s| s.name.as_str()).collect();
        assert!(live_names.contains(&"new-name"));
        assert!(!live_names.contains(&"old-name"));
    }
}
