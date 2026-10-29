//! Operator-disabled services: the marks that keep a service disabled
//! across daemon restarts until the operator enables it again.

use ananke_errors::ExpectedError;
use rusqlite::{OptionalExtension, params};

use crate::Database;

impl Database {
    /// Mark a service as disabled by the operator. Disabling an already
    /// marked service keeps its original timestamp.
    pub async fn mark_service_disabled(
        &self,
        service_id: i64,
        now_ms: i64,
    ) -> Result<(), ExpectedError> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO disabled_services(service_id, disabled_at) VALUES (?1, ?2)
             ON CONFLICT(service_id) DO NOTHING",
            params![service_id, now_ms],
        )
        .map_err(|e| self.db_err(e))?;
        Ok(())
    }

    /// Clear a service's disabled mark. Returns whether one existed.
    pub async fn clear_service_disabled(&self, service_id: i64) -> Result<bool, ExpectedError> {
        let conn = self.conn.lock();
        let n = conn
            .execute(
                "DELETE FROM disabled_services WHERE service_id = ?1",
                params![service_id],
            )
            .map_err(|e| self.db_err(e))?;
        Ok(n > 0)
    }

    /// When the operator disabled a service, or `None` if it is not marked.
    pub async fn service_disabled_at(&self, service_id: i64) -> Result<Option<i64>, ExpectedError> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT disabled_at FROM disabled_services WHERE service_id = ?1",
            params![service_id],
            |row| row.get(0),
        )
        .optional()
        .map_err(|e| self.db_err(e))
    }
}

#[cfg(test)]
mod tests {
    use crate::Database;

    #[tokio::test]
    async fn mark_clear_roundtrip_keeps_first_timestamp() {
        let db = Database::open_in_memory().await.unwrap();
        let id = db.upsert_service("demo", 1).await.unwrap();

        assert_eq!(db.service_disabled_at(id).await.unwrap(), None);
        db.mark_service_disabled(id, 10).await.unwrap();
        db.mark_service_disabled(id, 20).await.unwrap();
        assert_eq!(db.service_disabled_at(id).await.unwrap(), Some(10));

        assert!(db.clear_service_disabled(id).await.unwrap());
        assert!(!db.clear_service_disabled(id).await.unwrap());
        assert_eq!(db.service_disabled_at(id).await.unwrap(), None);
    }

    #[tokio::test]
    async fn mark_survives_reopen_and_rename() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.sqlite");
        {
            let db = Database::open(&path).await.unwrap();
            let id = db.upsert_service("old", 1).await.unwrap();
            db.mark_service_disabled(id, 10).await.unwrap();
            db.reparent("old", "new", 2).await.unwrap();
        }
        let db = Database::open(&path).await.unwrap();
        let id = db.upsert_service("new", 3).await.unwrap();
        assert_eq!(db.service_disabled_at(id).await.unwrap(), Some(10));
    }

    /// Renaming onto a name that already has a row tombstones the old row
    /// and keeps the new id; the renamed service's disable state goes with it.
    #[tokio::test]
    async fn rename_onto_an_existing_name_moves_the_mark() {
        let db = Database::open_in_memory().await.unwrap();
        let old = db.upsert_service("old", 1).await.unwrap();
        let new = db.upsert_service("new", 1).await.unwrap();
        db.mark_service_disabled(old, 10).await.unwrap();
        db.reparent("old", "new", 2).await.unwrap();
        assert_eq!(db.service_disabled_at(new).await.unwrap(), Some(10));
        assert_eq!(db.service_disabled_at(old).await.unwrap(), None);

        // A stale mark on the new name doesn't outlive a rename from an
        // enabled service.
        let a = db.upsert_service("a", 3).await.unwrap();
        let b = db.upsert_service("b", 3).await.unwrap();
        db.mark_service_disabled(b, 10).await.unwrap();
        db.reparent("a", "b", 4).await.unwrap();
        assert_eq!(db.service_disabled_at(b).await.unwrap(), None);
        assert_eq!(db.service_disabled_at(a).await.unwrap(), None);
    }
}
