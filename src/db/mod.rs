//! Database bootstrap and migrations.

pub mod logs;
pub mod models;
pub mod schema;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;
use rusqlite::Connection;

use crate::errors::ExpectedError;

#[derive(Clone)]
pub struct Database {
    conn: Arc<Mutex<Connection>>,
    path: PathBuf,
}

impl Database {
    pub fn open(path: &Path) -> Result<Self, ExpectedError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                ExpectedError::database_open_failed(path.to_path_buf(), e.to_string())
            })?;
        }
        let conn = Connection::open(path)
            .map_err(|e| ExpectedError::database_open_failed(path.to_path_buf(), e.to_string()))?;
        conn.execute_batch(schema::MIGRATION_0001)
            .map_err(|e| ExpectedError::database_open_failed(path.to_path_buf(), e.to_string()))?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            path: path.to_path_buf(),
        })
    }

    pub fn schema_version(&self) -> Result<u32, ExpectedError> {
        let conn = self.conn.lock();
        let v: u32 = conn
            .query_row(
                "SELECT COALESCE(MAX(version), 0) FROM schema_version",
                [],
                |row| row.get(0),
            )
            .map_err(|e| ExpectedError::database_open_failed(self.path.clone(), e.to_string()))?;
        Ok(v)
    }

    pub fn upsert_service(&self, name: &str, now_ms: i64) -> Result<i64, ExpectedError> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO services(name, created_at, deleted_at) VALUES (?1, ?2, NULL)
             ON CONFLICT(name) DO UPDATE SET deleted_at = NULL",
            (name, now_ms),
        )
        .map_err(|e| ExpectedError::database_open_failed(self.path.clone(), e.to_string()))?;
        let id: i64 = conn
            .query_row(
                "SELECT service_id FROM services WHERE name = ?1",
                [name],
                |row| row.get(0),
            )
            .map_err(|e| ExpectedError::database_open_failed(self.path.clone(), e.to_string()))?;
        Ok(id)
    }

    /// Reparent a `service_id` from `old_name` to `new_name`. Spec §6.4.
    ///
    /// - If old_name has a live service_id and new_name does not, rename old to new.
    /// - If both exist, tombstone old (set deleted_at) — the new service keeps its own id.
    ///   This matches "migrate_from naming a service that also exists live in the same config → error",
    ///   which is enforced in validation; here we handle the post-load case where old was tombstoned previously.
    /// - If old_name doesn't exist, no-op.
    pub fn reparent(
        &self,
        old_name: &str,
        new_name: &str,
        now_ms: i64,
    ) -> Result<(), ExpectedError> {
        let conn = self.conn.lock();
        let old_id: Option<i64> = conn
            .query_row(
                "SELECT service_id FROM services WHERE name = ?1",
                [old_name],
                |row| row.get(0),
            )
            .ok();
        let new_id: Option<i64> = conn
            .query_row(
                "SELECT service_id FROM services WHERE name = ?1",
                [new_name],
                |row| row.get(0),
            )
            .ok();
        match (old_id, new_id) {
            (Some(_), None) => {
                conn.execute(
                    "UPDATE services SET name = ?1 WHERE name = ?2",
                    (new_name, old_name),
                )
                .map_err(|e| {
                    ExpectedError::database_open_failed(self.path.clone(), e.to_string())
                })?;
            }
            (Some(_), Some(_)) => {
                conn.execute(
                    "UPDATE services SET deleted_at = ?1 WHERE name = ?2",
                    (now_ms, old_name),
                )
                .map_err(|e| {
                    ExpectedError::database_open_failed(self.path.clone(), e.to_string())
                })?;
            }
            (None, _) => {
                // No-op; see spec §6.4 "missing source is a warning".
            }
        }
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn with_conn<T>(
        &self,
        f: impl FnOnce(&Connection) -> rusqlite::Result<T>,
    ) -> rusqlite::Result<T> {
        let conn = self.conn.lock();
        f(&conn)
    }

    pub fn with_conn_mut<T>(
        &self,
        f: impl FnOnce(&mut Connection) -> rusqlite::Result<T>,
    ) -> rusqlite::Result<T> {
        let mut conn = self.conn.lock();
        f(&mut conn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn opens_and_migrates() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("ananke.sqlite")).unwrap();
        assert_eq!(db.schema_version().unwrap(), 1);
    }

    #[test]
    fn upsert_service_is_idempotent() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("a.sqlite")).unwrap();
        let id1 = db.upsert_service("demo", 1000).unwrap();
        let id2 = db.upsert_service("demo", 2000).unwrap();
        assert_eq!(id1, id2);
    }

    #[test]
    fn reparent_renames_when_only_old_exists() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("a.sqlite")).unwrap();
        let _ = db.upsert_service("old-name", 1000).unwrap();
        db.reparent("old-name", "new-name", 2000).unwrap();
        // new-name should now resolve; old-name should not.
        let new_id: i64 = db
            .with_conn(|c| {
                c.query_row(
                    "SELECT service_id FROM services WHERE name = 'new-name'",
                    [],
                    |r| r.get(0),
                )
            })
            .unwrap();
        let old_query: rusqlite::Result<i64> = db.with_conn(|c| {
            c.query_row(
                "SELECT service_id FROM services WHERE name = 'old-name'",
                [],
                |r| r.get(0),
            )
        });
        assert!(new_id > 0);
        assert!(old_query.is_err());
    }
}
