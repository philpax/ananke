//! Database bootstrap and shared toasty handle.

pub mod logs;
pub mod models;
pub mod pragma;
pub mod retention;

use std::path::{Path, PathBuf};

use toasty::Db;

use crate::errors::ExpectedError;

/// Shared database handle.
///
/// Wraps a [`toasty::Db`], which is itself cheaply cloneable — the underlying
/// connection pool and schema live behind an `Arc`.
#[derive(Clone)]
pub struct Database {
    db: Db,
    path: PathBuf,
}

impl Database {
    pub async fn open(path: &Path) -> Result<Self, ExpectedError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                ExpectedError::database_open_failed(path.to_path_buf(), e.to_string())
            })?;
        }

        pragma::prepare_fresh_db(path)
            .map_err(|e| ExpectedError::database_open_failed(path.to_path_buf(), e.to_string()))?;

        let url = format!("sqlite://{}", path.display());
        Self::connect(url, path.to_path_buf()).await
    }

    /// Open a purely in-memory database. Intended for tests that want to
    /// exercise the full DB surface (schema + queries) without touching
    /// disk. Skips the file-persistent pragmas (`auto_vacuum`,
    /// `journal_mode = WAL`) since they're meaningless for an in-memory
    /// handle.
    pub async fn open_in_memory() -> Result<Self, ExpectedError> {
        let synthetic = PathBuf::from(":memory:");
        Self::connect("sqlite::memory:".to_string(), synthetic).await
    }

    async fn connect(url: String, path: PathBuf) -> Result<Self, ExpectedError> {
        let db = Db::builder()
            .models(toasty::models!(
                models::Service,
                models::ServiceConfigVersion,
                models::RunningService,
                models::ServiceLog,
                models::AllocationEvent,
                models::Oneshot,
            ))
            .connect(&url)
            .await
            .map_err(|e| ExpectedError::database_open_failed(path.clone(), e.to_string()))?;

        db.push_schema()
            .await
            .map_err(|e| ExpectedError::database_open_failed(path.clone(), e.to_string()))?;

        Ok(Self { db, path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// A fresh, cheaply-cloned toasty handle suitable for `.exec(&mut db)`.
    pub fn handle(&self) -> Db {
        self.db.clone()
    }

    pub async fn upsert_service(&self, name: &str, now_ms: i64) -> Result<i64, ExpectedError> {
        use models::Service;

        let mut db = self.db.clone();
        let existing = Service::filter_by_name(name)
            .first()
            .exec(&mut db)
            .await
            .map_err(|e| ExpectedError::database_open_failed(self.path.clone(), e.to_string()))?;
        if let Some(mut svc) = existing {
            if svc.deleted_at.is_some() {
                svc.update()
                    .deleted_at(None)
                    .exec(&mut db)
                    .await
                    .map_err(|e| {
                        ExpectedError::database_open_failed(self.path.clone(), e.to_string())
                    })?;
            }
            return Ok(svc.service_id as i64);
        }
        let created = toasty::create!(Service {
            name: name.to_string(),
            created_at: now_ms,
        })
        .exec(&mut db)
        .await
        .map_err(|e| ExpectedError::database_open_failed(self.path.clone(), e.to_string()))?;
        Ok(created.service_id as i64)
    }

    /// Reparent a `service_id` from `old_name` to `new_name`. Spec §6.4.
    ///
    /// - If old_name has a live service_id and new_name does not, rename old to new.
    /// - If both exist, tombstone old (set deleted_at) — the new service keeps its own id.
    /// - If old_name doesn't exist, no-op.
    pub async fn reparent(
        &self,
        old_name: &str,
        new_name: &str,
        now_ms: i64,
    ) -> Result<(), ExpectedError> {
        use models::Service;

        let mut db = self.db.clone();
        let old = Service::filter_by_name(old_name)
            .first()
            .exec(&mut db)
            .await
            .map_err(|e| ExpectedError::database_open_failed(self.path.clone(), e.to_string()))?;
        let new = Service::filter_by_name(new_name)
            .first()
            .exec(&mut db)
            .await
            .map_err(|e| ExpectedError::database_open_failed(self.path.clone(), e.to_string()))?;
        match (old, new) {
            (Some(mut old), None) => {
                old.update()
                    .name(new_name.to_string())
                    .exec(&mut db)
                    .await
                    .map_err(|e| {
                        ExpectedError::database_open_failed(self.path.clone(), e.to_string())
                    })?;
            }
            (Some(mut old), Some(_)) => {
                old.update()
                    .deleted_at(Some(now_ms))
                    .exec(&mut db)
                    .await
                    .map_err(|e| {
                        ExpectedError::database_open_failed(self.path.clone(), e.to_string())
                    })?;
            }
            (None, _) => {}
        }
        Ok(())
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

        // old-name should resolve to a fresh row (it was renamed, so no row
        // exists under old-name); new-name should be the renamed row.
        let new_id = db.upsert_service("new-name", 3000).await.unwrap();
        assert_eq!(new_id, original);
    }
}
