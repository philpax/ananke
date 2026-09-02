//! SQLite-backed database. One process-wide [`Arc<Mutex<Connection>>`]
//! per [`Database`]; every query takes the short critical section and
//! returns. Schema is applied on open via the versioned migration chain
//! in [`migrations`], so re-opening an already-provisioned file applies
//! only pending migrations (empty set when up to date).

mod container;
mod devices;
#[cfg(target_os = "linux")]
mod lock;
mod metrics;
mod metrics_query;
mod oneshots;
mod restarts;
mod running;
mod service_logs;
mod services;

pub mod logs;
pub mod migrations;
pub mod models;
pub mod pragma;
pub mod retention;

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use ananke_errors::ExpectedError;
pub use metrics_query::MetricBucket;
use parking_lot::Mutex;
pub use restarts::SpecAcceptance;
use rusqlite::Connection;

/// Cloneable database handle. All queries go through the shared
/// `Connection` behind a `parking_lot::Mutex`. Lock durations stay in
/// the microsecond range because SQLite on local disk is fast and
/// nothing holds the lock across `.await` points.
#[derive(Clone)]
pub struct Database {
    conn: Arc<Mutex<Connection>>,
    path: PathBuf,
    #[cfg(target_os = "linux")]
    _lock: Option<Arc<lock::DatabaseLock>>,
}

impl Database {
    pub async fn open(path: &Path) -> Result<Self, ExpectedError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                ExpectedError::database_open_failed(path.to_path_buf(), e.to_string())
            })?;
        }
        let normalized = {
            #[cfg(target_os = "linux")]
            {
                lock::normalize_path(path, false).map_err(|e| {
                    ExpectedError::database_open_failed(path.to_path_buf(), e.to_string())
                })?
            }
            #[cfg(not(target_os = "linux"))]
            {
                path.to_path_buf()
            }
        };
        let is_new =
            !normalized.exists() || normalized.metadata().map(|m| m.len() == 0).unwrap_or(true);
        #[cfg(target_os = "linux")]
        let lock =
            Arc::new(lock::DatabaseLock::acquire(&normalized).map_err(|e| {
                ExpectedError::database_open_failed(normalized.clone(), e.to_string())
            })?);
        let mut conn = Connection::open(&normalized)
            .map_err(|e| ExpectedError::database_open_failed(normalized.clone(), e.to_string()))?;
        pragma::configure_connection(&conn)
            .and_then(|_| pragma::configure_file(&conn, is_new))
            .map_err(|e| ExpectedError::database_open_failed(normalized.clone(), e.to_string()))?;
        migrations::apply_pending(&mut conn, ananke_time::now_unix_ms())
            .map_err(|e| ExpectedError::database_open_failed(normalized.clone(), e.to_string()))?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            path: normalized,
            #[cfg(target_os = "linux")]
            _lock: Some(lock),
        })
    }

    /// Open a `:memory:` database with the same schema applied. Used by
    /// tests that want a full DB surface without touching disk.
    pub async fn open_in_memory() -> Result<Self, ExpectedError> {
        let mut conn = Connection::open_in_memory().map_err(|e| {
            ExpectedError::database_open_failed(PathBuf::from(":memory:"), e.to_string())
        })?;
        pragma::configure_connection(&conn).map_err(|e| {
            ExpectedError::database_open_failed(PathBuf::from(":memory:"), e.to_string())
        })?;
        migrations::apply_pending(&mut conn, ananke_time::now_unix_ms()).map_err(|e| {
            ExpectedError::database_open_failed(PathBuf::from(":memory:"), e.to_string())
        })?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            path: PathBuf::from(":memory:"),
            #[cfg(target_os = "linux")]
            _lock: None,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn with_connection<T>(
        &self,
        f: impl FnOnce(&Connection) -> rusqlite::Result<T>,
    ) -> rusqlite::Result<T> {
        let conn = self.conn.lock();
        f(&conn)
    }

    fn db_err(&self, e: rusqlite::Error) -> ExpectedError {
        ExpectedError::database_open_failed(self.path.clone(), e.to_string())
    }
}
