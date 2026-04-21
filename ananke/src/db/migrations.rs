//! Versioned schema migrations.
//!
//! Each [`Migration`] is an immutable (version, name, SQL) triple. At open
//! time, [`apply_pending`] bootstraps a `schema_version` table, reads which
//! versions are recorded there, and applies any unrecorded migration in
//! version order. Each migration runs inside a transaction alongside its
//! `schema_version` insert, so a mid-migration failure leaves the database
//! in its pre-migration state.
//!
//! Adding a migration:
//! 1. Drop a `NNNN_description.sql` file in `src/db/migrations/`.
//! 2. Append a [`Migration`] entry to [`MIGRATIONS`] with the next
//!    version number; version numbers are monotonic (gaps allowed but
//!    discouraged).
//! 3. Never mutate an existing entry — live databases have already
//!    recorded it as applied.

use std::collections::HashSet;

use rusqlite::{Connection, params};
use tracing::info;

/// One immutable schema change. Applied exactly once per database.
pub struct Migration {
    pub version: u32,
    pub name: &'static str,
    pub sql: &'static str,
}

/// Chain of migrations in application order. Append-only.
pub const MIGRATIONS: &[Migration] = &[Migration {
    version: 1,
    name: "initial_schema",
    sql: include_str!("migrations/0001_initial_schema.sql"),
}];

/// SQL applied before the first migration ever runs. Creates the version
/// tracker and sets the file-persistent pragmas. Idempotent — reruns every
/// open, which is fine because all operations here are no-ops on an
/// already-prepared file.
const BOOTSTRAP_SQL: &str = "\
PRAGMA auto_vacuum = INCREMENTAL;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS schema_version (
  version    INTEGER NOT NULL PRIMARY KEY,
  name       TEXT NOT NULL,
  applied_at INTEGER NOT NULL
);
";

/// Ensure the version tracker exists and apply any pending migrations.
/// Returns the versions that were applied this call (empty on a
/// fully-up-to-date database).
pub fn apply_pending(conn: &mut Connection, now_ms: i64) -> Result<Vec<u32>, rusqlite::Error> {
    conn.execute_batch(BOOTSTRAP_SQL)?;

    let applied: HashSet<u32> = {
        let mut stmt = conn.prepare("SELECT version FROM schema_version")?;
        let rows = stmt.query_map([], |r| r.get::<_, u32>(0))?;
        rows.collect::<rusqlite::Result<_>>()?
    };

    let mut applied_now = Vec::new();
    for m in MIGRATIONS {
        if applied.contains(&m.version) {
            continue;
        }
        let tx = conn.transaction()?;
        tx.execute_batch(m.sql)?;
        tx.execute(
            "INSERT INTO schema_version(version, name, applied_at) VALUES (?1, ?2, ?3)",
            params![m.version, m.name, now_ms],
        )?;
        tx.commit()?;
        info!(version = m.version, name = m.name, "applied migration");
        applied_now.push(m.version);
    }
    Ok(applied_now)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrations_apply_once_and_are_idempotent() {
        let mut conn = Connection::open_in_memory().unwrap();

        let first = apply_pending(&mut conn, 1_000).unwrap();
        assert!(!first.is_empty(), "fresh DB must apply at least one");

        let second = apply_pending(&mut conn, 2_000).unwrap();
        assert!(
            second.is_empty(),
            "re-open on an up-to-date DB must apply nothing, got {second:?}"
        );
    }

    #[test]
    fn versions_are_strictly_monotonic_and_unique() {
        let mut seen = HashSet::new();
        let mut last = 0;
        for m in MIGRATIONS {
            assert!(
                m.version > last,
                "migration {} ({}): versions must strictly increase",
                m.version,
                m.name
            );
            assert!(
                seen.insert(m.version),
                "duplicate version {} in MIGRATIONS",
                m.version
            );
            last = m.version;
        }
    }
}
