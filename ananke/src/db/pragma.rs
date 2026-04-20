//! Side-channel for the one pragma that doesn't belong in the migration
//! blob: `incremental_vacuum(N)` is issued by the retention loop on a
//! fresh connection so it doesn't contend with other writers.
//!
//! The file-persistent pragmas (`auto_vacuum = INCREMENTAL`,
//! `journal_mode = WAL`, `synchronous = NORMAL`, `foreign_keys = ON`)
//! live at the top of [`crate::db::schema::MIGRATION_0001`] and are
//! applied on every `Database::open`.

use std::path::Path;

use rusqlite::Connection;

pub fn incremental_vacuum(path: &Path, pages: u64) -> rusqlite::Result<()> {
    let conn = Connection::open(path)?;
    conn.execute_batch(&format!("PRAGMA incremental_vacuum({pages})"))?;
    Ok(())
}
