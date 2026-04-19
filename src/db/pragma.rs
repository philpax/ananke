//! Thin rusqlite side-channel for pragmas toasty cannot issue.
//!
//! Only two pragmas live here:
//! - `auto_vacuum = INCREMENTAL` — must run once on a fresh DB file before
//!   any `CREATE TABLE`. File-persistent.
//! - `incremental_vacuum(N)` — issued by retention to reclaim pages.
//!
//! Setting `journal_mode = WAL` here as well; it's file-persistent and
//! improves concurrent read/write behaviour for our workload.

use std::path::Path;

use rusqlite::Connection;

pub fn prepare_fresh_db(path: &Path) -> rusqlite::Result<()> {
    let conn = Connection::open(path)?;
    conn.execute_batch(
        "PRAGMA auto_vacuum = INCREMENTAL;\
         PRAGMA journal_mode = WAL;",
    )?;
    Ok(())
}

pub fn incremental_vacuum(path: &Path, pages: u64) -> rusqlite::Result<()> {
    let conn = Connection::open(path)?;
    conn.execute_batch(&format!("PRAGMA incremental_vacuum({pages})"))?;
    Ok(())
}
