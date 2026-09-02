//! SQLite connection and file-level policy.
//!
//! Connection-local settings (`foreign_keys`, `busy_timeout`, and
//! `synchronous`) are applied to every connection. File initialization sets
//! `auto_vacuum` before schema creation and negotiates persistent WAL.

use std::time::Duration;

use rusqlite::Connection;

/// Apply settings that are local to this SQLite connection.
pub fn configure_connection(conn: &Connection) -> rusqlite::Result<()> {
    conn.pragma_update(None, "foreign_keys", true)?;
    conn.busy_timeout(Duration::from_secs(5))?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    Ok(())
}

/// Configure persistent file settings before the schema is created.
///
/// A populated legacy database keeps its existing auto-vacuum mode because
/// changing it requires a full SQLite rewrite. Startup never performs that
/// rewrite implicitly.
pub fn configure_file(conn: &Connection, is_new: bool) -> rusqlite::Result<()> {
    let auto_vacuum: i64 = conn.query_row("PRAGMA auto_vacuum", [], |r| r.get(0))?;
    if is_new {
        conn.pragma_update(None, "auto_vacuum", "INCREMENTAL")?;
        let configured: i64 = conn.query_row("PRAGMA auto_vacuum", [], |r| r.get(0))?;
        if configured != 2 {
            return Err(rusqlite::Error::InvalidQuery);
        }
    } else if !(0..=2).contains(&auto_vacuum) {
        return Err(rusqlite::Error::InvalidQuery);
    }
    let mode: String = conn.query_row("PRAGMA journal_mode = WAL", [], |r| r.get(0))?;
    if !mode.eq_ignore_ascii_case("wal") {
        return Err(rusqlite::Error::InvalidQuery);
    }
    Ok(())
}

pub fn incremental_vacuum_connection(conn: &Connection, pages: u64) -> rusqlite::Result<()> {
    conn.execute_batch(&format!("PRAGMA incremental_vacuum({pages})"))?;
    Ok(())
}
