//! Database row types. Plain Rust structs with `from_row` helpers; no ORM.
//!
//! Kept here rather than inlined into queries so the row shape and its mapping
//! from `rusqlite::Row` are defined in one place and every caller reads the
//! same definition.

use rusqlite::Row;

#[derive(Debug, Clone)]
pub struct Service {
    pub service_id: i64,
    pub name: String,
    pub created_at: i64,
    pub deleted_at: Option<i64>,
}

impl Service {
    pub fn from_row(row: &Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            service_id: row.get(0)?,
            name: row.get(1)?,
            created_at: row.get(2)?,
            deleted_at: row.get(3)?,
        })
    }

    /// Columns in the order `from_row` expects.
    pub const COLUMNS: &'static str = "service_id, name, created_at, deleted_at";
}

#[derive(Debug, Clone)]
pub struct RunningService {
    pub service_id: i64,
    pub run_id: i64,
    pub pid: i64,
    pub spawned_at: i64,
    pub command_line: String,
    pub allocation: String,
    pub state: String,
}

impl RunningService {
    pub fn from_row(row: &Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            service_id: row.get(0)?,
            run_id: row.get(1)?,
            pid: row.get(2)?,
            spawned_at: row.get(3)?,
            command_line: row.get(4)?,
            allocation: row.get(5)?,
            state: row.get(6)?,
        })
    }

    pub const COLUMNS: &'static str =
        "service_id, run_id, pid, spawned_at, command_line, allocation, state";
}

#[derive(Debug, Clone)]
pub struct ServiceLog {
    pub service_id: i64,
    pub run_id: i64,
    pub timestamp_ms: i64,
    pub seq: i64,
    pub stream: String,
    pub line: String,
}

impl ServiceLog {
    pub fn from_row(row: &Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            service_id: row.get(0)?,
            run_id: row.get(1)?,
            timestamp_ms: row.get(2)?,
            seq: row.get(3)?,
            stream: row.get(4)?,
            line: row.get(5)?,
        })
    }

    pub const COLUMNS: &'static str = "service_id, run_id, timestamp_ms, seq, stream, line";
}
