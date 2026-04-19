//! Database models. Schema is owned by these derives; toasty's
//! `Db::push_schema()` issues the CREATE TABLE / CREATE INDEX statements
//! derived from them at bootstrap.

#[derive(Debug, toasty::Model)]
pub struct Service {
    #[key]
    #[auto]
    pub service_id: i64,

    #[unique]
    pub name: String,

    pub created_at: i64,

    pub deleted_at: Option<i64>,
}

#[derive(Debug, toasty::Model)]
pub struct ServiceConfigVersion {
    #[key]
    pub service_id: i64,
    #[key]
    pub version: i64,
    pub effective_config: String,
    pub recorded_at: i64,
}

#[derive(Debug, toasty::Model)]
pub struct RunningService {
    #[key]
    pub service_id: i64,
    #[key]
    pub run_id: i64,
    pub pid: i64,
    pub spawned_at: i64,
    pub command_line: String,
    pub allocation: String,
    pub state: String,
}

#[derive(Debug, toasty::Model)]
pub struct ServiceLog {
    #[key]
    pub service_id: i64,
    #[key]
    pub run_id: i64,
    #[key]
    pub seq: i64,
    #[index]
    pub timestamp_ms: i64,
    pub stream: String,
    pub line: String,
}

#[derive(Debug, toasty::Model)]
pub struct AllocationEvent {
    #[key]
    #[auto]
    pub event_id: i64,

    #[index]
    pub service_id: i64,
    pub run_id: i64,
    pub event_type: String,
    pub device: String,
    pub bytes: i64,
    pub at: i64,
}

#[derive(Debug, toasty::Model)]
pub struct Oneshot {
    #[key]
    pub id: String,

    #[index]
    pub service_id: i64,
    pub submitted_at: i64,
    pub started_at: Option<i64>,
    pub ended_at: Option<i64>,
    pub exit_code: Option<i32>,
    pub ttl_ms: i64,
}
