//! Database row types. Plain Rust structs with `from_row` helpers; no ORM.
//!
//! Kept here rather than inlined into queries so the row shape and its mapping
//! from `rusqlite::Row` are defined in one place and every caller reads the
//! same definition.

use ananke_api::internal::log_line::LogLine;
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
    pub pid: Option<i64>,
    pub spawned_at: i64,
    pub command_line: String,
    pub allocation: String,
    pub state: String,
    pub workload_kind: Option<String>,
    pub runtime: Option<String>,
    pub container_name: Option<String>,
    pub container_id: Option<String>,
    /// The binary this container was launched with. `None` for a native
    /// process, and for container rows predating the column — those fall
    /// back to the runtime's default name.
    pub runtime_executable: Option<String>,
}

impl RunningService {
    pub fn from_row(row: &Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            service_id: row.get(0)?,
            run_id: row.get(1)?,
            pid: row.get::<_, Option<i64>>(2)?,
            spawned_at: row.get(3)?,
            command_line: row.get(4)?,
            allocation: row.get(5)?,
            state: row.get(6)?,
            workload_kind: row.get(7)?,
            runtime: row.get(8)?,
            container_name: row.get(9)?,
            container_id: row.get(10)?,
            runtime_executable: row.get(11)?,
        })
    }

    pub const COLUMNS: &'static str = "service_id, run_id, pid, spawned_at, command_line, \
         allocation, state, workload_kind, runtime, container_name, container_id, \
         runtime_executable";
}

/// A durable container launch intent recorded before any runtime invocation.
#[derive(Debug, Clone)]
pub struct ContainerLaunchIntent {
    pub intent_id: i64,
    pub service_id: i64,
    pub run_id: i64,
    pub owner_uuid: String,
    pub workload_kind: String,
    pub runtime: String,
    pub runtime_executable: String,
    pub container_name: String,
    pub labels_json: String,
    pub spec_json: String,
    pub container_id: Option<String>,
    pub state: String,
    pub created_at: i64,
}

impl ContainerLaunchIntent {
    pub fn from_row(row: &Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            intent_id: row.get(0)?,
            service_id: row.get(1)?,
            run_id: row.get(2)?,
            owner_uuid: row.get(3)?,
            workload_kind: row.get(4)?,
            runtime: row.get(5)?,
            runtime_executable: row.get(6)?,
            container_name: row.get(7)?,
            labels_json: row.get(8)?,
            spec_json: row.get(9)?,
            container_id: row.get(10)?,
            state: row.get(11)?,
            created_at: row.get(12)?,
        })
    }

    pub const COLUMNS: &'static str = "intent_id, service_id, run_id, owner_uuid, \
         workload_kind, runtime, runtime_executable, container_name, labels_json, spec_json, \
         container_id, state, created_at";
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

    /// The row as the wire carries it, without its service id.
    pub fn into_line(self) -> LogLine {
        LogLine {
            timestamp_ms: self.timestamp_ms,
            stream: self.stream,
            line: self.line,
            run_id: self.run_id,
            seq: self.seq,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RequestMetric {
    pub metric_id: i64,
    pub service_id: i64,
    pub run_id: Option<i64>,
    pub timestamp_ms: i64,
    pub endpoint: String,
    pub model: String,
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
    /// Engine-reported count of prompt tokens actually evaluated during
    /// prefill (`timings.prompt_n`), llama.cpp only. Excludes tokens served
    /// from the KV cache, unlike the billed [`Self::prompt_tokens`]. Used as
    /// the input/aggregate TPS numerator so prompt caching doesn't inflate
    /// prefill throughput.
    pub prompt_eval_tokens: Option<i64>,
    pub duration_ms: Option<i64>,
    pub ttft_ms: Option<i64>,
    /// Engine-reported prefill time (`timings.prompt_ms`), llama.cpp only.
    pub prompt_ms: Option<i64>,
    /// Engine-reported decode time (`timings.predicted_ms`), llama.cpp only.
    pub predicted_ms: Option<i64>,
    /// Engine-reported count of tokens proposed by the speculative draft
    /// (`timings.draft_n`), llama.cpp with speculative decoding only.
    pub draft_tokens: Option<i64>,
    /// Engine-reported count of draft tokens the target model accepted
    /// (`timings.draft_n_accepted`). Sustained zero across drafting requests
    /// is the spec_collapse watchdog's trip condition.
    pub draft_tokens_accepted: Option<i64>,
    pub status_code: i64,
}

/// One auto-restart watchdog firing, persisted so the history survives the
/// live event stream. See migration `0006_service_restarts`.
#[derive(Debug, Clone)]
pub struct ServiceRestart {
    pub restart_id: i64,
    pub service_id: i64,
    /// The run that was drained by the firing.
    pub run_id: Option<i64>,
    pub at_ms: i64,
    /// Which watchdog fired (`"error_rate"`, `"ttft_stall"`,
    /// `"generation_stall"`, `"spec_collapse"`, or `"periodic"`).
    pub trigger: String,
    /// Human-readable reason carried by the event.
    pub detail: String,
}

impl ServiceRestart {
    pub fn from_row(row: &Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            restart_id: row.get(0)?,
            service_id: row.get(1)?,
            run_id: row.get(2)?,
            at_ms: row.get(3)?,
            trigger: row.get(4)?,
            detail: row.get(5)?,
        })
    }

    pub const COLUMNS: &'static str = "restart_id, service_id, run_id, at_ms, trigger_name, detail";
}

#[derive(Debug, Clone)]
pub struct DeviceSample {
    pub sample_id: i64,
    pub device: String,
    pub timestamp_ms: i64,
    pub total_bytes: i64,
    pub free_bytes: i64,
    pub used_bytes: i64,
}
