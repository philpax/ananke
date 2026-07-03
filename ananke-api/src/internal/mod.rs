//! Types shared beyond the HTTP boundary — broadcast bus events, placement
//! verdicts, and log rows that are also used by the supervisor, allocator,
//! and database layers.

pub mod event;
pub mod fit_verdict;
pub mod log_line;
