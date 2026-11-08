//! `/api/events` endpoints — the system-wide event bus's history.

pub mod history;

pub use history::EventHistory;

pub use crate::internal::event::Event;
