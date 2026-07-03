//! `WS /api/events` — re-exports the shared [`Event`] type.
//!
//! The `Event` enum is defined in [`crate::internal::event`] because it
//! is used as a broadcast bus message beyond the HTTP boundary. This
//! module is the per-endpoint home for the WebSocket surface.

pub use crate::internal::event::Event;
