//! `/api/events` WebSocket endpoint — system-wide event bus.

pub mod stream;

pub use crate::internal::event::Event;
