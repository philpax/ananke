//! Daemon-wide broadcast channel, change signals, and the shared estimate
//! cache.
//!
//! `EventBus` is the infallible-publisher broadcast bus every subsystem
//! (config reloads, tracking, the allocator, supervision, the HTTP surface)
//! publishes lifecycle events to. `ChangeSignal` covers state that changes
//! without an event, so live readers can still wake on it. The estimate
//! cache is shared between the supervisor's spawn-time estimator run and the
//! management `ServiceDetail` handler so successive detail polls don't
//! re-parse the GGUF.

pub mod change;
pub mod estimate_cache;
pub mod events;

pub use change::ChangeSignal;
pub use estimate_cache::{CacheEntry, EstimateCache};
pub use events::EventBus;
