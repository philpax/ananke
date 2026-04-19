//! Per-service activity timestamps, shared across tasks via
//! `Arc<Mutex<tokio::time::Instant>>`.
//!
//! Uses the tokio monotonic clock rather than wall-clock millis so
//! `tokio::time::pause()` can advance it virtually in tests. Idle-deadline
//! arithmetic then lives entirely on one clock, making reload/drain/timeout
//! behaviour deterministic under `start_paused = true`.
//!
//! A stale read only delays idle transitions, which is harmless for the
//! scheduler — so we don't need to hold the lock across the whole
//! deadline-compute window.

use std::{collections::BTreeMap, sync::Arc};

use parking_lot::{Mutex, RwLock};
use smol_str::SmolStr;

/// Per-service activity stamp. Cloneable handle around a shared tokio
/// `Instant`.
pub type ActivityStamp = Arc<Mutex<tokio::time::Instant>>;

#[derive(Clone, Default)]
pub struct ActivityTable {
    inner: Arc<RwLock<BTreeMap<SmolStr, ActivityStamp>>>,
}

impl ActivityTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the stamp for `service`, creating it if missing. A fresh
    /// stamp is seeded to `Instant::now()` so the first `idle_deadline`
    /// reading doesn't fire immediately.
    pub fn get_or_init(&self, service: &SmolStr) -> ActivityStamp {
        {
            let guard = self.inner.read();
            if let Some(existing) = guard.get(service) {
                return existing.clone();
            }
        }
        let mut guard = self.inner.write();
        guard
            .entry(service.clone())
            .or_insert_with(|| Arc::new(Mutex::new(tokio::time::Instant::now())))
            .clone()
    }

    /// Bump the activity stamp for `service` to the current tokio instant.
    pub fn ping(&self, service: &SmolStr) {
        let stamp = self.get_or_init(service);
        *stamp.lock() = tokio::time::Instant::now();
    }

    /// Read the last activity instant for `service`. Returns `None` if
    /// the service has never been pinged and has no stamp yet.
    pub fn last(&self, service: &SmolStr) -> Option<tokio::time::Instant> {
        self.inner.read().get(service).map(|a| *a.lock())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn ping_updates_last() {
        let t = ActivityTable::new();
        let svc = SmolStr::new("demo");
        assert!(t.last(&svc).is_none());
        t.ping(&svc);
        let first = t.last(&svc).unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        t.ping(&svc);
        let second = t.last(&svc).unwrap();
        assert!(second >= first);
    }

    #[tokio::test]
    async fn get_or_init_returns_same_stamp() {
        let t = ActivityTable::new();
        let svc = SmolStr::new("demo");
        let a = t.get_or_init(&svc);
        let b = t.get_or_init(&svc);
        let marker = tokio::time::Instant::now();
        *a.lock() = marker;
        assert_eq!(*b.lock(), marker);
    }
}
