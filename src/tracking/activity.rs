//! Per-service activity timestamps, shared across tasks via `Arc<AtomicU64>`.
//!
//! Stores UNIX epoch milliseconds. Readers (supervisors computing idle
//! deadlines) use `load(Ordering::Relaxed)`; writers (proxy paths) use
//! `store(now_ms, Ordering::Relaxed)`. A monotonic wall clock is not
//! required: a stale value only delays idle transitions, which is
//! harmless for the scheduler.

use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use parking_lot::RwLock;
use smol_str::SmolStr;

#[derive(Clone, Default)]
pub struct ActivityTable {
    inner: Arc<RwLock<BTreeMap<SmolStr, Arc<AtomicU64>>>>,
}

impl ActivityTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the atomic for `service`, creating it if missing.
    pub fn get_or_init(&self, service: &SmolStr) -> Arc<AtomicU64> {
        {
            let guard = self.inner.read();
            if let Some(existing) = guard.get(service) {
                return existing.clone();
            }
        }
        let mut guard = self.inner.write();
        guard
            .entry(service.clone())
            .or_insert_with(|| Arc::new(AtomicU64::new(now_ms())))
            .clone()
    }

    /// Bump the activity timestamp for `service` to now.
    pub fn ping(&self, service: &SmolStr) {
        self.get_or_init(service).store(now_ms(), Ordering::Relaxed);
    }

    /// Read the last activity timestamp for `service`. Returns `None` if
    /// the service has never been pinged.
    pub fn last_ms(&self, service: &SmolStr) -> Option<u64> {
        self.inner
            .read()
            .get(service)
            .map(|a| a.load(Ordering::Relaxed))
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ping_updates_last_ms() {
        let t = ActivityTable::new();
        let svc = SmolStr::new("demo");
        assert!(t.last_ms(&svc).is_none());
        t.ping(&svc);
        let first = t.last_ms(&svc).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        t.ping(&svc);
        let second = t.last_ms(&svc).unwrap();
        assert!(second >= first);
    }

    #[test]
    fn get_or_init_returns_same_atomic() {
        let t = ActivityTable::new();
        let svc = SmolStr::new("demo");
        let a = t.get_or_init(&svc);
        let b = t.get_or_init(&svc);
        a.store(42, Ordering::Relaxed);
        assert_eq!(b.load(Ordering::Relaxed), 42);
    }
}
