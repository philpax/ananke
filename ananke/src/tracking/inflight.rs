//! Per-service in-flight request counter used by the drain pipeline.
//!
//! Proxies increment before forwarding and decrement on response
//! completion (including error and connection close). The supervisor's
//! drain state waits for the counter to reach zero, bounded by
//! `max_request_duration`.

use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use parking_lot::RwLock;
use smol_str::SmolStr;

#[derive(Clone, Default)]
pub struct InflightTable {
    inner: Arc<RwLock<BTreeMap<SmolStr, Arc<AtomicU64>>>>,
}

impl InflightTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return (or lazily create) the counter for the given service.
    pub fn counter(&self, service: &SmolStr) -> Arc<AtomicU64> {
        {
            let guard = self.inner.read();
            if let Some(c) = guard.get(service) {
                return c.clone();
            }
        }
        let mut guard = self.inner.write();
        guard
            .entry(service.clone())
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone()
    }

    /// Return the current in-flight count for the given service.
    pub fn current(&self, service: &SmolStr) -> u64 {
        self.inner
            .read()
            .get(service)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }
}

/// RAII guard that increments the counter on construction and decrements on drop.
pub struct InflightGuard {
    counter: Arc<AtomicU64>,
}

impl InflightGuard {
    pub fn new(counter: Arc<AtomicU64>) -> Self {
        counter.fetch_add(1, Ordering::Relaxed);
        Self { counter }
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_increments_and_decrements() {
        let t = InflightTable::new();
        let svc = SmolStr::new("demo");
        assert_eq!(t.current(&svc), 0);
        let c = t.counter(&svc);
        {
            let _g = InflightGuard::new(c.clone());
            assert_eq!(t.current(&svc), 1);
        }
        assert_eq!(t.current(&svc), 0);
    }

    #[test]
    fn multiple_guards_stack() {
        let t = InflightTable::new();
        let svc = SmolStr::new("demo");
        let c = t.counter(&svc);
        let _g1 = InflightGuard::new(c.clone());
        let _g2 = InflightGuard::new(c.clone());
        let _g3 = InflightGuard::new(c.clone());
        assert_eq!(t.current(&svc), 3);
    }
}
