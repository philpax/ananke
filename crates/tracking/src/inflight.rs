//! Per-service in-flight request counters used by the drain pipeline and
//! the management API.
//!
//! Proxies hold an [`InflightGuard`](ananke_proxy::InflightGuard) per
//! request. Every counter the table mints shares one table-wide
//! [`ChangeSignal`], so a reader that renders all services can wait on a
//! single receiver.

use std::{collections::BTreeMap, sync::Arc};

use ananke_events::ChangeSignal;
use ananke_proxy::inflight::InflightCounter;
use parking_lot::RwLock;
use smol_str::SmolStr;
use tokio::sync::watch;

#[derive(Clone, Default)]
pub struct InflightTable {
    inner: Arc<RwLock<BTreeMap<SmolStr, Arc<InflightCounter>>>>,
    changed: ChangeSignal,
}

impl InflightTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return (or lazily create) the counter for the given service.
    pub fn counter(&self, service: &SmolStr) -> Arc<InflightCounter> {
        {
            let guard = self.inner.read();
            if let Some(c) = guard.get(service) {
                return c.clone();
            }
        }
        let mut guard = self.inner.write();
        guard
            .entry(service.clone())
            .or_insert_with(|| Arc::new(InflightCounter::new(self.changed.clone())))
            .clone()
    }

    /// Return the current in-flight count for the given service.
    pub fn current(&self, service: &SmolStr) -> u64 {
        self.inner
            .read()
            .get(service)
            .map(|c| c.current())
            .unwrap_or(0)
    }

    /// Wakes whenever any service's count changes.
    pub fn changes(&self) -> watch::Receiver<u64> {
        self.changed.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use ananke_proxy::InflightGuard;

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
    fn every_counter_notifies_the_table_signal() {
        let t = InflightTable::new();
        let mut rx = t.changes();
        let alpha = InflightGuard::new(t.counter(&SmolStr::new("alpha")));
        assert!(rx.has_changed().unwrap());
        rx.borrow_and_update();

        let _beta = InflightGuard::new(t.counter(&SmolStr::new("beta")));
        assert!(rx.has_changed().unwrap());
        rx.borrow_and_update();

        drop(alpha);
        assert!(rx.has_changed().unwrap());
    }
}
