//! Per-service in-flight request counting.
//!
//! Proxies hold an [`InflightGuard`] for each request (and for each spliced
//! WebSocket session). The supervisor's drain waits for the count to reach
//! zero, and live readers wake on the counter's [`ChangeSignal`], since a
//! request finishing publishes no event of its own.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use ananke_events::ChangeSignal;

/// One service's in-flight count, paired with the signal it bumps on every
/// increment and decrement.
pub struct InflightCounter {
    count: AtomicU64,
    changed: ChangeSignal,
}

impl InflightCounter {
    pub fn new(changed: ChangeSignal) -> Self {
        Self {
            count: AtomicU64::new(0),
            changed,
        }
    }

    /// A counter whose signal nobody else holds, for callers that only need
    /// the count.
    pub fn detached() -> Arc<Self> {
        Arc::new(Self::new(ChangeSignal::new()))
    }

    pub fn current(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }
}

/// RAII guard that increments the counter on construction and decrements on
/// drop, notifying the counter's signal both times.
pub struct InflightGuard {
    counter: Arc<InflightCounter>,
}

impl InflightGuard {
    pub fn new(counter: Arc<InflightCounter>) -> Self {
        counter.count.fetch_add(1, Ordering::Relaxed);
        counter.changed.notify();
        Self { counter }
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.counter.count.fetch_sub(1, Ordering::Relaxed);
        self.counter.changed.notify();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_increments_and_decrements() {
        let c = InflightCounter::detached();
        {
            let _g = InflightGuard::new(c.clone());
            assert_eq!(c.current(), 1);
        }
        assert_eq!(c.current(), 0);
    }

    #[test]
    fn multiple_guards_stack() {
        let c = InflightCounter::detached();
        let _g1 = InflightGuard::new(c.clone());
        let _g2 = InflightGuard::new(c.clone());
        let _g3 = InflightGuard::new(c.clone());
        assert_eq!(c.current(), 3);
    }

    #[test]
    fn guard_notifies_on_increment_and_on_drop() {
        let signal = ChangeSignal::new();
        let c = Arc::new(InflightCounter::new(signal.clone()));
        let mut rx = signal.subscribe();

        let guard = InflightGuard::new(c.clone());
        assert!(rx.has_changed().unwrap());
        rx.borrow_and_update();

        drop(guard);
        assert!(rx.has_changed().unwrap());
        assert_eq!(c.current(), 0);
    }
}
