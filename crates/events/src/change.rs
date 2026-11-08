//! A coalescing "something changed" signal for state that moves without an
//! [`Event`](ananke_api::events::Event): in-flight counts, activity stamps,
//! and the supervisor mirror.
//!
//! It is a `watch` channel over a generation counter, so a burst of
//! notifications wakes each receiver once and a receiver never lags.

use tokio::sync::watch;

/// Cheap to clone; every clone notifies the same receivers.
#[derive(Clone)]
pub struct ChangeSignal {
    tx: watch::Sender<u64>,
}

impl ChangeSignal {
    pub fn new() -> Self {
        Self {
            tx: watch::channel(0).0,
        }
    }

    /// Mark a change. Never blocks, and succeeds with no receivers.
    pub fn notify(&self) {
        self.tx
            .send_modify(|generation| *generation = generation.wrapping_add(1));
    }

    /// Subscribe before reading the state the signal guards, so a change
    /// landing between the two still wakes the receiver.
    pub fn subscribe(&self) -> watch::Receiver<u64> {
        self.tx.subscribe()
    }
}

impl Default for ChangeSignal {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn notify_wakes_every_subscriber() {
        let signal = ChangeSignal::new();
        let mut a = signal.subscribe();
        let mut b = signal.clone().subscribe();
        signal.notify();
        a.changed().await.unwrap();
        b.changed().await.unwrap();
    }

    #[test]
    fn bursts_coalesce_into_one_wake() {
        let signal = ChangeSignal::new();
        let mut rx = signal.subscribe();
        signal.notify();
        signal.notify();
        signal.notify();
        assert!(rx.has_changed().unwrap());
        rx.borrow_and_update();
        assert!(!rx.has_changed().unwrap());
    }

    #[test]
    fn notify_without_receivers_is_kept_for_later_subscribers() {
        let signal = ChangeSignal::new();
        signal.notify();
        let rx = signal.subscribe();
        assert_eq!(*rx.borrow(), 1);
        assert!(!rx.has_changed().unwrap());
    }
}
