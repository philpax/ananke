//! Broadcast-based event bus. Publishers are infallible from the caller's
//! perspective; subscribers handle lag explicitly via `RecvError::Lagged`.
//! The bus also retains a bounded history so a client connecting late can
//! backfill what it missed.

use std::{collections::VecDeque, sync::Arc};

use ananke_api::events::{Event, EventHistory};
use parking_lot::Mutex;
use tokio::sync::broadcast;

/// Capacity of the per-daemon event broadcast channel.
///
/// Subscribers that lag beyond this buffer receive
/// [`tokio::sync::broadcast::error::RecvError::Lagged`] and decide for
/// themselves how to recover (resnapshot, log, reconnect).
const EVENT_BUS_CAPACITY: usize = 1024;

/// How many recent events [`EventBus::history`] retains. It matches the
/// frontend's event-feed cap, so a full history fills the feed exactly.
const EVENT_HISTORY_CAPACITY: usize = 1000;

/// Cheap to clone; internally `Arc`-backed.
#[derive(Clone)]
pub struct EventBus {
    tx: broadcast::Sender<Event>,
    history: Arc<Mutex<History>>,
}

impl EventBus {
    pub fn new() -> Self {
        let (tx, _rx) = broadcast::channel(EVENT_BUS_CAPACITY);
        Self {
            tx,
            history: Arc::new(Mutex::new(History {
                since_ms: ananke_time::now_unix_ms(),
                events: VecDeque::with_capacity(EVENT_HISTORY_CAPACITY),
            })),
        }
    }

    /// Publish an event and retain it in the history. The broadcast half
    /// silently drops it if there are no subscribers.
    pub fn publish(&self, event: Event) {
        // Recording and sending under one lock keeps the history and the
        // broadcast in the same order for every observer.
        let mut history = self.history.lock();
        history.push(event.clone());
        let _ = self.tx.send(event);
    }

    /// The retained window of recent events, oldest first.
    pub fn history(&self) -> EventHistory {
        let history = self.history.lock();
        EventHistory {
            since_ms: history.since_ms,
            events: history.events.iter().cloned().collect(),
        }
    }

    /// Subscribe to the bus. Each subscriber has its own cursor.
    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.tx.subscribe()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

struct History {
    /// The daemon's start time until the first eviction, then the oldest
    /// retained event's timestamp.
    since_ms: i64,
    events: VecDeque<Event>,
}

impl History {
    fn push(&mut self, event: Event) {
        if self.events.len() == EVENT_HISTORY_CAPACITY
            && let Some(evicted) = self.events.pop_front()
        {
            self.since_ms = self.events.front().unwrap_or(&evicted).at_ms();
        }
        self.events.push_back(event);
    }
}

#[cfg(test)]
mod tests {
    use ananke_api::events::Event;
    use smol_str::SmolStr;

    use super::*;

    #[test]
    fn history_keeps_the_newest_events_and_advances_since() {
        let bus = EventBus::new();
        let started = bus.history().since_ms;
        bus.publish(Event::ConfigReloaded {
            at_ms: 0,
            changed_services: vec![],
        });
        assert_eq!(bus.history().since_ms, started);

        for at_ms in 1..(EVENT_HISTORY_CAPACITY as i64 + 5) {
            bus.publish(Event::ConfigReloaded {
                at_ms,
                changed_services: vec![],
            });
        }
        let history = bus.history();
        assert_eq!(history.events.len(), EVENT_HISTORY_CAPACITY);
        assert_eq!(history.since_ms, 5);
        assert_eq!(history.events[0].at_ms(), 5);
        assert_eq!(
            history.events.last().map(Event::at_ms),
            Some(EVENT_HISTORY_CAPACITY as i64 + 4)
        );
    }

    #[tokio::test]
    async fn publish_then_receive() {
        let bus = EventBus::new();
        let mut rx = bus.subscribe();
        bus.publish(Event::ConfigReloaded {
            at_ms: 1,
            changed_services: vec![SmolStr::new("demo")],
        });
        match rx.recv().await.unwrap() {
            Event::ConfigReloaded { at_ms, .. } => assert_eq!(at_ms, 1),
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[tokio::test]
    async fn lag_surfaces_as_recverror() {
        let bus = EventBus::new();
        let mut rx = bus.subscribe();
        for i in 0..(EVENT_BUS_CAPACITY + 5) {
            bus.publish(Event::EstimatorDrift {
                class: "vram".to_string(),
                service: SmolStr::new("demo"),
                rolling_mean: i as f32,
                at_ms: i as i64,
            });
        }
        match rx.recv().await {
            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                assert!(n >= 5, "expected at least 5 dropped, got {n}");
            }
            other => panic!("expected lag, got {other:?}"),
        }
    }
}
