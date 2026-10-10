//! Broadcast-based event bus. Publishers are infallible from the caller's
//! perspective; subscribers handle lag explicitly via `RecvError::Lagged`.

use std::{collections::VecDeque, sync::Arc};

use ananke_api::{dashboard::DashboardEvents, events::Event};
use parking_lot::Mutex;
use tokio::sync::broadcast;

/// Capacity of the per-daemon event broadcast channel.
///
/// Subscribers that lag beyond this buffer receive
/// [`tokio::sync::broadcast::error::RecvError::Lagged`]. The
/// `/api/events` WebSocket handler translates that into an
/// `Event::Overflow` frame for the client; other subscribers are free
/// to handle lag however they want (logging, reconnecting, etc.).
const EVENT_BUS_CAPACITY: usize = 1024;

/// Cheap to clone; internally `Arc`-backed via `broadcast::Sender`.
#[derive(Clone)]
pub struct EventBus {
    tx: broadcast::Sender<Event>,
    history: Arc<Mutex<VecDeque<Event>>>,
    since_ms: i64,
}

impl EventBus {
    pub fn new() -> Self {
        let (tx, _rx) = broadcast::channel(EVENT_BUS_CAPACITY);
        Self {
            tx,
            history: Arc::new(Mutex::new(VecDeque::new())),
            since_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0),
        }
    }

    /// Publish and retain an event even when no browser is connected.
    pub fn publish(&self, event: Event) {
        let mut history = self.history.lock();
        if history.len() == EVENT_BUS_CAPACITY {
            history.pop_front();
        }
        history.push_back(event.clone());
        let _ = self.tx.send(event);
    }

    /// Read the bounded capture window for initial load and reconnect backfill.
    pub fn history(&self) -> DashboardEvents {
        let history = self.history.lock();
        let since_ms = history
            .front()
            .map(|event| match event {
                Event::StateChanged { at_ms, .. }
                | Event::AllocationChanged { at_ms, .. }
                | Event::ConfigReloaded { at_ms, .. }
                | Event::EstimatorDrift { at_ms, .. }
                | Event::AutoRestarted { at_ms, .. } => *at_ms,
                Event::Overflow { .. } => self.since_ms,
            })
            .unwrap_or(self.since_ms);
        DashboardEvents {
            since_ms,
            events: history.iter().cloned().collect(),
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

#[cfg(test)]
mod tests {
    use ananke_api::events::Event;
    use smol_str::SmolStr;

    use super::*;

    #[test]
    fn capture_precedes_connections_and_keeps_the_newest_events() {
        let bus = EventBus::new();
        for at_ms in 0..(EVENT_BUS_CAPACITY as i64 + 5) {
            bus.publish(Event::ConfigReloaded {
                at_ms,
                changed_services: vec![],
            });
        }
        let history = bus.history();
        assert_eq!(history.events.len(), EVENT_BUS_CAPACITY);
        assert_eq!(history.since_ms, 5);
        assert!(
            matches!(history.events.last(), Some(Event::ConfigReloaded { at_ms, .. }) if *at_ms == EVENT_BUS_CAPACITY as i64 + 4)
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
