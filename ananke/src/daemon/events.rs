//! Broadcast-based event bus. Publishers are infallible from the caller's
//! perspective; subscribers handle lag explicitly via `RecvError::Lagged`.

use ananke_api::Event;
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
}

impl EventBus {
    pub fn new() -> Self {
        let (tx, _rx) = broadcast::channel(EVENT_BUS_CAPACITY);
        Self { tx }
    }

    /// Publish an event. Silently drops if there are no subscribers.
    pub fn publish(&self, event: Event) {
        let _ = self.tx.send(event);
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
    use ananke_api::Event;
    use smol_str::SmolStr;

    use super::*;

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
