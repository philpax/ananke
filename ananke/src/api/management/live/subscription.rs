//! One subscription's task: snapshot, send if the body changed, wait for a
//! trigger, settle, repeat.

use std::time::Duration;

use ananke_api::live::{LiveMessage, LiveResource};
use axum::extract::ws::Message;
use tokio::sync::mpsc;
use tracing::warn;

use crate::{
    api::management::live::{
        protocol::error_message,
        resource::{snapshot, triggers},
        triggers::TriggerStream,
    },
    daemon::app_state::AppState,
};

/// How long to let a burst of changes settle before recomputing. Bounds
/// the recompute rate under load (an activity ping per request, a device
/// tick, and a state change often land together).
const COALESCE: Duration = Duration::from_millis(100);

/// Run until the session's outbox closes. Blocking on a full outbox is the
/// backpressure: the task recomputes once it unblocks, so nothing stale
/// sits in the queue for long.
pub(crate) async fn run(
    state: AppState,
    id: String,
    resource: LiveResource,
    out: mpsc::Sender<Message>,
) {
    let mut triggers = TriggerStream::new(&state, resource.clone(), triggers(&resource));
    // The id is fixed for the task's lifetime, so comparing whole frames
    // compares bodies; snapshot serialisation is deterministic (no DTO
    // carries a HashMap), so an unchanged body is byte-identical.
    let mut last_sent: Option<String> = None;
    loop {
        if let Some(frame) = render(&state, &id, &resource).await
            && last_sent.as_ref() != Some(&frame)
        {
            if out.send(Message::Text(frame.clone())).await.is_err() {
                return;
            }
            last_sent = Some(frame);
        }
        if triggers.changed().await.is_err() {
            return;
        }
        tokio::time::sleep(COALESCE).await;
        triggers.drain();
    }
}

async fn render(state: &AppState, id: &str, resource: &LiveResource) -> Option<String> {
    let message = match snapshot(state, resource).await {
        Ok(data) => LiveMessage::Snapshot {
            id: id.to_owned(),
            data,
        },
        Err(code) => error_message(Some(id.to_owned()), code),
    };
    match serde_json::to_string(&message) {
        Ok(frame) => Some(frame),
        Err(e) => {
            warn!(error = %e, ?resource, "live: failed to serialise a snapshot");
            None
        }
    }
}
