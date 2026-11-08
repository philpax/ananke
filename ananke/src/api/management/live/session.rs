//! The live connection's shell: a reader that feeds [`Session`] and starts
//! or aborts subscription and follow tasks, and a writer that drains the shared outbox
//! with a stall timeout and keepalive pings.

use std::{borrow::Cow, collections::BTreeMap, time::Duration};

use ananke_api::live::{LiveFeed, LiveMessage};
use axum::extract::ws::{CloseFrame, Message, WebSocket, close_code};
use futures::{SinkExt, StreamExt, stream::SplitSink};
use tokio::{
    sync::mpsc,
    task::{AbortHandle, JoinSet},
};
use tracing::{debug, warn};

use crate::{
    api::{
        errors::ApiErrorCode,
        frontend::build_id,
        management::live::{
            logs,
            protocol::{Effect, LiveTask, Session, error_message},
            subscription,
        },
    },
    daemon::app_state::AppState,
};

/// Frames queued for the writer. Subscription tasks block when it is full.
const OUTBOX_CAPACITY: usize = 32;

/// How long one outbound frame may take to send before the client counts
/// as stalled and the connection is closed.
const SEND_TIMEOUT: Duration = Duration::from_secs(10);

/// Interval between server pings, so intermediaries keep an idle socket.
const KEEPALIVE: Duration = Duration::from_secs(30);

pub(crate) async fn serve(socket: WebSocket, state: AppState) {
    let (sink, mut stream) = socket.split();
    let (out, outbox) = mpsc::channel::<Message>(OUTBOX_CAPACITY);
    let mut writer = tokio::spawn(write_loop(sink, outbox));

    // Queued before any subscription exists, so it is always the first frame.
    let hello = LiveMessage::Hello { build: build_id() };
    match serde_json::to_string(&hello) {
        Ok(frame) => {
            if out.send(Message::Text(frame)).await.is_err() {
                writer.abort();
                return;
            }
        }
        Err(e) => warn!(error = %e, "live: failed to serialise the hello"),
    }

    let mut session = Session::default();
    let mut tasks = JoinSet::new();
    let mut handles: BTreeMap<String, AbortHandle> = BTreeMap::new();

    loop {
        let inbound = tokio::select! {
            inbound = stream.next() => inbound,
            _ = &mut writer => break,
            // Reap finished (aborted) tasks so a cancel-heavy client does
            // not grow the set without bound.
            Some(_) = tasks.join_next(), if !tasks.is_empty() => continue,
        };
        let effect = match inbound {
            Some(Ok(Message::Text(text))) => session.apply_text(&text),
            Some(Ok(Message::Binary(_))) => Effect::Reply(error_message(
                None,
                ApiErrorCode::InvalidLiveRequest {
                    reason: "frames must be JSON text".into(),
                },
            )),
            Some(Ok(Message::Ping(payload))) => {
                if out.send(Message::Pong(payload)).await.is_err() {
                    break;
                }
                Effect::Ignore
            }
            Some(Ok(Message::Pong(_))) => Effect::Ignore,
            Some(Ok(Message::Close(_))) | None => break,
            Some(Err(e)) => {
                debug!(error = %e, "live: read failed");
                break;
            }
        };
        match effect {
            Effect::Spawn { id, task } => {
                let handle = match task {
                    LiveTask::Snapshot(resource) => tasks.spawn(subscription::run(
                        state.clone(),
                        id.clone(),
                        resource,
                        out.clone(),
                    )),
                    LiveTask::Follow(LiveFeed::Logs(feed)) => {
                        tasks.spawn(logs::run(state.clone(), id.clone(), feed, out.clone()))
                    }
                };
                handles.insert(id, handle);
            }
            Effect::Abort { id } => {
                if let Some(handle) = handles.remove(&id) {
                    handle.abort();
                }
            }
            Effect::Reply(message) => match serde_json::to_string(&message) {
                Ok(frame) => {
                    if out.send(Message::Text(frame)).await.is_err() {
                        break;
                    }
                }
                Err(e) => warn!(error = %e, "live: failed to serialise a reply"),
            },
            Effect::Ignore => {}
        }
    }

    tasks.abort_all();
    drop(out);
    writer.abort();
}

async fn write_loop(mut sink: SplitSink<WebSocket, Message>, mut outbox: mpsc::Receiver<Message>) {
    let mut keepalive = tokio::time::interval(KEEPALIVE);
    keepalive.tick().await;
    loop {
        let message = tokio::select! {
            message = outbox.recv() => match message {
                Some(message) => message,
                None => return,
            },
            _ = keepalive.tick() => Message::Ping(Vec::new()),
        };
        match tokio::time::timeout(SEND_TIMEOUT, sink.send(message)).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                debug!(error = %e, "live: write failed");
                return;
            }
            Err(_) => {
                warn!("live: the client stopped reading; closing the connection");
                let close = Message::Close(Some(CloseFrame {
                    code: close_code::POLICY,
                    reason: Cow::Borrowed("live: the client is not reading"),
                }));
                let _ = tokio::time::timeout(SEND_TIMEOUT, sink.send(close)).await;
                return;
            }
        }
    }
}
