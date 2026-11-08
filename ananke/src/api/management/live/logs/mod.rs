//! A `logs` follow on `/api/live`: backfill from the store after the client's
//! position, then send new lines from the batcher's broadcast.
//!
//! The store is the source of truth and the broadcast is only a wake-up and a
//! fast path. Each sync subscribes before it reads, so every line committed
//! after the read arrives on the broadcast, and [`FollowCursor`] drops the
//! ones the read already covered. A follower that lags the broadcast resyncs
//! from the store, losing nothing; only a backfill over
//! [`MAX_BACKFILL_LINES`] leaves a gap, which is reported as `log_gap`.
//!
//! One edge is accepted: a line from an older run committed after a newer
//! run's line has passed the resume point is not sent, since a position
//! compares the run first. It needs a restart and a straggling pump in the
//! same sub-second window.

mod cursor;

use std::{sync::Arc, time::Duration};

use ananke_api::{
    events::Event,
    internal::log_line::LogLine,
    live::{LiveMessage, LogFeed},
};
use ananke_db::{LogFilter, logs::LogBatch};
use axum::extract::ws::Message;
use smol_str::SmolStr;
use tokio::sync::{broadcast::error::RecvError, mpsc};
use tracing::warn;

use crate::{
    api::{
        errors::ApiErrorCode,
        management::live::{
            logs::cursor::{FollowCursor, frames, gap},
            protocol::error_message,
        },
    },
    daemon::app_state::AppState,
};

/// Most lines one backfill sends; older ones are reported as a gap. Matches
/// the HTTP logs endpoint's page limit.
pub(crate) const MAX_BACKFILL_LINES: usize = 1000;

/// How long to wait before rereading the store after a failed read.
const RETRY_AFTER_FAILURE: Duration = Duration::from_secs(1);

/// How often to look for a configured service's row while it is provisioned.
const PROVISION_POLL: Duration = Duration::from_millis(100);

/// The outbox closed: the connection is gone.
struct Disconnected;

/// Run until the session's outbox closes or the daemon shuts down.
pub(crate) async fn run(state: AppState, id: String, feed: LogFeed, out: mpsc::Sender<Message>) {
    let out = Outbox { id, out };
    let Some(service_id) = resolve(&state, &feed.service, &out).await else {
        return;
    };
    let filter = LogFilter {
        run: feed.run,
        stream: feed.stream,
    };
    let mut cursor = FollowCursor::new(feed.after, filter);
    while let Ok(SyncEnd::Resync) = sync(&state, service_id, &mut cursor, &out).await {}
}

/// Wait until the service has a log, reporting it missing in the meantime. A
/// config reload may add it, as for the `service` resource.
async fn resolve(state: &AppState, name: &str, out: &Outbox) -> Option<i64> {
    loop {
        // Subscribed before the lookup, so a reload between the two wakes it.
        let mut events = state.events.subscribe();
        match state.db.resolve_service_id(name).await {
            Ok(Some(id)) => return Some(id),
            // The reload that adds a service is published before the service
            // is provisioned and given its row, so a configured service with
            // no row yet is about to have one.
            Ok(None) if configured(state, name) => {
                tokio::time::sleep(PROVISION_POLL).await;
                continue;
            }
            Ok(None) => {
                let code = ApiErrorCode::ServiceNotFound {
                    name: SmolStr::new(name),
                };
                out.error(code).await.ok()?;
            }
            Err(e) => {
                let code = ApiErrorCode::QueryFailed {
                    reason: e.to_string(),
                };
                out.error(code).await.ok()?;
                tokio::time::sleep(RETRY_AFTER_FAILURE).await;
                continue;
            }
        }
        loop {
            match events.recv().await {
                Ok(Event::ConfigReloaded { .. }) | Err(RecvError::Lagged(_)) => break,
                Ok(_) => {}
                Err(RecvError::Closed) => return None,
            }
        }
    }
}

fn configured(state: &AppState, name: &str) -> bool {
    state
        .config
        .effective()
        .services
        .iter()
        .any(|s| s.name.as_str() == name)
}

/// How one sync ended, when the follow should carry on.
enum SyncEnd {
    /// The broadcast lagged or the read failed; read the store again.
    Resync,
    /// The broadcast closed: the daemon is shutting down.
    Done,
}

/// Backfill from the store, then forward the broadcast until it lags.
async fn sync(
    state: &AppState,
    service_id: i64,
    cursor: &mut FollowCursor,
    out: &Outbox,
) -> Result<SyncEnd, Disconnected> {
    let mut rx = state.batcher.subscribe();
    let after = cursor.resume_point();
    let page = match state
        .db
        .fetch_logs_after(service_id, after, cursor.filter(), MAX_BACKFILL_LINES)
        .await
    {
        Ok(page) => page,
        Err(e) => {
            let code = ApiErrorCode::QueryFailed {
                reason: e.to_string(),
            };
            out.error(code).await?;
            tokio::time::sleep(RETRY_AFTER_FAILURE).await;
            return Ok(SyncEnd::Resync);
        }
    };
    if let Some((before, skipped)) = gap(&page) {
        out.send(&LiveMessage::LogGap {
            id: out.id.clone(),
            after,
            before,
            skipped,
        })
        .await?;
    }
    cursor.absorb_backfill(&page.rows);
    out.lines(&page.rows).await?;
    out.send(&LiveMessage::LogCaughtUp { id: out.id.clone() })
        .await?;

    loop {
        match rx.recv().await {
            Ok(batch) => {
                let lines = admitted(&batch, service_id, cursor);
                out.lines(&lines).await?;
            }
            Err(RecvError::Lagged(_)) => return Ok(SyncEnd::Resync),
            Err(RecvError::Closed) => return Ok(SyncEnd::Done),
        }
    }
}

fn admitted(batch: &Arc<LogBatch>, service_id: i64, cursor: &mut FollowCursor) -> Vec<LogLine> {
    batch
        .for_service(service_id)
        .filter(|line| cursor.admit(line))
        .cloned()
        .collect()
}

/// The follow's half of the session outbox.
struct Outbox {
    id: String,
    out: mpsc::Sender<Message>,
}

impl Outbox {
    /// Send `lines` in bounded frames; nothing if empty.
    async fn lines(&self, lines: &[LogLine]) -> Result<(), Disconnected> {
        for chunk in frames(lines) {
            self.send(&LiveMessage::LogLines {
                id: self.id.clone(),
                lines: chunk.to_vec(),
            })
            .await?;
        }
        Ok(())
    }

    async fn error(&self, code: ApiErrorCode) -> Result<(), Disconnected> {
        self.send(&error_message(Some(self.id.clone()), code)).await
    }

    /// Blocking on a full outbox is the backpressure: a follower held up long
    /// enough lags the broadcast and resyncs from the store.
    async fn send(&self, message: &LiveMessage) -> Result<(), Disconnected> {
        match serde_json::to_string(message) {
            Ok(frame) => self
                .out
                .send(Message::Text(frame))
                .await
                .map_err(|_| Disconnected),
            Err(e) => {
                warn!(error = %e, "live: failed to serialise a log frame");
                Ok(())
            }
        }
    }
}
