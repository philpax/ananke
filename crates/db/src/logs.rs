//! Log batching writer.
//!
//! Contract: `BatcherHandle::push` is fire-and-forget. A single writer task
//! owns a clone of the `Database` handle and commits every 200 ms or every
//! 100 lines, whichever first. A shutdown signal triggers a final flush.
//! After each successful commit the flush is broadcast as one [`LogBatch`] on
//! a `tokio::sync::broadcast` channel, so log followers see new lines without
//! polling the database. Broadcast order is commit order: a follower that
//! subscribes before reading the database is sent every line committed after
//! that read.

use std::{collections::HashMap, sync::Arc, time::Duration};

use ananke_api::internal::log_line::LogLine as ApiLogLine;
use tokio::{
    sync::{broadcast, mpsc, oneshot},
    time::Instant,
};
use tracing::warn;

use crate::{Database, models::ServiceLog};

const BATCH_LINES: usize = 100;
const BATCH_INTERVAL: Duration = Duration::from_millis(200);
/// Capacity of the live-log broadcast channel, in flushes. Subscribers that
/// fall behind by more than this many receive `RecvError::Lagged`.
pub const LOG_BROADCAST_CAPACITY: usize = 256;

/// Which output stream a log line came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    Stdout,
    Stderr,
    /// Merged stdout+stderr from a container log follower. The common
    /// Docker/Podman CLI contract does not document lossless stdout/stderr
    /// framing, so container output is tagged `combined` rather than
    /// mislabeled as `stdout`.
    Combined,
}

impl Stream {
    fn as_str(self) -> &'static str {
        match self {
            Stream::Stdout => "stdout",
            Stream::Stderr => "stderr",
            Stream::Combined => "combined",
        }
    }
}

/// A single captured log line from a running service.
#[derive(Debug)]
pub struct LogLine {
    pub service_id: i64,
    pub run_id: i64,
    pub timestamp_ms: i64,
    pub stream: Stream,
    pub line: String,
}

/// One committed flush: every line it wrote, across services, in seq order
/// within each `(service_id, run_id)`.
#[derive(Debug, Default)]
pub struct LogBatch {
    /// `(service_id, line)` pairs.
    pub lines: Vec<(i64, ApiLogLine)>,
}

impl LogBatch {
    /// The lines that belong to one service.
    pub fn for_service(&self, service_id: i64) -> impl Iterator<Item = &ApiLogLine> {
        self.lines
            .iter()
            .filter(move |(id, _)| *id == service_id)
            .map(|(_, line)| line)
    }
}

/// Cloneable handle to the background batcher task.
#[derive(Clone)]
pub struct BatcherHandle {
    tx: mpsc::UnboundedSender<Msg>,
    broadcast: broadcast::Sender<Arc<LogBatch>>,
}

impl BatcherHandle {
    /// Subscribe to the live-log broadcast, one message per committed flush.
    pub fn subscribe(&self) -> broadcast::Receiver<Arc<LogBatch>> {
        self.broadcast.subscribe()
    }
}

enum Msg {
    Line(LogLine),
    Flush(oneshot::Sender<()>),
}

impl BatcherHandle {
    /// Enqueue a log line. Fire-and-forget; silently drops if the writer task has exited.
    pub fn push(&self, line: LogLine) {
        let _ = self.tx.send(Msg::Line(line));
    }

    /// Flush any buffered lines to the database and wait for the ack.
    pub async fn flush(&self) {
        let (tx, rx) = oneshot::channel();
        if self.tx.send(Msg::Flush(tx)).is_err() {
            return;
        }
        let _ = rx.await;
    }
}

/// Spawn the background writer task and return a handle to it.
pub fn spawn(db: Database) -> BatcherHandle {
    let (tx, rx) = mpsc::unbounded_channel();
    let (broadcast, _) = broadcast::channel(LOG_BROADCAST_CAPACITY);
    tokio::spawn(run(db, rx, broadcast.clone()));
    BatcherHandle { tx, broadcast }
}

async fn run(
    db: Database,
    mut rx: mpsc::UnboundedReceiver<Msg>,
    broadcast: broadcast::Sender<Arc<LogBatch>>,
) {
    let mut buffer: Vec<LogLine> = Vec::with_capacity(BATCH_LINES);
    let mut seq_counters: HashMap<(i64, i64), i64> = HashMap::new();
    // tokio's Instant so paused-time tests advance correctly.
    let mut deadline = Instant::now() + BATCH_INTERVAL;

    loop {
        let tick = tokio::time::sleep_until(deadline);
        tokio::pin!(tick);

        tokio::select! {
            msg = rx.recv() => match msg {
                None => {
                    // Channel closed; do a final flush and exit.
                    flush(&db, &mut buffer, &mut seq_counters, &broadcast).await;
                    return;
                }
                Some(Msg::Line(line)) => {
                    buffer.push(line);
                    if buffer.len() >= BATCH_LINES {
                        flush(&db, &mut buffer, &mut seq_counters, &broadcast).await;
                        deadline = Instant::now() + BATCH_INTERVAL;
                    }
                }
                Some(Msg::Flush(ack)) => {
                    flush(&db, &mut buffer, &mut seq_counters, &broadcast).await;
                    let _ = ack.send(());
                    deadline = Instant::now() + BATCH_INTERVAL;
                }
            },
            _ = &mut tick => {
                if !buffer.is_empty() {
                    flush(&db, &mut buffer, &mut seq_counters, &broadcast).await;
                }
                deadline = Instant::now() + BATCH_INTERVAL;
            }
        }
    }
}

/// Write all buffered lines to the database in a single transaction.
///
/// The `seq` counter is per `(service_id, run_id)` and starts at 1,
/// incrementing monotonically across flush calls so ordering is preserved
/// even across batches. After a successful commit, the flush is sent on
/// `broadcast` as one batch so followers see output without polling.
async fn flush(
    db: &Database,
    buffer: &mut Vec<LogLine>,
    seq: &mut HashMap<(i64, i64), i64>,
    broadcast: &broadcast::Sender<Arc<LogBatch>>,
) {
    if buffer.is_empty() {
        return;
    }
    let lines = std::mem::take(buffer);

    // Allocate sequence numbers + build the rows to insert and the
    // messages to broadcast in one pass. Broadcast only fires after the
    // commit succeeds.
    let mut rows: Vec<ServiceLog> = Vec::with_capacity(lines.len());
    let mut to_broadcast: Vec<(i64, ApiLogLine)> = Vec::with_capacity(lines.len());

    for line in lines {
        let counter = seq.entry((line.service_id, line.run_id)).or_insert(0);
        *counter += 1;
        let api_line = ApiLogLine {
            timestamp_ms: line.timestamp_ms,
            stream: line.stream.as_str().to_string(),
            line: line.line.clone(),
            run_id: line.run_id,
            seq: *counter,
        };
        rows.push(ServiceLog {
            service_id: line.service_id,
            run_id: line.run_id,
            timestamp_ms: line.timestamp_ms,
            seq: *counter,
            stream: line.stream.as_str().to_string(),
            line: line.line,
        });
        to_broadcast.push((line.service_id, api_line));
    }

    if let Err(e) = db.insert_log_batch(&rows).await {
        warn!(error = %e, "log batch: insert failed");
        return;
    }

    // Errors from send are benign — no subscribers just means no
    // listeners.
    let _ = broadcast.send(Arc::new(LogBatch {
        lines: to_broadcast,
    }));
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::Database;

    async fn count_logs(db: &Database, service_id: i64) -> usize {
        db.fetch_service_logs(service_id).await.unwrap().len()
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn flushes_on_threshold() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let h = spawn(db.clone());

        for i in 0..BATCH_LINES as i64 {
            h.push(LogLine {
                service_id: svc,
                run_id: 1,
                timestamp_ms: i,
                stream: Stream::Stdout,
                line: format!("line {i}"),
            });
        }
        h.flush().await;

        assert_eq!(count_logs(&db, svc).await, BATCH_LINES);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn flushes_on_interval() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let h = spawn(db.clone());

        h.push(LogLine {
            service_id: svc,
            run_id: 1,
            timestamp_ms: 0,
            stream: Stream::Stdout,
            line: "first".into(),
        });
        // Advance paused time past the 200 ms deadline so the interval tick fires.
        tokio::time::sleep(Duration::from_millis(250)).await;
        h.flush().await;

        assert_eq!(count_logs(&db, svc).await, 1);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn seq_is_per_service_run_monotonic() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let h = spawn(db.clone());

        for i in 0..3 {
            h.push(LogLine {
                service_id: svc,
                run_id: 1,
                timestamp_ms: i,
                stream: Stream::Stdout,
                line: format!("{i}"),
            });
        }
        h.flush().await;

        let mut rows = db.fetch_service_logs(svc).await.unwrap();
        rows.sort_by_key(|r| r.timestamp_ms);
        let seqs: Vec<i64> = rows.into_iter().map(|r| r.seq).collect();
        assert_eq!(seqs, vec![1, 2, 3]);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_flush_is_broadcast_as_one_batch() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let other = db.upsert_service("other", 0).await.unwrap();
        let h = spawn(db.clone());
        let mut rx = h.subscribe();

        for (service_id, i) in [(svc, 0), (other, 1), (svc, 2)] {
            h.push(LogLine {
                service_id,
                run_id: 1,
                timestamp_ms: i,
                stream: Stream::Stdout,
                line: format!("{i}"),
            });
        }
        h.flush().await;

        let batch = rx.recv().await.unwrap();
        assert_eq!(batch.lines.len(), 3);
        let mine: Vec<i64> = batch.for_service(svc).map(|l| l.seq).collect();
        assert_eq!(mine, [1, 2]);
        assert!(rx.try_recv().is_err(), "one flush, one message");
    }
}
