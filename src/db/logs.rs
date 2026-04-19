//! Log batching writer.
//!
//! Contract: `BatcherHandle::push` is fire-and-forget. A single writer task owns
//! the SQLite connection and commits every 200 ms or every 100 lines, whichever
//! first. A shutdown signal triggers a final flush.

use std::collections::HashMap;
use std::time::Duration;

use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;
use tracing::warn;

use crate::db::Database;
use crate::db::models::ServiceLog;

const BATCH_LINES: usize = 100;
const BATCH_INTERVAL: Duration = Duration::from_millis(200);

/// Which output stream a log line came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    Stdout,
    Stderr,
}

impl Stream {
    fn as_str(self) -> &'static str {
        match self {
            Stream::Stdout => "stdout",
            Stream::Stderr => "stderr",
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

/// Cloneable handle to the background batcher task.
#[derive(Clone)]
pub struct BatcherHandle {
    tx: mpsc::UnboundedSender<Msg>,
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
    tokio::spawn(run(db, rx));
    BatcherHandle { tx }
}

async fn run(db: Database, mut rx: mpsc::UnboundedReceiver<Msg>) {
    let mut buffer: Vec<LogLine> = Vec::with_capacity(BATCH_LINES);
    let mut seq_counters: HashMap<(i64, i64), i64> = HashMap::new();
    // Use tokio's time::Instant so paused-time tests work correctly.
    let mut deadline = Instant::now() + BATCH_INTERVAL;

    loop {
        let tick = tokio::time::sleep_until(deadline);
        tokio::pin!(tick);

        tokio::select! {
            msg = rx.recv() => match msg {
                None => {
                    // Channel closed; do a final flush and exit.
                    flush(&db, &mut buffer, &mut seq_counters).await;
                    return;
                }
                Some(Msg::Line(line)) => {
                    buffer.push(line);
                    if buffer.len() >= BATCH_LINES {
                        flush(&db, &mut buffer, &mut seq_counters).await;
                        deadline = Instant::now() + BATCH_INTERVAL;
                    }
                }
                Some(Msg::Flush(ack)) => {
                    flush(&db, &mut buffer, &mut seq_counters).await;
                    let _ = ack.send(());
                    deadline = Instant::now() + BATCH_INTERVAL;
                }
            },
            _ = &mut tick => {
                if !buffer.is_empty() {
                    flush(&db, &mut buffer, &mut seq_counters).await;
                }
                deadline = Instant::now() + BATCH_INTERVAL;
            }
        }
    }
}

/// Write all buffered lines to the database in a single transaction.
///
/// The `seq` counter is per `(service_id, run_id)` and starts at 1, incrementing
/// monotonically across flush calls so ordering is preserved even across batches.
async fn flush(db: &Database, buffer: &mut Vec<LogLine>, seq: &mut HashMap<(i64, i64), i64>) {
    if buffer.is_empty() {
        return;
    }
    let lines = std::mem::take(buffer);

    let mut handle = db.handle();
    let mut tx = match handle.transaction().await {
        Ok(tx) => tx,
        Err(e) => {
            warn!(error = %e, "log batch: begin transaction failed");
            return;
        }
    };

    for line in lines {
        let counter = seq.entry((line.service_id, line.run_id)).or_insert(0);
        *counter += 1;
        let res = toasty::create!(ServiceLog {
            service_id: line.service_id,
            run_id: line.run_id,
            seq: *counter,
            timestamp_ms: line.timestamp_ms,
            stream: line.stream.as_str().to_string(),
            line: line.line,
        })
        .exec(&mut tx)
        .await;
        if let Err(e) = res {
            warn!(error = %e, "log batch: insert failed");
        }
    }

    if let Err(e) = tx.commit().await {
        warn!(error = %e, "log batch: commit failed");
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tempfile::tempdir;

    use super::*;
    use crate::db::Database;

    async fn count_logs(db: &Database) -> usize {
        let mut handle = db.handle();
        ServiceLog::all().exec(&mut handle).await.unwrap().len()
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn flushes_on_threshold() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("a.sqlite")).await.unwrap();
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

        assert_eq!(count_logs(&db).await, BATCH_LINES);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn flushes_on_interval() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("b.sqlite")).await.unwrap();
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

        assert_eq!(count_logs(&db).await, 1);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn seq_is_per_service_run_monotonic() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("c.sqlite")).await.unwrap();
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

        let mut handle = db.handle();
        let mut rows: Vec<ServiceLog> = ServiceLog::all().exec(&mut handle).await.unwrap();
        rows.sort_by_key(|r| r.timestamp_ms);
        let seqs: Vec<i64> = rows.into_iter().map(|r| r.seq).collect();
        assert_eq!(seqs, vec![1, 2, 3]);
    }
}
