//! Pump child stdout/stderr into the log batcher.

use tokio::io::{AsyncBufReadExt, BufReader};

use crate::{
    db::logs::{BatcherHandle, LogLine, Stream},
    system::DynAsyncRead,
};

/// Spawn a task that reads lines from `stdout` and forwards them to the batcher.
pub fn spawn_pump_stdout(
    stdout: DynAsyncRead,
    service_id: i64,
    run_id: i64,
    batcher: BatcherHandle,
) {
    tokio::spawn(pump(
        BufReader::new(stdout),
        service_id,
        run_id,
        Stream::Stdout,
        batcher,
    ));
}

/// Spawn a task that reads lines from `stderr` and forwards them to the batcher.
pub fn spawn_pump_stderr(
    stderr: DynAsyncRead,
    service_id: i64,
    run_id: i64,
    batcher: BatcherHandle,
) {
    tokio::spawn(pump(
        BufReader::new(stderr),
        service_id,
        run_id,
        Stream::Stderr,
        batcher,
    ));
}

/// Read lines from `reader` until EOF, pushing each into the batcher.
async fn pump<R: AsyncBufReadExt + Unpin>(
    mut reader: R,
    service_id: i64,
    run_id: i64,
    stream: Stream,
    batcher: BatcherHandle,
) {
    let mut buf = String::new();
    loop {
        buf.clear();
        match reader.read_line(&mut buf).await {
            Ok(0) => return,
            Ok(_) => {
                let line = buf.trim_end_matches(['\n', '\r']).to_string();
                batcher.push(LogLine {
                    service_id,
                    run_id,
                    timestamp_ms: crate::tracking::now_unix_ms(),
                    stream,
                    line,
                });
            }
            Err(_) => return,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::pin::Pin;

    use super::*;
    use crate::db::{Database, logs::spawn as spawn_batcher};

    /// Wrap a static byte buffer as a `DynAsyncRead` for tests that want to
    /// exercise the pump without spawning a real process.
    fn fixed(bytes: &'static [u8]) -> DynAsyncRead {
        Box::pin(Cursor::new(bytes)) as Pin<Box<_>>
    }

    #[tokio::test(flavor = "current_thread")]
    async fn pumps_lines_from_dyn_reader() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let batcher = spawn_batcher(db.clone());

        spawn_pump_stdout(fixed(b"hello\nworld\n"), svc, 1, batcher.clone());
        spawn_pump_stderr(fixed(b""), svc, 1, batcher.clone());

        // The pump tasks drain their readers and exit on EOF. Give them one
        // scheduler tick, then flush the batcher.
        tokio::task::yield_now().await;
        batcher.flush().await;

        let mut rows = db.fetch_service_logs(svc).await.unwrap();
        rows.sort_by_key(|r| r.seq);
        let lines: Vec<String> = rows.into_iter().map(|r| r.line).collect();
        assert_eq!(lines, vec!["hello".to_string(), "world".to_string()]);
    }
}
