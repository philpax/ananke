//! Pump child stdout/stderr into the log batcher.

use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::{ChildStderr, ChildStdout},
};

use crate::db::logs::{BatcherHandle, LogLine, Stream};

/// Spawn a task that reads lines from `stdout` and forwards them to the batcher.
pub fn spawn_pump_stdout(
    stdout: ChildStdout,
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
    stderr: ChildStderr,
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
    use tempfile::tempdir;
    use tokio::process::Command;

    use super::*;
    use crate::db::{Database, logs::spawn as spawn_batcher};

    #[tokio::test(flavor = "current_thread")]
    async fn pumps_echoed_lines() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("a.sqlite")).await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let batcher = spawn_batcher(db.clone());

        let mut child = Command::new("/bin/sh")
            .arg("-c")
            .arg("printf 'hello\\nworld\\n'; exit 0")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        spawn_pump_stdout(stdout, svc, 1, batcher.clone());
        spawn_pump_stderr(stderr, svc, 1, batcher.clone());

        let _ = child.wait().await;
        // Wait a tick for the pump tasks to drain before flushing.
        batcher.flush().await;
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        batcher.flush().await;

        let mut handle = db.handle();
        let mut rows: Vec<crate::db::models::ServiceLog> = crate::db::models::ServiceLog::filter(
            crate::db::models::ServiceLog::fields().service_id().eq(svc),
        )
        .exec(&mut handle)
        .await
        .unwrap();
        rows.sort_by_key(|r| r.seq);
        let lines: Vec<String> = rows.into_iter().map(|r| r.line).collect();
        assert_eq!(lines, vec!["hello".to_string(), "world".to_string()]);
    }
}
