//! `anankectl logs` command — paginated historical fetch with optional live
//! follow over `/api/live`.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use ananke_api::{
    internal::log_line::{LogLine, LogStream},
    live::{LiveFeed, LiveMessage, LiveRequest, LogFeed},
    services::logs::LogsResponse,
};
use futures::{SinkExt, StreamExt};
use reqwest::StatusCode;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

use crate::{
    client::{ApiClient, ApiClientError},
    output,
};

/// The follow's id on the live socket; the CLI holds only the one.
const FOLLOW_ID: &str = "logs";

// The signature mirrors the clap CLI surface, so the argument count is owned
// by the flag definitions rather than this function; grouping them into a
// struct would fight the derive.
#[expect(clippy::too_many_arguments)]
pub async fn run(
    client: &ApiClient,
    json: bool,
    name: &str,
    follow: bool,
    run: Option<i64>,
    since: Option<i64>,
    until: Option<i64>,
    limit: u32,
    stream: Option<LogStream>,
) -> Result<(), ApiClientError> {
    let mut query: Vec<String> = Vec::new();
    if let Some(v) = run {
        query.push(format!("run={v}"));
    }
    if let Some(v) = since {
        query.push(format!("since={v}"));
    }
    if let Some(v) = until {
        query.push(format!("until={v}"));
    }
    query.push(format!("limit={limit}"));
    if let Some(v) = stream {
        query.push(format!("stream={}", v.as_str()));
    }
    let path = format!("/api/services/{name}/logs?{}", query.join("&"));

    let response: LogsResponse = client.get_json(&path).await?;
    if json && !follow {
        output::print_json(&response);
        return Ok(());
    }
    // The response is newest-first; print oldest-first.
    for line in response.logs.iter().rev() {
        println!("{}", format_line(line, json));
    }
    if !follow {
        return Ok(());
    }

    let (mut ws, _) = connect_async(client.live_url()?)
        .await
        .map_err(|e| ApiClientError::WebSocket(e.to_string()))?;
    let request = LiveRequest::Follow {
        id: FOLLOW_ID.into(),
        feed: LiveFeed::Logs(LogFeed {
            service: name.into(),
            after: response.follow_after,
            run,
            stream,
        }),
    };
    let text = serde_json::to_string(&request).map_err(|e| ApiClientError::Parse(e.to_string()))?;
    ws.send(Message::Text(text.into()))
        .await
        .map_err(|e| ApiClientError::WebSocket(e.to_string()))?;

    while let Some(frame) = ws.next().await {
        let text = match frame {
            Ok(Message::Text(text)) => text,
            Ok(Message::Close(_)) => break,
            Ok(_) => continue,
            Err(e) => return Err(ApiClientError::WebSocket(e.to_string())),
        };
        let message = serde_json::from_str::<LiveMessage>(&text)
            .map_err(|e| ApiClientError::Parse(format!("live frame: {e}")))?;
        match render_frame(&message, json) {
            FrameAction::Print(lines) => {
                for line in lines {
                    println!("{line}");
                }
            }
            FrameAction::Warn(note) => eprintln!("{note}"),
            FrameAction::Fail(e) => return Err(e),
            FrameAction::Ignore => {}
        }
    }
    Ok(())
}

/// What to do with one frame from the live socket.
#[derive(Debug)]
enum FrameAction {
    /// Print these lines to stdout.
    Print(Vec<String>),
    /// Print this note to stderr.
    Warn(String),
    /// Stop with this error.
    Fail(ApiClientError),
    Ignore,
}

fn render_frame(message: &LiveMessage, json: bool) -> FrameAction {
    match message {
        LiveMessage::LogLines { lines, .. } => {
            FrameAction::Print(lines.iter().map(|l| format_line(l, json)).collect())
        }
        LiveMessage::LogGap { skipped, .. } => {
            FrameAction::Warn(format!("[{skipped} lines skipped]"))
        }
        // A resource error carries the HTTP status and envelope the same
        // request would get over HTTP, so it fails the same way.
        LiveMessage::Error { status, error, .. } => FrameAction::Fail(ApiClientError::Http {
            status: StatusCode::from_u16(*status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            body: serde_json::to_string(error).unwrap_or_default(),
        }),
        LiveMessage::Hello { .. } | LiveMessage::LogCaughtUp { .. } => FrameAction::Ignore,
        // The CLI subscribes to nothing, so a snapshot is not for it.
        LiveMessage::Snapshot { .. } => FrameAction::Ignore,
    }
}

/// Render one captured line: `[stream] text`, or the line as one JSON
/// object for `--json --follow`.
fn format_line(line: &LogLine, json: bool) -> String {
    if json {
        serde_json::to_string(line).unwrap_or_default()
    } else {
        format!("[{}] {}", line.stream, line.line)
    }
}

#[cfg(test)]
mod tests {
    use ananke_api::{
        internal::log_line::LogPosition,
        shared::errors::{ApiError, ApiErrorBody, ApiErrorCodeSlug, ApiErrorKind},
    };

    use super::*;

    fn line(stream: &str) -> LogLine {
        LogLine {
            timestamp_ms: 0,
            stream: stream.to_string(),
            line: "loading model".to_string(),
            run_id: 1,
            seq: 1,
        }
    }

    #[test]
    fn anankectl_accepts_and_prints_combined_stream() {
        // The label is printed verbatim, so a container's merged output is
        // never displayed as though it came from the child's stdout.
        assert_eq!(
            format_line(&line("combined"), false),
            "[combined] loading model"
        );
        assert_eq!(
            format_line(&line("stdout"), false),
            "[stdout] loading model"
        );
        assert_eq!(
            format_line(&line("stderr"), false),
            "[stderr] loading model"
        );
    }

    #[test]
    fn json_lines_are_one_object_each() {
        let rendered = format_line(&line("stdout"), true);
        assert!(!rendered.contains('\n'));
        let parsed: LogLine = serde_json::from_str(&rendered).unwrap();
        assert_eq!(parsed, line("stdout"));
    }

    #[test]
    fn log_lines_print_in_order() {
        let message = LiveMessage::LogLines {
            id: FOLLOW_ID.into(),
            lines: vec![line("stdout"), line("stderr")],
        };
        let FrameAction::Print(printed) = render_frame(&message, false) else {
            panic!("log lines must print");
        };
        assert_eq!(
            printed,
            ["[stdout] loading model", "[stderr] loading model"]
        );
    }

    #[test]
    fn a_gap_warns_with_its_count() {
        let message = LiveMessage::LogGap {
            id: FOLLOW_ID.into(),
            after: None,
            before: LogPosition {
                run_id: 1,
                seq: 501,
            },
            skipped: 500,
        };
        assert!(matches!(
            render_frame(&message, false),
            FrameAction::Warn(note) if note == "[500 lines skipped]"
        ));
    }

    #[test]
    fn an_error_fails_with_its_status() {
        let message = LiveMessage::Error {
            id: Some(FOLLOW_ID.into()),
            status: 404,
            error: ApiError {
                error: ApiErrorBody {
                    code: ApiErrorCodeSlug::ServiceNotFound,
                    message: "service `ghost` not found".into(),
                    kind: ApiErrorKind::InvalidRequestError,
                },
            },
        };
        assert!(matches!(
            render_frame(&message, false),
            FrameAction::Fail(ApiClientError::Http { status, .. }) if status == StatusCode::NOT_FOUND
        ));
    }

    #[test]
    fn markers_are_ignored() {
        for message in [
            LiveMessage::Hello {
                build: "0.2.0".into(),
            },
            LiveMessage::LogCaughtUp {
                id: FOLLOW_ID.into(),
            },
        ] {
            assert!(matches!(render_frame(&message, false), FrameAction::Ignore));
        }
    }
}
