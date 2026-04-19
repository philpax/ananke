//! `anankectl logs` command — paginated historical fetch with optional live tail.

use ananke_api::{LogLine, LogsResponse};
use futures::StreamExt;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

use crate::{
    client::{ApiClient, ApiClientError},
    output,
};

#[allow(clippy::too_many_arguments)]
pub async fn run(
    client: &ApiClient,
    json: bool,
    name: &str,
    follow: bool,
    run: Option<i64>,
    since: Option<i64>,
    until: Option<i64>,
    limit: u32,
    stream: Option<String>,
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
    if let Some(v) = stream.as_deref() {
        query.push(format!("stream={v}"));
    }
    let path = format!("/api/services/{name}/logs?{}", query.join("&"));

    // The response is newest-first; print oldest-first by iterating in reverse.
    let response: LogsResponse = client.get_json(&path).await?;
    // Track the highest seq seen so we can dedup during the live tail.
    let mut max_seq: Option<i64> = response.logs.first().map(|l| l.seq);
    if json {
        output::print_json(&response);
    } else {
        for line in response.logs.iter().rev() {
            print_line(line);
        }
    }

    if !follow {
        return Ok(());
    }

    // Upgrade to a WebSocket for the live tail.
    let ws_url = client
        .endpoint
        .join(&format!("/api/services/{name}/logs/stream"))
        .expect("valid path")
        .to_string()
        .replace("http://", "ws://")
        .replace("https://", "wss://");

    let (mut ws, _) = connect_async(ws_url)
        .await
        .map_err(|e| ApiClientError::WebSocket(e.to_string()))?;

    while let Some(Ok(Message::Text(s))) = ws.next().await {
        let Ok(line) = serde_json::from_str::<LogLine>(&s) else {
            continue;
        };
        // Skip lines we already printed from the historical fetch.
        if let Some(prev) = max_seq
            && line.seq <= prev
        {
            continue;
        }
        max_seq = Some(line.seq);
        print_line(&line);
    }

    Ok(())
}

fn print_line(line: &LogLine) {
    println!("[{}] {}", line.stream, line.line);
}
