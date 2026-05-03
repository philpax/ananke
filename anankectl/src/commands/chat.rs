//! Chat command — talks to a model via the OpenAI-compatible API.

use std::io::Write;

use ananke_api::ServicesResponse;
use futures::StreamExt;
use reqwest::Url;
use serde::Serialize;

use crate::client::{ApiClient, ApiClientError};

#[derive(Serialize)]
struct Message {
    role: &'static str,
    content: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
}

pub async fn run(
    client: &ApiClient,
    json: bool,
    model: &str,
    prompt: &str,
    system_prompt: &str,
) -> Result<(), ApiClientError> {
    // Discover the OpenAI port from the management API.
    let resp: ServicesResponse = client.get_json("/api/services").await?;
    let port = resp.openai_api_port;

    // Construct the OpenAI endpoint from the management endpoint's host and scheme.
    let openai_url = construct_openai_url(&client.endpoint, port)?;

    if json {
        run_non_streaming(openai_url, model, prompt, system_prompt).await
    } else {
        run_streaming(openai_url, model, prompt, system_prompt).await
    }
}

fn construct_openai_url(mgmt: &Url, port: u16) -> Result<Url, ApiClientError> {
    let host = mgmt
        .host_str()
        .ok_or_else(|| ApiClientError::Usage("management endpoint has no host".into()))?;
    let mut openai = mgmt.clone();
    openai.set_scheme(mgmt.scheme()).ok();
    openai.set_host(Some(host)).ok();
    let _ = openai.set_port(Some(port));
    Ok(openai)
}

async fn run_streaming(
    base: Url,
    model: &str,
    prompt: &str,
    system_prompt: &str,
) -> Result<(), ApiClientError> {
    let request = ChatRequest {
        model: model.to_string(),
        messages: vec![
            Message {
                role: "system",
                content: system_prompt.to_string(),
            },
            Message {
                role: "user",
                content: prompt.to_string(),
            },
        ],
        stream: true,
    };

    let body = serde_json::to_vec(&request)
        .map_err(|e| ApiClientError::Parse(format!("serialise chat request: {e}")))?;

    let url = base
        .join("v1/chat/completions")
        .map_err(|e| ApiClientError::Usage(format!("invalid openai path: {e}")))?;

    let resp = reqwest::Client::new()
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .body(body)
        .send()
        .await
        .map_err(ApiClientError::Connect)?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        return Err(ApiClientError::Http {
            status,
            body: body_text,
        });
    }

    let mut stream = resp.bytes_stream();
    let stdout = std::io::stdout();
    let mut locked = stdout.lock();

    // Accumulate a partial line buffer to handle SSE lines that span HTTP chunks.
    let mut buf = String::new();
    // Track whether we've seen the first non-empty content token to handle
    // leading whitespace that some models emit on the first chunk.
    let mut got_first = false;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(ApiClientError::Connect)?;
        buf.push_str(&String::from_utf8_lossy(&chunk));

        // Process each complete line in the buffer.
        while let Some(newline_pos) = buf.find('\n') {
            let line = buf[..newline_pos].to_string();
            buf.replace_range(..=newline_pos, "");

            if let Some(data) = line.trim().strip_prefix("data: ") {
                if data == "[DONE]" {
                    let _ = writeln!(locked);
                    let _ = locked.flush();
                    return Ok(());
                }
                if let Some(raw) = extract_content(data) {
                    if raw.is_empty() {
                        continue;
                    }
                    // Strip leading whitespace from the very first content token
                    // to handle models that emit "\n" in the initial delta event.
                    let token = if got_first { raw.as_str() } else { raw.trim_start() };
                    got_first = !token.is_empty();
                    if !token.is_empty() {
                        let _ = write!(locked, "{token}");
                        let _ = locked.flush();
                    }
                }
            }
        }
    }
    Ok(())
}

async fn run_non_streaming(
    base: Url,
    model: &str,
    prompt: &str,
    system_prompt: &str,
) -> Result<(), ApiClientError> {
    let request = ChatRequest {
        model: model.to_string(),
        messages: vec![
            Message {
                role: "system",
                content: system_prompt.to_string(),
            },
            Message {
                role: "user",
                content: prompt.to_string(),
            },
        ],
        stream: false,
    };

    let body = serde_json::to_vec(&request)
        .map_err(|e| ApiClientError::Parse(format!("serialise chat request: {e}")))?;

    let url = base
        .join("v1/chat/completions")
        .map_err(|e| ApiClientError::Usage(format!("invalid openai path: {e}")))?;

    let resp = reqwest::Client::new()
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .map_err(ApiClientError::Connect)?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        return Err(ApiClientError::Http {
            status,
            body: body_text,
        });
    }

    let json_val: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| ApiClientError::Parse(format!("parse response: {e}")))?;

    let content = json_val
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    println!("{}", serde_json::to_string(content).unwrap_or_default());
    Ok(())
}

/// Extract `delta.content` from a parsed SSE JSON payload.
fn extract_content(data: &str) -> Option<String> {
    let val: serde_json::Value = serde_json::from_str(data).ok()?;
    val.get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("delta"))
        .and_then(|d| d.get("content"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}
