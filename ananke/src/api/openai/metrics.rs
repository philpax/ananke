//! Response body inspection for per-request metrics recording.
//!
//! Wraps the proxied response body to extract `usage` (token counts) and
//! time-to-first-token from the SSE stream (or JSON body) as it passes
//! through to the client. When the stream ends, the recorded data is
//! written to the `request_metrics` table via a spawned task.

use std::time::Instant;

use bytes::Bytes;
use hyper::body::Frame;
use serde_json::Value;
use tracing::warn;

use crate::db::{Database, models::RequestMetric};

/// Collects metrics from a proxied response body. For streaming responses,
/// parses SSE `data:` lines incrementally. For non-streaming, buffers the
/// body and parses as JSON on completion.
pub struct MetricsRecorder {
    start: Instant,
    service_id: i64,
    run_id: Option<i64>,
    model: String,
    endpoint: &'static str,
    is_streaming: bool,
    first_token_at: Option<Instant>,
    prompt_tokens: Option<i64>,
    completion_tokens: Option<i64>,
    /// For SSE parsing: accumulates a partial line that spans a chunk
    /// boundary. For non-streaming: accumulates the full body (capped).
    buf: String,
    /// Cap on the buffer size. For non-streaming, most responses fit in
    /// well under this. For streaming, the `usage` field is in the final
    /// SSE chunk, so we never need more than the last chunk — but keeping
    /// the full buffer simplifies parsing. 256 KiB is generous.
    buf_cap: usize,
}

impl MetricsRecorder {
    pub fn new(
        start: Instant,
        service_id: i64,
        run_id: Option<i64>,
        model: String,
        endpoint: &'static str,
        is_streaming: bool,
    ) -> Self {
        Self {
            start,
            service_id,
            run_id,
            model,
            endpoint,
            is_streaming,
            first_token_at: None,
            prompt_tokens: None,
            completion_tokens: None,
            buf: String::new(),
            buf_cap: 256 * 1024,
        }
    }

    /// Feed response data bytes into the recorder. Passes through
    /// unchanged — the caller still sends `data` to the client.
    pub fn ingest(&mut self, data: &Bytes) {
        if self.buf.len() >= self.buf_cap {
            // Past the cap: for streaming, clear and keep only new data
            // (usage is in the final chunk). For non-streaming, stop
            // accumulating — we won't be able to parse the full JSON.
            if self.is_streaming {
                self.buf.clear();
            } else {
                return;
            }
        }
        self.buf.push_str(&String::from_utf8_lossy(data));
        if self.is_streaming {
            self.process_sse_lines();
        }
    }

    /// Split the buffer on newlines and process each complete SSE line.
    fn process_sse_lines(&mut self) {
        while let Some(pos) = self.buf.find('\n') {
            let line = self.buf[..pos].to_string();
            self.buf.replace_range(..=pos, "");
            self.process_sse_line(&line);
        }
    }

    fn process_sse_line(&mut self, line: &str) {
        let Some(data) = line.trim().strip_prefix("data: ") else {
            return;
        };
        if data == "[DONE]" {
            return;
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            return;
        };
        // Extract usage (usually in the final chunk).
        if let Some(usage) = v.get("usage") {
            self.prompt_tokens = usage.get("prompt_tokens").and_then(|t| t.as_i64());
            self.completion_tokens = usage.get("completion_tokens").and_then(|t| t.as_i64());
        }
        // Record TTFT on the first content or reasoning chunk.
        if self.first_token_at.is_none() {
            let has_content = v
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("delta"))
                .and_then(|d| {
                    d.get("content")
                        .or(d.get("reasoning_content"))
                        .or(d.get("reasoning"))
                })
                .and_then(|c| c.as_str())
                .map(|s| !s.is_empty())
                .unwrap_or(false);
            if has_content {
                self.first_token_at = Some(Instant::now());
            }
        }
    }

    /// Called when the response stream ends. Extracts any remaining
    /// data (e.g. usage from a non-streaming JSON body) and spawns a
    /// task to write the metric to the database.
    pub fn finish(self, db: Database, status_code: u16) {
        let mut rec = self;
        // For non-streaming, parse the accumulated body as JSON.
        if !rec.is_streaming
            && !rec.buf.is_empty()
            && let Ok(v) = serde_json::from_str::<Value>(&rec.buf)
            && let Some(usage) = v.get("usage")
        {
            rec.prompt_tokens = usage.get("prompt_tokens").and_then(|t| t.as_i64());
            rec.completion_tokens = usage.get("completion_tokens").and_then(|t| t.as_i64());
        }

        let duration_ms = rec.start.elapsed().as_millis() as i64;
        let ttft_ms = rec
            .first_token_at
            .map(|t| t.duration_since(rec.start).as_millis() as i64);
        let timestamp_ms = crate::tracking::now_unix_ms() - duration_ms;

        let metric = RequestMetric {
            metric_id: 0,
            service_id: rec.service_id,
            run_id: rec.run_id,
            timestamp_ms,
            endpoint: rec.endpoint.to_string(),
            model: rec.model.clone(),
            prompt_tokens: rec.prompt_tokens,
            completion_tokens: rec.completion_tokens,
            duration_ms: Some(duration_ms),
            ttft_ms,
            status_code: status_code as i64,
        };

        tokio::spawn(async move {
            if let Err(e) = db.insert_request_metric(&metric).await {
                warn!(error = %e, "failed to record request metric");
            }
        });
    }
}

pin_project_lite::pin_project! {
    /// Wraps a body with a [`MetricsRecorder`]. Passes all data through
    /// unchanged while feeding bytes to the recorder. When the stream
    /// ends, the recorder writes the metric to the database.
    pub struct MetricsBody<B> {
        #[pin]
        body: B,
        recorder: Option<MetricsRecorder>,
        db: Option<Database>,
        status_code: u16,
        recorded: bool,
    }
}

impl<B: hyper::body::Body> MetricsBody<B> {
    pub fn new(body: B, recorder: MetricsRecorder, db: Database, status_code: u16) -> Self {
        Self {
            body,
            recorder: Some(recorder),
            db: Some(db),
            status_code,
            recorded: false,
        }
    }
}

impl<B: hyper::body::Body<Data = Bytes>> hyper::body::Body for MetricsBody<B> {
    type Data = B::Data;
    type Error = B::Error;

    fn poll_frame(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.project();
        match this.body.poll_frame(cx) {
            std::task::Poll::Ready(None) => {
                if !*this.recorded {
                    *this.recorded = true;
                    if let (Some(recorder), Some(db)) = (this.recorder.take(), this.db.take()) {
                        recorder.finish(db, *this.status_code);
                    }
                }
                std::task::Poll::Ready(None)
            }
            std::task::Poll::Ready(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref()
                    && let Some(recorder) = this.recorder.as_mut()
                {
                    recorder.ingest(data);
                }
                std::task::Poll::Ready(Some(Ok(frame)))
            }
            std::task::Poll::Ready(Some(Err(e))) => {
                if !*this.recorded {
                    *this.recorded = true;
                    if let (Some(recorder), Some(db)) = (this.recorder.take(), this.db.take()) {
                        recorder.finish(db, *this.status_code);
                    }
                }
                std::task::Poll::Ready(Some(Err(e)))
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }

    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }

    fn size_hint(&self) -> hyper::body::SizeHint {
        self.body.size_hint()
    }
}
