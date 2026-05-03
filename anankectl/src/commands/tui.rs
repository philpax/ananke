//! Interactive TUI chat interface using ratatui.

use std::{
    io,
    time::{Duration, Instant},
};

use ananke_api::ServicesResponse;
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, MouseEventKind,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use futures::StreamExt;
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};
use tokio::sync::mpsc;

use crate::client::{ApiClient, ApiClientError};

const INPUT_MIN_CONTENT_ROWS: u16 = 3;
const INPUT_MAX_CONTENT_ROWS: u16 = 10;

#[derive(Debug, Clone, Copy, PartialEq)]
enum MsgRole {
    System,
    Assistant,
    User,
}

impl MsgRole {
    fn label(&self) -> &'static str {
        match self {
            Self::System => "System",
            Self::Assistant => "Assistant",
            Self::User => "User",
        }
    }

    fn color(&self) -> Color {
        match self {
            Self::System => Color::Magenta,
            Self::Assistant => Color::Cyan,
            Self::User => Color::Yellow,
        }
    }

    fn wire_role(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Assistant => "assistant",
            Self::User => "user",
        }
    }
}

struct TuiMsg {
    role: MsgRole,
    /// The visible final message body — what the model "says".
    content: String,
    /// Subdued reasoning/thinking content (if the model emits it via
    /// `delta.reasoning_content` or `delta.reasoning`). Rendered above
    /// `content` in a dim style.
    reasoning: String,
    streaming: bool,
    /// Live decode/usage timing for assistant messages. Frozen once the
    /// turn finishes. `None` for user/system messages.
    stats: Option<TurnStats>,
}

/// Live and final timing data for a single assistant turn.
struct TurnStats {
    /// When the request was dispatched.
    start: Instant,
    /// When the first decoded delta (content or reasoning) arrived.
    first_token_at: Option<Instant>,
    /// When the stream finished (either `[DONE]` or transport close).
    end: Option<Instant>,
    /// Decode chunks observed (content + reasoning), used as the live
    /// fallback when the server doesn't echo `usage`.
    content_chunks: u32,
    reasoning_chunks: u32,
    /// Token counts from the streamed `usage` chunk (if the server emits
    /// one — most llama.cpp-style servers do when `stream_options.include_usage`
    /// is set).
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
}

impl TurnStats {
    fn new(start: Instant) -> Self {
        Self {
            start,
            first_token_at: None,
            end: None,
            content_chunks: 0,
            reasoning_chunks: 0,
            prompt_tokens: None,
            completion_tokens: None,
        }
    }

    fn ttft(&self) -> Option<Duration> {
        self.first_token_at.map(|t| t.duration_since(self.start))
    }

    /// Tokens-per-second for decode, computed live from chunk counts.
    /// Returns `None` until at least 200 ms of decode have elapsed so the
    /// number doesn't whiplash on the first chunk.
    fn live_decode_rate(&self, now: Instant) -> Option<f64> {
        let first = self.first_token_at?;
        let elapsed = now.duration_since(first).as_secs_f64();
        if elapsed < 0.2 {
            return None;
        }
        let total = (self.content_chunks + self.reasoning_chunks) as f64;
        if total == 0.0 {
            return None;
        }
        Some(total / elapsed)
    }

    /// Final tokens-per-second for decode. Prefers `completion_tokens` from
    /// the usage chunk, falling back to chunk counts.
    fn final_decode_rate(&self) -> Option<f64> {
        let first = self.first_token_at?;
        let end = self.end?;
        let elapsed = end.duration_since(first).as_secs_f64();
        if elapsed <= 0.0 {
            return None;
        }
        let n = self
            .completion_tokens
            .map(|t| t as f64)
            .unwrap_or((self.content_chunks + self.reasoning_chunks) as f64);
        Some(n / elapsed)
    }

    /// Prompt-processing rate: `prompt_tokens / ttft`. Only available once
    /// usage has arrived.
    fn pp_rate(&self) -> Option<f64> {
        let pt = self.prompt_tokens? as f64;
        let ttft = self.ttft()?.as_secs_f64();
        if ttft <= 0.0 {
            return None;
        }
        Some(pt / ttft)
    }

    /// Live or final decoded-token count to display next to the rate.
    fn decoded_tokens(&self) -> u32 {
        self.completion_tokens
            .unwrap_or(self.content_chunks + self.reasoning_chunks)
    }
}

struct TuiState {
    messages: Vec<TuiMsg>,
    input: String,
    /// Lines to scroll back from the bottom. 0 means stuck to the latest output.
    scroll_offset: u16,
    streaming: bool,
    first_token: bool,
    error: Option<String>,
    model: String,
}

impl TuiState {
    fn new(system_prompt: &str, model: String) -> Self {
        let messages = if system_prompt.is_empty() {
            Vec::new()
        } else {
            vec![TuiMsg {
                role: MsgRole::System,
                content: system_prompt.to_string(),
                reasoning: String::new(),
                streaming: false,
                stats: None,
            }]
        };
        Self {
            messages,
            input: String::new(),
            scroll_offset: 0,
            streaming: false,
            first_token: false,
            error: None,
            model,
        }
    }

    /// Push the user's message and an empty streaming assistant message,
    /// and return the wire-format history to send (excluding the empty
    /// assistant we just appended).
    fn submit(&mut self, content: String) -> Vec<WireMessage> {
        self.error = None;
        self.messages.push(TuiMsg {
            role: MsgRole::User,
            content,
            reasoning: String::new(),
            streaming: false,
            stats: None,
        });
        let history = self
            .messages
            .iter()
            .map(|m| WireMessage {
                role: m.role.wire_role(),
                content: m.content.clone(),
            })
            .collect();
        self.messages.push(TuiMsg {
            role: MsgRole::Assistant,
            content: String::new(),
            reasoning: String::new(),
            streaming: true,
            stats: Some(TurnStats::new(Instant::now())),
        });
        self.streaming = true;
        self.first_token = false;
        self.scroll_offset = 0;
        history
    }

    fn append_token(&mut self, token: String) {
        let Some(last) = self.messages.last_mut() else {
            return;
        };
        if !last.streaming {
            return;
        }
        let pushed = if !self.first_token {
            // Trim leading whitespace from the first content token to handle
            // models that emit "\n" in the initial delta event.
            let trimmed = token.trim_start();
            if trimmed.is_empty() {
                return;
            }
            last.content.push_str(trimmed);
            self.first_token = true;
            true
        } else {
            last.content.push_str(&token);
            true
        };
        if pushed && let Some(stats) = last.stats.as_mut() {
            stats.first_token_at.get_or_insert_with(Instant::now);
            stats.content_chunks = stats.content_chunks.saturating_add(1);
        }
        self.scroll_offset = 0;
    }

    fn append_reasoning(&mut self, token: String) {
        let Some(last) = self.messages.last_mut() else {
            return;
        };
        if !last.streaming {
            return;
        }
        // Trim leading whitespace if reasoning is starting fresh.
        let pushed = if last.reasoning.is_empty() {
            let trimmed = token.trim_start();
            if trimmed.is_empty() {
                return;
            }
            last.reasoning.push_str(trimmed);
            true
        } else {
            last.reasoning.push_str(&token);
            true
        };
        if pushed && let Some(stats) = last.stats.as_mut() {
            stats.first_token_at.get_or_insert_with(Instant::now);
            stats.reasoning_chunks = stats.reasoning_chunks.saturating_add(1);
        }
        self.scroll_offset = 0;
    }

    fn apply_usage(&mut self, prompt: Option<u32>, completion: Option<u32>) {
        let Some(last) = self.messages.last_mut() else {
            return;
        };
        let Some(stats) = last.stats.as_mut() else {
            return;
        };
        if let Some(p) = prompt {
            stats.prompt_tokens = Some(p);
        }
        if let Some(c) = completion {
            stats.completion_tokens = Some(c);
        }
    }

    fn finish_streaming(&mut self) {
        if let Some(last) = self.messages.last_mut()
            && last.streaming
        {
            last.streaming = false;
            if let Some(stats) = last.stats.as_mut() {
                stats.end.get_or_insert_with(Instant::now);
            }
        }
        self.streaming = false;
    }

    fn set_error(&mut self, error: String) {
        self.error = Some(error);
        // Drop the placeholder assistant message if it carries no body and
        // no reasoning — there is nothing worth keeping in history.
        if let Some(last) = self.messages.last()
            && last.streaming
            && last.content.is_empty()
            && last.reasoning.is_empty()
        {
            self.messages.pop();
        } else if let Some(last) = self.messages.last_mut()
            && last.streaming
        {
            last.streaming = false;
            if let Some(stats) = last.stats.as_mut() {
                stats.end.get_or_insert_with(Instant::now);
            }
        }
        self.streaming = false;
    }

    fn scroll_up(&mut self, amount: u16) {
        self.scroll_offset = self.scroll_offset.saturating_add(amount);
    }

    fn scroll_down(&mut self, amount: u16) {
        self.scroll_offset = self.scroll_offset.saturating_sub(amount);
    }

    fn scroll_to_bottom(&mut self) {
        self.scroll_offset = 0;
    }
}

pub async fn run(
    client: &ApiClient,
    model: &str,
    system_prompt: &str,
) -> Result<(), ApiClientError> {
    // Discover the OpenAI port from the management API.
    let resp: ServicesResponse = client.get_json("/api/services").await?;
    let port = resp.openai_api_port;

    let openai_url = construct_openai_url(&client.endpoint, port)?;

    // Channels: TUI -> dispatcher carries new conversation submissions;
    // dispatcher -> TUI carries SSE updates.
    let (req_tx, req_rx) = mpsc::channel::<Vec<WireMessage>>(8);
    let (sse_tx, sse_rx) = mpsc::channel::<SSEUpdate>(64);

    let chat_handle = tokio::spawn(chat_dispatcher(
        openai_url,
        model.to_string(),
        req_rx,
        sse_tx,
    ));

    // Set up terminal.
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    if let Err(e) = execute!(stdout, EnterAlternateScreen, EnableMouseCapture) {
        let _ = disable_raw_mode();
        return Err(e.into());
    }

    let state = TuiState::new(system_prompt, model.to_string());

    let result = tokio::task::spawn_blocking(move || {
        let backend = ratatui::backend::CrosstermBackend::new(io::stdout());
        let mut terminal = ratatui::Terminal::new(backend)?;
        let mut state = state;
        let mut sse_rx = sse_rx;
        let req_tx = req_tx;
        let inner = run_tui(&mut terminal, &mut state, &mut sse_rx, &req_tx);
        // Restore terminal regardless of inner result.
        let _ = execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        );
        let _ = disable_raw_mode();
        let _ = terminal.show_cursor();
        inner
    })
    .await
    .map_err(|e| ApiClientError::Usage(format!("TUI panicked: {e}")))?;

    // Closing req_tx (dropped above) signals the dispatcher to exit; await it.
    let _ = chat_handle.await;

    result
}

fn construct_openai_url(mgmt: &reqwest::Url, port: u16) -> Result<reqwest::Url, ApiClientError> {
    let host = mgmt
        .host_str()
        .ok_or_else(|| ApiClientError::Usage("management endpoint has no host".into()))?;
    let mut openai = mgmt.clone();
    openai.set_scheme(mgmt.scheme()).ok();
    openai.set_host(Some(host)).ok();
    let _ = openai.set_port(Some(port));
    Ok(openai)
}

enum SSEUpdate {
    Content(String),
    Reasoning(String),
    Usage {
        prompt_tokens: Option<u32>,
        completion_tokens: Option<u32>,
    },
    Done,
    Error(String),
}

#[derive(serde::Serialize, Clone)]
struct WireMessage {
    role: &'static str,
    content: String,
}

#[derive(serde::Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<WireMessage>,
    stream: bool,
    stream_options: StreamOptions,
}

#[derive(serde::Serialize)]
struct StreamOptions {
    include_usage: bool,
}

async fn chat_dispatcher(
    base: reqwest::Url,
    model: String,
    mut req_rx: mpsc::Receiver<Vec<WireMessage>>,
    sse_tx: mpsc::Sender<SSEUpdate>,
) {
    let client = reqwest::Client::new();
    while let Some(messages) = req_rx.recv().await {
        if let Err(e) = stream_one(&client, &base, &model, messages, &sse_tx).await {
            let _ = sse_tx.send(SSEUpdate::Error(e)).await;
        }
    }
}

async fn stream_one(
    client: &reqwest::Client,
    base: &reqwest::Url,
    model: &str,
    messages: Vec<WireMessage>,
    sse_tx: &mpsc::Sender<SSEUpdate>,
) -> Result<(), String> {
    let request = ChatRequest {
        model: model.to_string(),
        messages,
        stream: true,
        stream_options: StreamOptions {
            include_usage: true,
        },
    };

    let body = serde_json::to_vec(&request).map_err(|e| format!("serialise chat request: {e}"))?;
    let url = base
        .join("v1/chat/completions")
        .map_err(|e| format!("invalid openai path: {e}"))?;

    let resp = client
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .body(body)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        return Err(format!("HTTP {status}: {body_text}"));
    }

    let mut stream = resp.bytes_stream();
    let mut buf = String::new();

    loop {
        tokio::select! {
            chunk_result = stream.next() => {
                let Some(chunk_result) = chunk_result else {
                    let _ = sse_tx.send(SSEUpdate::Done).await;
                    return Ok(());
                };
                let chunk = chunk_result.map_err(|e| e.to_string())?;
                buf.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(newline_pos) = buf.find('\n') {
                    let line = buf[..newline_pos].to_string();
                    buf.replace_range(..=newline_pos, "");

                    if let Some(data) = line.trim().strip_prefix("data: ") {
                        if data == "[DONE]" {
                            let _ = sse_tx.send(SSEUpdate::Done).await;
                            return Ok(());
                        }
                        let DeltaParts { content, reasoning } = extract_delta(data);
                        if let Some(r) = reasoning
                            && !r.is_empty()
                            && sse_tx.send(SSEUpdate::Reasoning(r)).await.is_err()
                        {
                            return Ok(());
                        }
                        if let Some(c) = content
                            && !c.is_empty()
                            && sse_tx.send(SSEUpdate::Content(c)).await.is_err()
                        {
                            return Ok(());
                        }
                        if let Some(usage) = extract_usage(data)
                            && sse_tx
                                .send(SSEUpdate::Usage {
                                    prompt_tokens: usage.prompt_tokens,
                                    completion_tokens: usage.completion_tokens,
                                })
                                .await
                                .is_err()
                        {
                            return Ok(());
                        }
                    }
                }
            }
            _ = sse_tx.closed() => {
                return Ok(());
            }
        }
    }
}

struct DeltaParts {
    content: Option<String>,
    reasoning: Option<String>,
}

/// Pull both `delta.content` and reasoning text from one SSE payload.
/// Reasoning may arrive under `reasoning_content` (DeepSeek style) or
/// plain `reasoning`; we accept either.
fn extract_delta(data: &str) -> DeltaParts {
    let Ok(val) = serde_json::from_str::<serde_json::Value>(data) else {
        return DeltaParts {
            content: None,
            reasoning: None,
        };
    };
    let Some(delta) = val
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("delta"))
    else {
        return DeltaParts {
            content: None,
            reasoning: None,
        };
    };
    let content = delta
        .get("content")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let reasoning = delta
        .get("reasoning_content")
        .and_then(|v| v.as_str())
        .or_else(|| delta.get("reasoning").and_then(|v| v.as_str()))
        .map(str::to_string);
    DeltaParts { content, reasoning }
}

struct UsageInfo {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
}

/// Pull `usage.{prompt,completion}_tokens` from an SSE payload. OpenAI-style
/// servers emit these in a final delta when `stream_options.include_usage`
/// is set; chunks without a `usage` object return `None`.
fn extract_usage(data: &str) -> Option<UsageInfo> {
    let val = serde_json::from_str::<serde_json::Value>(data).ok()?;
    let usage = val.get("usage")?;
    let prompt = usage
        .get("prompt_tokens")
        .and_then(|v| v.as_u64())
        .map(|n| n as u32);
    let completion = usage
        .get("completion_tokens")
        .and_then(|v| v.as_u64())
        .map(|n| n as u32);
    if prompt.is_none() && completion.is_none() {
        return None;
    }
    Some(UsageInfo {
        prompt_tokens: prompt,
        completion_tokens: completion,
    })
}

fn run_tui(
    terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>,
    state: &mut TuiState,
    sse_rx: &mut mpsc::Receiver<SSEUpdate>,
    req_tx: &mpsc::Sender<Vec<WireMessage>>,
) -> Result<(), ApiClientError> {
    loop {
        // Drain any pending SSE updates before drawing.
        while let Ok(update) = sse_rx.try_recv() {
            match update {
                SSEUpdate::Content(token) => state.append_token(token),
                SSEUpdate::Reasoning(token) => state.append_reasoning(token),
                SSEUpdate::Usage {
                    prompt_tokens,
                    completion_tokens,
                } => state.apply_usage(prompt_tokens, completion_tokens),
                SSEUpdate::Done => state.finish_streaming(),
                SSEUpdate::Error(e) => state.set_error(e),
            }
        }

        terminal.draw(|f| render(f, state))?;

        if !event::poll(std::time::Duration::from_millis(50))? {
            continue;
        }
        match event::read()? {
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                if let Some(action) = handle_key(key, state) {
                    match action {
                        KeyAction::Quit => return Ok(()),
                        KeyAction::Submit => {
                            let user_input = std::mem::take(&mut state.input);
                            let trimmed = user_input.trim();
                            if trimmed.is_empty() {
                                continue;
                            }
                            let history = state.submit(trimmed.to_string());
                            if req_tx.blocking_send(history).is_err() {
                                state.set_error("chat dispatcher closed".to_string());
                            }
                        }
                    }
                }
            }
            Event::Mouse(me) => match me.kind {
                MouseEventKind::ScrollUp => state.scroll_up(3),
                MouseEventKind::ScrollDown => state.scroll_down(3),
                _ => {}
            },
            _ => {}
        }
    }
}

enum KeyAction {
    Quit,
    Submit,
}

/// Handle a single key press, mutating `state` directly for in-line edits
/// and returning a `KeyAction` for things the caller has to coordinate
/// (channel sends, returning from the loop).
///
/// Note on Shift+Enter: many terminals don't deliver a `KeyCode::Enter`
/// with the SHIFT modifier — instead the user must bind Shift+Enter to
/// transmit a literal `\n`. That arrives at crossterm as `Ctrl+J`
/// (since 0x0A == ASCII Ctrl+J), so we treat both as "insert newline".
fn handle_key(key: crossterm::event::KeyEvent, state: &mut TuiState) -> Option<KeyAction> {
    use crossterm::event::KeyModifiers;
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);

    match key.code {
        KeyCode::Char('c') if ctrl => Some(KeyAction::Quit),
        KeyCode::Char('d') if ctrl && state.input.is_empty() => Some(KeyAction::Quit),
        // Shift+Enter via terminal modifier reporting (rare).
        KeyCode::Enter if shift => {
            state.input.push('\n');
            None
        }
        // Shift+Enter via terminal config sending a literal LF (0x0A),
        // which crossterm decodes as Ctrl+J.
        KeyCode::Char('j') if ctrl => {
            state.input.push('\n');
            None
        }
        KeyCode::Enter => {
            if state.streaming {
                None
            } else {
                Some(KeyAction::Submit)
            }
        }
        KeyCode::Backspace => {
            state.input.pop();
            None
        }
        KeyCode::Up => {
            state.scroll_up(1);
            None
        }
        KeyCode::Down => {
            state.scroll_down(1);
            None
        }
        KeyCode::End if ctrl => {
            state.scroll_to_bottom();
            None
        }
        KeyCode::Esc => {
            state.input.clear();
            None
        }
        KeyCode::Char(c) => {
            state.input.push(c);
            None
        }
        _ => None,
    }
}

fn render(f: &mut Frame, state: &TuiState) {
    // Layout: header (1) | messages (min) | input (grows) | status (1).
    // No borders on header/status, so Length(1) fits a single line. The
    // input grows with its content (between INPUT_MIN_CONTENT_ROWS and
    // INPUT_MAX_CONTENT_ROWS rows of content, plus two border rows); past
    // the cap the input area scrolls internally to keep the latest line
    // visible.
    let area = f.area();
    let inner_width = area.width.saturating_sub(2);
    let content_rows = input_content_rows(&state.input, inner_width);
    let input_height = content_rows.saturating_add(2);

    let chunks = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(1),
        Constraint::Length(input_height),
        Constraint::Length(1),
    ])
    .split(area);

    render_header(f, chunks[0], state);
    render_messages(f, chunks[1], state);
    render_input(f, chunks[2], state);
    render_status(f, chunks[3]);
}

fn render_header(f: &mut Frame, area: ratatui::layout::Rect, state: &TuiState) {
    let (sym, sym_color, label, label_color) = if let Some(err) = &state.error {
        ("✗", Color::Red, format!("Error: {err}"), Color::Red)
    } else if state.streaming {
        ("●", Color::Cyan, "Streaming…".to_string(), Color::White)
    } else {
        ("●", Color::Green, "Ready".to_string(), Color::White)
    };
    let model_label = format!(" · {}", state.model);
    let line = Line::from(vec![
        Span::styled(format!("{sym} "), Style::default().fg(sym_color)),
        Span::styled(
            label,
            Style::default()
                .fg(label_color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(model_label, Style::default().fg(Color::DarkGray)),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

fn render_messages(f: &mut Frame, area: Rect, state: &TuiState) {
    if area.width < 4 || area.height == 0 {
        return;
    }
    // Each message gets its own bordered block; the inner content width is
    // 2 less than the area's width.
    let content_width = area.width.saturating_sub(2);

    // Build per-message lines + total bordered height up front so we can
    // determine scroll. Messages butt up directly so adjacent borders share
    // a row.
    let rendered: Vec<(Vec<Line>, u16)> = state
        .messages
        .iter()
        .map(|m| {
            let lines = build_message_lines(m);
            let rows = visual_line_count(&lines, content_width);
            // u32 -> u16 with headroom for the +2 borders, never overflowing.
            let content = rows.min((u16::MAX - 2) as u32) as u16;
            (lines, content.saturating_add(2))
        })
        .collect();

    let total_height: u32 = rendered.iter().map(|(_, h)| *h as u32).sum();
    let visible_height = area.height as u32;

    // Anchor to the bottom; user scroll offset moves the window upward.
    let max_window_top = total_height.saturating_sub(visible_height);
    let window_top = max_window_top.saturating_sub(state.scroll_offset as u32);

    // Walk the virtual stack, emitting any visible portion of each message.
    let now = Instant::now();
    let mut virtual_y: u32 = 0;
    for (i, (lines, h)) in rendered.iter().enumerate() {
        let msg_top = virtual_y;
        let msg_bottom = virtual_y + *h as u32;
        virtual_y = msg_bottom;

        let win_top = window_top;
        let win_bottom = window_top + visible_height;
        if msg_bottom <= win_top || msg_top >= win_bottom {
            continue;
        }

        let clipped_top = msg_top.max(win_top);
        let clipped_bottom = msg_bottom.min(win_bottom);
        let render_y = area.y + (clipped_top - win_top) as u16;
        let render_h = (clipped_bottom - clipped_top) as u16;
        let internal_scroll = (clipped_top - msg_top) as u16;

        let rect = Rect {
            x: area.x,
            y: render_y,
            width: area.width,
            height: render_h,
        };

        let msg = &state.messages[i];
        let title_line = build_title_line(msg, now);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(msg.role.color()))
            .title(title_line);

        let paragraph = Paragraph::new(lines.clone())
            .wrap(Wrap { trim: false })
            .scroll((internal_scroll, 0))
            .block(block);
        f.render_widget(paragraph, rect);
    }
}

/// Build a title line for a message box: role label followed by any
/// per-turn stats (live during streaming, frozen after `Done`).
fn build_title_line(msg: &TuiMsg, now: Instant) -> Line<'static> {
    let dim = Style::default().fg(Color::DarkGray);
    let mut spans = vec![Span::styled(
        format!(" {} ", msg.role.label()),
        Style::default()
            .fg(msg.role.color())
            .add_modifier(Modifier::BOLD),
    )];
    if let Some(stats) = &msg.stats {
        let streaming = msg.streaming;
        let tokens = stats.decoded_tokens();
        if tokens > 0 {
            spans.push(Span::styled(format!("· {tokens} tok "), dim));
        }
        let rate = if streaming {
            stats.live_decode_rate(now)
        } else {
            stats.final_decode_rate()
        };
        if let Some(r) = rate {
            spans.push(Span::styled(format!("· {r:.1} tok/s "), dim));
        }
        if let Some(ttft) = stats.ttft() {
            spans.push(Span::styled(
                format!("· ttft {:.2}s ", ttft.as_secs_f64()),
                dim,
            ));
        }
        if let Some(pp) = stats.pp_rate() {
            let pt = stats.prompt_tokens.unwrap_or(0);
            spans.push(Span::styled(format!("· {pt} prompt @ {pp:.0} tok/s "), dim));
        }
    }
    Line::from(spans)
}

/// Build the visual lines for a single message: subdued reasoning first
/// (if any), then the content, with a streaming spinner on the trailing
/// line where appropriate.
fn build_message_lines(msg: &TuiMsg) -> Vec<Line<'static>> {
    let dim = Style::default().fg(Color::DarkGray);
    let reasoning_label = Style::default()
        .fg(Color::DarkGray)
        .add_modifier(Modifier::ITALIC | Modifier::BOLD);

    let mut lines: Vec<Line<'static>> = Vec::new();

    if !msg.reasoning.is_empty() {
        lines.push(Line::from(Span::styled("thinking", reasoning_label)));
        for raw in msg.reasoning.split('\n') {
            lines.push(Line::from(Span::styled(raw.to_string(), dim)));
        }
        // If the model is still mid-reasoning (no content yet), mark the
        // last reasoning line with the spinner.
        if msg.streaming
            && msg.content.is_empty()
            && let Some(last) = lines.last_mut()
        {
            last.spans.push(Span::styled(" ⟳", dim));
        }
        // Only insert a separator once content actually exists.
        if !msg.content.is_empty() {
            lines.push(Line::from(""));
        }
    }

    if msg.content.is_empty() && msg.streaming && msg.reasoning.is_empty() {
        lines.push(Line::from(Span::styled("⟳ thinking…", dim)));
    } else if !msg.content.is_empty() {
        for raw in msg.content.split('\n') {
            lines.push(Line::from(raw.to_string()));
        }
        if msg.streaming
            && let Some(last) = lines.last_mut()
        {
            last.spans.push(Span::styled(" ⟳", dim));
        }
    } else if !msg.streaming && msg.content.is_empty() && msg.reasoning.is_empty() {
        // System message with empty content (rare); render an empty line so
        // the box isn't zero-height.
        lines.push(Line::from(""));
    }

    if lines.is_empty() {
        lines.push(Line::from(""));
    }
    lines
}

/// Estimate how many rendered rows a list of `Line`s occupies when wrapped
/// to `width`. Accounts for empty lines (1 row) and rounds up partial rows.
fn visual_line_count(lines: &[Line<'_>], width: u16) -> u32 {
    if width == 0 {
        return lines.len() as u32;
    }
    let w = width as u32;
    let mut total = 0u32;
    for line in lines {
        let len: u32 = line
            .spans
            .iter()
            .map(|s| s.content.chars().count() as u32)
            .sum();
        let rows = if len == 0 { 1 } else { len.div_ceil(w) };
        total = total.saturating_add(rows);
    }
    total
}

fn render_input(f: &mut Frame, area: ratatui::layout::Rect, state: &TuiState) {
    let title = if state.streaming {
        "Input (waiting for response…)"
    } else {
        "Input"
    };
    let style = if state.streaming {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default().fg(Color::White)
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Gray))
        .title(title);
    let inner_width = block.inner(area).width;
    let inner_height = block.inner(area).height as u32;
    let total = wrapped_rows(&state.input, inner_width);
    // Anchor the cursor row (always at the end of input) to the bottom of
    // the box once content overflows.
    let scroll = total.saturating_sub(inner_height) as u16;
    f.render_widget(
        Paragraph::new(state.input.as_str())
            .style(style)
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0))
            .block(block),
        area,
    );
}

/// Total wrapped row count for the input string at `width`. Empty lines
/// still occupy one row; trailing newline reserves a final empty row so
/// the cursor sits visibly on it.
fn wrapped_rows(s: &str, width: u16) -> u32 {
    if width == 0 {
        return 1;
    }
    let w = width as u32;
    // `split('\n')` yields a trailing "" after a terminal '\n', which is
    // exactly the row we want the cursor to live on.
    s.split('\n')
        .map(|seg| {
            let len = seg.chars().count() as u32;
            if len == 0 { 1 } else { len.div_ceil(w) }
        })
        .sum()
}

/// Visible content-row count for the input box, clamped between min and
/// max row constants. Adds one trailing row when the buffer is empty so
/// the box never starts at 0 content rows.
fn input_content_rows(input: &str, inner_width: u16) -> u16 {
    let total = wrapped_rows(input, inner_width).min(u16::MAX as u32) as u16;
    total.clamp(INPUT_MIN_CONTENT_ROWS, INPUT_MAX_CONTENT_ROWS)
}

fn render_status(f: &mut Frame, area: ratatui::layout::Rect) {
    let line = Line::from(vec![
        Span::styled("Enter", Style::default().fg(Color::White)),
        Span::raw(" send · "),
        Span::styled("Shift+Enter", Style::default().fg(Color::White)),
        Span::raw(" newline · "),
        Span::styled("↑/↓/wheel", Style::default().fg(Color::White)),
        Span::raw(" scroll · "),
        Span::styled("Esc", Style::default().fg(Color::White)),
        Span::raw(" clear · "),
        Span::styled("Ctrl+C", Style::default().fg(Color::White)),
        Span::raw(" quit"),
    ])
    .style(Style::default().fg(Color::DarkGray));
    f.render_widget(Paragraph::new(line), area);
}
