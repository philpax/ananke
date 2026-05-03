//! Interactive TUI chat interface using ratatui.

use std::io;

use ananke_api::ServicesResponse;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
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
        if !self.first_token {
            // Trim leading whitespace from the first content token to handle
            // models that emit "\n" in the initial delta event.
            let trimmed = token.trim_start();
            if trimmed.is_empty() {
                return;
            }
            last.content.push_str(trimmed);
            self.first_token = true;
        } else {
            last.content.push_str(&token);
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
        if last.reasoning.is_empty() {
            let trimmed = token.trim_start();
            if trimmed.is_empty() {
                return;
            }
            last.reasoning.push_str(trimmed);
        } else {
            last.reasoning.push_str(&token);
        }
        self.scroll_offset = 0;
    }

    fn finish_streaming(&mut self) {
        if let Some(last) = self.messages.last_mut()
            && last.streaming
        {
            last.streaming = false;
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
                SSEUpdate::Done => state.finish_streaming(),
                SSEUpdate::Error(e) => state.set_error(e),
            }
        }

        terminal.draw(|f| render(f, state))?;

        if !event::poll(std::time::Duration::from_millis(50))? {
            continue;
        }
        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }

        use crossterm::event::KeyModifiers;
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);

        match key.code {
            KeyCode::Char('c') if ctrl => return Ok(()),
            KeyCode::Char('d') if ctrl && state.input.is_empty() => return Ok(()),
            KeyCode::Enter if shift => state.input.push('\n'),
            KeyCode::Enter => {
                if state.streaming {
                    continue;
                }
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
            KeyCode::Backspace => {
                state.input.pop();
            }
            KeyCode::Up => state.scroll_up(1),
            KeyCode::Down => state.scroll_down(1),
            KeyCode::PageUp => state.scroll_up(10),
            KeyCode::PageDown => state.scroll_down(10),
            KeyCode::End if ctrl => state.scroll_to_bottom(),
            KeyCode::Esc => state.input.clear(),
            KeyCode::Char(c) => state.input.push(c),
            _ => {}
        }
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
    // determine scroll. A single blank row separates messages so adjacent
    // borders don't visually fuse.
    let spacer: u16 = 1;
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

    let total_height: u32 = rendered.iter().map(|(_, h)| *h as u32).sum::<u32>()
        + rendered.len().saturating_sub(1) as u32 * spacer as u32;
    let visible_height = area.height as u32;

    // Anchor to the bottom; user scroll offset moves the window upward.
    let max_window_top = total_height.saturating_sub(visible_height);
    let window_top = max_window_top.saturating_sub(state.scroll_offset as u32);

    // Walk the virtual stack, emitting any visible portion of each message.
    let mut virtual_y: u32 = 0;
    for (i, (lines, h)) in rendered.iter().enumerate() {
        let msg_top = virtual_y;
        let msg_bottom = virtual_y + *h as u32;
        virtual_y = msg_bottom + spacer as u32;

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
        let title_span = Span::styled(
            format!(" {} ", msg.role.label()),
            Style::default()
                .fg(msg.role.color())
                .add_modifier(Modifier::BOLD),
        );
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(msg.role.color()))
            .title(title_span);

        let paragraph = Paragraph::new(lines.clone())
            .wrap(Wrap { trim: false })
            .scroll((internal_scroll, 0))
            .block(block);
        f.render_widget(paragraph, rect);
    }
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
        Span::styled("PgUp/PgDn", Style::default().fg(Color::White)),
        Span::raw(" scroll · "),
        Span::styled("Esc", Style::default().fg(Color::White)),
        Span::raw(" clear · "),
        Span::styled("Ctrl+C", Style::default().fg(Color::White)),
        Span::raw(" quit"),
    ])
    .style(Style::default().fg(Color::DarkGray));
    f.render_widget(Paragraph::new(line), area);
}
