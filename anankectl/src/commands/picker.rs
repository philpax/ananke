//! Interactive service-name picker used when `anankectl chat` is invoked
//! without an explicit model.

use std::io;

use ananke_api::{ServiceSummary, ServicesResponse};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};

use crate::client::{ApiClient, ApiClientError};

/// Fetch services and let the user pick one interactively. Returns
/// `Ok(Some(name))` on confirm, `Ok(None)` on user-cancel (Esc/Ctrl+C),
/// and short-circuits when there's nothing meaningful to pick:
///   - no enabled services → usage error,
///   - exactly one enabled service → returns it without showing the UI.
pub async fn pick_service(client: &ApiClient) -> Result<Option<String>, ApiClientError> {
    let resp: ServicesResponse = client.get_json("/api/services").await?;
    let candidates: Vec<ServiceSummary> = resp
        .services
        .into_iter()
        .filter(|s| !s.state.starts_with("disabled"))
        .collect();

    if candidates.is_empty() {
        return Err(ApiClientError::Usage(
            "no enabled services available — enable one first with `anankectl enable <name>`"
                .into(),
        ));
    }
    if candidates.len() == 1 {
        return Ok(Some(candidates.into_iter().next().unwrap().name));
    }

    tokio::task::spawn_blocking(move || run_picker(candidates))
        .await
        .map_err(|e| ApiClientError::Usage(format!("picker panicked: {e}")))?
}

fn run_picker(services: Vec<ServiceSummary>) -> Result<Option<String>, ApiClientError> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    if let Err(e) = execute!(stdout, EnterAlternateScreen) {
        let _ = disable_raw_mode();
        return Err(e.into());
    }

    let backend = ratatui::backend::CrosstermBackend::new(io::stdout());
    let mut terminal = ratatui::Terminal::new(backend)?;

    let result = picker_loop(&mut terminal, &services);

    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
    let _ = disable_raw_mode();
    let _ = terminal.show_cursor();
    result
}

fn picker_loop(
    terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>,
    services: &[ServiceSummary],
) -> Result<Option<String>, ApiClientError> {
    let mut selected = 0usize;
    loop {
        terminal.draw(|f| render(f, services, selected))?;
        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Esc => return Ok(None),
            KeyCode::Char('c') if ctrl => return Ok(None),
            KeyCode::Char('q') => return Ok(None),
            KeyCode::Up | KeyCode::Char('k') => {
                selected = selected.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') if selected + 1 < services.len() => {
                selected += 1;
            }
            KeyCode::Home | KeyCode::Char('g') => {
                selected = 0;
            }
            KeyCode::End | KeyCode::Char('G') => {
                selected = services.len().saturating_sub(1);
            }
            KeyCode::Enter => {
                return Ok(Some(services[selected].name.clone()));
            }
            _ => {}
        }
    }
}

fn render(f: &mut Frame, services: &[ServiceSummary], selected: usize) {
    let area = f.area();
    let chunks = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(1),
        Constraint::Length(1),
    ])
    .split(area);

    let header = Line::from(vec![
        Span::styled(
            "Select a model",
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(" ({} available)", services.len()),
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    f.render_widget(Paragraph::new(header), chunks[0]);

    render_list(f, chunks[1], services, selected);

    let hint = Line::from(vec![
        Span::styled("↑/↓", Style::default().fg(Color::White)),
        Span::raw(" select · "),
        Span::styled("Enter", Style::default().fg(Color::White)),
        Span::raw(" confirm · "),
        Span::styled("Esc", Style::default().fg(Color::White)),
        Span::raw(" cancel"),
    ])
    .style(Style::default().fg(Color::DarkGray));
    f.render_widget(Paragraph::new(hint), chunks[2]);
}

fn render_list(f: &mut Frame, area: Rect, services: &[ServiceSummary], selected: usize) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height == 0 {
        return;
    }

    // Pin the selection in view by sliding the visible window when needed.
    let height = inner.height as usize;
    let window_top = if selected >= height {
        selected - height + 1
    } else {
        0
    };
    let visible = services.iter().enumerate().skip(window_top).take(height);

    for (row, (i, svc)) in visible.enumerate() {
        let y = inner.y + row as u16;
        let rect = Rect {
            x: inner.x,
            y,
            width: inner.width,
            height: 1,
        };
        let is_selected = i == selected;
        let prefix = if is_selected { "▶ " } else { "  " };
        let row_style = if is_selected {
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        let state_color = state_color(&svc.state);
        let line = Line::from(vec![
            Span::styled(prefix, row_style),
            Span::styled(
                format!("{:<10}", svc.state),
                Style::default().fg(state_color),
            ),
            Span::raw(" "),
            Span::styled(
                format!("{:<11}", svc.lifecycle),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled(svc.name.clone(), row_style),
        ]);
        f.render_widget(Paragraph::new(line).style(row_style), rect);
    }
}

fn state_color(state: &str) -> Color {
    match state {
        "running" => Color::Green,
        "idle" => Color::Yellow,
        s if s.starts_with("disabled") => Color::DarkGray,
        s if s.starts_with("error") || s.contains("failed") => Color::Red,
        _ => Color::White,
    }
}
