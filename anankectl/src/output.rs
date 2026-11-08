//! Text / JSON formatters for CLI subcommands.
//!
//! Tabular output is rendered with `comfy-table`, which auto-sizes
//! columns and emits ANSI styling cross-platform via crossterm. When
//! stdout isn't a tty (e.g. `anankectl status | grep running`) the
//! library suppresses styling automatically, so piping stays clean.

use ananke_api::{
    devices::list::DeviceSummary,
    internal::log_line::LogLine,
    services::{detail::ServiceDetail, list::ServiceSummary},
    shared::{DisableReason, ServiceStateKind},
};
use ananke_config::units::{GIB, MIB};
use comfy_table::{Attribute, Cell, CellAlignment, Color, ContentArrangement, Table, presets};
use serde::Serialize;

pub fn print_json<T: Serialize>(value: &T) {
    match serde_json::to_string_pretty(value) {
        Ok(s) => println!("{s}"),
        Err(e) => eprintln!("failed to serialise response: {e}"),
    }
}

pub fn print_devices_table(devices: &[DeviceSummary]) {
    let mut table = base_table();
    table.set_header(header_row(["ID", "NAME", "TOTAL", "FREE", "RESERVATIONS"]));

    for d in devices {
        let reservations = if d.reservations.is_empty() {
            String::new()
        } else {
            d.reservations
                .iter()
                .map(|r| format!("{}: {}", r.service, format_bytes(r.bytes)))
                .collect::<Vec<_>>()
                .join(", ")
        };
        table.add_row(vec![
            Cell::new(&d.id),
            Cell::new(&d.name),
            Cell::new(format_bytes(d.total_bytes)).set_alignment(CellAlignment::Right),
            Cell::new(format_bytes(d.free_bytes)).set_alignment(CellAlignment::Right),
            Cell::new(reservations),
        ]);
    }

    println!("{table}");
}

pub fn print_services_table(services: &[ServiceSummary], show_all: bool) {
    let mut rows: Vec<&ServiceSummary> = services
        .iter()
        .filter(|s| show_all || s.state != ServiceStateKind::Disabled)
        .collect();
    // Hot first (running, then idle), then everything else; alphabetical
    // tiebreaker so the order is stable across calls.
    rows.sort_by(|a, b| {
        state_rank(a.state)
            .cmp(&state_rank(b.state))
            .then_with(|| a.name.cmp(&b.name))
    });

    let mut table = base_table();
    table.set_header(header_row([
        "NAME",
        "STATE",
        "LIFECYCLE",
        "PRI",
        "PORT",
        "PID",
    ]));

    for s in rows {
        let label = state_label(s.state, s.disabled_reason);
        let state_cell = match state_color(s.state, s.disabled_reason) {
            Some(c) => Cell::new(label).fg(c),
            None => Cell::new(label),
        };
        table.add_row(vec![
            Cell::new(&s.name),
            state_cell,
            Cell::new(s.lifecycle),
            Cell::new(s.priority).set_alignment(CellAlignment::Right),
            Cell::new(s.port).set_alignment(CellAlignment::Right),
            Cell::new(s.pid.map(|p| p.to_string()).unwrap_or_else(|| "—".into()))
                .set_alignment(CellAlignment::Right),
        ]);
    }

    println!("{table}");
}

/// Print a service's detail, with `recent_logs` newest first as the logs
/// endpoint returns them.
pub fn print_service_detail(detail: &ServiceDetail, recent_logs: &[LogLine]) {
    println!("{}", detail.name);
    match detail.disabled_reason {
        Some(reason) => println!("  state:     {} ({reason})", detail.state),
        None => println!("  state:     {}", detail.state),
    }
    println!("  lifecycle: {}", detail.lifecycle);
    println!("  priority:  {}", detail.priority);
    println!("  template:  {}", detail.template);
    println!(
        "  port:      {} (private {})",
        detail.port, detail.private_port
    );
    if let Some(run_id) = detail.run_id {
        println!("  run_id:    {run_id}");
    }
    if let Some(pid) = detail.pid {
        println!("  pid:       {pid}");
    }
    if !detail.placement_override.is_empty() {
        println!("  placement_override:");
        for (k, v) in &detail.placement_override {
            println!("    {k} = {v}");
        }
    }
    if !detail.recent_restarts.is_empty() {
        println!("  auto-restarts (most recent first):");
        for r in &detail.recent_restarts {
            println!(
                "    {} [{}] {}",
                format_restart_time(r.at_ms),
                r.trigger,
                r.detail
            );
        }
    }
    if !recent_logs.is_empty() {
        println!("  recent logs (last {}):", recent_logs.len());
        for line in recent_logs.iter().rev() {
            println!("    [{}] {}", line.stream, line.line);
        }
    }
}

/// Render a wall-clock timestamp (ms since epoch) as the local datetime plus
/// a coarse relative age: `2026-07-24 15:41 (3h ago)`. The absolute time is
/// what an operator correlates against other logs; the age is what they scan
/// for at a glance.
fn format_restart_time(at_ms: i64) -> String {
    let age = {
        let now_ms = jiff::Timestamp::now().as_millisecond();
        let delta_s = (now_ms - at_ms).max(0) / 1000;
        if delta_s < 60 {
            format!("{delta_s}s ago")
        } else if delta_s < 3600 {
            format!("{}m ago", delta_s / 60)
        } else if delta_s < 86_400 {
            format!("{}h ago", delta_s / 3600)
        } else {
            format!("{}d ago", delta_s / 86_400)
        }
    };
    match jiff::Timestamp::from_millisecond(at_ms) {
        Ok(ts) => {
            let local = ts.to_zoned(jiff::tz::TimeZone::system());
            format!("{} ({age})", local.strftime("%Y-%m-%d %H:%M"))
        }
        Err(_) => age,
    }
}

/// Format `bytes` with an automatically-chosen unit (GiB / MiB / B).
/// Used wherever VRAM totals or reservation amounts get printed.
pub fn format_bytes(bytes: u64) -> String {
    if bytes >= GIB {
        format!("{:.1} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{} MiB", bytes / MIB)
    } else {
        format!("{bytes} B")
    }
}

/// A state cell's text: the state, plus the reason when disabled
/// (`disabled · oom`).
pub fn state_label(state: ServiceStateKind, reason: Option<DisableReason>) -> String {
    match reason {
        Some(reason) => format!("{state} · {reason}"),
        None => state.to_string(),
    }
}

/// Map a service state to a display colour; `None` for states that
/// shouldn't draw the eye (defaults to terminal-default).
fn state_color(state: ServiceStateKind, reason: Option<DisableReason>) -> Option<Color> {
    match state_tone(state, reason) {
        StateTone::Hot => Some(Color::Green),
        StateTone::Warm => Some(Color::Yellow),
        StateTone::Quiet => Some(Color::DarkGrey),
        StateTone::Fault => Some(Color::Red),
        StateTone::Plain => None,
    }
}

/// How a state should draw the eye, shared by the table and the picker so
/// the two colour a state the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateTone {
    Hot,
    Warm,
    Quiet,
    Fault,
    Plain,
}

pub fn state_tone(state: ServiceStateKind, reason: Option<DisableReason>) -> StateTone {
    match state {
        ServiceStateKind::Running => StateTone::Hot,
        ServiceStateKind::Idle => StateTone::Warm,
        ServiceStateKind::Failed => StateTone::Fault,
        ServiceStateKind::Disabled if reason.is_some_and(DisableReason::is_fault) => {
            StateTone::Fault
        }
        ServiceStateKind::Disabled => StateTone::Quiet,
        ServiceStateKind::Starting
        | ServiceStateKind::Draining
        | ServiceStateKind::Stopped
        | ServiceStateKind::Evicted
        | ServiceStateKind::Unknown => StateTone::Plain,
    }
}

/// Order services by how "ready" they are: running first (already hot),
/// idle next (warm), then anything else.
pub fn state_rank(state: ServiceStateKind) -> u8 {
    match state {
        ServiceStateKind::Running => 0,
        ServiceStateKind::Idle => 1,
        _ => 2,
    }
}

/// Build a borderless, dynamically-arranged table. Borders are avoided so the
/// output keeps the visual density of plain-text tables; comfy-table still
/// gives us auto-sizing and per-cell styling.
fn base_table() -> Table {
    let mut table = Table::new();
    table.load_preset(presets::NOTHING);
    table.set_content_arrangement(ContentArrangement::Dynamic);
    table
}

fn header_row<const N: usize>(cells: [&str; N]) -> Vec<Cell> {
    cells
        .into_iter()
        .map(|c| Cell::new(c).add_attribute(Attribute::Bold))
        .collect()
}
