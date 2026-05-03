//! Text / JSON formatters for CLI subcommands.
//!
//! Tabular output is rendered with `comfy-table`, which auto-sizes
//! columns and emits ANSI styling cross-platform via crossterm. When
//! stdout isn't a tty (e.g. `anankectl status | grep running`) the
//! library suppresses styling automatically, so piping stays clean.

use ananke_api::{DeviceSummary, ServiceDetail, ServiceSummary};
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
        .filter(|s| show_all || !s.state.starts_with("disabled"))
        .collect();
    // Hot first (running, then idle), then everything else; alphabetical
    // tiebreaker so the order is stable across calls.
    rows.sort_by(|a, b| {
        state_rank(&a.state)
            .cmp(&state_rank(&b.state))
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
        let state_cell = match state_color(&s.state) {
            Some(c) => Cell::new(&s.state).fg(c),
            None => Cell::new(&s.state),
        };
        table.add_row(vec![
            Cell::new(&s.name),
            state_cell,
            Cell::new(&s.lifecycle),
            Cell::new(s.priority).set_alignment(CellAlignment::Right),
            Cell::new(s.port).set_alignment(CellAlignment::Right),
            Cell::new(s.pid.map(|p| p.to_string()).unwrap_or_else(|| "—".into()))
                .set_alignment(CellAlignment::Right),
        ]);
    }

    println!("{table}");
}

pub fn print_service_detail(detail: &ServiceDetail) {
    println!("{}", detail.name);
    println!("  state:     {}", detail.state);
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
    if !detail.recent_logs.is_empty() {
        println!("  recent logs (last {}):", detail.recent_logs.len());
        for line in detail.recent_logs.iter().rev().take(10).rev() {
            println!("    [{}] {}", line.stream, line.line);
        }
    }
}

/// Format `bytes` with an automatically-chosen unit (GiB / MiB / B).
/// Used wherever VRAM totals or reservation amounts get printed.
pub fn format_bytes(bytes: u64) -> String {
    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;
    if bytes >= GIB {
        format!("{:.1} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{} MiB", bytes / MIB)
    } else {
        format!("{bytes} B")
    }
}

/// Map a service state to a display colour; `None` for states that
/// shouldn't draw the eye (defaults to terminal-default).
pub fn state_color(state: &str) -> Option<Color> {
    match state {
        "running" => Some(Color::Green),
        "idle" => Some(Color::Yellow),
        s if s.starts_with("disabled") => Some(Color::DarkGrey),
        s if s.starts_with("error") || s.contains("failed") => Some(Color::Red),
        _ => None,
    }
}

/// Order services by how "ready" they are: running first (already hot),
/// idle next (warm), then anything else.
fn state_rank(state: &str) -> u8 {
    match state {
        "running" => 0,
        "idle" => 1,
        _ => 2,
    }
}

/// Build a borderless, dynamically-arranged table. We avoid borders so
/// the output keeps the same visual density as the legacy plain-text
/// tables; comfy-table still gives us auto-sizing and per-cell styling.
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
