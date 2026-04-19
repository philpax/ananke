//! Text / JSON formatters for CLI subcommands.

use ananke_api::{DeviceSummary, ServiceDetail, ServiceSummary};
use serde::Serialize;

pub fn print_json<T: Serialize>(value: &T) {
    match serde_json::to_string_pretty(value) {
        Ok(s) => println!("{s}"),
        Err(e) => eprintln!("failed to serialise response: {e}"),
    }
}

pub fn print_devices_table(devices: &[DeviceSummary]) {
    println!(
        "{:<10} {:<28} {:>12} {:>12}   RESERVATIONS",
        "ID", "NAME", "TOTAL", "FREE"
    );
    for d in devices {
        let total_gib = d.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let free_gib = d.free_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let resv = d
            .reservations
            .iter()
            .map(|r| format!("{}: {} MiB", r.service, r.bytes / (1024 * 1024)))
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "{:<10} {:<28} {:>10.1}G {:>10.1}G   {}",
            d.id, d.name, total_gib, free_gib, resv
        );
    }
}

pub fn print_services_table(services: &[ServiceSummary], show_all: bool) {
    let rows: Vec<&ServiceSummary> = services
        .iter()
        .filter(|s| show_all || !s.state.starts_with("disabled"))
        .collect();
    println!(
        "{:<24} {:<12} {:<12} {:>4} {:>6} {:>8}",
        "NAME", "STATE", "LIFECYCLE", "PRI", "PORT", "PID"
    );
    for s in rows {
        println!(
            "{:<24} {:<12} {:<12} {:>4} {:>6} {:>8}",
            s.name,
            s.state,
            s.lifecycle,
            s.priority,
            s.port,
            s.pid.map(|p| p.to_string()).unwrap_or_else(|| "—".into()),
        );
    }
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
