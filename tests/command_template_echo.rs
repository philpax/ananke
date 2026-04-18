use std::collections::BTreeMap;

use ananke::config::parse::{RawAllocation, RawService};
use ananke::config::validate::{
    DeviceSlot, Filters, HealthSettings, Lifecycle, PlacementPolicy, ServiceConfig,
};
use ananke::config::{AllocationMode, Template};
use ananke::devices::Allocation;
use ananke::supervise::render_argv;
use smol_str::SmolStr;

#[test]
fn command_argv_substitutes_port() {
    let mut placement = BTreeMap::new();
    placement.insert(DeviceSlot::Gpu(0), 6144);

    let raw = RawService {
        name: Some(SmolStr::new("comfy")),
        template: Some(SmolStr::new("command")),
        command: Some(vec![
            "python".into(),
            "main.py".into(),
            "--port".into(),
            "{port}".into(),
        ]),
        port: Some(8188),
        allocation: Some(RawAllocation {
            mode: Some(SmolStr::new("static")),
            vram_gb: Some(6.0),
            ..Default::default()
        }),
        ..Default::default()
    };

    let svc = ServiceConfig {
        name: SmolStr::new("comfy"),
        template: Template::Command,
        port: 8188,
        private_port: 48188,
        lifecycle: Lifecycle::OnDemand,
        priority: 50,
        health: HealthSettings {
            http_path: "/system_stats".into(),
            timeout_ms: 60_000,
            probe_interval_ms: 500,
        },
        placement_override: placement.clone(),
        placement_policy: PlacementPolicy::GpuOnly,
        idle_timeout_ms: 600_000,
        warming_grace_ms: 30_000,
        drain_timeout_ms: 5_000,
        extended_stream_drain_ms: 5_000,
        max_request_duration_ms: 60_000,
        filters: Filters::default(),
        allocation_mode: AllocationMode::Static { vram_mb: 6144 },
        command: Some(vec![
            "python".into(),
            "main.py".into(),
            "--port".into(),
            "{port}".into(),
        ]),
        workdir: None,
        openai_compat: false,
        raw,
    };

    let alloc = Allocation::from_override(&placement);
    let cfg = render_argv(&svc, &alloc, None);
    assert_eq!(cfg.binary, "python");
    // The private port (48188) is substituted into {port}.
    assert!(
        cfg.args.iter().any(|a| a == "48188"),
        "expected 48188 in args: {:?}",
        cfg.args
    );
}
