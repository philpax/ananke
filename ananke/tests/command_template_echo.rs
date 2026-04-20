use std::collections::BTreeMap;

use ananke::{
    config::{
        AllocationMode, CommandConfig, TemplateConfig,
        parse::DEFAULT_START_QUEUE_DEPTH,
        validate::{
            DeviceSlot, Filters, HealthSettings, Lifecycle, PlacementPolicy, ServiceConfig,
        },
    },
    devices::Allocation,
    supervise::render_argv,
};
use smol_str::SmolStr;

#[test]
fn command_argv_substitutes_port() {
    let mut placement = BTreeMap::new();
    placement.insert(DeviceSlot::Gpu(0), 6144);

    let argv = vec![
        "python".into(),
        "main.py".into(),
        "--port".into(),
        "{port}".into(),
    ];

    let svc = ServiceConfig {
        name: SmolStr::new("comfy"),
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
        gpu_allow: Vec::new(),
        idle_timeout_ms: 600_000,
        drain_timeout_ms: 5_000,
        extended_stream_drain_ms: 5_000,
        max_request_duration_ms: 60_000,
        filters: Filters::default(),
        allocation_mode: AllocationMode::Static { vram_mb: 6144 },
        openai_compat: false,
        description: None,
        start_queue_depth: DEFAULT_START_QUEUE_DEPTH,
        extra_args: Vec::new(),
        env: BTreeMap::new(),
        metadata: ananke_api::AnankeMetadata::new(),
        template_config: TemplateConfig::Command(CommandConfig {
            command: argv,
            workdir: None,
            shutdown_command: None,
            private_port_override: None,
        }),
    };

    let alloc = Allocation::from_override(&placement);
    let cfg = render_argv(&svc, &alloc, None).unwrap();
    assert_eq!(cfg.binary, "python");
    // The private port (48188) is substituted into {port}.
    assert!(
        cfg.args.iter().any(|a| a == "48188"),
        "expected 48188 in args: {:?}",
        cfg.args
    );
}
