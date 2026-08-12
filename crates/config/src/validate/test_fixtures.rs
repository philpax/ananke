//! Shared `ServiceConfig` factory and TOML scaffolding for unit tests.
//!
//! Centralised so individual test modules don't hand-roll fixtures that
//! range over the full struct surface and have to be updated in lockstep
//! every time a field is added.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use ananke_api::shared::modality::Modality;
use ananke_errors::ExpectedError;
use smol_str::SmolStr;

use crate::{
    merge::resolve_inheritance,
    parse::{EstimationConfig, RawConfig, SamplingConfig, parse_toml},
    validate::{
        AllocationMode, AutoRestartSettings, CommandConfig, DEFAULT_SERVICE_PRIORITY,
        DeviceReserves, DeviceSlot, Filters, HealthSettings, Lifecycle, LlamaCppConfig,
        OffloadMode, PlacementPolicy, ServiceConfig, SplitMode, Template, TemplateConfig,
        TrackingSettings, validate,
    },
};

/// Parse a TOML fragment and resolve its `extends` inheritance, yielding the
/// `RawConfig` that `validate` expects as its input.
pub fn parse_and_merge(src: &str) -> RawConfig {
    let mut cfg = parse_toml(src, Path::new("/t")).unwrap();
    resolve_inheritance(&mut cfg).unwrap();
    cfg
}

/// Validate a single llama-cpp service carrying the given `[service]`-level
/// TOML `block`, so an auto-restart test can state only the keys it exercises.
pub fn svc_with_auto_restart(block: &str) -> Result<ServiceConfig, ExpectedError> {
    let src = format!(
        r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "cpu-only"
{block}
"#
    );
    let cfg = parse_and_merge(&src);
    validate(&cfg)
        .map(|ec| ec.services.into_iter().next().unwrap())
        .map_err(|report| report.into_expected_error(PathBuf::from("<config>")))
}

/// Build a minimal `ServiceConfig` with CPU-only placement, suitable for
/// unit tests that need a well-formed config but don't care about its
/// specific field values. The caller is free to mutate the returned
/// struct to customise individual fields.
pub fn minimal_service(name: &str) -> ServiceConfig {
    minimal_llama_cpp_service(name)
}

/// Minimal llama-cpp service config for tests, placed on CPU only. See
/// [`minimal_service`].
pub fn minimal_llama_cpp_service(name: &str) -> ServiceConfig {
    let mut placement = BTreeMap::new();
    placement.insert(DeviceSlot::Cpu, 100);
    ServiceConfig {
        name: SmolStr::new(name),
        port: 0,
        private_port: 0,
        lifecycle: Lifecycle::OnDemand,
        priority: DEFAULT_SERVICE_PRIORITY,
        health: HealthSettings {
            http_path: Some("/health".into()),
            timeout_ms: 5_000,
            probe_interval_ms: 200,
        },
        placement_override: placement,
        placement_policy: PlacementPolicy::CpuOnly,
        gpu_allow: Vec::new(),
        split_mode: SplitMode::Layer,
        tensor_split_weights: None,
        gpu_headroom_mb: 0,
        reserves: Arc::new(DeviceReserves::default()),
        idle_timeout_ms: 60_000,
        drain_timeout_ms: 1_000,
        extended_stream_drain_ms: 1_000,
        max_request_duration_ms: 5_000,
        auto_restart: AutoRestartSettings::disabled(),
        filters: Filters::default(),
        allocation_mode: AllocationMode::None,
        openai_compat: true,
        description: None,
        modality: Modality::Chat,
        start_queue_depth: 10,
        extra_args: Vec::new(),
        env: BTreeMap::new(),
        env_inherit: true,
        tracking: TrackingSettings::default(),
        metadata: ananke_api::shared::metadata::AnankeMetadata::new(),
        template_config: TemplateConfig::LlamaCpp(Box::new(llama_cpp_fixture())),
    }
}

/// Minimal command service config for tests, built on
/// [`minimal_llama_cpp_service`] with the template swapped.
pub fn minimal_command_service(name: &str, argv: Vec<String>) -> ServiceConfig {
    let mut svc = minimal_llama_cpp_service(name);
    svc.template_config = TemplateConfig::Command(CommandConfig {
        command: argv,
        workdir: None,
        shutdown_command: None,
        private_port_override: None,
        openai_proxy: None,
    });
    svc.openai_compat = false;
    svc
}

/// Borrow the LlamaCpp variant or panic. Convenience for tests that set
/// up a service via `minimal_service` and need to tweak llama-cpp fields.
pub fn expect_llama_cpp(svc: &mut ServiceConfig) -> &mut LlamaCppConfig {
    match &mut svc.template_config {
        TemplateConfig::LlamaCpp(lc) => lc.as_mut(),
        TemplateConfig::Command(_) => panic!("expected LlamaCpp template_config"),
    }
}

/// Borrow the Command variant or panic.
pub fn expect_command(svc: &mut ServiceConfig) -> &mut CommandConfig {
    match &mut svc.template_config {
        TemplateConfig::LlamaCpp(_) => panic!("expected Command template_config"),
        TemplateConfig::Command(cmd) => cmd,
    }
}

fn llama_cpp_fixture() -> LlamaCppConfig {
    LlamaCppConfig {
        runtime: Default::default(),
        model: PathBuf::from("/fake/model.gguf"),
        mmproj: None,
        context: None,
        n_gpu_layers: None,
        expert_offload: OffloadMode::Off,
        flash_attn: None,
        cache_type_k: None,
        cache_type_v: None,
        mmap: None,
        mlock: None,
        parallel: None,
        spec_type: None,
        spec_draft_n_max: None,
        draft_model: None,
        kv_unified: None,
        cache_idle_slots: None,
        cache_ram_mb: None,
        metrics: None,
        slots: None,
        batch_size: None,
        ubatch_size: None,
        threads: None,
        threads_batch: None,
        numa: None,
        jinja: None,
        chat_template_file: None,
        override_tensor: Vec::new(),
        sampling: SamplingConfig::default(),
        estimation: EstimationConfig::default(),
        binary: PathBuf::from("llama-server"),
        launcher: None,
    }
}

// Silence unused warnings on types that only specific tests use.
fn _coerce_template_used() {
    let _ = Template::LlamaCpp;
}
