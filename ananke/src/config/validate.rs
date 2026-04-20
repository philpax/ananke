//! Validate a post-merge `RawConfig`, producing an `EffectiveConfig` of
//! per-service validated configs plus daemon-global settings.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

use ananke_api::AnankeMetadata;
use smol_str::SmolStr;
use tracing::warn;

use crate::{
    config::parse::{
        EstimationConfig, RawCommandService, RawConfig, RawLlamaCppService, RawService,
        RawServiceCommon, SamplingConfig,
    },
    errors::ExpectedError,
};

/// Default idle-before-drain timeout for on-demand services (10 minutes).
pub const DEFAULT_IDLE_TIMEOUT_MS: u64 = 600_000;

/// Default cadence for the health-probe loop (5 seconds).
pub const DEFAULT_HEALTH_PROBE_INTERVAL_MS: u64 = 5_000;

/// Default per-probe timeout for health checks (3 minutes).
pub const DEFAULT_HEALTH_TIMEOUT_MS: u64 = 180_000;

/// Default grace window after spawn during which health-probe failures do not
/// count as hard failures (1 minute).
pub const DEFAULT_WARMING_GRACE_MS: u64 = 60_000;

/// Default drain timeout before the supervisor escalates to SIGKILL (30 seconds).
pub const DEFAULT_DRAIN_TIMEOUT_MS: u64 = 30_000;

/// Default extra grace granted to in-flight streaming requests during drain
/// (30 seconds).
pub const DEFAULT_EXTENDED_STREAM_DRAIN_MS: u64 = 30_000;

/// Default cap on the wall-clock duration of a single proxied request
/// (10 minutes).
pub const DEFAULT_MAX_REQUEST_DURATION_MS: u64 = 600_000;

/// Default service scheduling priority (higher wins eviction contests).
pub const DEFAULT_SERVICE_PRIORITY: u8 = 50;

/// Default minimum runtime a borrower must accumulate before the balloon
/// resolver may fast-kill it (1 minute).
pub const DEFAULT_MIN_BORROWER_RUNTIME_MS: u64 = 60_000;

/// Convert GiB (as declared by users in config) to MiB using the same
/// truncating cast the validator has always used. Centralised so the oneshot
/// API path and the TOML path agree on rounding.
pub fn gib_to_mib(gib: f32) -> u64 {
    (gib * 1024.0) as u64
}

#[derive(Debug, Clone)]
pub struct EffectiveConfig {
    pub daemon: DaemonSettings,
    pub services: Vec<ServiceConfig>,
}

#[derive(Debug, Clone)]
pub struct DaemonSettings {
    pub management_listen: String,
    pub openai_listen: String,
    pub data_dir: PathBuf,
    pub shutdown_timeout_ms: u64,
    pub allow_external_management: bool,
    pub allow_external_services: bool,
}

#[derive(Debug, Clone)]
pub struct ServiceConfig {
    pub name: SmolStr,
    pub port: u16,
    pub private_port: u16,
    pub lifecycle: Lifecycle,
    pub priority: u8,
    pub health: HealthSettings,
    pub placement_override: BTreeMap<DeviceSlot, u64>,
    pub placement_policy: PlacementPolicy,
    pub gpu_allow: Vec<u32>,
    pub filters: Filters,
    pub idle_timeout_ms: u64,
    pub warming_grace_ms: u64,
    pub drain_timeout_ms: u64,
    pub extended_stream_drain_ms: u64,
    pub max_request_duration_ms: u64,
    pub allocation_mode: AllocationMode,
    pub openai_compat: bool,
    pub description: Option<String>,
    pub start_queue_depth: usize,
    pub extra_args: Vec<String>,
    pub env: BTreeMap<String, String>,
    /// Passthrough entries from `[[service]] metadata.*`. The
    /// `openai_compat` key (if present) is special-cased into
    /// `openai_compat` above; every other entry is opaque to the daemon
    /// and exists only to be echoed back through `/v1/models` and
    /// `/api/services`.
    pub metadata: AnankeMetadata,
    pub template_config: TemplateConfig,
}

impl ServiceConfig {
    pub fn template(&self) -> Template {
        self.template_config.template()
    }

    /// Borrow the llama-cpp configuration, or `None` if this is a command
    /// service. Intended for code paths that are only reachable for
    /// llama-cpp services (estimator, llama-server argv rendering).
    pub fn llama_cpp(&self) -> Option<&LlamaCppConfig> {
        match &self.template_config {
            TemplateConfig::LlamaCpp(lc) => Some(lc.as_ref()),
            TemplateConfig::Command(_) => None,
        }
    }

    /// Borrow the command configuration, or `None` if this is a llama-cpp
    /// service.
    pub fn command(&self) -> Option<&CommandConfig> {
        match &self.template_config {
            TemplateConfig::LlamaCpp(_) => None,
            TemplateConfig::Command(cmd) => Some(cmd),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TemplateConfig {
    /// Boxed so the llama-cpp variant (~272 bytes) doesn't dominate the size of
    /// every `ServiceConfig`. Command services are ~48 bytes; boxing keeps the
    /// enum small for both.
    LlamaCpp(Box<LlamaCppConfig>),
    Command(CommandConfig),
}

impl TemplateConfig {
    pub fn template(&self) -> Template {
        match self {
            TemplateConfig::LlamaCpp(_) => Template::LlamaCpp,
            TemplateConfig::Command(_) => Template::Command,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LlamaCppConfig {
    pub model: PathBuf,
    pub mmproj: Option<PathBuf>,
    pub context: Option<u32>,
    pub n_gpu_layers: Option<i32>,
    pub n_cpu_moe: Option<u32>,
    pub flash_attn: Option<bool>,
    pub cache_type_k: Option<SmolStr>,
    pub cache_type_v: Option<SmolStr>,
    pub mmap: Option<bool>,
    pub mlock: Option<bool>,
    pub parallel: Option<u32>,
    pub batch_size: Option<u32>,
    pub ubatch_size: Option<u32>,
    pub threads: Option<u32>,
    pub threads_batch: Option<u32>,
    pub jinja: Option<bool>,
    pub chat_template_file: Option<PathBuf>,
    pub override_tensor: Vec<String>,
    pub sampling: SamplingConfig,
    pub estimation: EstimationConfig,
}

#[derive(Debug, Clone)]
pub struct CommandConfig {
    pub command: Vec<String>,
    pub workdir: Option<PathBuf>,
    /// Optional argv to run after the SIGTERM/SIGKILL drain pipeline
    /// exits. Used for external services that can't stop via signal
    /// alone — e.g. a docker-run wrapper whose container needs an
    /// explicit `docker stop` sibling command.
    pub shutdown_command: Option<Vec<String>>,
    /// When `Some`, ananke's reverse proxy forwards to this port rather
    /// than one picked from the private-port pool. Lets operators point
    /// at a fixed upstream (docker host binding, a service managed
    /// externally, etc). `None` = auto-assign.
    pub private_port_override: Option<u16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Template {
    LlamaCpp,
    Command,
}

impl Template {
    pub fn as_str(self) -> &'static str {
        match self {
            Template::LlamaCpp => "llamacpp",
            Template::Command => "command",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationMode {
    /// Llama-cpp services: placement decided by estimator/override; mode absent.
    None,
    Static {
        vram_mb: u64,
    },
    Dynamic {
        min_mb: u64,
        max_mb: u64,
        min_borrower_runtime_ms: u64,
    },
}

impl AllocationMode {
    /// Resolve an allocation mode from a `(template, mode)` pair plus the
    /// associated VRAM knobs. Shared by the TOML validator and the oneshot
    /// API so both paths agree on the semantics of `"static"`, `"dynamic"`,
    /// and the llama-cpp exclusions.
    ///
    /// The returned error is a bare sentence fragment; the caller is
    /// expected to prepend context (e.g. `service {name}: `).
    pub fn from_parts(
        template: Template,
        mode: Option<&str>,
        vram_gb: Option<f32>,
        min_vram_gb: Option<f32>,
        max_vram_gb: Option<f32>,
        min_borrower_runtime_ms: u64,
    ) -> Result<AllocationMode, String> {
        match (template, mode) {
            (Template::LlamaCpp, Some(m)) => Err(format!(
                "allocation.mode `{m}` invalid for llama-cpp (use placement_override or estimator)"
            )),
            (Template::LlamaCpp, None) => Ok(AllocationMode::None),
            (Template::Command, Some("static")) => {
                let gb =
                    vram_gb.ok_or_else(|| "allocation.mode=static requires vram_gb".to_string())?;
                Ok(AllocationMode::Static {
                    vram_mb: gib_to_mib(gb),
                })
            }
            (Template::Command, Some("dynamic")) => {
                let min = min_vram_gb
                    .ok_or_else(|| "allocation.mode=dynamic requires min_vram_gb".to_string())?;
                let max = max_vram_gb
                    .ok_or_else(|| "allocation.mode=dynamic requires max_vram_gb".to_string())?;
                if max <= min {
                    return Err("max_vram_gb must be > min_vram_gb".to_string());
                }
                Ok(AllocationMode::Dynamic {
                    min_mb: gib_to_mib(min),
                    max_mb: gib_to_mib(max),
                    min_borrower_runtime_ms,
                })
            }
            (Template::Command, Some(other)) => Err(format!("unknown allocation.mode `{other}`")),
            (Template::Command, None) => {
                Err("command template requires allocation.mode (static|dynamic)".to_string())
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lifecycle {
    Persistent,
    OnDemand,
}

impl Lifecycle {
    pub fn as_str(self) -> &'static str {
        match self {
            Lifecycle::Persistent => "persistent",
            Lifecycle::OnDemand => "ondemand",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Filters {
    pub strip_params: Vec<String>,
    pub set_params: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementPolicy {
    GpuOnly,
    CpuOnly,
    Hybrid,
}

#[derive(Debug, Clone)]
pub struct HealthSettings {
    pub http_path: String,
    pub timeout_ms: u64,
    pub probe_interval_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeviceSlot {
    Cpu,
    Gpu(u32),
}

pub fn validate(cfg: &RawConfig) -> Result<EffectiveConfig, ExpectedError> {
    let data_dir = cfg.daemon.data_dir.clone().unwrap_or_else(|| {
        std::env::var("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()))
                    .join(".local")
                    .join("share")
            })
            .join("ananke")
    });

    let shutdown_timeout_str = if cfg.daemon.shutdown_timeout.is_empty() {
        "120s"
    } else {
        &cfg.daemon.shutdown_timeout
    };
    let shutdown_timeout_ms = parse_duration_ms(shutdown_timeout_str)
        .map_err(|e| fail(format!("daemon.shutdown_timeout: {e}")))?;

    let management_addr = if cfg.daemon.management_listen.is_empty() {
        "127.0.0.1:7777".into()
    } else {
        cfg.daemon.management_listen.clone()
    };
    let mgmt_socket_addr: std::net::SocketAddr = management_addr
        .parse()
        .map_err(|e: std::net::AddrParseError| fail(format!("daemon.management_listen: {e}")))?;
    if !mgmt_socket_addr.ip().is_loopback() && !cfg.daemon.allow_external_management {
        return Err(fail(
            "daemon.management_listen is non-loopback but daemon.allow_external_management is false; \
             see §11 of the spec before enabling this — the management API has no authentication".into(),
        ));
    }
    let management_port = management_addr
        .rsplit(':')
        .next()
        .and_then(|p| p.parse::<u16>().ok());
    let openai_listen = cfg
        .openai_api
        .listen
        .clone()
        .unwrap_or_else(|| "127.0.0.1:8080".into());

    let private_port_range =
        PrivatePortRange::from_config(cfg.daemon.private_port_start, cfg.daemon.private_port_end)?;
    let mut private_ports = PrivatePortAllocator::new(private_port_range);

    let mut names: BTreeSet<SmolStr> = BTreeSet::new();
    let mut ports: BTreeSet<u16> = BTreeSet::new();
    let mut out = Vec::new();

    for (i, raw) in cfg.services.iter().enumerate() {
        let svc = validate_service(
            i,
            raw,
            &cfg.defaults,
            management_port,
            &mut names,
            &mut ports,
            &mut private_ports,
        )?;
        out.push(svc);
    }

    Ok(EffectiveConfig {
        daemon: DaemonSettings {
            management_listen: management_addr,
            openai_listen,
            data_dir,
            shutdown_timeout_ms,
            allow_external_management: cfg.daemon.allow_external_management,
            allow_external_services: cfg.daemon.allow_external_services,
        },
        services: out,
    })
}

fn validate_service(
    index: usize,
    raw: &RawService,
    defaults: &crate::config::parse::DefaultsConfig,
    management_port: Option<u16>,
    names: &mut BTreeSet<SmolStr>,
    ports: &mut BTreeSet<u16>,
    private_ports: &mut PrivatePortAllocator,
) -> Result<ServiceConfig, ExpectedError> {
    let common = raw.common();
    let name = common
        .name
        .clone()
        .ok_or_else(|| fail(format!("service[{index}] missing name")))?;
    let port = common
        .port
        .ok_or_else(|| fail(format!("service {name} missing port")))?;

    if !names.insert(name.clone()) {
        return Err(fail(format!("duplicate service name `{name}`")));
    }
    if !ports.insert(port) {
        return Err(fail(format!("duplicate service port {port}")));
    }
    if Some(port) == management_port {
        return Err(fail(format!(
            "service {name} port {port} collides with daemon.management_listen"
        )));
    }

    let template = match raw {
        RawService::LlamaCpp(_) => Template::LlamaCpp,
        RawService::Command(_) => Template::Command,
    };

    let (allocation_mode, template_config) = match raw {
        RawService::LlamaCpp(lc) => {
            let tc = validate_llama_cpp(&name, lc)?;
            // llama-cpp never takes an allocation.mode; none of the dynamic
            // knobs apply here.
            let alloc = AllocationMode::from_parts(
                Template::LlamaCpp,
                None,
                None,
                None,
                None,
                DEFAULT_MIN_BORROWER_RUNTIME_MS,
            )
            .map_err(|e| fail(format!("service {name}: {e}")))?;
            (alloc, TemplateConfig::LlamaCpp(Box::new(tc)))
        }
        RawService::Command(cmd) => {
            let raw_alloc = cmd.allocation.clone().unwrap_or_default();
            let runtime_ms = raw_alloc
                .min_borrower_runtime
                .as_deref()
                .map(parse_duration_ms)
                .transpose()
                .map_err(|e| fail(format!("service {name} min_borrower_runtime: {e}")))?
                .unwrap_or(DEFAULT_MIN_BORROWER_RUNTIME_MS);
            let alloc = AllocationMode::from_parts(
                Template::Command,
                raw_alloc.mode.as_deref(),
                raw_alloc.vram_gb,
                raw_alloc.min_vram_gb,
                raw_alloc.max_vram_gb,
                runtime_ms,
            )
            .map_err(|e| fail(format!("service {name}: {e}")))?;
            let tc = validate_command(&name, cmd)?;
            (alloc, TemplateConfig::Command(tc))
        }
    };

    let lifecycle_str = common
        .lifecycle
        .clone()
        .unwrap_or_else(|| SmolStr::new("on_demand"));
    let lifecycle = match lifecycle_str.as_str() {
        "persistent" => Lifecycle::Persistent,
        "on_demand" => Lifecycle::OnDemand,
        "oneshot" => {
            return Err(fail(format!(
                "service {name}: lifecycle `oneshot` is invalid in a [[service]] block (API-only)"
            )));
        }
        other => return Err(fail(format!("service {name}: unknown lifecycle `{other}`"))),
    };

    let metadata = build_ananke_metadata(common.metadata.as_ref())
        .map_err(|e| fail(format!("service {name} metadata: {e}")))?;
    // llama-cpp always speaks OpenAI; command services never do. No
    // config knob — if you need a command service routed through
    // `/v1/models`, wrap it in a llama-cpp-compatible proxy instead.
    let openai_compat = template == Template::LlamaCpp;

    let dev = common.devices.clone().unwrap_or_default();
    let n_gpu_layers = match &template_config {
        TemplateConfig::LlamaCpp(lc) => lc.n_gpu_layers,
        TemplateConfig::Command(_) => None,
    };
    let placement_policy = match dev.placement.as_deref().unwrap_or("gpu-only") {
        "gpu-only" => PlacementPolicy::GpuOnly,
        "cpu-only" => {
            if n_gpu_layers.unwrap_or(0) != 0 {
                return Err(fail(format!(
                    "service {name}: devices.placement=cpu-only with n_gpu_layers={} is invalid",
                    n_gpu_layers.unwrap()
                )));
            }
            PlacementPolicy::CpuOnly
        }
        "hybrid" => PlacementPolicy::Hybrid,
        other => return Err(fail(format!("service {name}: unknown placement `{other}`"))),
    };

    if let TemplateConfig::LlamaCpp(lc) = &template_config
        && lc.context.is_none()
    {
        warn!(
            service = %name,
            "no context set; the estimator will default to 4096 tokens"
        );
    }

    let raw_override = dev.placement_override.clone().unwrap_or_default();
    if dev.placement_override.is_some() && raw_override.is_empty() {
        return Err(fail(format!(
            "service {name}: devices.placement_override is empty"
        )));
    }
    let mut placement_override = BTreeMap::new();
    for (k, v) in raw_override {
        let slot = match k.as_str() {
            "cpu" => DeviceSlot::Cpu,
            s if s.starts_with("gpu:") => {
                let n: u32 = s[4..].parse().map_err(|_| {
                    fail(format!(
                        "service {name}: invalid placement_override key `{s}`"
                    ))
                })?;
                DeviceSlot::Gpu(n)
            }
            other => {
                return Err(fail(format!(
                    "service {name}: invalid placement_override key `{other}`"
                )));
            }
        };
        if v == 0 {
            return Err(fail(format!(
                "service {name}: placement_override for {k} is zero"
            )));
        }
        placement_override.insert(slot, v);
    }

    if placement_policy == PlacementPolicy::GpuOnly
        && placement_override.contains_key(&DeviceSlot::Cpu)
    {
        return Err(fail(format!(
            "service {name}: placement=gpu-only but placement_override includes cpu"
        )));
    }

    let gpu_allow = dev.gpu_allow.clone().unwrap_or_default();

    let health_raw = common.health.clone().unwrap_or_default();
    let health = HealthSettings {
        http_path: health_raw.http.unwrap_or_else(|| "/v1/models".into()),
        timeout_ms: health_raw
            .timeout
            .map(|s| {
                parse_duration_ms(&s)
                    .map_err(|e| fail(format!("service {name} health.timeout: {e}")))
            })
            .transpose()?
            .unwrap_or(DEFAULT_HEALTH_TIMEOUT_MS),
        probe_interval_ms: health_raw
            .probe_interval
            .map(|s| {
                parse_duration_ms(&s)
                    .map_err(|e| fail(format!("service {name} health.probe_interval: {e}")))
            })
            .transpose()?
            .unwrap_or(DEFAULT_HEALTH_PROBE_INTERVAL_MS),
    };

    let priority = common
        .priority
        .or(defaults.priority)
        .unwrap_or(DEFAULT_SERVICE_PRIORITY);
    let idle_timeout_ms = common
        .idle_timeout
        .as_deref()
        .or(defaults.idle_timeout.as_deref())
        .map(parse_duration_ms)
        .transpose()
        .map_err(|e| fail(format!("service {name} idle_timeout: {e}")))?
        .unwrap_or(DEFAULT_IDLE_TIMEOUT_MS);
    let warming_grace_ms = common
        .warming_grace
        .as_deref()
        .or(defaults.warming_grace.as_deref())
        .map(parse_duration_ms)
        .transpose()
        .map_err(|e| fail(format!("service {name} warming_grace: {e}")))?
        .unwrap_or(DEFAULT_WARMING_GRACE_MS);
    let drain_timeout_ms = common
        .drain_timeout
        .as_deref()
        .map(parse_duration_ms)
        .transpose()
        .map_err(|e| fail(format!("service {name} drain_timeout: {e}")))?
        .unwrap_or(DEFAULT_DRAIN_TIMEOUT_MS);
    let extended_stream_drain_ms = common
        .extended_stream_drain
        .as_deref()
        .map(parse_duration_ms)
        .transpose()
        .map_err(|e| fail(format!("service {name} extended_stream_drain: {e}")))?
        .unwrap_or(DEFAULT_EXTENDED_STREAM_DRAIN_MS);
    let max_request_duration_ms = common
        .max_request_duration
        .as_deref()
        .map(parse_duration_ms)
        .transpose()
        .map_err(|e| fail(format!("service {name} max_request_duration: {e}")))?
        .unwrap_or(DEFAULT_MAX_REQUEST_DURATION_MS);

    let mut filters = Filters::default();
    if let Some(raw_filters) = &common.filters {
        if let Some(strip) = &raw_filters.strip_params {
            filters.strip_params = strip.clone();
        }
        if let Some(set) = &raw_filters.set_params {
            for (k, v) in set {
                let json_val = toml_value_to_json(v.clone())
                    .map_err(|e| fail(format!("service {name} filters.set_params[{k}]: {e}")))?;
                filters.set_params.insert(k.clone(), json_val);
            }
        }
    }

    let start_queue_depth = common
        .start_queue_depth
        .unwrap_or(crate::config::parse::DEFAULT_START_QUEUE_DEPTH);

    let extra_args = common.extra_args.clone().unwrap_or_default();
    // extra_args_append is consumed into extra_args during merge for extending
    // services, but for non-extending services it's still present here. Fold it
    // in so downstream sees a single list.
    let mut all_extra = extra_args;
    if let Some(append) = &common.extra_args_append {
        all_extra.extend(append.iter().cloned());
    }
    let env = common.env.clone().unwrap_or_default();

    // Allocate a private loopback port. Default is auto-assignment from
    // the daemon's pool; a command service may override with a fixed
    // port (used when the external service binds a predictable host
    // port, e.g. a docker container). If the operator didn't override,
    // warn when their `command`/`env` never substitutes `{port}` — that
    // suggests the child binds a fixed port ananke doesn't know about.
    let private_port_override = match &template_config {
        TemplateConfig::Command(cmd) => cmd.private_port_override,
        TemplateConfig::LlamaCpp(_) => None,
    };
    let private_port = if let Some(fixed) = private_port_override {
        if private_ports.contains(fixed) {
            warn!(
                service = %name,
                port = fixed,
                range_start = private_ports.range.start,
                range_end = private_ports.range.end,
                "private_port override falls inside the auto-assignment pool; a later auto-assigned service may collide — move this port outside [private_port_start, private_port_end]"
            );
        }
        fixed
    } else {
        let p = private_ports.allocate(&name)?;
        if let TemplateConfig::Command(cmd) = &template_config
            && !command_uses_port_placeholder(cmd, common.env.as_ref())
        {
            warn!(
                service = %name,
                private_port = p,
                "auto-assigned private_port is never referenced via {{port}} in the command or env — the child likely binds a different port and ananke's proxy will fail to forward. Either substitute {{port}} or set `private_port` to match the child's actual port"
            );
        }
        p
    };

    Ok(ServiceConfig {
        name,
        port,
        private_port,
        lifecycle,
        priority,
        health,
        placement_override,
        placement_policy,
        gpu_allow,
        filters,
        idle_timeout_ms,
        warming_grace_ms,
        drain_timeout_ms,
        extended_stream_drain_ms,
        max_request_duration_ms,
        allocation_mode,
        openai_compat,
        description: common.description.clone(),
        start_queue_depth,
        extra_args: all_extra,
        env,
        metadata,
        template_config,
    })
}

fn validate_llama_cpp(
    name: &SmolStr,
    lc: &RawLlamaCppService,
) -> Result<LlamaCppConfig, ExpectedError> {
    let model = lc.model.clone().ok_or_else(|| {
        fail(format!(
            "service {name}: template llama-cpp requires `model`"
        ))
    })?;
    let flash = lc.flash_attn.unwrap_or(false);
    for (key, val) in [
        ("cache_type_k", lc.cache_type_k.as_deref()),
        ("cache_type_v", lc.cache_type_v.as_deref()),
    ] {
        if let Some(v) = val
            && v != "f16"
            && !flash
        {
            return Err(fail(format!(
                "service {name}: {key}={v} requires flash_attn=true \
                 (llama.cpp requires FA for quantised KV)"
            )));
        }
    }

    Ok(LlamaCppConfig {
        model,
        mmproj: lc.mmproj.clone(),
        context: lc.context,
        n_gpu_layers: lc.n_gpu_layers,
        n_cpu_moe: lc.n_cpu_moe,
        flash_attn: lc.flash_attn,
        cache_type_k: lc.cache_type_k.clone(),
        cache_type_v: lc.cache_type_v.clone(),
        mmap: lc.mmap,
        mlock: lc.mlock,
        parallel: lc.parallel,
        batch_size: lc.batch_size,
        ubatch_size: lc.ubatch_size,
        threads: lc.threads,
        threads_batch: lc.threads_batch,
        jinja: lc.jinja,
        chat_template_file: lc.chat_template_file.clone(),
        override_tensor: lc.override_tensor.clone().unwrap_or_default(),
        sampling: lc.sampling.clone().unwrap_or_default(),
        estimation: lc.estimation.clone().unwrap_or_default(),
    })
}

fn validate_command(
    name: &SmolStr,
    cmd: &RawCommandService,
) -> Result<CommandConfig, ExpectedError> {
    let command = cmd.command.clone().ok_or_else(|| {
        fail(format!(
            "service {name}: command template requires `command`"
        ))
    })?;
    if command.is_empty() {
        return Err(fail(format!("service {name}: command is empty")));
    }
    if let Some(sd) = &cmd.shutdown_command
        && sd.is_empty()
    {
        return Err(fail(format!(
            "service {name}: shutdown_command is present but empty"
        )));
    }
    // Dry-run the placeholder substitution so typos surface now rather
    // than at spawn/drain time. Uses a synthetic context — values are
    // arbitrary but cover every placeholder the supervisor will later
    // supply, so anything the runtime will accept also passes here.
    check_placeholders(name, "command", &command)?;
    if let Some(sd) = &cmd.shutdown_command {
        check_placeholders(name, "shutdown_command", sd)?;
    }
    Ok(CommandConfig {
        command,
        workdir: cmd.workdir.clone(),
        shutdown_command: cmd.shutdown_command.clone(),
        private_port_override: cmd.private_port,
    })
}

/// Resolve every `{placeholder}` in `argv` against a synthetic context
/// covering every substitution the supervisor can produce. Propagates
/// the first [`SubstituteError`] as a config error with `field` + `name`
/// context, so a typo like `{prot}` fails `config validate` rather than
/// slipping through to a runtime `StartFailure`.
fn check_placeholders(name: &SmolStr, field: &str, argv: &[String]) -> Result<(), ExpectedError> {
    use crate::{
        devices::{Allocation, DeviceId},
        templates::{PlaceholderContext, substitute},
    };
    let mut alloc_bytes = std::collections::BTreeMap::new();
    alloc_bytes.insert(DeviceId::Gpu(0), 1);
    let alloc = Allocation { bytes: alloc_bytes };
    let ctx = PlaceholderContext {
        name,
        port: 0,
        model: Some("/m/x.gguf"),
        allocation: &alloc,
        // `None` so a `{vram_mb}` placeholder on a dynamic allocation
        // trips the `VramMbOnDynamic` branch at config time, not
        // later. Static allocations re-validate at spawn time against
        // the real static_vram_mb.
        static_vram_mb: None,
    };
    for (i, arg) in argv.iter().enumerate() {
        substitute(arg, &ctx)
            .map_err(|e| fail(format!("service {name}: {field}[{i}] {arg:?}: {e}")))?;
    }
    Ok(())
}

fn fail(msg: String) -> ExpectedError {
    ExpectedError::config_unparseable(PathBuf::from("<config>"), msg)
}

/// Convert the raw `[[service]] metadata.*` table into the JSON-valued
/// map that [`ServiceConfig`] carries through to the OpenAI and
/// management API responses. Keeps the TOML→JSON coercion in one place
/// so it matches `filters.set_params` and doesn't drift.
fn build_ananke_metadata(
    raw: Option<&BTreeMap<String, toml::Value>>,
) -> Result<AnankeMetadata, String> {
    let Some(m) = raw else {
        return Ok(AnankeMetadata::new());
    };
    m.iter()
        .map(|(k, v)| toml_value_to_json(v.clone()).map(|j| (k.clone(), j)))
        .collect()
}

fn toml_value_to_json(v: toml::Value) -> Result<serde_json::Value, String> {
    Ok(match v {
        toml::Value::String(s) => serde_json::Value::String(s),
        toml::Value::Integer(i) => serde_json::Value::Number(i.into()),
        toml::Value::Float(f) => serde_json::Number::from_f64(f)
            .map(serde_json::Value::Number)
            .ok_or_else(|| "non-finite float".to_string())?,
        toml::Value::Boolean(b) => serde_json::Value::Bool(b),
        toml::Value::Array(a) => serde_json::Value::Array(
            a.into_iter()
                .map(toml_value_to_json)
                .collect::<Result<_, _>>()?,
        ),
        toml::Value::Table(t) => {
            let mut m = serde_json::Map::new();
            for (k, v) in t {
                m.insert(k, toml_value_to_json(v)?);
            }
            serde_json::Value::Object(m)
        }
        toml::Value::Datetime(dt) => serde_json::Value::String(dt.to_string()),
    })
}

/// Default private-port window. Hand out loopback ports llama-server will
/// bind for its private HTTP listener. Configurable via
/// `daemon.private_port_start` / `daemon.private_port_end`.
const DEFAULT_PRIVATE_PORT_START: u16 = 40_000;
const DEFAULT_PRIVATE_PORT_END: u16 = 59_999;

/// Inclusive `start..=end` range of loopback ports assigned to supervised
/// children. Derived from `daemon.private_port_start` / `_end` or the
/// compiled-in default.
#[derive(Debug, Clone, Copy)]
struct PrivatePortRange {
    start: u16,
    end: u16,
}

impl PrivatePortRange {
    fn width(self) -> u32 {
        (self.end as u32) - (self.start as u32) + 1
    }

    fn from_config(start: Option<u16>, end: Option<u16>) -> Result<Self, ExpectedError> {
        let start = start.unwrap_or(DEFAULT_PRIVATE_PORT_START);
        let end = end.unwrap_or(DEFAULT_PRIVATE_PORT_END);
        if end <= start {
            return Err(fail(format!(
                "daemon.private_port_end ({end}) must exceed daemon.private_port_start ({start})"
            )));
        }
        Ok(Self { start, end })
    }
}

/// Hand out unique private ports from a bounded range. First-come
/// first-served from `range.start` upward. External-process collisions are
/// detected at spawn time by llama-server's bind failure, not by this
/// allocator — probing here would only narrow a race window that the
/// supervisor already surfaces as a `StartFailure`.
struct PrivatePortAllocator {
    range: PrivatePortRange,
    next: u32,
}

impl PrivatePortAllocator {
    fn new(range: PrivatePortRange) -> Self {
        Self {
            range,
            next: range.start as u32,
        }
    }

    fn allocate(&mut self, svc_name: &SmolStr) -> Result<u16, ExpectedError> {
        if self.next > self.range.end as u32 {
            return Err(fail(format!(
                "service {svc_name}: private_port_range [{}, {}] exhausted ({} slots) — widen the range or reduce service count",
                self.range.start,
                self.range.end,
                self.range.width()
            )));
        }
        let port = self.next as u16;
        self.next += 1;
        Ok(port)
    }

    /// `true` when `port` is within the allocator's range (and would be
    /// a candidate for auto-assignment). Used to warn operators whose
    /// `private_port` override happens to overlap the auto-pool.
    fn contains(&self, port: u16) -> bool {
        port >= self.range.start && port <= self.range.end
    }
}

/// Returns `true` when the command service's argv or any env value
/// references `{port}`. Heuristic for warning about an auto-assigned
/// `private_port` that the child never receives.
fn command_uses_port_placeholder(
    cmd: &CommandConfig,
    env: Option<&BTreeMap<String, String>>,
) -> bool {
    const PLACEHOLDER: &str = "{port}";
    cmd.command.iter().any(|a| a.contains(PLACEHOLDER))
        || env
            .map(|m| m.values().any(|v| v.contains(PLACEHOLDER)))
            .unwrap_or(false)
}

pub(crate) fn parse_duration_ms(s: &str) -> Result<u64, String> {
    // Accepts "10m", "30s", "500ms", "2h". Returns milliseconds.
    let s = s.trim();
    if let Some(rest) = s.strip_suffix("ms") {
        return rest.parse::<u64>().map_err(|e| e.to_string());
    }
    if let Some(rest) = s.strip_suffix('s') {
        return rest
            .parse::<u64>()
            .map(|n| n * 1000)
            .map_err(|e| e.to_string());
    }
    if let Some(rest) = s.strip_suffix('m') {
        return rest
            .parse::<u64>()
            .map(|n| n * 60_000)
            .map_err(|e| e.to_string());
    }
    if let Some(rest) = s.strip_suffix('h') {
        return rest
            .parse::<u64>()
            .map(|n| n * 3_600_000)
            .map_err(|e| e.to_string());
    }
    Err(format!("unrecognised duration: {s}"))
}

// Silence the unused-import warning when the `test_fixtures` module is not
// compiled (i.e. outside of tests).
#[allow(dead_code)]
const _: () = ();

#[cfg(test)]
pub mod test_fixtures {
    //! Shared `ServiceConfig` factory for unit tests.
    //!
    //! Centralised so individual test modules don't drift in their hand-rolled
    //! fixtures, which previously ranged over the full struct surface and had
    //! to be updated in lockstep every time a field was added.

    use std::{collections::BTreeMap, path::PathBuf};

    use smol_str::SmolStr;

    use super::{
        AllocationMode, CommandConfig, DEFAULT_SERVICE_PRIORITY, DeviceSlot, Filters,
        HealthSettings, Lifecycle, LlamaCppConfig, PlacementPolicy, ServiceConfig, Template,
        TemplateConfig,
    };
    use crate::config::parse::{EstimationConfig, SamplingConfig};

    /// Build a minimal `ServiceConfig` with CPU-only placement, suitable for
    /// unit tests that need a well-formed config but don't care about its
    /// specific field values. The caller is free to mutate the returned
    /// struct to customise individual fields.
    pub fn minimal_service(name: &str) -> ServiceConfig {
        minimal_llama_cpp_service(name)
    }

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
                http_path: "/health".into(),
                timeout_ms: 5_000,
                probe_interval_ms: 200,
            },
            placement_override: placement,
            placement_policy: PlacementPolicy::CpuOnly,
            gpu_allow: Vec::new(),
            idle_timeout_ms: 60_000,
            warming_grace_ms: 100,
            drain_timeout_ms: 1_000,
            extended_stream_drain_ms: 1_000,
            max_request_duration_ms: 5_000,
            filters: Filters::default(),
            allocation_mode: AllocationMode::None,
            openai_compat: true,
            description: None,
            start_queue_depth: 10,
            extra_args: Vec::new(),
            env: BTreeMap::new(),
            metadata: ananke_api::AnankeMetadata::new(),
            template_config: TemplateConfig::LlamaCpp(Box::new(llama_cpp_fixture())),
        }
    }

    pub fn minimal_command_service(name: &str, argv: Vec<String>) -> ServiceConfig {
        let mut svc = minimal_llama_cpp_service(name);
        svc.template_config = TemplateConfig::Command(CommandConfig {
            command: argv,
            workdir: None,
            shutdown_command: None,
            private_port_override: None,
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
            model: PathBuf::from("/fake/model.gguf"),
            mmproj: None,
            context: None,
            n_gpu_layers: None,
            n_cpu_moe: None,
            flash_attn: None,
            cache_type_k: None,
            cache_type_v: None,
            mmap: None,
            mlock: None,
            parallel: None,
            batch_size: None,
            ubatch_size: None,
            threads: None,
            threads_batch: None,
            jinja: None,
            chat_template_file: None,
            override_tensor: Vec::new(),
            sampling: SamplingConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    // Silence unused warnings on types that only specific tests use.
    #[allow(dead_code)]
    fn _coerce_template_used() {
        let _ = Template::LlamaCpp;
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::config::{merge::resolve_inheritance, parse::parse_toml};

    fn parse_and_merge(src: &str) -> RawConfig {
        let mut cfg = parse_toml(src, Path::new("/t")).unwrap();
        resolve_inheritance(&mut cfg).unwrap();
        cfg
    }

    const GOOD: &str = r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 8192
flash_attn = true
cache_type_k = "q8_0"
cache_type_v = "q8_0"
devices.placement = "gpu-only"
devices.placement_override = { "gpu:0" = 18944 }
lifecycle = "persistent"
"#;

    #[test]
    fn validates_good() {
        let cfg = parse_and_merge(GOOD);
        let ec = validate(&cfg).unwrap();
        assert_eq!(ec.services.len(), 1);
        assert_eq!(ec.services[0].name, "demo");
        assert_eq!(ec.services[0].port, 11435);
        assert!(ec.services[0].private_port != 11435);
        assert_eq!(
            ec.services[0].placement_override[&DeviceSlot::Gpu(0)],
            18944
        );
        assert!(matches!(
            ec.services[0].template_config,
            TemplateConfig::LlamaCpp(_)
        ));
    }

    #[test]
    fn phase3_accepts_missing_placement_override() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "gpu-only"
lifecycle = "persistent"
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert!(ec.services[0].placement_override.is_empty());
    }

    #[test]
    fn rejects_duplicate_port() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }

[[service]]
name = "b"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("duplicate") && format!("{err}").contains("port"));
    }

    #[test]
    fn rejects_quantised_kv_without_flash_attn() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
flash_attn = false
cache_type_k = "q8_0"
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("flash_attn"));
    }

    #[test]
    fn rejects_cpu_only_with_ngl_nonzero() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
n_gpu_layers = 10
devices.placement = "cpu-only"
devices.placement_override = { "cpu" = 1000 }
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("cpu-only"));
    }

    #[test]
    fn rejects_oneshot_lifecycle_in_service_block() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "oneshot"
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("oneshot"));
    }

    #[test]
    fn phase2_accepts_on_demand() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "on_demand"
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert_eq!(ec.services[0].lifecycle, Lifecycle::OnDemand);
    }

    #[test]
    fn default_lifecycle_is_on_demand() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert_eq!(ec.services[0].lifecycle, Lifecycle::OnDemand);
    }

    #[test]
    fn parses_filters() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }
filters.strip_params = ["temperature"]
filters.set_params = { max_tokens = 4096 }
"#,
        );
        let ec = validate(&cfg).unwrap();
        let s = &ec.services[0];
        assert_eq!(s.filters.strip_params, vec!["temperature"]);
        assert!(s.filters.set_params.contains_key("max_tokens"));
    }

    #[test]
    fn parses_idle_timeout() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "on_demand"
idle_timeout = "5m"
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert_eq!(ec.services[0].idle_timeout_ms, 300_000);
    }

    #[test]
    fn duration_parser() {
        assert_eq!(parse_duration_ms("500ms").unwrap(), 500);
        assert_eq!(parse_duration_ms("30s").unwrap(), 30_000);
        assert_eq!(parse_duration_ms("10m").unwrap(), 600_000);
        assert_eq!(parse_duration_ms("2h").unwrap(), 7_200_000);
        assert!(parse_duration_ms("bogus").is_err());
    }

    #[test]
    fn command_template_with_static_allocation() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "comfy"
template = "command"
command = ["python", "main.py"]
port = 8188
lifecycle = "on_demand"
allocation.mode = "static"
allocation.vram_gb = 6
"#,
        );
        let ec = validate(&cfg).unwrap();
        let svc = &ec.services[0];
        assert_eq!(svc.template(), Template::Command);
        assert!(matches!(
            svc.allocation_mode,
            AllocationMode::Static { vram_mb: 6144 }
        ));
    }

    #[test]
    fn command_template_with_dynamic_allocation() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "comfy"
template = "command"
command = ["python", "main.py"]
port = 8188
lifecycle = "on_demand"
allocation.mode = "dynamic"
allocation.min_vram_gb = 4
allocation.max_vram_gb = 20
"#,
        );
        let ec = validate(&cfg).unwrap();
        let svc = &ec.services[0];
        assert!(matches!(
            svc.allocation_mode,
            AllocationMode::Dynamic {
                min_mb: 4096,
                max_mb: 20480,
                ..
            }
        ));
    }

    #[test]
    fn llama_cpp_allocation_mode_rejected_at_parse() {
        // With a tagged enum, `allocation` isn't a field on the llama-cpp
        // variant; serde rejects it before the validator runs.
        let res = parse_toml(
            r#"
[[service]]
name = "llama"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
allocation.mode = "static"
allocation.vram_gb = 4
"#,
            Path::new("/t"),
        );
        assert!(
            res.is_err(),
            "expected parse error for allocation on llama-cpp; got {:?}",
            res.ok()
        );
    }

    #[test]
    fn command_rejects_missing_command() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "comfy"
template = "command"
port = 8188
allocation.mode = "static"
allocation.vram_gb = 6
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("requires `command`"));
    }

    #[test]
    fn dynamic_rejects_max_le_min() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "comfy"
template = "command"
command = ["python"]
port = 8188
allocation.mode = "dynamic"
allocation.min_vram_gb = 10
allocation.max_vram_gb = 5
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("max_vram_gb"));
    }

    #[test]
    fn non_loopback_without_flag_is_rejected() {
        let cfg = parse_and_merge(
            r#"
[daemon]
management_listen = "0.0.0.0:17777"

[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
devices.placement_override = { "gpu:0" = 18944 }
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("allow_external_management"));
    }

    #[test]
    fn private_port_range_is_configurable_and_exhausts_cleanly() {
        // A two-port window must fit exactly two services; the third triggers
        // the exhausted-range error with the requested bounds echoed back so
        // the operator knows which knobs to widen. The default 40_000–59_999
        // window is deliberately large, so this guard only matters on hosts
        // that shrink it to dodge a port collision.
        let cfg = parse_and_merge(
            r#"
[daemon]
private_port_start = 50000
private_port_end = 50001

[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }

[[service]]
name = "b"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11001
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }

[[service]]
name = "c"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11002
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("exhausted") && msg.contains("50000") && msg.contains("50001"),
            "expected range-exhausted error naming [50000, 50001]; got: {msg}"
        );
    }

    #[test]
    fn private_port_range_assigns_in_order_from_start() {
        // Two services in a custom window should get start, start+1 — not the
        // 40000-base default, and not duplicates. Regression: an earlier
        // formulation derived the private port from the public port via
        // `40_000 + (port - 11_000)` and wrapped to 65535 for every service.
        let cfg = parse_and_merge(
            r#"
[daemon]
private_port_start = 45000
private_port_end = 45099

[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }

[[service]]
name = "b"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11001
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let ec = validate(&cfg).unwrap();
        let ports: Vec<u16> = ec.services.iter().map(|s| s.private_port).collect();
        assert_eq!(ports, vec![45000, 45001]);
    }

    #[test]
    fn private_port_range_rejects_inverted_bounds() {
        let cfg = parse_and_merge(
            r#"
[daemon]
private_port_start = 50000
private_port_end = 49999

[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("private_port_end") && msg.contains("must exceed"),
            "expected inverted-bounds error; got: {msg}"
        );
    }

    #[test]
    fn non_loopback_with_flag_is_accepted() {
        let cfg = parse_and_merge(
            r#"
[daemon]
management_listen = "0.0.0.0:17777"
allow_external_management = true

[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
devices.placement_override = { "gpu:0" = 18944 }
lifecycle = "persistent"
"#,
        );
        assert!(validate(&cfg).is_ok());
    }

    #[test]
    fn command_service_honours_private_port_override() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "ext"
template = "command"
command = ["/bin/true"]
port = 8500
private_port = 18188
allocation.mode = "static"
allocation.vram_gb = 1
"#,
        );
        let eff = validate(&cfg).expect("validate");
        let svc = &eff.services[0];
        assert_eq!(svc.private_port, 18188);
    }

    #[test]
    fn command_service_rejects_empty_shutdown_command() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "ext"
template = "command"
command = ["/bin/true"]
port = 8500
shutdown_command = []
allocation.mode = "static"
allocation.vram_gb = 1
"#,
        );
        let err = validate(&cfg).expect_err("empty shutdown_command is rejected");
        assert!(
            format!("{err}").contains("shutdown_command is present but empty"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn command_service_rejects_typo_in_placeholder() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "ext"
template = "command"
command = ["run", "--port={prot}"]
port = 8500
allocation.mode = "static"
allocation.vram_gb = 1
"#,
        );
        let err = validate(&cfg).expect_err("typoed placeholder is rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("command[1]") && msg.contains("{prot}"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn command_service_rejects_typo_in_shutdown_placeholder() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "ext"
template = "command"
command = ["run", "--port={port}"]
shutdown_command = ["stop", "{bogus}"]
port = 8500
allocation.mode = "static"
allocation.vram_gb = 1
"#,
        );
        let err = validate(&cfg).expect_err("typoed shutdown placeholder is rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("shutdown_command[1]") && msg.contains("{bogus}"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn private_port_override_outside_pool_does_not_warn() {
        // Smoke-test the code path; we don't capture tracing output,
        // but this at least exercises the branch.
        let cfg = parse_and_merge(
            r#"
[daemon]
private_port_start = 40000
private_port_end = 40100

[[service]]
name = "ext"
template = "command"
command = ["/bin/true"]
port = 8500
private_port = 18188
allocation.mode = "static"
allocation.vram_gb = 1
"#,
        );
        assert!(validate(&cfg).is_ok());
    }
}

// The `_: RawServiceCommon` usage below keeps the import live even if
// downstream test modules don't touch the type directly.
#[allow(dead_code)]
fn _use_common(c: &RawServiceCommon) -> bool {
    c.name.is_some()
}
