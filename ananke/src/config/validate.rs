//! Validate a post-merge `RawConfig`, producing an `EffectiveConfig` of
//! per-service validated configs plus daemon-global settings.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    sync::Arc,
};

use ananke_api::shared::{metadata::AnankeMetadata, modality::Modality};
use smol_str::SmolStr;
use tracing::warn;

use crate::{
    config::parse::{
        EstimationConfig, RawAutoRestart, RawCommandService, RawConfig, RawErrorRateSettings,
        RawExpertOffload, RawLlamaCppService, RawPeriodicSettings, RawService, SamplingConfig,
        Toggle,
    },
    errors::ExpectedError,
};

/// Default idle-before-drain timeout for on-demand services (10 minutes).
pub const DEFAULT_IDLE_TIMEOUT_MS: u64 = 600_000;

/// Default OpenAI request body limit (64 MiB). Generous so multi-megabyte
/// base64 vision payloads pass; axum's own default is only 2 MiB, which
/// rejects most real images with `413 Payload Too Large`.
pub const DEFAULT_OPENAI_MAX_BODY_MB: u64 = 64;

/// [`DEFAULT_OPENAI_MAX_BODY_MB`] expressed in bytes, for the contexts that
/// want a byte count directly (e.g. the [`DaemonSettings`] default).
pub const DEFAULT_OPENAI_MAX_BODY_BYTES: usize = DEFAULT_OPENAI_MAX_BODY_MB as usize * 1024 * 1024;

/// Default cadence for the health-probe loop (5 seconds).
pub const DEFAULT_HEALTH_PROBE_INTERVAL_MS: u64 = 5_000;

/// Default per-probe timeout for health checks (3 minutes).
pub const DEFAULT_HEALTH_TIMEOUT_MS: u64 = 180_000;

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

/// Default rolling window for the auto-restart error-rate watchdog (2
/// minutes). Validated against production data: a service that wedges into
/// 100 % 5xx is caught ~60 s after the first error at typical traffic.
pub const DEFAULT_AUTO_RESTART_WINDOW_MS: u64 = 120_000;

/// Default error-rate threshold (fraction of the window) that trips the
/// watchdog.
pub const DEFAULT_AUTO_RESTART_MAX_ERROR_RATE: f64 = 0.5;

/// Default minimum request count in the window before the ratio is trusted.
/// Never fired across 8.5 hours of healthy production traffic; fired within
/// a minute of a real wedge.
pub const DEFAULT_AUTO_RESTART_MIN_REQUESTS: u32 = 20;

/// Default cadence at which the watchdog polls the metrics store (30 s).
pub const DEFAULT_AUTO_RESTART_POLL_INTERVAL_MS: u64 = 30_000;

/// Default anti-flap cooldown: a fresh run must live this long before
/// another auto-restart may fire (5 minutes).
pub const DEFAULT_AUTO_RESTART_MIN_UPTIME_MS: u64 = 300_000;

/// Default number of auto-restarts tolerated within
/// [`DEFAULT_AUTO_RESTART_FLAP_WINDOW_MS`] before the service is disabled.
pub const DEFAULT_AUTO_RESTART_MAX_RESTARTS: u32 = 3;

/// Default sliding window over which [`DEFAULT_AUTO_RESTART_MAX_RESTARTS`]
/// is counted (30 minutes).
pub const DEFAULT_AUTO_RESTART_FLAP_WINDOW_MS: u64 = 1_800_000;

/// Default periodic-restart interval — only consulted when a service enables
/// periodic restarts without spelling out an interval, which is rejected;
/// present for completeness alongside the other knobs.
pub const DEFAULT_AUTO_RESTART_PERIODIC_MODE: PeriodicMode = PeriodicMode::OnRequest;

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
    pub openai_allow_cors: bool,
    pub openai_max_body_bytes: usize,
}

/// Neutral settings for test construction. Production always derives
/// `DaemonSettings` from validated config, never from `Default`, so this is
/// gated to test builds.
#[cfg(any(test, feature = "test-fakes"))]
impl Default for DaemonSettings {
    fn default() -> Self {
        Self {
            management_listen: String::new(),
            openai_listen: String::new(),
            data_dir: PathBuf::new(),
            shutdown_timeout_ms: 5_000,
            allow_external_management: false,
            allow_external_services: false,
            openai_allow_cors: false,
            openai_max_body_bytes: DEFAULT_OPENAI_MAX_BODY_BYTES,
        }
    }
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
    /// Inter-GPU split strategy for multi-GPU llama.cpp services. See
    /// [`SplitMode`]. Default [`SplitMode::Layer`] preserves the historical
    /// first-fit pipeline behaviour.
    pub split_mode: SplitMode,
    /// Extra per-GPU VRAM (MiB) this service keeps free when packing, layered
    /// on top of [`Self::reserves`]. From `[service.devices] gpu_headroom_mb`.
    pub gpu_headroom_mb: u64,
    /// Global device reserves (resolved from `[devices]`), shared here so the
    /// packer reads them without a separate config handle. Identical across
    /// every service in a config, so the `Arc` is cloned per service rather than
    /// the map.
    pub reserves: Arc<DeviceReserves>,
    pub filters: Filters,
    pub idle_timeout_ms: u64,
    pub drain_timeout_ms: u64,
    pub extended_stream_drain_ms: u64,
    pub max_request_duration_ms: u64,
    /// Self-healing restart policy. Error-rate watchdog on by default;
    /// periodic restart off by default. See [`AutoRestartSettings`].
    pub auto_restart: AutoRestartSettings,
    pub allocation_mode: AllocationMode,
    pub openai_compat: bool,
    pub description: Option<String>,
    /// What kind of model the service exposes (chat or embedding).
    /// Default is [`Modality::Chat`] so configs and JSON shipped before
    /// the field landed are unchanged. Embedding services opt in with
    /// `modality = "embedding"` in their `[[service]]` block.
    pub modality: Modality,
    pub start_queue_depth: usize,
    pub extra_args: Vec<String>,
    pub env: BTreeMap<String, String>,
    /// Whether the child process inherits the daemon's environment
    /// (default `true`). When `false`, the child sees only the
    /// variables in `env` plus `CUDA_VISIBLE_DEVICES`.
    pub env_inherit: bool,
    /// Per-service hints that adjust how the snapshotter attributes
    /// observed VRAM/RSS to this service. See [`TrackingSettings`].
    pub tracking: TrackingSettings,
    /// Passthrough entries from `[[service]] metadata.*`. Opaque to the
    /// daemon — these exist only to be echoed back through `/v1/models`
    /// and `/api/services` for clients (Discord rotation, residence
    /// flags, …).
    pub metadata: AnankeMetadata,
    pub template_config: TemplateConfig,
}

/// Snapshotter attribution hints. Empty (`Default::default()`) by default,
/// in which case the snapshotter falls back to "registered pid +
/// transitive descendants" attribution. Set when the service runs in a
/// container (or otherwise out of the daemon's process tree).
#[derive(Debug, Clone, Default)]
pub struct TrackingSettings {
    /// Cgroup v2 path that contains every pid the service's workload
    /// runs in. The snapshotter unions in any pid whose cgroup matches
    /// this path or sits inside its subtree.
    pub cgroup_parent: Option<SmolStr>,
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
    /// MoE expert-offload policy. See [`OffloadMode`].
    pub expert_offload: OffloadMode,
    pub flash_attn: Option<bool>,
    pub cache_type_k: Option<SmolStr>,
    pub cache_type_v: Option<SmolStr>,
    pub mmap: Option<bool>,
    pub mlock: Option<bool>,
    pub parallel: Option<u32>,
    /// `--spec-type` value (e.g. `"draft-mtp"`). See
    /// [`crate::config::parse::RawLlamaCppService::spec_type`].
    pub spec_type: Option<SmolStr>,
    /// `--spec-draft-n-max` value.
    pub spec_draft_n_max: Option<u32>,
    /// Separate draft-model GGUF (`-md` / `--model-draft`). See
    /// [`crate::config::parse::RawLlamaCppService::draft_model`].
    pub draft_model: Option<PathBuf>,
    /// `-kvu` / `--kv-unified` unified KV pool toggle.
    pub kv_unified: Option<bool>,
    /// When `Some(false)`, emit `--no-cache-idle-slots`.
    pub cache_idle_slots: Option<bool>,
    /// `--metrics` endpoint toggle.
    pub metrics: Option<bool>,
    /// `--slots` endpoint toggle.
    pub slots: Option<bool>,
    pub batch_size: Option<u32>,
    pub ubatch_size: Option<u32>,
    pub threads: Option<u32>,
    pub threads_batch: Option<u32>,
    pub jinja: Option<bool>,
    pub chat_template_file: Option<PathBuf>,
    pub override_tensor: Vec<String>,
    pub sampling: SamplingConfig,
    pub estimation: EstimationConfig,
    /// Resolved executable used to launch the service. Defaults to
    /// `"llama-server"` (looked up on `$PATH`); a per-service
    /// `llama_server` overrides that, falling back to the daemon-level
    /// `daemon.llama_server`. Ignored when [`Self::launcher`] is set —
    /// the launcher's first element becomes the executable.
    pub binary: PathBuf,
    /// Optional argv template that replaces the default
    /// `llama-server -m <model> …` invocation. `launcher[0]` becomes
    /// the executable; `launcher[1..]` is substituted with the standard
    /// placeholders and the splat `{args}` (which expands to every
    /// other llama-server flag ananke would have emitted).
    pub launcher: Option<Vec<String>>,
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
    /// When `Some`, the service is fronted by the OpenAI-compatible
    /// multiplexer at `:7070`: it appears in `/v1/models`, accepts
    /// `/v1/chat/completions` (etc.) addressed to its service `name`,
    /// and the multiplexer rewrites the JSON `model` field to
    /// `upstream_model` before forwarding. `None` = ordinary command
    /// service that's only reachable via its per-service reverse proxy.
    pub openai_proxy: Option<OpenAiProxyConfig>,
}

#[derive(Debug, Clone)]
pub struct OpenAiProxyConfig {
    /// Model name written into the upstream's JSON `model` field. The
    /// service's `name` is what clients see in `/v1/models`; this is
    /// what the upstream (vLLM, TGI, …) is asked to serve.
    pub upstream_model: SmolStr,
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

/// How a multi-GPU llama.cpp service divides the model across the GPUs it
/// spans. Orthogonal to [`PlacementPolicy`], which decides CPU-vs-GPU and
/// whether CPU spill is allowed; this decides the *inter-GPU* strategy and
/// maps straight onto llama.cpp's `--split-mode`.
///
/// - `Layer` (default): pipeline — each GPU holds whole layers and the
///   first-fit packer fills one GPU before spilling to the next. Minimal
///   inter-GPU traffic, but only one GPU computes at a time for a single
///   request.
/// - `Row` / `Tensor`: tensor parallelism — every layer is sharded across
///   all spanned GPUs, which compute in parallel and reduce per layer.
///   `tensor` is llama.cpp's newer, faster implementation; `row` is the
///   older one, kept for parity. Both require [`PlacementPolicy::GpuOnly`]
///   (no CPU spill) and a llama-cpp service.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplitMode {
    #[default]
    Layer,
    Row,
    Tensor,
}

impl SplitMode {
    /// The `--split-mode` flag value, also used verbatim in operator-facing
    /// validation errors.
    pub fn as_flag(self) -> &'static str {
        match self {
            SplitMode::Layer => "layer",
            SplitMode::Row => "row",
            SplitMode::Tensor => "tensor",
        }
    }

    /// Whether this mode shards every layer across all spanned GPUs (as
    /// opposed to `Layer`'s whole-layer pipeline). Drives the packer's
    /// balanced-distribution path.
    pub fn is_sharded(self) -> bool {
        matches!(self, SplitMode::Row | SplitMode::Tensor)
    }
}

/// MoE expert-offload policy for a llama-cpp service. Resolved from the
/// `expert_offload` config value. The packer reads this to decide whether and
/// how much expert weight to move off the GPU when the model doesn't fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OffloadMode {
    /// No expert offload. The model packs whole layers, spilling entire layers
    /// to CPU only under a CPU-allowing placement.
    #[default]
    Off,
    /// The packer keeps each expert on its layer's home GPU while that GPU has
    /// room, then greedily spills the experts that don't fit — to the most-free
    /// other GPU first, then to CPU — so only the surplus over live VRAM moves.
    Auto,
    /// The packer offloads the experts of exactly the `N` tail-most
    /// expert-bearing layers, regardless of fit.
    Layers(u32),
}

impl OffloadMode {
    /// Whether any expert offload is requested (i.e. not [`OffloadMode::Off`]).
    pub fn is_enabled(self) -> bool {
        !matches!(self, OffloadMode::Off)
    }
}

/// Per-device VRAM/RAM the daemon keeps free, resolved from the global
/// `[devices]` config. Copied onto each [`ServiceConfig`] so the (pure) packer
/// can read reserves without a separate config handle. The per-service
/// `gpu_headroom_mb` is layered on top of these by the packer.
#[derive(Debug, Clone, Default)]
pub struct DeviceReserves {
    /// VRAM (MiB) kept free on every GPU that lacks a `per_gpu_mb` entry.
    pub default_gpu_mb: u64,
    /// VRAM (MiB) kept free on specific GPUs, keyed by GPU id.
    pub per_gpu_mb: BTreeMap<u32, u64>,
    /// Host RAM (bytes) kept free; bounds the packer's CPU expert offload.
    pub cpu_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct HealthSettings {
    /// HTTP path to probe for readiness. `None` means no health check —
    /// the service transitions to Running immediately after spawn.
    pub http_path: Option<String>,
    pub timeout_ms: u64,
    pub probe_interval_ms: u64,
}

/// Resolved self-healing restart policy for a service. Both triggers are
/// independent `Option`s: the error-rate watchdog is `Some` by default, the
/// periodic timer `None` by default. The guardrail fields apply to whichever
/// trigger fires.
#[derive(Debug, Clone, PartialEq)]
pub struct AutoRestartSettings {
    /// Error-rate watchdog, or `None` if disabled (`error_rate = false`).
    pub error_rate: Option<ErrorRateTrigger>,
    /// Periodic restart, or `None` if disabled (the default).
    pub periodic: Option<PeriodicTrigger>,
    /// Anti-flap cooldown: minimum uptime of a fresh run before another
    /// auto-restart may fire.
    pub min_uptime_ms: u64,
    /// Auto-restarts tolerated within [`Self::flap_window_ms`] before the
    /// service is disabled with `AutoRestartLoop`.
    pub max_restarts: u32,
    /// Sliding window over which [`Self::max_restarts`] is counted.
    pub flap_window_ms: u64,
}

impl AutoRestartSettings {
    /// Whether any trigger is active — a cheap gate the supervisor uses to
    /// skip watchdog setup entirely for services that opted fully out.
    pub fn any_enabled(&self) -> bool {
        self.error_rate.is_some() || self.periodic.is_some()
    }

    /// Both triggers off, guardrails at their defaults. Used by test
    /// fixtures so a supervisor under test gets no watchdog unless the test
    /// opts in explicitly.
    pub fn disabled() -> Self {
        Self {
            error_rate: None,
            periodic: None,
            ..Self::default()
        }
    }
}

impl Default for AutoRestartSettings {
    fn default() -> Self {
        Self {
            error_rate: Some(ErrorRateTrigger::default()),
            periodic: None,
            min_uptime_ms: DEFAULT_AUTO_RESTART_MIN_UPTIME_MS,
            max_restarts: DEFAULT_AUTO_RESTART_MAX_RESTARTS,
            flap_window_ms: DEFAULT_AUTO_RESTART_FLAP_WINDOW_MS,
        }
    }
}

/// Resolved error-rate watchdog thresholds.
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorRateTrigger {
    pub window_ms: u64,
    pub max_error_rate: f64,
    pub min_requests: u32,
    pub poll_interval_ms: u64,
    pub statuses: ErrorStatusClass,
}

impl Default for ErrorRateTrigger {
    fn default() -> Self {
        Self {
            window_ms: DEFAULT_AUTO_RESTART_WINDOW_MS,
            max_error_rate: DEFAULT_AUTO_RESTART_MAX_ERROR_RATE,
            min_requests: DEFAULT_AUTO_RESTART_MIN_REQUESTS,
            poll_interval_ms: DEFAULT_AUTO_RESTART_POLL_INTERVAL_MS,
            statuses: ErrorStatusClass::ServerOnly,
        }
    }
}

/// Which HTTP statuses the error-rate watchdog counts as errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorStatusClass {
    /// Server errors only (500–599). The default — a wedged upstream 5xxs,
    /// whereas 4xx is usually the client's fault and should not self-restart.
    ServerOnly,
    /// Any status ≥ 400 (client and server errors alike).
    ClientAndServer,
}

impl ErrorStatusClass {
    /// Whether a recorded status code counts as an error under this class.
    pub fn is_error(self, status: u16) -> bool {
        match self {
            ErrorStatusClass::ServerOnly => (500..600).contains(&status),
            ErrorStatusClass::ClientAndServer => status >= 400,
        }
    }

    /// The inclusive lower bound on status codes that count as errors, for the
    /// SQL `status_code >= ?` predicate. There are no ≥ 600 statuses in
    /// practice, so `ServerOnly`'s upper bound needs no separate clause.
    pub fn min_status_code(self) -> u16 {
        match self {
            ErrorStatusClass::ServerOnly => 500,
            ErrorStatusClass::ClientAndServer => 400,
        }
    }
}

/// Resolved periodic-restart settings.
#[derive(Debug, Clone, PartialEq)]
pub struct PeriodicTrigger {
    pub interval_ms: u64,
    pub mode: PeriodicMode,
}

/// How a periodic restart is timed once the interval elapses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeriodicMode {
    /// Drain and respawn the moment the interval elapses, interrupting any
    /// in-flight traffic (gracefully, via the normal drain pipeline).
    Immediate,
    /// Wait for a quiet window (no in-flight requests) after the interval
    /// elapses, then restart. Zero disruption, but may never fire under
    /// continuous load.
    OnIdle,
    /// Mark the run stale when the interval elapses; the next request
    /// triggers the restart and blocks on the fresh process. Guarantees the
    /// restart happens even under continuous load.
    OnRequest,
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
        ananke_api::shared::defaults::MANAGEMENT_LISTEN.into()
    } else {
        cfg.daemon.management_listen.clone()
    };
    let mgmt_socket_addr: std::net::SocketAddr = management_addr
        .parse()
        .map_err(|e: std::net::AddrParseError| fail(format!("daemon.management_listen: {e}")))?;
    if !mgmt_socket_addr.ip().is_loopback() && !cfg.daemon.allow_external_management {
        return Err(fail(
            "daemon.management_listen is non-loopback but daemon.allow_external_management is false; \
             the management API has no authentication".into(),
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
        .unwrap_or_else(|| ananke_api::shared::defaults::OPENAI_LISTEN.into());

    let openai_max_body_bytes = cfg
        .openai_api
        .max_body_mb
        .unwrap_or(DEFAULT_OPENAI_MAX_BODY_MB)
        .saturating_mul(1024 * 1024)
        .min(usize::MAX as u64) as usize;

    let private_port_range =
        PrivatePortRange::from_config(cfg.daemon.private_port_start, cfg.daemon.private_port_end)?;
    let mut private_ports = PrivatePortAllocator::new(private_port_range);
    let daemon_llama_server = cfg.daemon.llama_server.clone();
    let device_reserves = Arc::new(resolve_device_reserves(&cfg.devices)?);

    let mut names: BTreeSet<SmolStr> = BTreeSet::new();
    let mut ports: BTreeSet<u16> = BTreeSet::new();
    let mut out = Vec::new();

    let daemon_ctx = DaemonValidationCtx {
        defaults: &cfg.defaults,
        management_port,
        daemon_llama_server: daemon_llama_server.as_deref(),
        reserves: &device_reserves,
    };
    let mut svc_state = ServiceValidationState {
        names: &mut names,
        ports: &mut ports,
        private_ports: &mut private_ports,
    };

    for (i, raw) in cfg.services.iter().enumerate() {
        let svc = validate_service(i, raw, &daemon_ctx, &mut svc_state)?;
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
            openai_allow_cors: cfg.openai_api.allow_cors,
            openai_max_body_bytes,
        },
        services: out,
    })
}

/// Resolve the global `[devices]` reserve knobs into a [`DeviceReserves`].
/// `gpu_reserved_mb` keys are GPU id strings (`"0"`); a non-numeric key is a
/// hard config error rather than a silently ignored reservation.
fn resolve_device_reserves(
    dev: &crate::config::parse::DevicesConfig,
) -> Result<DeviceReserves, ExpectedError> {
    let mut per_gpu_mb = BTreeMap::new();
    for (key, mb) in &dev.gpu_reserved_mb {
        let id: u32 = key.parse().map_err(|_| {
            fail(format!(
                "devices.gpu_reserved_mb: invalid GPU id key `{key}` (expected a number like \"0\")"
            ))
        })?;
        per_gpu_mb.insert(id, *mb);
    }
    Ok(DeviceReserves {
        default_gpu_mb: dev.default_gpu_reserved_mb.unwrap_or(0),
        per_gpu_mb,
        cpu_bytes: dev
            .cpu
            .reserved_gb
            .unwrap_or(0)
            .saturating_mul(1024 * 1024 * 1024),
    })
}

/// Daemon-scoped inputs that don't change across services within a
/// single `validate` call. Grouped into a struct so per-service
/// validation doesn't need a long arg list (and so clippy stops
/// flagging it).
struct DaemonValidationCtx<'a> {
    defaults: &'a crate::config::parse::DefaultsConfig,
    management_port: Option<u16>,
    daemon_llama_server: Option<&'a std::path::Path>,
    /// Global device reserves resolved from `[devices]`, shared with every
    /// service so the packer can read them. The `Arc` is cloned per service.
    reserves: &'a Arc<DeviceReserves>,
}

/// Mutable bookkeeping that accumulates across the per-service loop:
/// the set of names seen so duplicates can be rejected, the same for
/// ports, and the allocator that hands out private loopback ports.
struct ServiceValidationState<'a> {
    names: &'a mut BTreeSet<SmolStr>,
    ports: &'a mut BTreeSet<u16>,
    private_ports: &'a mut PrivatePortAllocator,
}

fn validate_service(
    index: usize,
    raw: &RawService,
    daemon: &DaemonValidationCtx<'_>,
    state: &mut ServiceValidationState<'_>,
) -> Result<ServiceConfig, ExpectedError> {
    let common = raw.common();
    let name = common
        .name
        .clone()
        .ok_or_else(|| fail(format!("service[{index}] missing name")))?;
    let port = common
        .port
        .ok_or_else(|| fail(format!("service {name} missing port")))?;

    if !state.names.insert(name.clone()) {
        return Err(fail(format!("duplicate service name `{name}`")));
    }
    if !state.ports.insert(port) {
        return Err(fail(format!("duplicate service port {port}")));
    }
    if Some(port) == daemon.management_port {
        return Err(fail(format!(
            "service {name} port {port} collides with daemon.management_listen"
        )));
    }

    let (allocation_mode, template_config) = match raw {
        RawService::LlamaCpp(lc) => {
            let tc = validate_llama_cpp(&name, lc, daemon.daemon_llama_server)?;
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
    let modality = match common.modality.as_deref() {
        None | Some("chat") => Modality::Chat,
        Some("embedding") => Modality::Embedding,
        Some(other) => {
            return Err(fail(format!(
                "service {name}: unknown modality `{other}` (valid: `chat`, `embedding`)"
            )));
        }
    };
    // llama-cpp always speaks OpenAI. Command services opt in by
    // setting `[service.openai_proxy] upstream_model = ...`; that's
    // also where the model-name rewrite to the upstream lives.
    let openai_compat = match &template_config {
        TemplateConfig::LlamaCpp(_) => true,
        TemplateConfig::Command(cmd) => cmd.openai_proxy.is_some(),
    };

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
    let gpu_headroom_mb = dev.gpu_headroom_mb.unwrap_or(0);

    let split_mode = match dev.split.as_deref().unwrap_or("layer") {
        "layer" => SplitMode::Layer,
        "row" => SplitMode::Row,
        "tensor" => SplitMode::Tensor,
        other => {
            return Err(fail(format!(
                "service {name}: unknown devices.split `{other}` (expected layer, row, or tensor)"
            )));
        }
    };

    // Expert offload moves expert tensors to the CPU. That makes it
    // incompatible with a sharded (tensor/row) split — which divides every
    // layer across the GPUs in parallel with no CPU half — and it requires a
    // CPU-allowing placement. Reject both combinations at load time rather than
    // silently producing a placement that can't honour the request.
    if let TemplateConfig::LlamaCpp(lc) = &template_config
        && lc.expert_offload.is_enabled()
    {
        if split_mode.is_sharded() {
            return Err(fail(format!(
                "service {name}: expert_offload cannot be combined with devices.split=`{}` (sharded split is GPU-only; expert offload targets the CPU)",
                split_mode.as_flag()
            )));
        }
        if placement_policy != PlacementPolicy::Hybrid {
            return Err(fail(format!(
                "service {name}: expert_offload requires placement=hybrid (expert tensors offload to CPU)"
            )));
        }
    }
    if split_mode.is_sharded() {
        // Tensor/row split shards every layer across all spanned GPUs in
        // parallel; there is no CPU half and no per-tensor override to honour.
        if placement_policy != PlacementPolicy::GpuOnly {
            return Err(fail(format!(
                "service {name}: devices.split=`{}` requires placement=gpu-only (tensor/row split cannot spill to CPU)",
                split_mode.as_flag()
            )));
        }
        match &template_config {
            TemplateConfig::Command(_) => {
                return Err(fail(format!(
                    "service {name}: devices.split=`{}` is only valid for llama-cpp services",
                    split_mode.as_flag()
                )));
            }
            TemplateConfig::LlamaCpp(lc) if !lc.override_tensor.is_empty() => {
                return Err(fail(format!(
                    "service {name}: devices.split=`{}` cannot be combined with override_tensor",
                    split_mode.as_flag()
                )));
            }
            TemplateConfig::LlamaCpp(_) => {}
        }
    }

    let health_raw = common.health.clone().unwrap_or_default();
    let health = HealthSettings {
        http_path: match &health_raw.http {
            Some(s) if s.is_empty() => None,
            Some(s) => Some(s.clone()),
            None => Some("/v1/models".into()),
        },
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
        .or(daemon.defaults.priority)
        .unwrap_or(DEFAULT_SERVICE_PRIORITY);
    let idle_timeout_ms = common
        .idle_timeout
        .as_deref()
        .or(daemon.defaults.idle_timeout.as_deref())
        .map(parse_duration_ms)
        .transpose()
        .map_err(|e| fail(format!("service {name} idle_timeout: {e}")))?
        .unwrap_or(DEFAULT_IDLE_TIMEOUT_MS);
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
    let env_inherit = common.env_inherit.unwrap_or(true);

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
        if state.private_ports.contains(fixed) {
            warn!(
                service = %name,
                port = fixed,
                range_start = state.private_ports.range.start,
                range_end = state.private_ports.range.end,
                "private_port override falls inside the auto-assignment pool; a later auto-assigned service may collide — move this port outside [private_port_start, private_port_end]"
            );
        }
        fixed
    } else {
        let p = state.private_ports.allocate(&name)?;
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

    let tracking = validate_tracking(&name, common.tracking.as_ref())?;

    // auto_restart resolves as a whole block: a service's own block replaces
    // `[defaults.auto_restart]` entirely rather than merging field-by-field.
    let auto_restart = validate_auto_restart(
        &name,
        common
            .auto_restart
            .as_ref()
            .or(daemon.defaults.auto_restart.as_ref()),
    )?;

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
        split_mode,
        gpu_headroom_mb,
        reserves: Arc::clone(daemon.reserves),
        filters,
        idle_timeout_ms,
        drain_timeout_ms,
        extended_stream_drain_ms,
        max_request_duration_ms,
        auto_restart,
        allocation_mode,
        openai_compat,
        description: common.description.clone(),
        modality,
        start_queue_depth,
        extra_args: all_extra,
        env,
        env_inherit,
        tracking,
        metadata,
        template_config,
    })
}

fn validate_tracking(
    name: &SmolStr,
    raw: Option<&crate::config::parse::RawTracking>,
) -> Result<TrackingSettings, ExpectedError> {
    let Some(raw) = raw else {
        return Ok(TrackingSettings::default());
    };
    let cgroup_parent = match &raw.cgroup_parent {
        Some(s) if s.is_empty() => {
            return Err(fail(format!(
                "service {name}: tracking.cgroup_parent is empty — omit the field or supply a non-empty cgroup path"
            )));
        }
        Some(s) if !s.starts_with('/') => {
            return Err(fail(format!(
                "service {name}: tracking.cgroup_parent must be an absolute cgroup v2 path starting with `/` (got `{s}`)"
            )));
        }
        Some(s) if s.ends_with('/') => {
            return Err(fail(format!(
                "service {name}: tracking.cgroup_parent must not end with `/` (got `{s}`)"
            )));
        }
        Some(s)
            if !s
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '/' | '-')) =>
        {
            return Err(fail(format!(
                "service {name}: tracking.cgroup_parent contains invalid characters (allowed: alphanumeric, `.`, `_`, `/`, `-`); got `{s}`"
            )));
        }
        Some(s) => Some(s.clone()),
        None => None,
    };
    Ok(TrackingSettings { cgroup_parent })
}

fn validate_auto_restart(
    name: &SmolStr,
    raw: Option<&RawAutoRestart>,
) -> Result<AutoRestartSettings, ExpectedError> {
    let Some(raw) = raw else {
        return Ok(AutoRestartSettings::default());
    };

    let dur = |field: &str, s: &str| {
        parse_duration_ms(s).map_err(|e| fail(format!("service {name} auto_restart.{field}: {e}")))
    };

    // Error-rate watchdog is on by default; only an explicit `false` disables it.
    let error_rate = match &raw.error_rate {
        None | Some(Toggle::Enabled(true)) => Some(ErrorRateTrigger::default()),
        Some(Toggle::Enabled(false)) => None,
        Some(Toggle::Settings(s)) => Some(validate_error_rate(name, s)?),
    };

    // Periodic is off by default; a table (with an interval) enables it. A bare
    // `true` is rejected because there is no interval to restart on.
    let periodic = match &raw.periodic {
        None | Some(Toggle::Enabled(false)) => None,
        Some(Toggle::Enabled(true)) => {
            return Err(fail(format!(
                "service {name}: auto_restart.periodic = true needs an interval; write `periodic = {{ interval = \"6h\" }}`"
            )));
        }
        Some(Toggle::Settings(s)) => Some(validate_periodic(name, s)?),
    };

    let min_uptime_ms = raw
        .min_uptime
        .as_deref()
        .map(|s| dur("min_uptime", s))
        .transpose()?
        .unwrap_or(DEFAULT_AUTO_RESTART_MIN_UPTIME_MS);
    let max_restarts = raw
        .max_restarts
        .unwrap_or(DEFAULT_AUTO_RESTART_MAX_RESTARTS);
    let flap_window_ms = raw
        .flap_window
        .as_deref()
        .map(|s| dur("flap_window", s))
        .transpose()?
        .unwrap_or(DEFAULT_AUTO_RESTART_FLAP_WINDOW_MS);

    Ok(AutoRestartSettings {
        error_rate,
        periodic,
        min_uptime_ms,
        max_restarts,
        flap_window_ms,
    })
}

fn validate_error_rate(
    name: &SmolStr,
    s: &RawErrorRateSettings,
) -> Result<ErrorRateTrigger, ExpectedError> {
    let d = ErrorRateTrigger::default();
    let window_ms = s
        .window
        .as_deref()
        .map(|x| {
            parse_duration_ms(x).map_err(|e| {
                fail(format!(
                    "service {name} auto_restart.error_rate.window: {e}"
                ))
            })
        })
        .transpose()?
        .unwrap_or(d.window_ms);
    let max_error_rate = match s.max_error_rate {
        None => d.max_error_rate,
        Some(r) if r > 0.0 && r <= 1.0 => r,
        Some(r) => {
            return Err(fail(format!(
                "service {name}: auto_restart.error_rate.max_error_rate must be in (0.0, 1.0], got {r}"
            )));
        }
    };
    let poll_interval_ms = s
        .poll_interval
        .as_deref()
        .map(|x| {
            parse_duration_ms(x).map_err(|e| {
                fail(format!(
                    "service {name} auto_restart.error_rate.poll_interval: {e}"
                ))
            })
        })
        .transpose()?
        .unwrap_or(d.poll_interval_ms);
    let statuses = match s.error_statuses.as_deref() {
        None => d.statuses,
        Some("5xx") => ErrorStatusClass::ServerOnly,
        Some("4xx+5xx") => ErrorStatusClass::ClientAndServer,
        Some(other) => {
            return Err(fail(format!(
                "service {name}: auto_restart.error_rate.error_statuses must be `5xx` or `4xx+5xx`, got `{other}`"
            )));
        }
    };
    Ok(ErrorRateTrigger {
        window_ms,
        max_error_rate,
        min_requests: s.min_requests.unwrap_or(d.min_requests),
        poll_interval_ms,
        statuses,
    })
}

fn validate_periodic(
    name: &SmolStr,
    s: &RawPeriodicSettings,
) -> Result<PeriodicTrigger, ExpectedError> {
    let interval_ms = match s.interval.as_deref() {
        Some(x) => parse_duration_ms(x).map_err(|e| {
            fail(format!(
                "service {name} auto_restart.periodic.interval: {e}"
            ))
        })?,
        None => {
            return Err(fail(format!(
                "service {name}: auto_restart.periodic requires an `interval`"
            )));
        }
    };
    let mode = match s.mode.as_deref() {
        None => DEFAULT_AUTO_RESTART_PERIODIC_MODE,
        Some("immediate") => PeriodicMode::Immediate,
        Some("on-idle") => PeriodicMode::OnIdle,
        Some("on-request") => PeriodicMode::OnRequest,
        Some(other) => {
            return Err(fail(format!(
                "service {name}: auto_restart.periodic.mode must be `immediate`, `on-idle`, or `on-request`, got `{other}`"
            )));
        }
    };
    Ok(PeriodicTrigger { interval_ms, mode })
}

fn validate_llama_cpp(
    name: &SmolStr,
    lc: &RawLlamaCppService,
    daemon_llama_server: Option<&std::path::Path>,
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

    if lc.draft_model.is_some() && lc.spec_type.is_none() {
        return Err(fail(format!(
            "service {name}: draft_model requires spec_type to be set \
             (e.g. spec_type = \"draft-mtp\")"
        )));
    }

    let launcher = match &lc.launcher {
        None => None,
        Some(argv) => {
            if argv.is_empty() {
                return Err(fail(format!(
                    "service {name}: launcher is present but empty"
                )));
            }
            check_launcher_placeholders(name, argv)?;
            Some(argv.clone())
        }
    };
    let binary = lc
        .llama_server
        .clone()
        .or_else(|| daemon_llama_server.map(std::path::Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("llama-server"));

    let expert_offload = match &lc.expert_offload {
        None => OffloadMode::Off,
        Some(RawExpertOffload::Layers(n)) => OffloadMode::Layers(*n),
        Some(RawExpertOffload::Mode(s)) => match s.as_str() {
            "off" => OffloadMode::Off,
            "auto" => OffloadMode::Auto,
            other => {
                return Err(fail(format!(
                    "service {name}: expert_offload `{other}` is invalid \
                     (expected \"off\", \"auto\", or an integer layer count)"
                )));
            }
        },
    };

    Ok(LlamaCppConfig {
        model,
        mmproj: lc.mmproj.clone(),
        context: lc.context,
        n_gpu_layers: lc.n_gpu_layers,
        expert_offload,
        flash_attn: lc.flash_attn,
        cache_type_k: lc.cache_type_k.clone(),
        cache_type_v: lc.cache_type_v.clone(),
        mmap: lc.mmap,
        mlock: lc.mlock,
        parallel: lc.parallel,
        spec_type: lc.spec_type.clone(),
        spec_draft_n_max: lc.spec_draft_n_max,
        draft_model: lc.draft_model.clone(),
        kv_unified: lc.kv_unified,
        cache_idle_slots: lc.cache_idle_slots,
        metrics: lc.metrics,
        slots: lc.slots,
        batch_size: lc.batch_size,
        ubatch_size: lc.ubatch_size,
        threads: lc.threads,
        threads_batch: lc.threads_batch,
        jinja: lc.jinja,
        chat_template_file: lc.chat_template_file.clone(),
        override_tensor: lc.override_tensor.clone().unwrap_or_default(),
        sampling: lc.sampling.clone().unwrap_or_default(),
        estimation: lc.estimation.clone().unwrap_or_default(),
        binary,
        launcher,
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
    let openai_proxy = match &cmd.openai_proxy {
        None => None,
        Some(proxy) => {
            let upstream_model = proxy
                .upstream_model
                .as_ref()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    fail(format!(
                        "service {name}: openai_proxy.upstream_model must be a non-empty string"
                    ))
                })?
                .clone();
            Some(OpenAiProxyConfig { upstream_model })
        }
    };
    Ok(CommandConfig {
        command,
        workdir: cmd.workdir.clone(),
        shutdown_command: cmd.shutdown_command.clone(),
        private_port_override: cmd.private_port,
        openai_proxy,
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

/// Dry-run a llama-cpp `launcher` argv at validate time. Identical
/// purpose to [`check_placeholders`] but tolerates the `{args}` splat
/// (which would otherwise be rejected by [`substitute`]). Surfaces
/// typos like `{prot}` and misuses like `--foo={args}` as config errors
/// rather than runtime `StartFailure`s.
fn check_launcher_placeholders(name: &SmolStr, argv: &[String]) -> Result<(), ExpectedError> {
    use crate::{
        devices::{Allocation, DeviceId},
        templates::{PlaceholderContext, substitute_launcher_argv},
    };
    let mut alloc_bytes = std::collections::BTreeMap::new();
    alloc_bytes.insert(DeviceId::Gpu(0), 1);
    let alloc = Allocation { bytes: alloc_bytes };
    let ctx = PlaceholderContext {
        name,
        port: 0,
        model: Some("/m/x.gguf"),
        allocation: &alloc,
        static_vram_mb: None,
    };
    substitute_launcher_argv(argv, &[], &ctx)
        .map_err(|e| fail(format!("service {name}: launcher: {e}")))?;
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

#[cfg(any(test, feature = "test-fakes"))]
pub mod test_fixtures {
    //! Shared `ServiceConfig` factory for unit tests.
    //!
    //! Centralised so individual test modules don't drift in their hand-rolled
    //! fixtures, which previously ranged over the full struct surface and had
    //! to be updated in lockstep every time a field was added.

    use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

    use smol_str::SmolStr;

    use super::{
        AllocationMode, AutoRestartSettings, CommandConfig, DEFAULT_SERVICE_PRIORITY,
        DeviceReserves, DeviceSlot, Filters, HealthSettings, Lifecycle, LlamaCppConfig, Modality,
        OffloadMode, PlacementPolicy, ServiceConfig, SplitMode, Template, TemplateConfig,
        TrackingSettings,
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
                http_path: Some("/health".into()),
                timeout_ms: 5_000,
                probe_interval_ms: 200,
            },
            placement_override: placement,
            placement_policy: PlacementPolicy::CpuOnly,
            gpu_allow: Vec::new(),
            split_mode: SplitMode::Layer,
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
            metrics: None,
            slots: None,
            batch_size: None,
            ubatch_size: None,
            threads: None,
            threads_batch: None,
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

    fn svc_with_auto_restart(block: &str) -> Result<ServiceConfig, ExpectedError> {
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
        validate(&cfg).map(|ec| ec.services.into_iter().next().unwrap())
    }

    #[test]
    fn auto_restart_defaults_error_rate_on_periodic_off() {
        // No block at all → error-rate watchdog on with defaults, periodic off.
        let svc = svc_with_auto_restart("").unwrap();
        let ar = &svc.auto_restart;
        let er = ar.error_rate.as_ref().expect("error-rate on by default");
        assert_eq!(er.window_ms, DEFAULT_AUTO_RESTART_WINDOW_MS);
        assert_eq!(er.min_requests, DEFAULT_AUTO_RESTART_MIN_REQUESTS);
        assert_eq!(er.statuses, ErrorStatusClass::ServerOnly);
        assert!(ar.periodic.is_none(), "periodic off by default");
        assert_eq!(ar.min_uptime_ms, DEFAULT_AUTO_RESTART_MIN_UPTIME_MS);
        assert_eq!(ar.max_restarts, DEFAULT_AUTO_RESTART_MAX_RESTARTS);
    }

    #[test]
    fn auto_restart_error_rate_false_disables_it() {
        let svc = svc_with_auto_restart("auto_restart = { error_rate = false }").unwrap();
        assert!(svc.auto_restart.error_rate.is_none());
        assert!(!svc.auto_restart.any_enabled());
    }

    #[test]
    fn auto_restart_periodic_table_enables_with_defaults() {
        let svc = svc_with_auto_restart("auto_restart.periodic = { interval = \"6h\" }").unwrap();
        let p = svc
            .auto_restart
            .periodic
            .as_ref()
            .expect("periodic enabled");
        assert_eq!(p.interval_ms, 6 * 60 * 60 * 1000);
        assert_eq!(p.mode, PeriodicMode::OnRequest);
        // Error-rate stays on — the block only touched periodic.
        assert!(svc.auto_restart.error_rate.is_some());
    }

    #[test]
    fn auto_restart_error_rate_thresholds_override() {
        let svc = svc_with_auto_restart(
            "auto_restart.error_rate = { window = \"5m\", max_error_rate = 0.8, min_requests = 50, error_statuses = \"4xx+5xx\" }",
        )
        .unwrap();
        let er = svc.auto_restart.error_rate.as_ref().unwrap();
        assert_eq!(er.window_ms, 5 * 60 * 1000);
        assert_eq!(er.max_error_rate, 0.8);
        assert_eq!(er.min_requests, 50);
        assert_eq!(er.statuses, ErrorStatusClass::ClientAndServer);
    }

    #[test]
    fn auto_restart_rejects_bad_values() {
        assert!(
            svc_with_auto_restart("auto_restart.error_rate = { max_error_rate = 1.5 }").is_err()
        );
        assert!(
            svc_with_auto_restart("auto_restart.error_rate = { error_statuses = \"3xx\" }")
                .is_err()
        );
        assert!(
            svc_with_auto_restart(
                "auto_restart.periodic = { interval = \"6h\", mode = \"eager\" }"
            )
            .is_err()
        );
        // periodic without an interval is meaningless.
        assert!(svc_with_auto_restart("auto_restart.periodic = { mode = \"immediate\" }").is_err());
        assert!(svc_with_auto_restart("auto_restart.periodic = true").is_err());
    }

    #[test]
    fn auto_restart_resolves_from_defaults_whole_block() {
        // A service with no auto_restart block inherits `[defaults.auto_restart]`.
        let src = r#"
[defaults.auto_restart]
error_rate = false
periodic = { interval = "4h", mode = "immediate" }

[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "cpu-only"
"#;
        let cfg = parse_and_merge(src);
        let svc = &validate(&cfg).unwrap().services[0];
        assert!(svc.auto_restart.error_rate.is_none());
        let p = svc.auto_restart.periodic.as_ref().unwrap();
        assert_eq!(p.mode, PeriodicMode::Immediate);
        assert_eq!(p.interval_ms, 4 * 60 * 60 * 1000);
    }

    #[test]
    fn expert_offload_parses_auto_and_count() {
        let auto = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
expert_offload = "auto"
devices.placement = "hybrid"
lifecycle = "persistent"
"#,
        );
        let ec = validate(&auto).unwrap();
        assert_eq!(
            ec.services[0].llama_cpp().unwrap().expert_offload,
            OffloadMode::Auto
        );

        let count = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
expert_offload = 16
devices.placement = "hybrid"
lifecycle = "persistent"
"#,
        );
        let ec = validate(&count).unwrap();
        assert_eq!(
            ec.services[0].llama_cpp().unwrap().expert_offload,
            OffloadMode::Layers(16)
        );
    }

    #[test]
    fn draft_model_requires_spec_type() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
draft_model = "/m/mtp-draft.gguf"
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(
            format!("{err}").contains("draft_model requires spec_type"),
            "got: {err}"
        );
    }

    #[test]
    fn draft_model_with_spec_type_validates() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
flash_attn = true
spec_type = "draft-mtp"
draft_model = "/m/mtp-draft.gguf"
kv_unified = true
cache_idle_slots = false
metrics = true
slots = true
lifecycle = "persistent"
"#,
        );
        let ec = validate(&cfg).unwrap();
        let lc = ec.services[0].llama_cpp().unwrap();
        assert_eq!(
            lc.draft_model.as_deref(),
            Some(std::path::Path::new("/m/mtp-draft.gguf"))
        );
        assert_eq!(lc.kv_unified, Some(true));
        assert_eq!(lc.cache_idle_slots, Some(false));
        assert_eq!(lc.metrics, Some(true));
        assert_eq!(lc.slots, Some(true));
    }

    #[test]
    fn expert_offload_requires_hybrid_placement() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
expert_offload = "auto"
devices.placement = "gpu-only"
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(
            format!("{err}").contains("expert_offload requires placement=hybrid"),
            "got: {err}"
        );
    }

    #[test]
    fn expert_offload_rejects_sharded_split() {
        // A sharded (tensor/row) split has no CPU half, so it cannot honour an
        // expert offload to host RAM — reject the combination explicitly rather
        // than leaving the operator to infer it from the placement constraints.
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
expert_offload = "auto"
devices.placement = "gpu-only"
devices.split = "tensor"
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(
            format!("{err}").contains("expert_offload cannot be combined with devices.split"),
            "got: {err}"
        );
    }

    #[test]
    fn n_cpu_moe_is_rejected_as_unknown_field() {
        // The legacy knob no longer exists; deny_unknown_fields surfaces it.
        let err = parse_toml(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
n_cpu_moe = 8
"#,
            Path::new("/t"),
        )
        .unwrap_err();
        assert!(
            format!("{err}").contains("n_cpu_moe"),
            "unknown-field error should name n_cpu_moe, got: {err}"
        );
    }

    #[test]
    fn resolves_global_reserves_and_headroom() {
        let cfg = parse_and_merge(
            r#"
[devices]
default_gpu_reserved_mb = 512
gpu_reserved_mb = { "1" = 4096 }
[devices.cpu]
reserved_gb = 8

[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "hybrid"
devices.gpu_headroom_mb = 1024
lifecycle = "persistent"
"#,
        );
        let ec = validate(&cfg).unwrap();
        let svc = &ec.services[0];
        assert_eq!(svc.gpu_headroom_mb, 1024);
        assert_eq!(svc.reserves.default_gpu_mb, 512);
        assert_eq!(svc.reserves.per_gpu_mb.get(&1).copied(), Some(4096));
        assert_eq!(svc.reserves.cpu_bytes, 8 * 1024 * 1024 * 1024);
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
    fn parses_tensor_split_mode() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "gpu-only"
devices.split = "tensor"
lifecycle = "persistent"
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert_eq!(ec.services[0].split_mode, SplitMode::Tensor);
    }

    #[test]
    fn defaults_split_mode_to_layer() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
lifecycle = "persistent"
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert_eq!(ec.services[0].split_mode, SplitMode::Layer);
    }

    #[test]
    fn rejects_unknown_split_mode() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
devices.split = "diagonal"
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("unknown devices.split"));
    }

    #[test]
    fn rejects_tensor_split_with_cpu_spill() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "hybrid"
devices.split = "tensor"
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("requires placement=gpu-only"));
    }

    #[test]
    fn rejects_tensor_split_on_command_service() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "command"
command = ["/bin/true"]
port = 11435
allocation.mode = "static"
allocation.vram_gb = 4
devices.placement = "gpu-only"
devices.split = "row"
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("only valid for llama-cpp"));
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
    fn tracking_cgroup_parent_accepted_when_well_formed() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "comfy"
template = "command"
command = ["python"]
port = 8188
lifecycle = "on_demand"
allocation.mode = "dynamic"
allocation.min_vram_gb = 2
allocation.max_vram_gb = 20
tracking.cgroup_parent = "/system.slice/ananke-comfyui.slice"
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert_eq!(
            ec.services[0].tracking.cgroup_parent.as_deref(),
            Some("/system.slice/ananke-comfyui.slice"),
        );
    }

    #[test]
    fn tracking_rejects_relative_cgroup_path() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "comfy"
template = "command"
command = ["python"]
port = 8188
lifecycle = "on_demand"
allocation.mode = "static"
allocation.vram_gb = 4
tracking.cgroup_parent = "ananke-comfyui.slice"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(
            format!("{err}").contains("absolute cgroup v2 path"),
            "expected absolute-path error, got: {err}"
        );
    }

    #[test]
    fn tracking_rejects_trailing_slash() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "comfy"
template = "command"
command = ["python"]
port = 8188
lifecycle = "on_demand"
allocation.mode = "static"
allocation.vram_gb = 4
tracking.cgroup_parent = "/system.slice/ananke-comfyui.slice/"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(
            format!("{err}").contains("must not end with"),
            "expected trailing-slash error, got: {err}"
        );
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
    fn command_service_with_openai_proxy_is_listed() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "qwen3.6-27b-vllm"
template = "command"
command = ["/run/vllm.sh", "{port}"]
port = 8500
allocation.mode = "static"
allocation.vram_gb = 1
[service.openai_proxy]
upstream_model = "qwen3.6-27b-autoround"
"#,
        );
        let eff = validate(&cfg).expect("validate");
        let svc = &eff.services[0];
        assert!(svc.openai_compat, "openai_compat should be true");
        let cmd = svc.command().expect("command template");
        let proxy = cmd.openai_proxy.as_ref().expect("openai_proxy populated");
        assert_eq!(proxy.upstream_model, "qwen3.6-27b-autoround");
    }

    #[test]
    fn command_service_rejects_empty_openai_proxy_upstream_model() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "ext"
template = "command"
command = ["/bin/true"]
port = 8500
allocation.mode = "static"
allocation.vram_gb = 1
[service.openai_proxy]
upstream_model = ""
"#,
        );
        let err = validate(&cfg).expect_err("empty upstream_model is rejected");
        assert!(
            format!("{err}").contains("openai_proxy.upstream_model"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn command_service_rejects_missing_openai_proxy_upstream_model() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "ext"
template = "command"
command = ["/bin/true"]
port = 8500
allocation.mode = "static"
allocation.vram_gb = 1
[service.openai_proxy]
"#,
        );
        let err = validate(&cfg).expect_err("missing upstream_model is rejected");
        assert!(
            format!("{err}").contains("openai_proxy.upstream_model"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn command_service_without_openai_proxy_is_hidden() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "comfy"
template = "command"
command = ["/bin/comfyui"]
port = 8188
allocation.mode = "dynamic"
allocation.min_vram_gb = 2
allocation.max_vram_gb = 8
"#,
        );
        let eff = validate(&cfg).expect("validate");
        let svc = &eff.services[0];
        assert!(
            !svc.openai_compat,
            "openai_compat should default to false for command services without openai_proxy"
        );
        assert!(
            svc.command()
                .expect("command template")
                .openai_proxy
                .is_none()
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
    fn llama_server_defaults_to_path_lookup() {
        let cfg = parse_and_merge(GOOD);
        let ec = validate(&cfg).unwrap();
        let lc = ec.services[0].llama_cpp().unwrap();
        assert_eq!(lc.binary, PathBuf::from("llama-server"));
        assert!(lc.launcher.is_none());
    }

    #[test]
    fn daemon_llama_server_default_applies_when_service_unset() {
        let cfg = parse_and_merge(
            r#"
[daemon]
llama_server = "/opt/llama-build/llama-server"

[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let ec = validate(&cfg).unwrap();
        let lc = ec.services[0].llama_cpp().unwrap();
        assert_eq!(lc.binary, PathBuf::from("/opt/llama-build/llama-server"));
    }

    #[test]
    fn service_llama_server_overrides_daemon_default() {
        let cfg = parse_and_merge(
            r#"
[daemon]
llama_server = "/opt/global"

[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
llama_server = "/opt/per-service"
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let ec = validate(&cfg).unwrap();
        let lc = ec.services[0].llama_cpp().unwrap();
        assert_eq!(lc.binary, PathBuf::from("/opt/per-service"));
    }

    #[test]
    fn launcher_accepts_well_formed_template() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
launcher = ["/opt/podman-wrap.sh", "{model}", "{args}"]
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let ec = validate(&cfg).unwrap();
        let lc = ec.services[0].llama_cpp().unwrap();
        assert_eq!(
            lc.launcher.as_deref(),
            Some(
                &[
                    "/opt/podman-wrap.sh".to_string(),
                    "{model}".into(),
                    "{args}".into()
                ][..]
            )
        );
    }

    #[test]
    fn launcher_rejects_unknown_placeholder() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
launcher = ["wrap.sh", "{model}", "{bogus}", "{args}"]
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("{bogus}") && msg.contains("launcher"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn launcher_rejects_splat_embedded_in_arg() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
launcher = ["wrap.sh", "{model}", "--foo={args}"]
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("{args}"));
    }

    #[test]
    fn launcher_rejects_empty_argv() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
launcher = []
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("launcher"));
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

    #[test]
    fn env_inherit_defaults_to_true() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "svc"
template = "command"
command = ["/bin/true"]
port = 11500
allocation.mode = "static"
allocation.vram_gb = 1
"#,
        );
        let validated = validate(&cfg).unwrap();
        let svc = validated.services.iter().find(|s| s.name == "svc").unwrap();
        assert!(svc.env_inherit, "env_inherit should default to true");
    }

    #[test]
    fn env_inherit_false_parsed() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "svc"
template = "command"
command = ["/bin/true"]
port = 11500
allocation.mode = "static"
allocation.vram_gb = 1
env_inherit = false
"#,
        );
        let validated = validate(&cfg).unwrap();
        let svc = validated.services.iter().find(|s| s.name == "svc").unwrap();
        assert!(!svc.env_inherit, "env_inherit should be false");
    }
}
