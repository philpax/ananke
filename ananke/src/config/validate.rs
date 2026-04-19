//! Validate a post-merge `RawConfig`, producing an `EffectiveConfig` of
//! per-service validated configs plus daemon-global settings.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

use smol_str::SmolStr;
use tracing::warn;

use crate::{
    config::parse::{RawConfig, RawService},
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
}

#[derive(Debug, Clone)]
pub struct ServiceConfig {
    pub name: SmolStr,
    pub template: Template,
    pub port: u16,
    pub private_port: u16,
    pub lifecycle: Lifecycle,
    pub priority: u8,
    pub health: HealthSettings,
    pub placement_override: BTreeMap<DeviceSlot, u64>,
    pub placement_policy: PlacementPolicy,
    pub filters: Filters,
    pub idle_timeout_ms: u64,
    pub warming_grace_ms: u64,
    pub drain_timeout_ms: u64,
    pub extended_stream_drain_ms: u64,
    pub max_request_duration_ms: u64,
    pub allocation_mode: AllocationMode,
    pub command: Option<Vec<String>>,
    pub workdir: Option<PathBuf>,
    pub openai_compat: bool,
    pub raw: RawService,
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

    let mut names: BTreeSet<SmolStr> = BTreeSet::new();
    let mut ports: BTreeSet<u16> = BTreeSet::new();
    let mut out = Vec::new();

    for (i, raw) in cfg.services.iter().enumerate() {
        let name = raw
            .name
            .clone()
            .ok_or_else(|| fail(format!("service[{i}] missing name")))?;
        let port = raw
            .port
            .ok_or_else(|| fail(format!("service {name} missing port")))?;
        let template_str = raw
            .template
            .clone()
            .ok_or_else(|| fail(format!("service {name} missing template")))?;
        let template = match template_str.as_str() {
            "llama-cpp" => Template::LlamaCpp,
            "command" => Template::Command,
            other => return Err(fail(format!("service {name}: unknown template `{other}`"))),
        };

        let raw_alloc = raw.allocation.clone().unwrap_or_default();
        let runtime_ms = raw_alloc
            .min_borrower_runtime
            .as_deref()
            .map(parse_duration_ms)
            .transpose()
            .map_err(|e| fail(format!("service {name} min_borrower_runtime: {e}")))?
            .unwrap_or(DEFAULT_MIN_BORROWER_RUNTIME_MS);
        let allocation_mode = AllocationMode::from_parts(
            template,
            raw_alloc.mode.as_deref(),
            raw_alloc.vram_gb,
            raw_alloc.min_vram_gb,
            raw_alloc.max_vram_gb,
            runtime_ms,
        )
        .map_err(|e| fail(format!("service {name}: {e}")))?;

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

        let lifecycle_str = raw
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

        // Template-specific requirements.
        match template {
            Template::LlamaCpp => {
                if raw.model.is_none() {
                    return Err(fail(format!(
                        "service {name}: template llama-cpp requires `model`"
                    )));
                }
                let flash = raw.flash_attn.unwrap_or(false);
                for (key, val) in [
                    ("cache_type_k", raw.cache_type_k.as_deref()),
                    ("cache_type_v", raw.cache_type_v.as_deref()),
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
            }
            Template::Command => {
                let cmd = raw.command.as_ref().ok_or_else(|| {
                    fail(format!(
                        "service {name}: command template requires `command`"
                    ))
                })?;
                if cmd.is_empty() {
                    return Err(fail(format!("service {name}: command is empty")));
                }
            }
        }

        let openai_compat = raw
            .metadata
            .as_ref()
            .and_then(|m| m.get("openai_compat"))
            .and_then(|v| match v {
                toml::Value::Boolean(b) => Some(*b),
                _ => None,
            })
            // llama-cpp defaults to true; command defaults to false.
            .unwrap_or(template == Template::LlamaCpp);

        let dev = raw.devices.clone().unwrap_or_default();
        let placement_policy = match dev.placement.as_deref().unwrap_or("gpu-only") {
            "gpu-only" => PlacementPolicy::GpuOnly,
            "cpu-only" => {
                if raw.n_gpu_layers.unwrap_or(0) != 0 {
                    return Err(fail(format!(
                        "service {name}: devices.placement=cpu-only with n_gpu_layers={} is invalid",
                        raw.n_gpu_layers.unwrap()
                    )));
                }
                PlacementPolicy::CpuOnly
            }
            "hybrid" => PlacementPolicy::Hybrid,
            other => return Err(fail(format!("service {name}: unknown placement `{other}`"))),
        };

        if raw.context.is_none() {
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

        // Placement/override consistency check.
        if placement_policy == PlacementPolicy::GpuOnly
            && placement_override.contains_key(&DeviceSlot::Cpu)
        {
            return Err(fail(format!(
                "service {name}: placement=gpu-only but placement_override includes cpu"
            )));
        }

        let health_raw = raw.health.clone().unwrap_or_default();
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

        let priority = raw
            .priority
            .or(cfg.defaults.priority)
            .unwrap_or(DEFAULT_SERVICE_PRIORITY);
        let idle_timeout_ms = raw
            .idle_timeout
            .as_deref()
            .or(cfg.defaults.idle_timeout.as_deref())
            .map(parse_duration_ms)
            .transpose()
            .map_err(|e| fail(format!("service {name} idle_timeout: {e}")))?
            .unwrap_or(DEFAULT_IDLE_TIMEOUT_MS);
        let warming_grace_ms = raw
            .warming_grace
            .as_deref()
            .or(cfg.defaults.warming_grace.as_deref())
            .map(parse_duration_ms)
            .transpose()
            .map_err(|e| fail(format!("service {name} warming_grace: {e}")))?
            .unwrap_or(DEFAULT_WARMING_GRACE_MS);
        let drain_timeout_ms = raw
            .drain_timeout
            .as_deref()
            .map(parse_duration_ms)
            .transpose()
            .map_err(|e| fail(format!("service {name} drain_timeout: {e}")))?
            .unwrap_or(DEFAULT_DRAIN_TIMEOUT_MS);
        let extended_stream_drain_ms = raw
            .extended_stream_drain
            .as_deref()
            .map(parse_duration_ms)
            .transpose()
            .map_err(|e| fail(format!("service {name} extended_stream_drain: {e}")))?
            .unwrap_or(DEFAULT_EXTENDED_STREAM_DRAIN_MS);
        let max_request_duration_ms = raw
            .max_request_duration
            .as_deref()
            .map(parse_duration_ms)
            .transpose()
            .map_err(|e| fail(format!("service {name} max_request_duration: {e}")))?
            .unwrap_or(DEFAULT_MAX_REQUEST_DURATION_MS);

        let mut filters = Filters::default();
        if let Some(raw_filters) = &raw.filters {
            if let Some(strip) = &raw_filters.strip_params {
                filters.strip_params = strip.clone();
            }
            if let Some(set) = &raw_filters.set_params {
                for (k, v) in set {
                    let json_val = toml_value_to_json(v.clone()).map_err(|e| {
                        fail(format!("service {name} filters.set_params[{k}]: {e}"))
                    })?;
                    filters.set_params.insert(k.clone(), json_val);
                }
            }
        }

        // Allocate a private loopback port deterministically based on the external port plus a large
        // offset so two services with adjacent external ports don't collide on private ports.
        let private_port = 40_000u16.saturating_add(port.wrapping_sub(11_000));

        out.push(ServiceConfig {
            name,
            template,
            port,
            private_port,
            lifecycle,
            priority,
            health,
            placement_override,
            placement_policy,
            filters,
            idle_timeout_ms,
            warming_grace_ms,
            drain_timeout_ms,
            extended_stream_drain_ms,
            max_request_duration_ms,
            allocation_mode,
            command: raw.command.clone(),
            workdir: raw.workdir.clone(),
            openai_compat,
            raw: raw.clone(),
        });
    }

    Ok(EffectiveConfig {
        daemon: DaemonSettings {
            management_listen: management_addr,
            openai_listen,
            data_dir,
            shutdown_timeout_ms,
            allow_external_management: cfg.daemon.allow_external_management,
        },
        services: out,
    })
}

fn fail(msg: String) -> ExpectedError {
    ExpectedError::config_unparseable(PathBuf::from("<config>"), msg)
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
        AllocationMode, DEFAULT_SERVICE_PRIORITY, DeviceSlot, Filters, HealthSettings, Lifecycle,
        PlacementPolicy, ServiceConfig, Template,
    };
    use crate::config::parse::RawService;

    /// Build a minimal `ServiceConfig` with CPU-only placement, suitable for
    /// unit tests that need a well-formed config but don't care about its
    /// specific field values. The caller is free to mutate the returned
    /// struct to customise individual fields.
    pub fn minimal_service(name: &str) -> ServiceConfig {
        let mut placement = BTreeMap::new();
        placement.insert(DeviceSlot::Cpu, 100);
        ServiceConfig {
            name: SmolStr::new(name),
            template: Template::LlamaCpp,
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
            idle_timeout_ms: 60_000,
            warming_grace_ms: 100,
            drain_timeout_ms: 1_000,
            extended_stream_drain_ms: 1_000,
            max_request_duration_ms: 5_000,
            filters: Filters::default(),
            allocation_mode: AllocationMode::None,
            command: None,
            workdir: None,
            openai_compat: true,
            raw: RawService {
                name: Some(SmolStr::new(name)),
                template: Some(SmolStr::new("llama-cpp")),
                model: Some(PathBuf::from("/fake/model.gguf")),
                port: Some(0),
                ..Default::default()
            },
        }
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
        // placement_override is empty — the estimator will handle placement at runtime.
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
        assert_eq!(svc.template, Template::Command);
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
    fn llama_cpp_rejects_allocation_mode() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "llama"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
allocation.mode = "static"
allocation.vram_gb = 4
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("allocation.mode"));
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
}
