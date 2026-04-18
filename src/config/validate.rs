//! Validate a post-merge `RawConfig`, producing an `EffectiveConfig` of
//! per-service validated configs plus daemon-global settings.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use smol_str::SmolStr;

use crate::config::parse::{RawConfig, RawService};
use crate::errors::ExpectedError;

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
    pub raw: RawService,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Template {
    LlamaCpp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lifecycle {
    Persistent,
    OnDemand,
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

    let management_addr = cfg.daemon.management_listen.clone();
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
            "command" => {
                return Err(fail(format!(
                    "service {name}: template `command` is deferred to phase 4"
                )));
            }
            other => return Err(fail(format!("service {name}: unknown template `{other}`"))),
        };

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
        }

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

        let raw_override = dev.placement_override.clone().ok_or_else(|| {
            fail(format!(
                "service {name}: devices.placement_override is required in phase 1 \
                 (estimator lands in phase 3)"
            ))
        })?;
        if raw_override.is_empty() {
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
                .unwrap_or(180_000),
            probe_interval_ms: health_raw
                .probe_interval
                .map(|s| {
                    parse_duration_ms(&s)
                        .map_err(|e| fail(format!("service {name} health.probe_interval: {e}")))
                })
                .transpose()?
                .unwrap_or(5_000),
        };

        let priority = raw.priority.or(cfg.defaults.priority).unwrap_or(50);
        let idle_timeout_ms = raw
            .idle_timeout
            .as_deref()
            .or(cfg.defaults.idle_timeout.as_deref())
            .map(parse_duration_ms)
            .transpose()
            .map_err(|e| fail(format!("service {name} idle_timeout: {e}")))?
            .unwrap_or(600_000);
        let warming_grace_ms = raw
            .warming_grace
            .as_deref()
            .or(cfg.defaults.warming_grace.as_deref())
            .map(parse_duration_ms)
            .transpose()
            .map_err(|e| fail(format!("service {name} warming_grace: {e}")))?
            .unwrap_or(60_000);
        let drain_timeout_ms = raw
            .drain_timeout
            .as_deref()
            .map(parse_duration_ms)
            .transpose()
            .map_err(|e| fail(format!("service {name} drain_timeout: {e}")))?
            .unwrap_or(30_000);
        let extended_stream_drain_ms = raw
            .extended_stream_drain
            .as_deref()
            .map(parse_duration_ms)
            .transpose()
            .map_err(|e| fail(format!("service {name} extended_stream_drain: {e}")))?
            .unwrap_or(30_000);
        let max_request_duration_ms = raw
            .max_request_duration
            .as_deref()
            .map(parse_duration_ms)
            .transpose()
            .map_err(|e| fail(format!("service {name} max_request_duration: {e}")))?
            .unwrap_or(600_000);

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
            raw: raw.clone(),
        });
    }

    Ok(EffectiveConfig {
        daemon: DaemonSettings {
            management_listen: management_addr,
            openai_listen,
            data_dir,
            shutdown_timeout_ms,
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

fn parse_duration_ms(s: &str) -> Result<u64, String> {
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
mod tests {
    use super::*;
    use crate::config::merge::resolve_inheritance;
    use crate::config::parse::parse_toml;
    use std::path::Path;

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
    fn rejects_missing_placement_override() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
devices.placement = "gpu-only"
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("placement_override"));
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
}
