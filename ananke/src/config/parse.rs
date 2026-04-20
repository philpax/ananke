//! Parse a TOML string into a `RawConfig` typed tree (pre-merge, pre-validation).
//!
//! `RawService` is a `#[serde(tag = "template")]` enum with one variant per
//! template kind. Template-specific fields live on the corresponding variant's
//! struct; fields shared across templates live on `RawServiceCommon` and are
//! flattened into each variant. This makes wrong-template fields a parse error
//! rather than a runtime surprise, and lets downstream code pattern-match on a
//! typed variant instead of reaching through a bag of `Option`s.

use std::{collections::BTreeMap, path::PathBuf};

use serde::Deserialize;
use smol_str::SmolStr;

use crate::errors::ExpectedError;

/// Default concurrency cap on pending start requests waiting for the same
/// supervisor to finish warming before they are rejected with `QueueFull`.
pub const DEFAULT_START_QUEUE_DEPTH: usize = 10;

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawAllocation {
    pub mode: Option<SmolStr>,
    /// Static allocation only: VRAM in GiB.
    pub vram_gb: Option<f32>,
    /// Dynamic allocation only: minimum VRAM in GiB.
    pub min_vram_gb: Option<f32>,
    /// Dynamic allocation only: maximum VRAM in GiB.
    pub max_vram_gb: Option<f32>,
    /// Balloon resolver grace period (default 60s); dynamic only.
    pub min_borrower_runtime: Option<String>,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct RawConfig {
    #[serde(default)]
    pub daemon: DaemonConfig,
    #[serde(default)]
    pub devices: DevicesConfig,
    #[serde(default)]
    pub openai_api: OpenAiApiConfig,
    #[serde(default)]
    pub defaults: DefaultsConfig,
    #[serde(default, rename = "service")]
    pub services: Vec<RawService>,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct DaemonConfig {
    #[serde(default = "default_management_listen")]
    pub management_listen: String,
    pub data_dir: Option<PathBuf>,
    #[serde(default = "default_shutdown_timeout")]
    pub shutdown_timeout: String,
    #[serde(default)]
    pub allow_external_management: bool,
    /// Bind per-service reverse proxies on 0.0.0.0 instead of 127.0.0.1
    /// so direct hits to `<host>:<svc.port>` reach them over the network.
    /// The OpenAI multiplexed endpoint on `openai_api.listen` always
    /// honours its own bind address; this controls only the per-service
    /// proxies (one port per `[[service]]`). Same security posture as
    /// `allow_external_management`: unauthenticated, trust the perimeter.
    #[serde(default)]
    pub allow_external_services: bool,
    /// Inclusive lower bound of the loopback port range handed out to
    /// llama-server children for their private listener. Default: 40000.
    pub private_port_start: Option<u16>,
    /// Inclusive upper bound of the private-listener port range. Default:
    /// 59999. Override (together with `private_port_start`) when another
    /// process on the host occupies the default window.
    pub private_port_end: Option<u16>,
}

fn default_management_listen() -> String {
    "127.0.0.1:7777".into()
}

fn default_shutdown_timeout() -> String {
    "120s".into()
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct DevicesConfig {
    #[serde(default)]
    pub gpu_ids: Option<Vec<u32>>,
    #[serde(default)]
    pub gpu_reserved_mb: BTreeMap<String, u64>,
    #[serde(default)]
    pub default_gpu_reserved_mb: Option<u64>,
    #[serde(default)]
    pub cpu: CpuConfig,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct CpuConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub reserved_gb: Option<u64>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct OpenAiApiConfig {
    pub listen: Option<String>,
    #[serde(default)]
    pub enabled: Option<bool>,
    pub max_request_duration: Option<String>,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct DefaultsConfig {
    pub idle_timeout: Option<String>,
    pub priority: Option<u8>,
    pub warming_grace: Option<String>,
    pub start_queue_depth: Option<u32>,
}

/// Template-tagged service: the `template = "llama-cpp" | "command"` field
/// selects a variant. Each variant flattens `RawServiceCommon` so all shared
/// fields appear at the top level of the service table in TOML.
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "template", rename_all = "kebab-case")]
pub enum RawService {
    LlamaCpp(RawLlamaCppService),
    Command(RawCommandService),
}

impl RawService {
    pub fn common(&self) -> &RawServiceCommon {
        match self {
            RawService::LlamaCpp(s) => &s.common,
            RawService::Command(s) => &s.common,
        }
    }

    pub fn common_mut(&mut self) -> &mut RawServiceCommon {
        match self {
            RawService::LlamaCpp(s) => &mut s.common,
            RawService::Command(s) => &mut s.common,
        }
    }

    pub fn template_label(&self) -> &'static str {
        match self {
            RawService::LlamaCpp(_) => "llama-cpp",
            RawService::Command(_) => "command",
        }
    }

    /// Return the start queue depth, falling back to `DEFAULT_START_QUEUE_DEPTH`
    /// when unset.
    pub fn start_queue_depth(&self) -> usize {
        self.common()
            .start_queue_depth
            .unwrap_or(DEFAULT_START_QUEUE_DEPTH)
    }
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawLlamaCppService {
    #[serde(flatten)]
    pub common: RawServiceCommon,
    pub model: Option<PathBuf>,
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
    pub override_tensor: Option<Vec<String>>,
    pub sampling: Option<SamplingConfig>,
    pub estimation: Option<EstimationConfig>,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawCommandService {
    #[serde(flatten)]
    pub common: RawServiceCommon,
    /// argv to execute. Required; emptiness is caught by the validator.
    pub command: Option<Vec<String>>,
    pub workdir: Option<PathBuf>,
    pub allocation: Option<RawAllocation>,
    /// Upstream port ananke's reverse proxy should forward to. When
    /// absent, ananke picks one from the daemon's private-port pool and
    /// substitutes it into `command` / `env` via the `{port}`
    /// placeholder. Set it explicitly when the external service binds a
    /// fixed port (e.g. a docker container exposing 18188 on the host).
    pub private_port: Option<u16>,
    /// Optional argv run at drain time after SIGTERM-then-SIGKILL
    /// completes. Useful for external services that don't stop via
    /// signal — e.g. a docker-run wrapper where SIGTERM reaches the
    /// host shell but the container needs an explicit `docker stop`.
    /// Accepts the same placeholder substitutions as `command`.
    pub shutdown_command: Option<Vec<String>>,
}

/// Fields shared by every template variant. Flattened into each variant so
/// users write `name = "x"` at the top level of `[[service]]` rather than
/// under a nested table.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawServiceCommon {
    pub name: Option<SmolStr>,
    pub extends: Option<SmolStr>,
    pub migrate_from: Option<SmolStr>,
    pub port: Option<u16>,
    pub lifecycle: Option<SmolStr>,
    pub priority: Option<u8>,
    pub idle_timeout: Option<String>,
    pub warming_grace: Option<String>,
    pub description: Option<String>,
    pub filters: Option<RawFilters>,
    pub metadata: Option<BTreeMap<String, toml::Value>>,
    pub devices: Option<RawServiceDevices>,
    pub extra_args: Option<Vec<String>>,
    pub extra_args_append: Option<Vec<String>>,
    pub env: Option<BTreeMap<String, String>>,
    pub health: Option<RawHealth>,
    pub drain_timeout: Option<String>,
    pub extended_stream_drain: Option<String>,
    pub max_request_duration: Option<String>,
    pub start_queue_depth: Option<usize>,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawFilters {
    pub strip_params: Option<Vec<String>>,
    pub set_params: Option<BTreeMap<String, toml::Value>>,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawServiceDevices {
    pub placement: Option<SmolStr>,
    pub gpu_allow: Option<Vec<u32>>,
    pub placement_override: Option<BTreeMap<String, u64>>,
}

/// Estimator overrides. No transformation between parse and validate layers —
/// this type serves both.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct EstimationConfig {
    pub compute_buffer_mb: Option<u32>,
    pub safety_factor: Option<f32>,
    /// Accept the coarse fallback estimate when the GGUF's architecture
    /// isn't recognised by any per-family estimator. Defaults to `false`
    /// — unknown architectures hard-reject at config load so the
    /// operator either adds the arch to the right family list or
    /// explicitly opts in here. The silent fallback previously masked a
    /// 67× under-reservation on glm4moe before it was a recognised
    /// family. (See `ananke::estimator::fallback` for the current
    /// formula.)
    pub allow_fallback: Option<bool>,
}

/// Sampling parameters that map to `llama-server` CLI flags. Only the knobs
/// we actually forward are accepted; unknown keys surface as parse errors
/// rather than silently being dropped. Shared between parse and validate
/// layers — validation is a no-op for this type.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct SamplingConfig {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawHealth {
    pub http: Option<String>,
    pub timeout: Option<String>,
    pub probe_interval: Option<String>,
}

pub fn parse_toml(source: &str, origin_path: &std::path::Path) -> Result<RawConfig, ExpectedError> {
    toml_edit::de::from_str::<RawConfig>(source)
        .map_err(|e| ExpectedError::config_unparseable(origin_path.to_path_buf(), e.to_string()))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn parses_minimal_llama_cpp() {
        let toml = r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
"#;
        let cfg = parse_toml(toml, Path::new("/tmp/c.toml")).unwrap();
        assert_eq!(cfg.services.len(), 1);
        let svc = &cfg.services[0];
        assert_eq!(svc.common().name.as_deref(), Some("demo"));
        assert_eq!(svc.common().port, Some(11435));
        let RawService::LlamaCpp(lc) = svc else {
            panic!("expected LlamaCpp variant");
        };
        assert_eq!(lc.model.as_ref().unwrap().to_str(), Some("/m/x.gguf"));
    }

    #[test]
    fn parses_minimal_command() {
        let toml = r#"
[[service]]
name = "svc"
template = "command"
port = 11500
command = ["/bin/true"]
"#;
        let cfg = parse_toml(toml, Path::new("/tmp/c.toml")).unwrap();
        let svc = &cfg.services[0];
        let RawService::Command(cmd) = svc else {
            panic!("expected Command variant");
        };
        assert_eq!(cmd.command.as_ref().unwrap().as_slice(), ["/bin/true"]);
    }

    #[test]
    fn parses_dotted_keys() {
        let toml = r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
sampling.temperature = 0.7
devices.placement = "gpu-only"
devices.placement_override = { "gpu:0" = 18944 }
"#;
        let cfg = parse_toml(toml, Path::new("/tmp/c.toml")).unwrap();
        let RawService::LlamaCpp(lc) = &cfg.services[0] else {
            panic!("expected LlamaCpp");
        };
        assert_eq!(
            lc.common.devices.as_ref().unwrap().placement.as_deref(),
            Some("gpu-only")
        );
        assert_eq!(
            lc.common
                .devices
                .as_ref()
                .unwrap()
                .placement_override
                .as_ref()
                .unwrap()["gpu:0"],
            18944
        );
    }

    #[test]
    fn rejects_unparseable() {
        let toml = "this is not valid toml [[[";
        let err = parse_toml(toml, Path::new("/tmp/c.toml")).unwrap_err();
        assert!(format!("{err}").contains("parse"));
    }

    #[test]
    fn rejects_llama_cpp_field_on_command_template() {
        // `model` belongs to the llama-cpp variant; with a tagged enum it is
        // not a known field of the command variant, so serde rejects at parse.
        let toml = r#"
[[service]]
name = "svc"
template = "command"
port = 11500
command = ["/bin/true"]
model = "/m/x.gguf"
"#;
        let err = parse_toml(toml, Path::new("/tmp/c.toml"));
        assert!(
            err.is_err(),
            "expected parse error for llama-cpp field on command template, got {:?}",
            err.ok()
        );
    }

    #[test]
    fn rejects_unknown_template() {
        let toml = r#"
[[service]]
name = "svc"
template = "does-not-exist"
port = 11500
"#;
        let err = parse_toml(toml, Path::new("/tmp/c.toml")).unwrap_err();
        assert!(format!("{err}").contains("does-not-exist"));
    }

    #[test]
    fn rejects_missing_template() {
        // Tagged enum requires the discriminator; missing template is a parse error.
        let toml = r#"
[[service]]
name = "svc"
port = 11500
"#;
        let err = parse_toml(toml, Path::new("/tmp/c.toml")).unwrap_err();
        assert!(format!("{err}").contains("template"));
    }
}
