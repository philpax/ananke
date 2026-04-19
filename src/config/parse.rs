//! Parse a TOML string into a `RawConfig` typed tree (pre-merge, pre-validation).

use std::{collections::BTreeMap, path::PathBuf};

use serde::Deserialize;
use smol_str::SmolStr;

use crate::errors::ExpectedError;

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
    #[serde(default, rename = "persistent_service")]
    pub persistent_services: Vec<RawService>,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct DaemonConfig {
    #[serde(default = "default_management_listen")]
    pub management_listen: String,
    pub data_dir: Option<PathBuf>,
    #[serde(default = "default_shutdown_timeout")]
    pub shutdown_timeout: String,
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

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(default)]
pub struct RawService {
    pub name: Option<SmolStr>,
    pub template: Option<SmolStr>,
    pub extends: Option<SmolStr>,
    pub migrate_from: Option<SmolStr>,
    pub port: Option<u16>,
    pub model: Option<PathBuf>,
    pub mmproj: Option<PathBuf>,
    pub context: Option<u32>,
    pub lifecycle: Option<SmolStr>,
    pub priority: Option<u8>,
    pub idle_timeout: Option<String>,
    pub warming_grace: Option<String>,
    pub description: Option<String>,
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
    pub sampling: Option<BTreeMap<String, toml::Value>>,
    pub filters: Option<RawFilters>,
    pub metadata: Option<BTreeMap<String, toml::Value>>,
    pub devices: Option<RawServiceDevices>,
    pub estimation: Option<RawEstimation>,
    pub extra_args: Option<Vec<String>>,
    pub extra_args_append: Option<Vec<String>>,
    pub env: Option<BTreeMap<String, String>>,
    pub health: Option<RawHealth>,
    pub drain_timeout: Option<String>,
    pub extended_stream_drain: Option<String>,
    pub max_request_duration: Option<String>,
    #[serde(default)]
    pub start_queue_depth: Option<usize>,
    /// Command template: the argv to execute.
    pub command: Option<Vec<String>>,
    /// Command template: working directory for the spawned process.
    pub workdir: Option<PathBuf>,
    /// Allocation mode for command-template services.
    pub allocation: Option<RawAllocation>,
}

impl RawService {
    /// Return the start queue depth, falling back to 10 when unset.
    pub fn start_queue_depth(&self) -> usize {
        self.start_queue_depth.unwrap_or(10)
    }
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

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawEstimation {
    pub compute_buffer_mb: Option<u32>,
    pub safety_factor: Option<f32>,
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
    fn parses_minimal() {
        let toml = r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
"#;
        let cfg = parse_toml(toml, Path::new("/tmp/c.toml")).unwrap();
        assert_eq!(cfg.services.len(), 1);
        assert_eq!(cfg.services[0].name.as_deref(), Some("demo"));
        assert_eq!(cfg.services[0].port, Some(11435));
    }

    #[test]
    fn parses_persistent_service_alias() {
        let toml = r#"
[[persistent_service]]
name = "big"
template = "llama-cpp"
model = "/m/b.gguf"
port = 11500
"#;
        let cfg = parse_toml(toml, Path::new("/tmp/c.toml")).unwrap();
        assert_eq!(cfg.persistent_services.len(), 1);
        assert_eq!(cfg.persistent_services[0].name.as_deref(), Some("big"));
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
        let s = &cfg.services[0];
        assert_eq!(
            s.devices.as_ref().unwrap().placement.as_deref(),
            Some("gpu-only")
        );
        assert_eq!(
            s.devices
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
}
