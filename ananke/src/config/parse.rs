//! Parse a TOML string into a `RawConfig` typed tree (pre-merge, pre-validation).
//!
//! `RawService` is a `#[serde(tag = "template")]` enum with one variant per
//! template kind. Template-specific fields live on the corresponding variant's
//! struct; fields shared across templates live on `RawServiceCommon` and are
//! flattened into each variant. This makes wrong-template fields a parse error
//! rather than a runtime surprise, and lets downstream code pattern-match on a
//! typed variant instead of reaching through a bag of `Option`s.

use std::{collections::BTreeMap, path::PathBuf};

/// Default concurrency cap on pending start requests waiting for the same
/// supervisor to finish starting before they are rejected with `QueueFull`.
pub use ananke_config::docs::DEFAULT_START_QUEUE_DEPTH;
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
    /// Path (or `$PATH` lookup name) of the llama-server executable used
    /// when spawning llama-cpp services. Defaults to `"llama-server"`
    /// (looked up on `$PATH`). A per-service `llama_server` field
    /// overrides this. Useful when the llama-server binary lives outside
    /// `$PATH`, or when operators wrap it in a container/script that
    /// still accepts llama-server's CLI.
    pub llama_server: Option<PathBuf>,
}

fn default_management_listen() -> String {
    ananke_config::defaults::MANAGEMENT_LISTEN.into()
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
    /// Allow cross-origin requests from browsers. Defaults to `true`
    /// since ananke is unauthenticated and designed for trusted-network
    /// deployment; operators who want to block browser-based access to
    /// the OpenAI API can set `allow_cors = false`.
    #[serde(default = "default_true")]
    pub allow_cors: bool,
    /// Maximum request body size for the OpenAI endpoints, in mebibytes.
    /// Vision requests carry base64-encoded images that routinely exceed
    /// axum's 2 MiB default body limit, so ananke's default is generous.
    /// Raise it if a single request carries very large or many images.
    pub max_body_mb: Option<u64>,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct DefaultsConfig {
    pub idle_timeout: Option<String>,
    pub priority: Option<u8>,
    pub start_queue_depth: Option<u32>,
    /// Fleet-wide default auto-restart policy, applied to any service that
    /// does not set its own `[service.auto_restart]` block. See
    /// [`RawAutoRestart`].
    pub auto_restart: Option<RawAutoRestart>,
}

/// Template-tagged service: the `template = "llama-cpp" | "command"` field
/// selects a variant. Each variant flattens `RawServiceCommon` so all shared
/// fields appear at the top level of the service table in TOML.
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "template", rename_all = "kebab-case")]
pub enum RawService {
    /// Both variants are boxed: each inner struct flattens
    /// `RawServiceCommon` and so runs to several hundred bytes, and the
    /// llama-cpp side carries ~1 KiB of optional knobs on top. Boxing
    /// keeps `RawService` pointer-sized and the two variants uniform,
    /// mirroring the boxing on the validated
    /// [`crate::config::validate::TemplateConfig`].
    LlamaCpp(Box<RawLlamaCppService>),
    Command(Box<RawCommandService>),
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
    /// MoE expert-offload policy: `"off"` (no expert offload — whole-layer CPU
    /// spill only), `"auto"` (the packer offloads the minimum experts to fit
    /// live VRAM), or an integer `N` (offload the experts of the N tail-most
    /// expert layers). Validated into [`crate::config::OffloadMode`]. Requires
    /// a CPU-allowing placement (`placement = "hybrid"`) when not `"off"`.
    pub expert_offload: Option<RawExpertOffload>,
    pub flash_attn: Option<bool>,
    pub cache_type_k: Option<SmolStr>,
    pub cache_type_v: Option<SmolStr>,
    pub mmap: Option<bool>,
    pub mlock: Option<bool>,
    pub parallel: Option<u32>,
    /// Speculative-decoding type passed to llama-server's `--spec-type`
    /// (e.g. `"draft-mtp"` for multi-token prediction). When set to
    /// `"draft-mtp"` and the model carries an MTP head
    /// (`nextn_predict_layers > 0`), the estimator adds the MTP draft
    /// context's KV + compute overhead. MTP composes with `parallel > 1`
    /// and `mmproj` — both are supported by current llama.cpp.
    pub spec_type: Option<SmolStr>,
    /// Maximum number of draft tokens per step, passed to
    /// `--spec-draft-n-max`. Only meaningful when `spec_type` is set.
    pub spec_draft_n_max: Option<u32>,
    /// Separate draft-model GGUF for speculative decoding, passed to
    /// llama-server's `-md` / `--model-draft`. Used with
    /// `spec_type = "draft-mtp"` for model families that ship their MTP
    /// head as a standalone file (e.g. Gemma 4's `gemma4-assistant` head)
    /// rather than embedded in the target GGUF (Qwen 3.6). When set, the
    /// estimator reads this file to add the draft model's resident-weight
    /// plus compute-buffer overhead; its attention layers reuse the
    /// target's KV cache, so it adds no context-scaling KV. See
    /// [`crate::estimator::mtp`].
    pub draft_model: Option<PathBuf>,
    /// Use a single unified KV cache pool shared across all parallel
    /// slots (`-kvu` / `--kv-unified`) instead of statically partitioning
    /// the context window per slot. With `parallel > 1` this lets idle
    /// slots lend their share to active ones; the total KV footprint is
    /// unchanged, so the estimate does not depend on it.
    pub kv_unified: Option<bool>,
    /// When `false`, pass `--no-cache-idle-slots` so llama-server does not
    /// retain idle slots' prompt-cache state. Unset leaves llama-server's
    /// default (idle-slot caching on).
    pub cache_idle_slots: Option<bool>,
    /// Expose the Prometheus `/metrics` endpoint (`--metrics`).
    pub metrics: Option<bool>,
    /// Expose the `/slots` introspection endpoint (`--slots`).
    pub slots: Option<bool>,
    pub batch_size: Option<u32>,
    pub ubatch_size: Option<u32>,
    pub threads: Option<u32>,
    pub threads_batch: Option<u32>,
    pub jinja: Option<bool>,
    pub chat_template_file: Option<PathBuf>,
    pub override_tensor: Option<Vec<String>>,
    pub sampling: Option<SamplingConfig>,
    pub estimation: Option<EstimationConfig>,
    /// Per-service override of the llama-server executable. Overrides
    /// the daemon-level `daemon.llama_server` default. Has no effect
    /// when `launcher` is set — the launcher's first element is the
    /// executable in that case.
    pub llama_server: Option<PathBuf>,
    /// Full argv template that replaces the default
    /// `llama-server -m <model> …` invocation. When set, `launcher[0]`
    /// is the executable and `launcher[1..]` is its argv. Each entry is
    /// substituted with the standard placeholders (`{model}`,
    /// `{mmproj}`, `{port}`, `{name}`, `{gpu_ids}`) plus the splat
    /// `{args}`, which expands to every llama-server flag ananke would
    /// otherwise have emitted (excluding `-m <model>` — that lives in
    /// `{model}` so wrappers can position it freely). Lets operators
    /// front llama-server with a docker/podman wrapper that has its own
    /// argv shape.
    pub launcher: Option<Vec<String>>,
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
    /// Opt the command service into the OpenAI-compatible multiplexer.
    /// When present, the service shows up in `/v1/models` and accepts
    /// `/v1/chat/completions` and friends; the multiplexer rewrites the
    /// JSON `model` field to `upstream_model` before forwarding to the
    /// service's private port.
    pub openai_proxy: Option<RawOpenAiProxy>,
}

/// `[service.openai_proxy]` block. Marks a `command` service as fronting
/// an upstream OpenAI-compatible API (vLLM, TGI, SGLang, …) so ananke's
/// allocator and lifecycle apply uniformly with the llama.cpp services.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawOpenAiProxy {
    /// Model name passed to the upstream's OpenAI API. The service's
    /// `name` is what clients see; this is what ananke writes into the
    /// JSON `model` field before forwarding. Required; the validator
    /// rejects an empty/missing value.
    pub upstream_model: Option<SmolStr>,
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
    pub description: Option<String>,
    /// What kind of OpenAI endpoint the service serves: `"chat"`
    /// (default) or `"embedding"`. Embedding services advertise
    /// themselves through `/v1/models` + `/api/services` with a typed
    /// `modality` field clients can filter on, and the frontend renders
    /// an `embedding` badge alongside the service name. Validated into
    /// the [`ananke_api::shared::modality::Modality`] enum during config validation; an
    /// unknown string is a hard config error rather than a silent fall
    /// back to chat.
    pub modality: Option<SmolStr>,
    pub filters: Option<RawFilters>,
    pub metadata: Option<BTreeMap<String, toml::Value>>,
    pub devices: Option<RawServiceDevices>,
    pub extra_args: Option<Vec<String>>,
    pub extra_args_append: Option<Vec<String>>,
    pub env: Option<BTreeMap<String, String>>,
    /// Whether the child process inherits the daemon's environment
    /// (default `true`). When `false`, the child sees only the
    /// variables in `env` plus `CUDA_VISIBLE_DEVICES`.
    pub env_inherit: Option<bool>,
    pub health: Option<RawHealth>,
    pub drain_timeout: Option<String>,
    pub extended_stream_drain: Option<String>,
    pub max_request_duration: Option<String>,
    pub start_queue_depth: Option<usize>,
    pub tracking: Option<RawTracking>,
    /// Self-healing restart policy. See [`RawAutoRestart`]. Resolved as a
    /// whole block: a service that sets any `auto_restart` field replaces
    /// `[defaults.auto_restart]` entirely rather than merging field-by-field.
    pub auto_restart: Option<RawAutoRestart>,
}

/// `[service.tracking]` block. Optional per-service hints that adjust how
/// the snapshotter attributes observed VRAM/RSS to the service.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawTracking {
    /// Cgroup v2 path (e.g. `/system.slice/ananke-comfyui.slice`) under
    /// which the service's actual workload pids live. Used by services
    /// whose workload runs in a container and is therefore reparented
    /// out of the daemon's process tree, so descendant-pid attribution
    /// can't reach it. Pids whose `/proc/<pid>/cgroup` path equals this
    /// value or sits inside its subtree are summed into the service's
    /// observed peak.
    pub cgroup_parent: Option<SmolStr>,
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
    /// Extra per-GPU VRAM (MiB) to keep free when placing *this* service, added
    /// on top of the global `[devices]` reserve. Lets a single model be packed
    /// more conservatively (headroom for KV growth, a co-resident service, or
    /// estimator slack) without bypassing the estimator.
    pub gpu_headroom_mb: Option<u64>,
    /// `--split-mode` for multi-GPU llama.cpp services: `"layer"` (default),
    /// `"row"`, or `"tensor"`. Validated into [`crate::config::SplitMode`].
    pub split: Option<SmolStr>,
}

/// Raw `expert_offload` value before validation: a mode string (`"off"` /
/// `"auto"`) or an integer layer count. Validated into
/// [`crate::config::OffloadMode`].
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum RawExpertOffload {
    /// `expert_offload = N` — offload exactly N tail-most expert layers.
    Layers(u32),
    /// `expert_offload = "off" | "auto"`.
    Mode(SmolStr),
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

/// `[service.auto_restart]` block. Opt-in self-healing for a `Running`
/// service that is alive but degraded. Two independent triggers, both
/// feeding the existing drain → respawn cycle; guardrails below bound
/// the churn.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawAutoRestart {
    /// Error-rate watchdog. Enabled by default (write `error_rate = false`
    /// to opt a service out). A table tunes the thresholds.
    pub error_rate: Option<Toggle<RawErrorRateSettings>>,
    /// Periodic restart. Disabled by default; a table with an `interval`
    /// enables it (write `periodic = false` to be explicit).
    pub periodic: Option<Toggle<RawPeriodicSettings>>,
    /// Time-to-first-token stall watchdog. Enabled by default (write
    /// `ttft_stall = false` to opt a service out). A table tunes the timeout.
    pub ttft_stall: Option<Toggle<RawTtftStallSettings>>,
    /// Minimum uptime a fresh run must reach before another auto-restart
    /// may fire — the anti-flap cooldown.
    pub min_uptime: Option<String>,
    /// How many auto-restarts within [`Self::flap_window`] are tolerated
    /// before the service is disabled with `AutoRestartLoop` instead of
    /// restarted again.
    pub max_restarts: Option<u32>,
    /// Sliding window over which [`Self::max_restarts`] is counted.
    pub flap_window: Option<String>,
}

/// A trigger that accepts either a bare boolean toggle or a settings
/// table. `true` (or an empty table) means "enabled with defaults";
/// `false` means "disabled". A populated table both enables the trigger
/// and overrides individual thresholds.
///
/// Deserialized by hand rather than `#[serde(untagged)]` so the inner
/// settings struct's `deny_unknown_fields` is honoured — untagged enums
/// silently drop unknown keys, which would let a typo'd threshold pass.
#[derive(Debug, Clone)]
pub enum Toggle<T> {
    Enabled(bool),
    Settings(Box<T>),
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Toggle<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ToggleVisitor<T>(std::marker::PhantomData<T>);

        impl<'de, T: Deserialize<'de>> serde::de::Visitor<'de> for ToggleVisitor<T> {
            type Value = Toggle<T>;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a boolean or a settings table")
            }

            fn visit_bool<E>(self, v: bool) -> Result<Self::Value, E> {
                Ok(Toggle::Enabled(v))
            }

            fn visit_map<A>(self, map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let settings = T::deserialize(serde::de::value::MapAccessDeserializer::new(map))?;
                Ok(Toggle::Settings(Box::new(settings)))
            }
        }

        deserializer.deserialize_any(ToggleVisitor(std::marker::PhantomData))
    }
}

/// `[service.auto_restart.error_rate]` thresholds. Every field is optional
/// and falls back to a built-in default during validation.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawErrorRateSettings {
    /// Rolling window over which the error rate is measured.
    pub window: Option<String>,
    /// Fraction of requests in the window that must be errors to trigger
    /// (0.0–1.0).
    pub max_error_rate: Option<f64>,
    /// Minimum request count in the window before the ratio is trusted —
    /// stops a 2-of-2-failed service from restarting.
    pub min_requests: Option<u32>,
    /// How often the watchdog queries the metrics store.
    pub poll_interval: Option<String>,
    /// Which HTTP statuses count as errors: `"5xx"` (server errors only,
    /// the default) or `"4xx+5xx"` (any status ≥ 400).
    pub error_statuses: Option<SmolStr>,
}

/// `[service.auto_restart.ttft_stall]` settings. Only the timeout is tunable.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawTtftStallSettings {
    /// How long a request may stay in-flight with no response token before
    /// the service is restarted.
    pub timeout: Option<String>,
}

/// `[service.auto_restart.periodic]` settings.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawPeriodicSettings {
    /// How long a run may live before a periodic restart is due. Required
    /// when periodic is enabled.
    pub interval: Option<String>,
    /// When the interval elapses, how the restart is timed: `"immediate"`,
    /// `"on-idle"`, or `"on-request"` (the default).
    pub mode: Option<SmolStr>,
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
    fn parses_mtp_spec_keys() {
        let toml = r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
parallel = 4
spec_type = "draft-mtp"
spec_draft_n_max = 2
draft_model = "/m/mtp-draft.gguf"
kv_unified = true
cache_idle_slots = false
metrics = true
slots = true
"#;
        let cfg = parse_toml(toml, Path::new("/tmp/c.toml")).unwrap();
        let RawService::LlamaCpp(lc) = &cfg.services[0] else {
            panic!("expected LlamaCpp variant");
        };
        assert_eq!(lc.parallel, Some(4));
        assert_eq!(lc.spec_type.as_deref(), Some("draft-mtp"));
        assert_eq!(lc.spec_draft_n_max, Some(2));
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
    fn parses_auto_restart_block() {
        let toml = r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435

[service.auto_restart]
min_uptime = "10m"
max_restarts = 5
flap_window = "1h"
error_rate = { window = "2m", max_error_rate = 0.6, min_requests = 30, error_statuses = "4xx+5xx" }
periodic = { interval = "6h", mode = "on-idle" }
"#;
        let cfg = parse_toml(toml, Path::new("/tmp/c.toml")).unwrap();
        let ar = cfg.services[0].common().auto_restart.as_ref().unwrap();
        assert_eq!(ar.min_uptime.as_deref(), Some("10m"));
        assert_eq!(ar.max_restarts, Some(5));
        let Some(Toggle::Settings(er)) = &ar.error_rate else {
            panic!("expected error_rate settings table");
        };
        assert_eq!(er.window.as_deref(), Some("2m"));
        assert_eq!(er.max_error_rate, Some(0.6));
        assert_eq!(er.error_statuses.as_deref(), Some("4xx+5xx"));
        let Some(Toggle::Settings(p)) = &ar.periodic else {
            panic!("expected periodic settings table");
        };
        assert_eq!(p.interval.as_deref(), Some("6h"));
        assert_eq!(p.mode.as_deref(), Some("on-idle"));
    }

    #[test]
    fn parses_error_rate_bool_toggle() {
        let toml = r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435

[service.auto_restart]
error_rate = false
"#;
        let cfg = parse_toml(toml, Path::new("/tmp/c.toml")).unwrap();
        let ar = cfg.services[0].common().auto_restart.as_ref().unwrap();
        assert!(matches!(ar.error_rate, Some(Toggle::Enabled(false))));
    }

    #[test]
    fn parses_ttft_stall_bool_toggle() {
        let toml = r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435

[service.auto_restart]
ttft_stall = false
"#;
        let cfg = parse_toml(toml, Path::new("/tmp/c.toml")).unwrap();
        let ar = cfg.services[0].common().auto_restart.as_ref().unwrap();
        assert!(matches!(ar.ttft_stall, Some(Toggle::Enabled(false))));
    }

    #[test]
    fn ttft_stall_table_rejects_unknown_field() {
        let toml = r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435

[service.auto_restart]
ttft_stall = { timeuot = "90s" }
"#;
        let err = parse_toml(toml, Path::new("/tmp/c.toml"));
        assert!(
            err.is_err(),
            "expected parse error for typo'd ttft_stall field, got {:?}",
            err.ok()
        );
    }

    #[test]
    fn auto_restart_table_rejects_unknown_field() {
        // The custom `Toggle` deserialize must honour the settings struct's
        // deny_unknown_fields — a plain `#[serde(untagged)]` enum would
        // silently drop this typo.
        let toml = r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435

[service.auto_restart]
error_rate = { windwo = "2m" }
"#;
        let err = parse_toml(toml, Path::new("/tmp/c.toml"));
        assert!(
            err.is_err(),
            "expected parse error for typo'd error_rate field, got {:?}",
            err.ok()
        );
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
