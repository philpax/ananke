//! `GET /api/services/{name}` — service detail.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    internal::fit_verdict::FitVerdict,
    shared::{
        lifecycle::Lifecycle,
        metadata::AnankeMetadata,
        modality::Modality,
        service_state::{DisableReason, ServiceStateKind},
    },
};

/// `GET /api/services/{name}` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ServiceDetail {
    /// Service name.
    pub name: String,
    /// Current state.
    pub state: ServiceStateKind,
    /// Why the service is disabled; present exactly when `state` is
    /// `disabled`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disabled_reason: Option<DisableReason>,
    /// When the service starts and stops.
    pub lifecycle: Lifecycle,
    /// Eviction priority.
    pub priority: u8,
    /// Public port.
    pub port: u16,
    /// Private (upstream) port.
    pub private_port: u16,
    /// Template name, e.g. `"llamacpp"` or `"command"`.
    pub template: String,
    /// Free-form description from the service's config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Whether the service is reachable through the OpenAI-compatible API,
    /// and so can be chatted with or embedded against.
    pub openai_compat: bool,
    /// The validated placement constraints the packer works under.
    pub placement: PlacementConfig,
    /// Command-template settings. `None` for llama-cpp services.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<CommandDetail>,
    /// Manual placement override, keyed by device slot string.
    pub placement_override: BTreeMap<String, u64>,
    /// Idle-before-drain timeout.
    pub idle_timeout_ms: u64,
    /// Active run id if any.
    pub run_id: Option<i64>,
    /// Child PID if any.
    pub pid: Option<i32>,
    /// Rolling estimator correction factor for the service's VRAM footprint.
    /// `None` until the first observation lands.
    pub rolling_mean: Option<f32>,
    /// Sample count backing [`Self::rolling_mean`].
    pub rolling_samples: u64,
    /// Rolling estimator correction factor for the service's host-RAM
    /// footprint, learned independently of the VRAM one. `None` for a service
    /// that holds nothing in host RAM, which never accumulates a sample.
    pub rolling_mean_host: Option<f32>,
    /// Sample count backing [`Self::rolling_mean_host`].
    pub rolling_samples_host: u64,
    /// Observed memory peak (VRAM + RSS) across the service's lifetime.
    pub observed_peak_bytes: u64,
    /// Placeholder for elastic-borrower tracking.
    pub elastic_borrower: Option<String>,
    /// GGUF-derived facts about the model file. `None` for command-
    /// template services and for llama-cpp services whose GGUF couldn't
    /// be read (file missing, unrecognised, etc.). Cached in the daemon
    /// after the first successful read so repeated detail polls don't
    /// re-parse the file.
    pub model_info: Option<ModelInfo>,
    /// Memory estimate computed against the service's configured context
    /// and KV cache settings. Same caching rules as `model_info`.
    pub estimate: Option<EstimateSummary>,
    /// Per-device placement the service would take under current conditions,
    /// plus whether it fits without eviction. `None` for command-template
    /// services, services with a manual `placement_override`, and llama-cpp
    /// services whose GGUF couldn't be read.
    pub placement_preview: Option<PlacementPreview>,
    /// What pledge the service is currently holding on each device.
    /// Empty when idle. Keys are slot strings (`"cpu"`, `"gpu:0"`, …),
    /// values are MiB.
    pub current_allocation: BTreeMap<String, u64>,
    /// What kind of OpenAI endpoint the service serves. See
    /// [`crate::services::list::ServiceSummary::modality`] for the rendering rule.
    #[serde(default, skip_serializing_if = "Modality::is_chat")]
    pub modality: Modality,
    /// Passthrough entries from `[[service]] metadata.*`. See
    /// [`crate::services::list::ServiceSummary::ananke_metadata`].
    #[serde(default, skip_serializing_if = "AnankeMetadata::is_empty")]
    #[schema(value_type = Object)]
    pub ananke_metadata: AnankeMetadata,
    /// Wall-clock timestamp (ms since epoch) of the last time the
    /// service was provisioned or received a request. `None` if the
    /// service has never been started.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_used_ms: Option<i64>,
    /// Recent auto-restart watchdog firings, most recent first. Sourced
    /// from the daemon's durable `service_restarts` store, so firings are
    /// visible after the fact — not only to a client that was connected when it happened.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recent_restarts: Vec<RestartEvent>,
    /// Serving-runtime details for a llama-cpp service. `None` for
    /// command-template services.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<RuntimeInfo>,
    /// Curated serving knobs for a llama-cpp service. `None` for
    /// command-template services.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serving: Option<ServingConfig>,
    /// Container identity for a containerized service. `None` for native
    /// host processes. Kept beside the llama-cpp `runtime`/`serving` cards
    /// so containerized llama.cpp retains its engine semantics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<ContainerDetail>,
}

/// The placement constraints a service's config sets, as validated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct PlacementConfig {
    /// Which device classes the service may land on.
    pub policy: PlacementPolicy,
    /// GPU indices the service may use; empty means every GPU.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gpu_allow: Vec<u32>,
    /// VRAM this service keeps free on each GPU it uses, on top of the global
    /// reserves.
    pub gpu_headroom_bytes: u64,
    /// How a multi-GPU llama-cpp service splits across cards. `None` for
    /// command services, where the split mode means nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split: Option<SplitConfig>,
    /// How the allocator sizes the service's reservation.
    pub allocation: AllocationConfig,
}

/// Which device classes a service may be placed on, spelled as in config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "kebab-case")]
pub enum PlacementPolicy {
    /// GPUs only; a model that does not fit fails placement.
    GpuOnly,
    /// Host memory only.
    CpuOnly,
    /// GPUs first, spilling to the host.
    Hybrid,
}

/// A llama-cpp service's inter-GPU split.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct SplitConfig {
    /// The split strategy.
    pub mode: SplitMode,
    /// Per-GPU `--tensor-split` weights in ascending GPU order, when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor_split_weights: Option<Vec<f32>>,
}

/// How llama.cpp spreads a model across GPUs, spelled as in config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum SplitMode {
    /// Whole layers pipelined across the cards.
    Layer,
    /// Every layer's weights split by row across the cards.
    Row,
    /// Every layer's weights split across the cards as one fused device.
    Tensor,
}

/// How the allocator sizes a service's reservation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AllocationConfig {
    /// Sized by the estimator (llama-cpp services).
    Estimated,
    /// A fixed reservation.
    Static {
        /// The reservation, in bytes.
        reserve_bytes: u64,
    },
    /// A balloon that grows and shrinks between two bounds with observed use.
    Dynamic {
        /// The lower bound, in bytes.
        min_bytes: u64,
        /// The upper bound, in bytes.
        max_bytes: u64,
        /// How long a borrower must run before its memory may be reclaimed.
        min_borrower_runtime_ms: u64,
    },
}

/// A command-template service's settings that its launch command does not
/// show.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct CommandDetail {
    /// The model name rewritten into proxied OpenAI requests, when the
    /// service has an `openai_proxy` block.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_model: Option<String>,
    /// The command run to stop the workload, when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shutdown_command: Option<Vec<String>>,
    /// `true` when the config fixes the private port; `false` when it comes
    /// from the daemon's pool.
    pub private_port_fixed: bool,
    /// The working directory the command runs in, when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workdir: Option<String>,
}

/// Container identity for a service running under Docker/Podman.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct ContainerDetail {
    /// Container runtime (`docker` or `podman`).
    pub runtime: String,
    /// Container image reference.
    pub image: String,
    /// Network mode (`bridge` or `host`).
    pub network: String,
    /// Live container ID, present only while a container is running.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container_id: Option<String>,
    /// Live container name, present only while a container is running.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container_name: Option<String>,
}

/// One persisted auto-restart watchdog firing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct RestartEvent {
    /// Wall-clock timestamp of the firing (ms since epoch).
    pub at_ms: i64,
    /// Which watchdog fired: `"error_rate"`, `"ttft_stall"`,
    /// `"generation_stall"`, `"spec_collapse"`, or `"periodic"`.
    pub trigger: String,
    /// Human-readable reason (e.g. the observed error rate and window).
    pub detail: String,
    /// The run that was drained by the firing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<i64>,
}

/// Curated serving configuration for a llama-cpp service: the knobs an
/// operator reaches for when comparing services or debugging
/// performance, plus derived values (per-slot context) that no single
/// config key or argv flag states directly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct ServingConfig {
    /// The llama-server executable path serving this service. With
    /// multiple runtimes installed (mainline + ik_llama), which binary
    /// a service uses is a first-class fact.
    pub binary: String,
    /// KV cache type for keys (`f16` when unset).
    pub cache_type_k: String,
    /// KV cache type for values (`f16` when unset).
    pub cache_type_v: String,
    /// Flash attention enabled.
    pub flash_attn: bool,
    /// Request parallelism (`-np`).
    pub parallel: u32,
    /// Layers offloaded to the GPU (`-ngl`), when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_gpu_layers: Option<i32>,
    /// The configured context window, before any default applies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<u32>,
    /// The prompt cache's host RAM cap (`-cram`), resolved the way the
    /// launch command resolves it. Zero when the cache is disabled.
    pub cache_ram_bytes: u64,
    /// Unified KV pool across slots (`--kv-unified`).
    pub kv_unified: bool,
    /// The context window a single request actually gets: `context`
    /// divided by `parallel` when the KV pool is statically split,
    /// or the full window when unified (slots borrow from each other).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_context_per_slot: Option<u32>,
    /// `--spec-type` value, when speculative decoding is on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec_type: Option<String>,
    /// Draft model file name (not full path), when a separate draft
    /// GGUF is configured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_model: Option<String>,
    /// MoE expert-offload mode: `"off"`, `"auto"`, or a layer count.
    pub expert_offload: String,
    /// Logical batch size (`-b`), when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<u32>,
    /// Physical batch size (`-ub`), when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ubatch_size: Option<u32>,
    /// Generation threads (`-t`), when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads: Option<u32>,
    /// Prompt-processing threads (`-tb`), when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads_batch: Option<u32>,
    /// `--numa` strategy, when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numa: Option<String>,
    /// Model file memory-mapped (llama.cpp's default is `true`).
    pub mmap: bool,
    /// Model locked in RAM (`--mlock`).
    pub mlock: bool,
}

/// Which llama-server implementation serves a llama-cpp service, plus
/// the fork-specific configuration when it is not mainline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct RuntimeInfo {
    /// `"llama-cpp"` (mainline) or `"ik-llama"` (ikawrakow fork).
    pub kind: String,
    /// ik_llama.cpp parameters; present only when `kind == "ik-llama"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ik: Option<IkParams>,
}

/// The ik_llama.cpp knobs of a `runtime = { kind = "ik-llama", ... }`
/// service, mirrored from the validated config.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct IkParams {
    /// MLA kernel mode (`-mla`, 0-3), when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mla: Option<u32>,
    /// DSA sparse attention (`-dsa -fidx`).
    pub dsa: bool,
    /// `-amb` attention scratch cap in MiB, when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attn_max_batch: Option<u32>,
    /// `-rtr` runtime repacking.
    pub runtime_repack: bool,
}

/// GGUF-derived facts about a model file. Read once per service per
/// daemon run and cached; the file isn't re-parsed on every detail
/// poll.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ModelInfo {
    /// `general.architecture` from the GGUF (`"qwen2"`, `"llama"`,
    /// `"qwen3moe"`, …).
    pub architecture: String,
    /// Friendly name declared in `general.name`, when the publisher
    /// set one. Typically the upstream model identifier ("Qwen3 32B",
    /// "Llama 3.3 70B Instruct", …) rather than ananke's service
    /// name. Absent for hand-built GGUFs that left the field blank.
    pub model_name: Option<String>,
    /// `general.license` if declared (`"apache-2.0"`, `"llama3.3"`,
    /// `"gemma"`, …). Surfaced so operators can see the licence
    /// without leaving the dashboard.
    pub license: Option<String>,
    /// `general.parameter_count` if declared by the publisher. Modern
    /// converter tools (Unsloth, bartowski's repacks, …) write this;
    /// older GGUFs may not. The field is the model's *parameter*
    /// count, not the on-disk byte size — quantisation makes the two
    /// diverge by a large factor.
    pub parameter_count: Option<u64>,
    /// Sum of every tensor's byte size across every shard. The
    /// reference number when sizing the model on disk.
    pub total_tensor_bytes: u64,
    /// Layer count exposed by the architecture's `{arch}.block_count`
    /// metadata key, when present.
    pub block_count: Option<u32>,
    /// Number of shards discovered. 1 for a single-file GGUF.
    pub shard_count: u32,
    /// The model's *trained* context window, read from
    /// `{arch}.context_length`. The service's configured context may
    /// be lower.
    pub trained_context_length: Option<u32>,
    /// File name (basename of shard 0). Useful for the UI to show the
    /// underlying filename without the full path.
    pub file_name: String,
    /// `true` when the service has a configured `mmproj` GGUF — the
    /// standard signal for vision / multimodal support.
    pub has_mmproj: bool,
}

/// Estimator output projected to the wire. Carries the components a
/// reader needs to answer "how much memory will this service take?"
/// without having to re-derive any of them client-side.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct EstimateSummary {
    /// Static weight bytes (including mmproj if configured).
    pub weights_bytes: u64,
    /// KV cache bytes per context token. Zero for SSM/Mamba
    /// architectures with no KV cache.
    pub kv_per_token: u64,
    /// Configured context window the estimate is sized against.
    pub configured_context: u32,
    /// `kv_per_token × configured_context`. Precomputed so the
    /// frontend can render the total directly.
    pub kv_bytes_for_context: u64,
    /// Compute-buffer reservation per active device, in bytes.
    pub compute_buffer_bytes_per_device: u64,
}

/// Where a service's memory would land per device under current conditions, and
/// whether it fits without the daemon having to evict or reclaim. Computed by
/// running the placement engine against the live snapshot and pledge book.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct PlacementPreview {
    /// Per-device memory the service would occupy, sorted by device. Keys are
    /// slot strings (`"cpu"`, `"gpu:0"`, …); values are bytes.
    pub devices: Vec<DevicePlacement>,
    /// Whether the placement fits right now without eviction or reclaim.
    pub verdict: FitVerdict,
    /// Total expert-tensor bytes the packer offloaded to the CPU for a MoE
    /// model (auto/manual `expert_offload`). Zero for non-MoE services or when
    /// nothing was offloaded.
    pub expert_offload_bytes: u64,
    /// Number of distinct layers with at least one expert offloaded to the CPU.
    pub expert_offload_layers: u32,
}

/// One device's share of a [`PlacementPreview`], with enough context to draw a
/// utilisation bar: this service's share, what is already in use by everything
/// else, and the device's total capacity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct DevicePlacement {
    /// Slot string: `"cpu"` or `"gpu:N"`.
    pub device: String,
    /// Memory bytes this service reserves on the device — the pledge floor for a
    /// dynamic service.
    pub bytes: u64,
    /// Upper bound this service could grow to on the device. Equals `bytes`
    /// for fixed-size services; larger for a dynamic command service that may
    /// borrow up to its configured maximum.
    pub max_bytes: u64,
    /// Memory bytes already in use on the device by everything except this
    /// service (for a running service, its own resident memory is excluded so
    /// `used_by_others_bytes + bytes` doesn't double-count it).
    pub used_by_others_bytes: u64,
    /// Total memory capacity of the device, in bytes. Zero if unknown.
    pub total_bytes: u64,
}
