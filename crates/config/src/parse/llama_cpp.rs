//! The `llama-cpp` template variant: `RawLlamaCppService` and its nested
//! runtime, expert-offload, estimation, and sampling tables.

use std::path::PathBuf;

use serde::Deserialize;
use smol_str::SmolStr;

use crate::parse::RawServiceCommon;

/// A raw `llama-cpp`-template service: model, runtime, and serving knobs.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawLlamaCppService {
    /// Fields shared with every template variant, flattened at top level.
    #[serde(flatten)]
    pub common: RawServiceCommon,
    /// Serving runtime, as a tagged table:
    /// `runtime = { kind = "ik-llama", mla = 1, dsa = true, ... }`.
    /// Absent means mainline llama.cpp. Runtime-specific options live
    /// inside the table, so fork-only knobs are *unrepresentable* on the
    /// mainline runtime rather than merely validated away. Fork-only
    /// behaviour (flag dialect, estimator shapes) keys
    /// off the variant. Point `llama_server` at a matching binary — the
    /// validator can't check that, but each runtime rejects the other's
    /// flags at spawn.
    pub runtime: Option<RawRuntime>,
    /// Path to the model GGUF.
    pub model: Option<PathBuf>,
    /// Path to the multimodal projector GGUF, for vision models.
    pub mmproj: Option<PathBuf>,
    /// Context window size in tokens.
    pub context: Option<u32>,
    /// Number of GPU layers (negative offloads the last layers to CPU).
    pub n_gpu_layers: Option<i32>,
    /// MoE expert-offload policy: `"off"` (no expert offload — whole-layer CPU
    /// spill only), `"auto"` (the packer offloads the minimum experts to fit
    /// live VRAM), or an integer `N` (offload the experts of the N tail-most
    /// expert layers). Validated into [`crate::OffloadMode`]. Requires
    /// a CPU-allowing placement (`placement = "hybrid"`) when not `"off"`.
    pub expert_offload: Option<RawExpertOffload>,
    /// Whether to use flash attention.
    pub flash_attn: Option<bool>,
    /// KV cache quantization format for the K tensors.
    pub cache_type_k: Option<SmolStr>,
    /// KV cache quantization format for the V tensors.
    pub cache_type_v: Option<SmolStr>,
    /// Whether to memory-map the model file.
    pub mmap: Option<bool>,
    /// Whether to lock the model in RAM.
    pub mlock: Option<bool>,
    /// Number of parallel decoding slots.
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
    /// `ananke_estimate::mtp`.
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
    /// Host RAM cap for the server's prompt cache (`-cram`, MiB), which
    /// holds serialized evicted prompts so a returning conversation skips
    /// reprocessing. Unset means llama.cpp's 8192 MiB default. The cap is
    /// always passed through explicitly so the reservation and the runtime
    /// agree on the same number; `0` disables the cache.
    pub cache_ram_mb: Option<u32>,
    /// Expose the Prometheus `/metrics` endpoint (`--metrics`).
    pub metrics: Option<bool>,
    /// Expose the `/slots` introspection endpoint (`--slots`).
    pub slots: Option<bool>,
    /// Context batch size (`-b`).
    pub batch_size: Option<u32>,
    /// Physical batch size (`-ub`).
    pub ubatch_size: Option<u32>,
    /// CPU threads for prompt processing.
    pub threads: Option<u32>,
    /// CPU threads for generation.
    pub threads_batch: Option<u32>,
    /// NUMA thread-and-memory placement strategy passed to llama-server's
    /// `--numa`: `"distribute"` (spread threads and interleave memory across
    /// all nodes), `"isolate"` (pin to one node), or `"numactl"` (defer to
    /// an external `numactl` mask). Validated into
    /// [`crate::NumaStrategy`]. Unset leaves llama-server's default
    /// (no `--numa` flag).
    pub numa: Option<SmolStr>,
    /// Whether to use Jinja chat templating.
    pub jinja: Option<bool>,
    /// Custom chat-template file override (`--chat-template-file`).
    pub chat_template_file: Option<PathBuf>,
    /// Per-tensor overrides of the model's tensor layout.
    pub override_tensor: Option<Vec<String>>,
    /// Sampling knobs forwarded as llama-server CLI flags.
    pub sampling: Option<SamplingConfig>,
    /// Estimator overrides (compute-buffer headroom, safety factor).
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

/// The `runtime` table of a llama-cpp-template service, tagged by
/// `kind`. Each variant carries only the options that runtime actually
/// has; see [`RawLlamaCppService::runtime`].
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum RawRuntime {
    /// Mainline llama.cpp — the implicit default; carries no options
    /// beyond what the service struct already exposes.
    LlamaCpp,
    /// ikawrakow's ik_llama.cpp fork.
    IkLlama(RawIkSettings),
}

/// ik_llama.cpp-specific knobs (the `runtime` table's fields when
/// `kind = "ik-llama"`).
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawIkSettings {
    /// MLA kernel mode (`-mla`, 0-3). Mode 1 is the calibrated
    /// recommendation alongside `dsa = true` — mode 3 gains ~8% shallow
    /// prefill but collapses DSA's deep-prefill advantage (measured
    /// 143 → 61 t/s at 58k depth on GLM-5.2).
    pub mla: Option<u32>,
    /// Enable DSA sparse attention (`-dsa -fidx`). Requires f16 KV —
    /// the fork rejects quantised cache types alongside it.
    pub dsa: Option<bool>,
    /// Attention scratch cap in MiB (`-amb`).
    pub attn_max_batch: Option<u32>,
    /// Repack quants for CPU at load (`-rtr`). Adds load time;
    /// unnecessary for the already-CPU-fast KS quants.
    pub runtime_repack: Option<bool>,
}

/// Raw `expert_offload` value before validation: a mode string (`"off"` /
/// `"auto"`) or an integer layer count. Validated into
/// [`crate::OffloadMode`].
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
    /// Extra compute-buffer headroom (MiB) added to the estimator's own.
    pub compute_buffer_mb: Option<u32>,
    /// Multiplier applied to the estimator's VRAM prediction.
    pub safety_factor: Option<f32>,
}

/// Sampling parameters that map to `llama-server` CLI flags. Only the knobs
/// we actually forward are accepted; unknown keys surface as parse errors
/// rather than silently being dropped. Shared between parse and validate
/// layers — validation is a no-op for this type.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct SamplingConfig {
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus-sampling probability mass (`--top-p`).
    pub top_p: Option<f32>,
    /// Top-k candidate cutoff (`--top-k`).
    pub top_k: Option<u32>,
    /// Minimum token probability (`--min-p`).
    pub min_p: Option<f32>,
    /// Repetition penalty.
    pub repeat_penalty: Option<f32>,
}

#[cfg(test)]
mod tests {

    use crate::parse::{RawService, parse_toml};

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
        let cfg = parse_toml(toml).unwrap();
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
}
