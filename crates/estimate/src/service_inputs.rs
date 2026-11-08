//! Distil the estimator-relevant fields out of a [`ananke_config::ServiceConfig`].
//!
//! The `extra_args` readers live here rather than beside the estimator proper:
//! they parse llama.cpp's own command line the way the runtime does — last flag
//! wins, and `extra_args` is appended after everything the daemon generates —
//! which is a property of the configuration, not of the estimate.

use ananke_config::{PlacementPolicy, ServiceConfig, validate::OffloadMode};
use ananke_gguf::GgufType;
use tracing::warn;

use crate::{EstimatorInputs, Fork, Speculation};

/// Distil the estimator-relevant fields out of a `ServiceConfig`.
///
/// A free function rather than an `EstimatorInputs` method because the direction
/// of the dependency matters: `EstimatorInputs` is a pure input struct that
/// standalone callers — the calibration tools, the `estimate` example — build
/// directly, and it should not have to know what a `ServiceConfig` is. Reading
/// one is the daemon's business.
///
/// Returns `None` if `svc` is a command-template service — the estimator only
/// applies to llama-cpp workloads.
pub fn estimator_inputs(svc: &ServiceConfig) -> Option<EstimatorInputs<'_>> {
    let lc = svc.llama_cpp()?;
    Some(EstimatorInputs {
        name: svc.name.as_str(),
        model: lc.model.as_path(),
        mmproj: lc.mmproj.as_deref(),
        context: lc.context.unwrap_or(4096),
        ubatch: extra_arg_value(&svc.extra_args, &["-ub", "--ubatch-size", "--ubatch_size"])
            .and_then(|v| v.parse().ok())
            .or(lc.ubatch_size),
        // One device unless a caller that can see the machine says
        // otherwise: under-stating this under-reserves by ~20 MiB a card,
        // which the rolling correction can absorb, where over-stating it
        // on a single-card host could not be recovered.
        visible_devices: 1,
        // Any expert offload at all puts the model in the hybrid regime.
        host_resident_experts: !matches!(lc.expert_offload, OffloadMode::Off),
        split_mode: svc.split_mode,
        cache_type_k: cache_type(&svc.name, "cache_type_k", lc.cache_type_k.as_deref()),
        cache_type_v: cache_type(&svc.name, "cache_type_v", lc.cache_type_v.as_deref()),
        override_tensor: &lc.override_tensor,
        compute_buffer_mb: lc.estimation.compute_buffer_mb,
        // Mainline's spelling only. ik's `mtp:n_max=…` reads as no speculation,
        // so an ik draft context costs nothing in the estimate. See
        // `ananke_estimate::mtp`.
        //
        // A separate draft GGUF costs something under any `--spec-type` —
        // `draft-mtp`, `draft-dflash`, whatever comes next — so it is not
        // gated on which mechanism, only on being set at all; only an
        // *embedded* head (no `-md`) is mtp-specific, since every other
        // mechanism seen so far requires a separate draft context.
        speculation: match (lc.spec_type.as_deref(), lc.draft_model.as_deref()) {
            (Some(spec_type), Some(draft)) => Speculation::SeparateDraft {
                path: draft,
                spec_type,
            },
            (Some("draft-mtp"), None) => Speculation::EmbeddedMtp,
            _ => Speculation::None,
        },
        fork: match lc.runtime.ik() {
            Some(ik) => Fork::Ik { dsa: ik.dsa },
            None => Fork::Mainline,
        },
        parallel: extra_arg_value(&svc.extra_args, &["-np", "--parallel"])
            .and_then(|v| v.parse().ok())
            .or(lc.parallel),
        // Unset means llama.cpp's `-fa auto`, which resolves to *on* for
        // CUDA — the only backend ananke supports — and off when there is
        // no device to support it. Assuming off for a GPU service doubles
        // the modelled mask and puts the prediction outside the rolling
        // correction's reach; assuming on for a CPU-only one would halve
        // it. Resolve the same way the runtime will.
        flash_attn: flash_attn_from_extra_args(&svc.extra_args)
            .or(lc.flash_attn)
            .or(Some(!matches!(
                svc.placement_policy,
                PlacementPolicy::CpuOnly
            ))),
        kv_unified: kv_unified_from_extra_args(&svc.extra_args).or(lc.kv_unified),
        // An operator who set the flag by hand in `extra_args` gets that
        // value in the estimate too, so the reservation matches the
        // runtime rather than the default the flag overrode. The flag wins
        // over `cache_ram_mb` because the launch command omits the config
        // value when the flag is present.
        cache_ram_mb: cache_ram_from_extra_args(&svc.extra_args).or(lc.cache_ram_mb),
    })
}

/// A configured `cache_type_*` as the ggml type it names.
///
/// The config key is free-form: it is passed to llama.cpp verbatim, and the two
/// forks accept different sets, so an unrecognised name is not a config error
/// here. The child will refuse it at startup. Until then the estimate warns and
/// falls back to f16.
fn cache_type(service: &str, key: &str, configured: Option<&str>) -> Option<GgufType> {
    let configured = configured?;
    let ty = GgufType::from_name(configured);
    if ty.is_none() {
        warn!(
            service,
            key,
            value = configured,
            "unknown KV cache type; estimating as f16"
        );
    }
    ty
}

/// Names llama.cpp accepts for the prompt-cache cap. `--` arguments have
/// underscores normalised to dashes, so `--cache_ram` is valid too.
const CACHE_RAM_FLAGS: &[&str] = &["--cache-ram", "-cram", "--cache_ram"];

/// The value an operator passed for one of `names` in `extra_args`.
///
/// Returns the **last** occurrence, because that is what llama.cpp uses: a
/// repeated argument logs "only last value will be used"
/// (`common/arg.cpp`). `extra_args` is appended after every flag the daemon
/// renders, so whatever is found here is what the child will actually run
/// with — and therefore what the estimate has to be built from.
///
/// The `--flag=value` form is deliberately not handled: llama.cpp requires an
/// exact whole-token match and exits with `invalid argument`, so it fails
/// loudly rather than diverging silently.
pub fn extra_arg_value<'a>(extra_args: &'a [String], names: &[&str]) -> Option<&'a str> {
    let mut found = None;
    let mut it = extra_args.iter().peekable();
    while let Some(arg) = it.next() {
        if names.contains(&arg.as_str())
            && let Some(v) = it.peek()
        {
            found = Some(v.as_str());
        }
    }
    found
}

/// Find a `--cache-ram` / `-cram` value an operator passed through
/// `extra_args`. Configs in the wild set it this way rather than through the
/// dedicated key; without this the daemon reserves the 8 GiB default for a
/// service that runs with the cache switched off.
pub fn cache_ram_from_extra_args(extra_args: &[String]) -> Option<u32> {
    extra_arg_value(extra_args, CACHE_RAM_FLAGS).and_then(|v| v.parse().ok())
}

/// `-kvu` / `--kv-unified` (a bare flag) or `-no-kvu` / `--no-kv-unified`
/// from `extra_args`.
fn kv_unified_from_extra_args(extra_args: &[String]) -> Option<bool> {
    extra_args.iter().rev().find_map(|a| match a.as_str() {
        "-kvu" | "--kv-unified" => Some(true),
        "-no-kvu" | "--no-kv-unified" => Some(false),
        _ => None,
    })
}

/// `-fa` / `--flash-attn`, which takes `on`/`off`/`auto`/`1`/`0`, or the bare
/// `-no-fa`. `auto` yields `None` so the caller falls through to resolving it
/// the way the runtime will.
fn flash_attn_from_extra_args(extra_args: &[String]) -> Option<bool> {
    if extra_args
        .iter()
        .any(|a| a == "-no-fa" || a == "--no-flash-attn")
    {
        return Some(false);
    }
    match extra_arg_value(extra_args, &["-fa", "--flash-attn"])? {
        "on" | "1" | "true" => Some(true),
        "off" | "0" | "false" => Some(false),
        _ => None,
    }
}
