//! Run the estimator against a GGUF and dump JSON.
//!
//! Usage:
//!   cargo run --example estimate -- \
//!     --model /path/to/model.gguf \
//!     --context 8192 \
//!     [--mmproj /path/to/mmproj.gguf] \
//!     [--cache-type-k q8_0 --cache-type-v q8_0] \
//!     [--override-tensor '<regex>=<device>' ...] \
//!     [--n-cpu-moe N] \
//!     [--compute-buffer-mb N] \
//!     [--active-devices N] \
//!     [--mtp] \
//!     [--draft-model /path/to/draft.gguf] \
//!     [--allow-fallback]
//!
//! Unknown architectures now hard-reject by default; pass `--allow-fallback`
//! to accept the coarse fallback (see `ananke::estimator::fallback`).
//!
//! The estimator is a pure function over the GGUF bytes plus a small set
//! of service-level knobs (context, cache-type, override rules, mmproj,
//! compute_buffer override). This example builds an
//! [`EstimatorInputs`] from CLI args and prints the resulting `Estimate`
//! as JSON — same code path the daemon uses at spawn time, without any
//! packer / placement / NVML involvement.
//!
//! Used by `scripts/stress/calibrate.py` to record predicted VRAM at each
//! (model, context) and compare against llama-server's real footprint.

use std::{path::PathBuf, process};

use ananke::{
    estimator::{self, EstimatorInputs},
    system::LocalFs,
};
use serde_json::json;

struct Args {
    model: PathBuf,
    mmproj: Option<PathBuf>,
    context: u32,
    ubatch: Option<u32>,
    cache_type_k: Option<String>,
    cache_type_v: Option<String>,
    override_tensor: Vec<String>,
    compute_buffer_mb: Option<u32>,
    active_devices: Option<u64>,
    allow_fallback: bool,
    mtp: bool,
    draft_model: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut it = std::env::args().skip(1);
    let mut model: Option<PathBuf> = None;
    let mut mmproj: Option<PathBuf> = None;
    let mut context: u32 = 4096;
    let mut ubatch: Option<u32> = None;
    let mut cache_type_k: Option<String> = None;
    let mut cache_type_v: Option<String> = None;
    let mut override_tensor: Vec<String> = Vec::new();
    let mut compute_buffer_mb: Option<u32> = None;
    let mut active_devices: Option<u64> = None;
    let mut allow_fallback = false;
    let mut mtp = false;
    let mut draft_model: Option<PathBuf> = None;
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model = it.next().map(PathBuf::from),
            "--mmproj" => mmproj = it.next().map(PathBuf::from),
            "--context" => {
                context = it.next().and_then(|s| s.parse().ok()).unwrap_or(context);
            }
            "--ubatch" | "--ubatch-size" => ubatch = it.next().and_then(|s| s.parse().ok()),
            "--cache-type-k" => cache_type_k = it.next(),
            "--cache-type-v" => cache_type_v = it.next(),
            "--override-tensor" => {
                if let Some(rule) = it.next() {
                    override_tensor.push(rule);
                }
            }
            "--compute-buffer-mb" => compute_buffer_mb = it.next().and_then(|s| s.parse().ok()),
            "--active-devices" => active_devices = it.next().and_then(|s| s.parse().ok()),
            "--allow-fallback" => allow_fallback = true,
            "--mtp" => mtp = true,
            "--draft-model" => draft_model = it.next().map(PathBuf::from),
            _ => {
                eprintln!("unknown argument: {arg}");
                process::exit(2);
            }
        }
    }
    let Some(model) = model else {
        eprintln!("--model is required");
        process::exit(2);
    };
    Args {
        model,
        mmproj,
        context,
        ubatch,
        cache_type_k,
        cache_type_v,
        override_tensor,
        compute_buffer_mb,
        active_devices,
        allow_fallback,
        // A separate draft model only contributes overhead under MTP, so
        // `--draft-model` implies `--mtp` for convenience.
        mtp: mtp || draft_model.is_some(),
        draft_model,
    }
}

fn main() {
    let args = parse_args();
    let inputs = EstimatorInputs {
        name: "estimate-example",
        model: args.model.as_path(),
        mmproj: args.mmproj.as_deref(),
        context: args.context,
        ubatch: args.ubatch,
        cache_type_k: args.cache_type_k.as_deref(),
        cache_type_v: args.cache_type_v.as_deref(),
        override_tensor: &args.override_tensor,
        compute_buffer_mb: args.compute_buffer_mb,
        allow_fallback: args.allow_fallback,
        mtp: args.mtp,
        draft_model: args.draft_model.as_deref(),
    };

    let estimate = match estimator::estimate_from_path(&LocalFs, &inputs) {
        Ok(e) => e,
        Err(e) => {
            println!("{}", json!({"estimator_error": e.to_string()}));
            process::exit(1);
        }
    };

    let kv_total_bytes = estimate
        .kv_per_token
        .saturating_mul(estimate.context as u64);

    // The packer adds `compute_buffer_mb` to every device it lands the
    // model on. Caller passes `--active-devices N` for the placement
    // they're modelling (1 for a single-GPU-only fit, 2 for dual-GPU,
    // 3 for dual-GPU + CPU embedding/offload). Default 3 matches the
    // common "two GPUs plus CPU-resident embeddings" layout for large
    // llama-family models.
    let active_devices = args.active_devices.unwrap_or(3);
    let cb_total_bytes = (estimate.compute_buffer_mb as u64)
        .saturating_mul(active_devices)
        .saturating_mul(1024 * 1024);
    let total_bytes = estimate
        .weights_bytes
        .saturating_add(kv_total_bytes)
        .saturating_add(cb_total_bytes)
        .saturating_add(estimate.mtp_bytes);

    // "GPU VRAM" estimate: subtract the tensors llama.cpp keeps on CPU
    // by default (token embeddings — plus the per-layer token embeddings
    // for gemma4 E-variants). Expert offload is a placement-time decision
    // the packer makes from live VRAM, so this estimator-only view counts
    // experts on the GPU; run the daemon to see the offloaded placement.
    let cpu_resident_bytes = estimate.non_layer.token_embd_bytes;
    let gpu_weights_bytes = estimate.weights_bytes.saturating_sub(cpu_resident_bytes);
    let gpu_total_bytes = gpu_weights_bytes
        .saturating_add(kv_total_bytes)
        .saturating_add(
            (estimate.compute_buffer_mb as u64)
                .saturating_mul(active_devices.min(2))
                .saturating_mul(1024 * 1024),
        )
        .saturating_add(estimate.mtp_bytes);

    let out = json!({
        "architecture": estimate.architecture.as_str(),
        "context": estimate.context,
        "weights_bytes": estimate.weights_bytes,
        "weights_gib": estimate.weights_bytes as f64 / 1024.0_f64.powi(3),
        "kv_per_token_bytes": estimate.kv_per_token,
        "kv_total_bytes": kv_total_bytes,
        "kv_total_mib": kv_total_bytes / (1024 * 1024),
        "compute_buffer_mb": estimate.compute_buffer_mb,
        "mtp_bytes": estimate.mtp_bytes,
        "mtp_mib": estimate.mtp_bytes / (1024 * 1024),
        "per_layer_count": estimate.per_layer_bytes.as_ref().map(|v| v.len()),
        "non_layer_output_head_bytes": estimate.non_layer.output_head_bytes,
        "non_layer_token_embd_bytes": estimate.non_layer.token_embd_bytes,
        "non_layer_other_bytes": estimate.non_layer.other_bytes,
        "expert_layer_count": estimate.expert_layers.len(),
        "expert_tensor_count": estimate.expert_tensors.as_ref().map(|v| v.len()),
        "expert_total_bytes": estimate
            .expert_tensors
            .as_ref()
            .map(|v| v.iter().map(|e| e.bytes).sum::<u64>()),
        "override_tensor_bytes": estimate.override_tensor_bytes
            .iter()
            .map(|(k, v)| (format!("{k:?}"), serde_json::Value::from(*v)))
            .collect::<serde_json::Map<_, _>>(),
        // Rough "sum of accounted bytes" number — weights + kv(total) +
        // compute buffer. Doesn't include the packer's tensor-split fudge
        // or per-device compute-buffer doubling; treat as a lower bound.
        "total_accounted_bytes": total_bytes,
        "total_accounted_mib": total_bytes / (1024 * 1024),
        // GPU-only estimate: what nvidia-smi will report summed across
        // GPUs. Excludes llama.cpp's CPU-resident embedding tensors and
        // caps cb device count at 2 (CPU doesn't contribute to GPU VRAM).
        "gpu_vram_bytes": gpu_total_bytes,
        "gpu_vram_mib": gpu_total_bytes / (1024 * 1024),
    });

    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
