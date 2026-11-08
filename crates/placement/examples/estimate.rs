//! Run the estimator against a GGUF and dump JSON.
//!
//! ```text
//! cargo run -p ananke-placement --example estimate -- --model model.gguf --context 8192
//! cargo run -p ananke-placement --example estimate -- --help
//! ```
//!
//! `--pack` runs the packer against the supplied device capacities and reports
//! the per-device allocation alongside the raw estimate. Without `--pack`, only
//! the estimator output is printed.
//!
//! `--n-cpu-moe N` forces N expert layers to CPU (`expert_offload = N`),
//! matching llama-server's `--n-cpu-moe` flag. `--host-resident-experts` is
//! shorthand for `expert_offload = auto`, which spills only the surplus that
//! does not fit on the GPUs.
//!
//! An architecture no family estimator covers is refused: there is nothing here
//! that can price its KV cache or its graph.

use std::{collections::BTreeMap, path::PathBuf, process};

use ananke_config::{
    placement::{OffloadMode, PlacementInputs, PlacementPolicy, SplitMode},
    units::{GIB_F64, MIB},
};
use ananke_estimate::{self as estimator, Estimate, EstimatorInputs, Fork, Speculation};
use ananke_fs::LocalFs;
use ananke_gguf::GgufType;
use ananke_placement::{
    Corrections, Packed,
    devices::{CpuSnapshot, DeviceSnapshot, GpuSnapshot},
    pack_demand,
};
use clap::Parser;
use serde::Serialize;

/// The name every report is filed under. Only reaches log lines.
const SERVICE: &str = "estimate-example";

#[derive(Parser)]
#[command(about = "Estimate a GGUF's memory footprint, and optionally pack it")]
struct Args {
    /// Path to the first GGUF shard.
    #[arg(long)]
    model: PathBuf,
    /// Vision projector to load alongside the model.
    #[arg(long)]
    mmproj: Option<PathBuf>,
    /// Context window the child would be launched with (`-c`).
    #[arg(long, default_value_t = 4096)]
    context: u32,
    /// Physical batch size (`-ub`).
    #[arg(long, alias = "ubatch-size")]
    ubatch: Option<u32>,
    /// K-cache quantisation, by ggml type name (`f16`, `q8_0`, …).
    #[arg(long, value_parser = parse_cache_type)]
    cache_type_k: Option<GgufType>,
    /// V-cache quantisation.
    #[arg(long, value_parser = parse_cache_type)]
    cache_type_v: Option<GgufType>,
    /// An `<regex>=<device>` rule, repeatable.
    #[arg(long)]
    override_tensor: Vec<String>,
    /// Override the estimated compute-buffer reservation, per device.
    #[arg(long)]
    compute_buffer_mb: Option<u32>,
    /// How many devices the totals below charge a compute buffer to.
    #[arg(long, default_value_t = 3)]
    active_devices: u64,
    /// Run `--spec-type draft-mtp` against the model's own head. Implied by
    /// `--draft-model`.
    #[arg(long)]
    mtp: bool,
    /// The draft head as a separate GGUF (`-md`).
    #[arg(long)]
    draft_model: Option<PathBuf>,
    /// The draft's `--spec-type`, when it isn't `draft-mtp` — `draft-dflash`,
    /// for instance. Only meaningful alongside `--draft-model`; an embedded
    /// head is always mtp.
    #[arg(long, default_value = "draft-mtp")]
    spec_type: String,
    /// Parallel slots (`-np`).
    #[arg(long)]
    parallel: Option<u32>,
    /// Flash attention (`-fa`). Unset resolves the way the runtime will.
    #[arg(long, value_name = "on|off", value_parser = parse_on_off)]
    flash_attn: Option<bool>,
    /// One KV cache shared across slots (`-kvu`).
    #[arg(long, value_name = "on|off", value_parser = parse_on_off)]
    kv_unified: Option<bool>,
    /// The prompt cache's host-RAM cap (`-cram`).
    #[arg(long)]
    cache_ram_mb: Option<u32>,
    /// How the model is split across cards (`-sm`).
    #[arg(long, value_name = "layer|row|tensor", value_parser = parse_split_mode, default_value = "layer")]
    split_mode: SplitMode,
    /// How many devices the child will see. Defaults to 1, or to 2 under a
    /// tensor or row split, which usually implies more than one card —
    /// llama.cpp accepts either on a single card, and estimating such a run as
    /// two charges it a second device's share of every per-device term.
    #[arg(long)]
    visible_devices: Option<u32>,
    /// Estimate for ik_llama.cpp rather than mainline.
    #[arg(long)]
    ik_llama: bool,
    /// ik's sparse attention (`-dsa`).
    #[arg(long)]
    ik_dsa: bool,
    /// `expert_offload = auto`.
    #[arg(long, conflicts_with = "n_cpu_moe")]
    host_resident_experts: bool,
    /// `expert_offload = N`.
    #[arg(long)]
    n_cpu_moe: Option<u32>,
    /// Pack the estimate against `--gpu` / `--cpu` and report the allocation.
    #[arg(long)]
    pack: bool,
    /// A GPU's capacity in MiB, repeated once per card.
    #[arg(long = "gpu")]
    gpu_capacities_mib: Vec<u64>,
    /// Host capacity in MiB.
    #[arg(long = "cpu")]
    cpu_capacity_mib: Option<u64>,
}

impl Args {
    fn expert_offload(&self) -> OffloadMode {
        match (self.host_resident_experts, self.n_cpu_moe) {
            (true, _) => OffloadMode::Auto,
            (false, Some(n)) => OffloadMode::Layers(n),
            (false, None) => OffloadMode::Off,
        }
    }

    fn visible_devices(&self) -> u32 {
        self.visible_devices
            .unwrap_or(if self.split_mode == SplitMode::Layer {
                1
            } else {
                2
            })
    }

    fn speculation(&self) -> Speculation<'_> {
        match (self.mtp, self.draft_model.as_deref()) {
            (_, Some(draft)) => Speculation::SeparateDraft {
                path: draft,
                spec_type: &self.spec_type,
            },
            (true, None) => Speculation::EmbeddedMtp,
            (false, None) => Speculation::None,
        }
    }
}

fn main() {
    let args = Args::parse();
    let inputs = EstimatorInputs {
        name: SERVICE,
        model: args.model.as_path(),
        mmproj: args.mmproj.as_deref(),
        context: args.context,
        ubatch: args.ubatch,
        worst_graph_tokens: None,
        ik_assignment_options_supported: true,
        visible_devices: args.visible_devices(),
        host_resident_experts: args.expert_offload().is_enabled(),
        split_mode: args.split_mode,
        cache_type_k: args.cache_type_k,
        cache_type_v: args.cache_type_v,
        override_tensor: &args.override_tensor,
        compute_buffer_mb: args.compute_buffer_mb,
        speculation: args.speculation(),
        fork: if args.ik_llama {
            Fork::Ik { dsa: args.ik_dsa }
        } else {
            Fork::Mainline
        },
        parallel: args.parallel,
        flash_attn: args.flash_attn,
        kv_unified: args.kv_unified,
        cache_ram_mb: args.cache_ram_mb,
    };

    let estimate = match estimator::estimate_from_path(&LocalFs, &inputs) {
        Ok(estimate) => estimate,
        Err(e) => {
            print(&EstimatorFailed {
                estimator_error: e.to_string(),
            });
            process::exit(1);
        }
    };

    let placement = args.pack.then(|| {
        match pack_demand(
            &estimate,
            &build_placement(&args),
            &build_snapshot(&args),
            Corrections::NEUTRAL,
        ) {
            Ok(packed) => PlacementReport::Packed(Box::new(packed_report(&packed))),
            Err(e) => PlacementReport::Failed {
                error: e.to_string(),
            },
        }
    });
    print(&report(&estimate, &args, placement));
}

/// The estimate, and the packing if one was asked for.
///
/// Byte counts are reported alongside their rounded MiB / GiB forms rather than
/// either alone: the exact figure is what a regression check compares, and the
/// rounded one is what an operator reads against an `nvidia-smi` line.
#[derive(Serialize)]
struct Report<'a> {
    architecture: &'a str,
    context: u32,
    weights_bytes: u64,
    weights_gib: f64,
    kv_per_token_bytes: u64,
    kv_total_bytes: u64,
    kv_total_mib: u64,
    compute_buffer_mb: u32,
    tensor_split_replicated_mib: u64,
    mtp_bytes: u64,
    mtp_mib: u64,
    output_buffer_bytes: u64,
    output_buffer_mib: u64,
    host_overhead_mib: u64,
    host_cache_mib: u64,
    per_layer_count: Option<usize>,
    non_layer_output_head_bytes: u64,
    non_layer_token_embd_bytes: u64,
    non_layer_other_bytes: u64,
    expert_layer_count: usize,
    expert_tensor_count: Option<usize>,
    expert_total_bytes: Option<u64>,
    override_tensor_bytes: BTreeMap<String, u64>,
    /// Weights + KV + one compute buffer per `--active-devices` + MTP.
    total_accounted_bytes: u64,
    total_accounted_mib: u64,
    /// The same, less the CPU-resident token embeddings and charging a compute
    /// buffer to at most two devices.
    gpu_vram_bytes: u64,
    gpu_vram_mib: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    placement: Option<PlacementReport>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum PlacementReport {
    Packed(Box<PackedReport>),
    Failed { error: String },
}

#[derive(Serialize)]
struct PackedReport {
    allocation: BTreeMap<String, DeviceBytes>,
    /// The allocation above carries deliberate slop — one layer's worth of
    /// headroom, the prompt cache's cap — that the process is not expected to
    /// use. The prediction is what the rolling correction divides an
    /// observation by, so it is the figure to compare against a measurement.
    predicted_vram_bytes: u64,
    predicted_vram_mib: u64,
    predicted_host_bytes: u64,
    gpu_weight_bytes: u64,
    expert_offload_bytes: u64,
    expert_offload_mib: u64,
    expert_offload_layers: u32,
}

#[derive(Serialize)]
struct DeviceBytes {
    bytes: u64,
    mib: u64,
    gib: f64,
}

#[derive(Serialize)]
struct EstimatorFailed {
    estimator_error: String,
}

fn report<'a>(
    estimate: &'a Estimate,
    args: &Args,
    placement: Option<PlacementReport>,
) -> Report<'a> {
    let kv_total_bytes = estimate
        .kv_per_token
        .saturating_mul(u64::from(estimate.context));
    let compute_bytes = |devices: u64| {
        u64::from(estimate.buffers.compute_mb)
            .saturating_mul(devices)
            .saturating_mul(MIB)
    };
    let total_bytes = estimate
        .weights_bytes
        .saturating_add(kv_total_bytes)
        .saturating_add(compute_bytes(args.active_devices))
        .saturating_add(estimate.mtp.bytes);
    // The embedding table stays on the CPU, and llama.cpp materialises logits
    // on the head card alone, so a second card is the most that can be charged
    // a full compute buffer here.
    let gpu_total_bytes = estimate
        .weights_bytes
        .saturating_sub(estimate.layout.non_layer.token_embd_bytes)
        .saturating_add(kv_total_bytes)
        .saturating_add(compute_bytes(args.active_devices.min(2)))
        .saturating_add(estimate.mtp.bytes);

    Report {
        architecture: estimate.architecture.as_str(),
        context: estimate.context,
        weights_bytes: estimate.weights_bytes,
        weights_gib: estimate.weights_bytes as f64 / GIB_F64,
        kv_per_token_bytes: estimate.kv_per_token,
        kv_total_bytes,
        kv_total_mib: kv_total_bytes / MIB,
        compute_buffer_mb: estimate.buffers.compute_mb,
        tensor_split_replicated_mib: estimate.layout.tensor_split_replicated_bytes / MIB,
        mtp_bytes: estimate.mtp.bytes,
        mtp_mib: estimate.mtp.bytes / MIB,
        output_buffer_bytes: estimate.buffers.output_bytes,
        output_buffer_mib: estimate.buffers.output_bytes / MIB,
        host_overhead_mib: estimate.host.overhead_bytes / MIB,
        host_cache_mib: estimate.host.cache_bytes / MIB,
        per_layer_count: estimate.layout.per_layer_bytes.as_ref().map(Vec::len),
        non_layer_output_head_bytes: estimate.layout.non_layer.output_head_bytes,
        non_layer_token_embd_bytes: estimate.layout.non_layer.token_embd_bytes,
        non_layer_other_bytes: estimate.layout.non_layer.other_bytes,
        expert_layer_count: estimate.layout.expert_layers.len(),
        expert_tensor_count: estimate.layout.expert_tensors.as_ref().map(Vec::len),
        expert_total_bytes: estimate
            .layout
            .expert_tensors
            .as_ref()
            .map(|tensors| tensors.iter().map(|e| e.bytes).sum()),
        override_tensor_bytes: estimate
            .layout
            .override_tensor_bytes
            .iter()
            .map(|(slot, bytes)| (format!("{slot:?}"), *bytes))
            .collect(),
        total_accounted_bytes: total_bytes,
        total_accounted_mib: total_bytes / MIB,
        gpu_vram_bytes: gpu_total_bytes,
        gpu_vram_mib: gpu_total_bytes / MIB,
        placement,
    }
}

fn packed_report(packed: &Packed) -> PackedReport {
    PackedReport {
        allocation: packed
            .allocation
            .bytes
            .iter()
            .map(|(id, bytes)| {
                (
                    id.as_display().to_string(),
                    DeviceBytes {
                        bytes: *bytes,
                        mib: bytes / MIB,
                        gib: *bytes as f64 / GIB_F64,
                    },
                )
            })
            .collect(),
        predicted_vram_bytes: packed.rolling.uncorrected_vram_bytes,
        predicted_vram_mib: packed.rolling.uncorrected_vram_bytes / MIB,
        predicted_host_bytes: packed.rolling.uncorrected_host_bytes,
        gpu_weight_bytes: packed.rolling.gpu_weight_bytes,
        expert_offload_bytes: packed.expert_offload_bytes,
        expert_offload_mib: packed.expert_offload_bytes / MIB,
        expert_offload_layers: packed.expert_offload_layers,
    }
}

/// The placement the packer needs: policy, split mode, the GPU allow list, and
/// the expert-offload decision.
fn build_placement(args: &Args) -> PlacementInputs {
    let expert_offload = args.expert_offload();
    PlacementInputs {
        policy: if expert_offload.is_enabled() {
            PlacementPolicy::Hybrid
        } else {
            PlacementPolicy::GpuOnly
        },
        split_mode: args.split_mode,
        gpu_allow: (0..args.visible_devices()).collect(),
        expert_offload,
        ik_llama: args.ik_llama,
        override_tensor: args.override_tensor.clone(),
        ..PlacementInputs::named(SERVICE)
    }
}

fn build_snapshot(args: &Args) -> DeviceSnapshot {
    DeviceSnapshot {
        gpus: args
            .gpu_capacities_mib
            .iter()
            .enumerate()
            .map(|(i, cap_mib)| GpuSnapshot {
                id: i as u32,
                name: format!("fake-{i}"),
                total_bytes: cap_mib * MIB,
                free_bytes: cap_mib * MIB,
            })
            .collect(),
        cpu: args.cpu_capacity_mib.map(|cap_mib| CpuSnapshot {
            total_bytes: cap_mib * MIB,
            available_bytes: cap_mib * MIB,
        }),
        taken_at_ms: 0,
    }
}

fn print<T: Serialize>(value: &T) {
    match serde_json::to_string_pretty(value) {
        Ok(json) => println!("{json}"),
        Err(e) => {
            eprintln!("failed to render the report: {e}");
            process::exit(1);
        }
    }
}

fn parse_cache_type(value: &str) -> Result<GgufType, String> {
    GgufType::from_name(value).ok_or_else(|| format!("`{value}` names no ggml type"))
}

fn parse_split_mode(value: &str) -> Result<SplitMode, String> {
    SplitMode::from_flag(value)
        .ok_or_else(|| format!("expected layer, row, or tensor; got {value}"))
}

fn parse_on_off(value: &str) -> Result<bool, String> {
    match value {
        "on" | "true" | "1" => Ok(true),
        "off" | "false" | "0" => Ok(false),
        other => Err(format!("expected on or off; got {other}")),
    }
}
