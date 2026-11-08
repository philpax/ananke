//! Compare the estimator's prediction against every comparable measured cell.
//!
//! This is the campaign's top-level accuracy signal. It runs the estimator and
//! the packer **in-process**: spawning `cargo run --example estimate` once per
//! cell instead would be some two hundred subprocess launches, each re-reading
//! the GGUF.
//!
//! What it compares is the *prediction*, not the reservation. The reservation
//! carries slop the process is not expected to use — one layer's headroom above
//! all — and comparing that to a measurement reads the slop as error. On the
//! production Qwen3.6-27B cell the two differ by 472 MiB, which is the
//! difference between +1.1% and −0.1%.

use ananke_config::{
    placement::{OffloadMode, PlacementInputs, PlacementPolicy, SplitMode},
    units::MIB,
};
use ananke_estimate::{EstimatorInputs, Fork, Speculation};
use ananke_gguf::GgufType;
use ananke_measure::record::Status;
use ananke_placement::{
    Corrections,
    devices::{DeviceSnapshot, GpuSnapshot},
};

use crate::record::{FlashAttn, KvType, Record, Runtime};

/// Host memory the packer is told it has. The campaign machine has 256 GiB and
/// nothing here depends on the exact figure — it only has to be large enough not
/// to bound an expert offload.
const CPU_CAPACITY_MIB: u64 = 256_000;

/// One cell's outcome.
#[derive(Debug, Clone)]
pub struct Comparison {
    pub label: String,
    pub arch: String,
    pub predicted_mib: u64,
    pub measured_mib: u64,
    pub reserved_mib: u64,
    pub drift_pct: f64,
}

/// Why a cell was not comparable. Counted and reported, never silently dropped.
pub fn skip_reason(record: &Record, known_models: &[String]) -> Option<String> {
    let f = &record.factors;
    if record.status != Status::Ok {
        return Some(format!("status {:?}", record.status));
    }
    if record.rss.gpu_used_mib.is_none_or(|v| v == 0) {
        return Some("no driver reading".into());
    }
    if !f.served {
        return Some("idle: no first-use allocations".into());
    }
    if !known_models.iter().any(|m| m == &f.model) {
        return Some("model not in models.toml".into());
    }
    // An operator-chosen `ngl` is a placement the packer did not choose, so the
    // two are not comparable. Expert offload is different: `--n-cpu-moe` leaves
    // `ngl` at 99 and the packer models it.
    if !f.fully_offloaded() && f.n_cpu_moe.is_none() {
        return Some(format!("operator-chosen placement (ngl {})", f.ngl));
    }
    if f.embeddings {
        return Some("embedding modality".into());
    }
    None
}

/// The estimator inputs this cell was measured under.
pub fn estimator_inputs<'a>(record: &'a Record, model: &'a std::path::Path) -> EstimatorInputs<'a> {
    let f = &record.factors;
    let cards = record.factors.gpu_ids().len().max(1) as u32;
    EstimatorInputs {
        name: &f.label,
        model,
        mmproj: f.mmproj.as_deref().map(std::path::Path::new),
        context: f.ctx,
        ubatch: Some(f.ubatch),
        worst_graph_tokens: None,
        ik_assignment_options_supported: true,
        visible_devices: cards,
        host_resident_experts: f.n_cpu_moe.is_some(),
        split_mode: f.split_or_layer(),
        // Every `KvType` names a type `GgufType` knows — `ananke_estimate`'s
        // `the_two_quantised_partitions_agree` holds the two vocabularies
        // together — so this never silently degrades to the f16 default.
        cache_type_k: GgufType::from_name(f.kv_type.name()),
        cache_type_v: GgufType::from_name(f.kv_type.name()),
        override_tensor: &[],
        compute_buffer_mb: None,
        speculation: match (f.spec_type.as_deref(), f.draft.as_deref()) {
            (Some(spec_type), Some(draft)) => Speculation::SeparateDraft {
                path: std::path::Path::new(draft),
                spec_type,
            },
            (Some("draft-mtp"), None) => Speculation::EmbeddedMtp,
            _ => Speculation::None,
        },
        fork: if f.runtime_is_ik() {
            // The fork's sparse-attention path is a separate flag, and the
            // campaign runs it only for the architecture that has one.
            Fork::Ik {
                dsa: record.parsed.arch == "glm-dsa",
            }
        } else {
            Fork::Mainline
        },
        parallel: Some(f.parallel),
        flash_attn: Some(f.flash_attn_on()),
        kv_unified: Some(f.kv_unified),
        cache_ram_mb: Some(f.cram),
    }
}

/// The placement this cell was measured under.
pub fn placement_inputs(record: &Record) -> PlacementInputs {
    let f = &record.factors;
    PlacementInputs {
        policy: if f.n_cpu_moe.is_some() {
            PlacementPolicy::Hybrid
        } else {
            PlacementPolicy::GpuOnly
        },
        split_mode: f.split_or_layer(),
        gpu_allow: record.factors.gpu_ids(),
        expert_offload: match f.n_cpu_moe {
            Some(n) => OffloadMode::Layers(n),
            None => OffloadMode::Off,
        },
        ik_llama: f.runtime_is_ik(),
        ..PlacementInputs::named(&f.label)
    }
}

/// The cards this cell ran on, at the capacities the campaign saw.
///
/// The driver reserves a little of each card, which is why a nominally 24576 MiB
/// 3090 reads ~24124 free. Pack against what was actually available.
pub fn snapshot(record: &Record) -> DeviceSnapshot {
    let capacities: Vec<u64> = record
        .hardware
        .gpus
        .iter()
        .map(|g| g.memory_total_mib)
        .collect();
    snapshot_for(&record.factors.gpu_ids(), &capacities)
}

/// A key identifying the configuration, so repeats of one cell are counted once.
///
/// Every factor `tests/factors.rs` classifies as read is pinned, [`KEY_EXCLUDED`]
/// aside, and `tests/factors.rs::the_validate_key_pins_every_read_factor` holds
/// the two in step.
///
/// A key that omits a factor is not a narrower key, it is a wrong one: cells
/// differing only in the omitted factor collide and all but the first are
/// discarded as duplicates. Omitting `extra` alone — it carries ik's `-dsa`,
/// worth gigabytes of VRAM — collides 53 otherwise comparable cells.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ConfigurationKey<'a> {
    model: &'a str,
    runtime: Runtime,
    gpus: &'a str,
    ctx: u32,
    ubatch: u32,
    parallel: u32,
    ngl: u32,
    split: Option<SplitMode>,
    kv_type: KvType,
    kv_unified: bool,
    flash_attn: FlashAttn,
    n_cpu_moe: Option<u32>,
    mmproj: Option<&'a str>,
    draft: Option<&'a str>,
    spec_type: Option<&'a str>,
    no_mmap: bool,
    rtr: bool,
    cram: u32,
    soak: u32,
    concurrency: u32,
    probe_prompt_tokens: u32,
    embeddings: bool,
    bench: bool,
    served: bool,
    numa: Option<&'a str>,
    extra: &'a [String],
}

/// The one read factor [`ConfigurationKey`] leaves out, and why.
///
/// A label names a cell; it does not describe the process. Two labels sharing a
/// configuration — a re-measurement, or the same point reached from two sweeps —
/// measure the same thing, and collapsing them is what the key is for.
pub const KEY_EXCLUDED: &[&str] = &["label"];

/// The configuration this cell was measured under.
pub fn configuration_key(record: &Record) -> ConfigurationKey<'_> {
    let f = &record.factors;
    ConfigurationKey {
        model: &f.model,
        runtime: f.runtime,
        gpus: &f.gpus,
        ctx: f.ctx,
        ubatch: f.ubatch,
        parallel: f.parallel,
        ngl: f.ngl,
        split: f.split,
        kv_type: f.kv_type,
        kv_unified: f.kv_unified,
        flash_attn: f.flash_attn,
        n_cpu_moe: f.n_cpu_moe,
        mmproj: f.mmproj.as_deref(),
        draft: f.draft.as_deref(),
        spec_type: f.spec_type.as_deref(),
        no_mmap: f.no_mmap,
        rtr: f.rtr,
        cram: f.cram,
        soak: f.soak,
        concurrency: f.concurrency,
        probe_prompt_tokens: f.probe_prompt_tokens,
        embeddings: f.embeddings,
        bench: f.bench,
        served: f.served,
        numa: f.numa.as_deref(),
        extra: &f.extra,
    }
}

/// A snapshot of the given cards at the given capacities.
///
/// `capacities_mib` is positional against `ids`; a card without an entry gets the
/// campaign's 24576 MiB default.
pub fn snapshot_for(ids: &[u32], capacities_mib: &[u64]) -> DeviceSnapshot {
    let gpus = ids
        .iter()
        .enumerate()
        .map(|(index, &id)| {
            let total = capacities_mib.get(index).copied().unwrap_or(24_576) * MIB;
            GpuSnapshot {
                id,
                name: format!("GPU {id}"),
                total_bytes: total,
                free_bytes: total,
            }
        })
        .collect();
    DeviceSnapshot {
        gpus,
        cpu: Some(ananke_placement::devices::CpuSnapshot {
            total_bytes: CPU_CAPACITY_MIB * MIB,
            available_bytes: CPU_CAPACITY_MIB * MIB,
        }),
        taken_at_ms: 0,
    }
}

/// The reservation summed over the GPU slots, in MiB.
///
/// Distinct from the *prediction* `validate` compares: this carries the slop the
/// packer adds and is what the cards are actually asked to hold.
pub fn gpu_reserved_mib(packed: &ananke_placement::Packed) -> u64 {
    packed
        .allocation
        .bytes
        .iter()
        .filter(|(id, _)| matches!(id, ananke_placement::devices::DeviceId::Gpu(_)))
        .map(|(_, &b)| b)
        .sum::<u64>()
        / MIB
}

/// The neutral corrections a validation run packs with: this compares the
/// estimator, not the rolling correction that trains on top of it.
pub const NEUTRAL: Corrections = Corrections::NEUTRAL;
