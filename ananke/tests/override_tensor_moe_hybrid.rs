//! Integration test: a MoE service with `override_tensor` rules passes those
//! rules through to `CommandArgs.override_tensor` after `placement::pack`.
#![cfg(feature = "test-fakes")]

mod common;

use std::collections::BTreeMap;

use ananke::{
    allocator::{AllocationTable, placement},
    config::{OffloadMode, PlacementPolicy, ServiceConfig, TemplateConfig},
    devices::{CpuSnapshot, DeviceId, DeviceSnapshot, GpuSnapshot},
    estimator,
};
use common::synth_gguf;

fn moe_svc_with_override_tensor(model_path: std::path::PathBuf) -> ServiceConfig {
    let mut svc = common::minimal_llama_service("moe-ot-svc", 0);
    svc.placement_override = BTreeMap::new();
    svc.placement_policy = PlacementPolicy::GpuOnly;
    let TemplateConfig::LlamaCpp(lc) = &mut svc.template_config else {
        unreachable!();
    };
    lc.model = model_path;
    lc.override_tensor = vec![
        r"\.ffn_up_exps\.=CPU".to_string(),
        r"\.ffn_down_exps\.=CPU".to_string(),
    ];
    svc
}

#[test]
fn override_tensor_rules_propagate_to_command_args() {
    let path = std::path::Path::new("/fake/moe-ot.gguf");
    let fs = synth_gguf::Builder::new()
        .kv_string("general.architecture", "qwen3moe")
        .kv_u32("qwen3moe.block_count", 2)
        .kv_u32("qwen3moe.attention.head_count_kv", 4)
        .kv_u32("qwen3moe.attention.key_length", 128)
        .kv_u32("qwen3moe.attention.value_length", 128)
        .tensor_f16("blk.0.attn_q.weight", 512 * 1024)
        .tensor_f16("blk.1.attn_q.weight", 512 * 1024)
        .tensor_f16("blk.0.ffn_gate_exps.weight", 4 * 512 * 1024)
        .tensor_f16("blk.1.ffn_up_exps.weight", 4 * 512 * 1024)
        .tensor_f16("blk.1.ffn_down_exps.weight", 4 * 512 * 1024)
        .into_in_memory_fs(path);

    let svc = moe_svc_with_override_tensor(path.to_path_buf());
    let inputs = estimator::EstimatorInputs::from_service(&svc).unwrap();
    let est =
        estimator::estimate_from_path(&fs, &inputs).expect("estimate must succeed on MoE GGUF");

    // Snapshot with one GPU that has ample free memory.
    let snap = DeviceSnapshot {
        gpus: vec![GpuSnapshot {
            id: 0,
            name: "GPU 0".into(),
            total_bytes: 24 * 1024 * 1024 * 1024,
            free_bytes: 24 * 1024 * 1024 * 1024,
        }],
        cpu: None,
        taken_at_ms: 0,
    };

    let reserved = AllocationTable::new();
    let packed = placement::pack(&est, &svc, &snap, &reserved)
        .expect("placement must succeed on single GPU");

    // The override_tensor rules declared in the service must be forwarded
    // verbatim to CommandArgs so the spawn renderer can emit -ot flags.
    assert_eq!(
        packed.args.override_tensor.len(),
        2,
        "both override_tensor rules must be present in CommandArgs"
    );
    assert!(
        packed.args.override_tensor[0].contains("ffn_up_exps"),
        "first rule must reference ffn_up_exps"
    );
    assert!(
        packed.args.override_tensor[1].contains("ffn_down_exps"),
        "second rule must reference ffn_down_exps"
    );
}

#[test]
fn auto_expert_offload_emits_n_cpu_moe_under_hybrid() {
    let path = std::path::Path::new("/fake/moe-auto.gguf");
    // 4 layers, tiny attention, 256 MiB per fused expert tensor (≈3 GiB of
    // experts) — far more than the 1 GiB card can hold.
    let exp_elems = 128 * 1024 * 1024; // f16 → 256 MiB each
    let mut builder = synth_gguf::Builder::new()
        .kv_string("general.architecture", "qwen3moe")
        .kv_u32("qwen3moe.block_count", 4)
        .kv_u32("qwen3moe.attention.head_count_kv", 4)
        .kv_u32("qwen3moe.attention.key_length", 128)
        .kv_u32("qwen3moe.attention.value_length", 128);
    for layer in 0..4 {
        builder = builder
            .tensor_f16(&format!("blk.{layer}.attn_q.weight"), 512 * 1024)
            .tensor_f16(&format!("blk.{layer}.ffn_gate_exps.weight"), exp_elems)
            .tensor_f16(&format!("blk.{layer}.ffn_up_exps.weight"), exp_elems)
            .tensor_f16(&format!("blk.{layer}.ffn_down_exps.weight"), exp_elems);
    }
    let fs = builder.into_in_memory_fs(path);

    let mut svc = common::minimal_llama_service("moe-auto-svc", 0);
    svc.placement_override = BTreeMap::new();
    svc.placement_policy = PlacementPolicy::Hybrid;
    let TemplateConfig::LlamaCpp(lc) = &mut svc.template_config else {
        unreachable!();
    };
    lc.model = path.to_path_buf();
    lc.expert_offload = OffloadMode::Auto;

    let inputs = estimator::EstimatorInputs::from_service(&svc).unwrap();
    let est = estimator::estimate_from_path(&fs, &inputs).expect("estimate must succeed");

    // 1 GiB card, generous host RAM.
    let snap = DeviceSnapshot {
        gpus: vec![GpuSnapshot {
            id: 0,
            name: "GPU 0".into(),
            total_bytes: 24 * 1024 * 1024 * 1024,
            free_bytes: 1024 * 1024 * 1024,
        }],
        cpu: Some(CpuSnapshot {
            total_bytes: 128 * 1024 * 1024 * 1024,
            available_bytes: 64 * 1024 * 1024 * 1024,
        }),
        taken_at_ms: 0,
    };

    let packed = placement::pack(&est, &svc, &snap, &AllocationTable::new())
        .expect("hybrid auto-offload must pack on a 1 GiB card");

    // -ngl 999: all layers to GPU, then --n-cpu-moe pulls the trailing
    // experts back to CPU (the runtime owns the cross-GPU split).
    assert_eq!(packed.args.ngl, Some(999));
    // Surplus experts landed on the CPU via coarse whole-layer offload.
    assert!(
        packed
            .allocation
            .bytes
            .get(&DeviceId::Cpu)
            .copied()
            .unwrap_or(0)
            > 0,
        "experts must offload to the CPU"
    );
    assert!(packed.expert_offload_bytes > 0);
    assert!(
        matches!(packed.args.n_cpu_moe, Some(n) if n > 0),
        "coarse --n-cpu-moe offload must be set, got {:?}",
        packed.args.n_cpu_moe
    );
    assert!(
        packed.args.override_tensor.is_empty(),
        "no per-tensor expert -ot is synthesised, got {:?}",
        packed.args.override_tensor
    );
}
