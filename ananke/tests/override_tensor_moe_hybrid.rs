//! Integration test: a MoE service with `override_tensor` rules passes those
//! rules through to `CommandArgs.override_tensor` after `placement::pack`.

mod common;

use std::collections::BTreeMap;

use ananke::{
    allocator::{AllocationTable, placement},
    config::{PlacementPolicy, ServiceConfig, TemplateConfig},
    devices::{DeviceSnapshot, GpuSnapshot},
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
    let file = synth_gguf::tempfile("moe-ot");
    synth_gguf::Builder::new()
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
        .write_to(file.path());

    let svc = moe_svc_with_override_tensor(file.path().to_path_buf());

    let est = estimator::estimate_from_path(file.path(), &svc)
        .expect("estimate must succeed on MoE GGUF");

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
