//! Integration test: MoE estimator reads a synthetic qwen3moe GGUF and
//! correctly identifies expert layers.

mod common;

use std::path::Path;

use ananke::estimator;
use common::synth_gguf;

#[test]
fn moe_estimator_identifies_expert_layers() {
    let path = Path::new("/fake/moe.gguf");
    let fs = synth_gguf::Builder::new()
        .kv_string("general.architecture", "qwen3moe")
        .kv_u32("qwen3moe.block_count", 3)
        .kv_u32("qwen3moe.attention.head_count_kv", 4)
        .kv_u32("qwen3moe.attention.key_length", 128)
        .kv_u32("qwen3moe.attention.value_length", 128)
        // Attention tensors (non-expert).
        .tensor_f16("blk.0.attn_q.weight", 512 * 1024)
        .tensor_f16("blk.1.attn_q.weight", 512 * 1024)
        .tensor_f16("blk.2.attn_q.weight", 512 * 1024)
        // Expert tensors — identified by the `_exps` suffix.
        .tensor_f16("blk.0.ffn_gate_exps.weight", 4 * 512 * 1024)
        .tensor_f16("blk.1.ffn_gate_exps.weight", 8 * 512 * 1024)
        .tensor_f16("blk.2.ffn_up_exps.weight", 2 * 512 * 1024)
        .tensor_f16("output.weight", 512 * 1024)
        .into_in_memory_fs(path);

    let svc = common::minimal_llama_service("demo", 0);
    let est = estimator::estimate_from_path(&fs, path, &svc).unwrap();
    assert!(
        !est.expert_layers.is_empty(),
        "expert_layers must be non-empty"
    );
    assert!(est.weights_bytes > 0, "weights_bytes must be positive");
}
