//! Integration test: llama-family estimator reads a synthetic GGUF and
//! produces positive weights and KV estimates.

mod common;

use ananke::estimator;
use common::synth_gguf;

#[test]
fn llama_family_weights_include_layers_and_non_layer() {
    let file = synth_gguf::tempfile("llama");
    synth_gguf::Builder::new()
        .kv_string("general.architecture", "qwen3")
        .kv_u32("qwen3.block_count", 2)
        .kv_u32("qwen3.attention.head_count_kv", 4)
        .kv_u32("qwen3.attention.key_length", 128)
        .kv_u32("qwen3.attention.value_length", 128)
        .tensor_f16("blk.0.attn_q.weight", 512 * 1024)
        .tensor_f16("blk.1.attn_q.weight", 512 * 1024)
        .tensor_f16("output.weight", 2 * 512 * 1024)
        .tensor_f16("token_embd.weight", 4 * 512 * 1024)
        .write_to(file.path());

    let svc = common::minimal_llama_service("demo", 0);
    let est = estimator::estimate_from_path(file.path(), &svc).unwrap();
    assert!(est.weights_bytes > 0, "weights_bytes must be positive");
    assert_eq!(est.architecture, "qwen3");
    assert!(est.kv_per_token > 0, "kv_per_token must be positive");
}
