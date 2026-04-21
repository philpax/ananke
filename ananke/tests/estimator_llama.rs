//! Integration test: llama-family estimator reads a synthetic GGUF and
//! produces positive weights and KV estimates.
#![cfg(feature = "test-fakes")]

mod common;

use std::path::Path;

use ananke::estimator;
use common::synth_gguf;

#[test]
fn llama_family_weights_include_layers_and_non_layer() {
    let path = Path::new("/fake/llama.gguf");
    let fs = synth_gguf::Builder::new()
        .kv_string("general.architecture", "qwen3")
        .kv_u32("qwen3.block_count", 2)
        .kv_u32("qwen3.attention.head_count_kv", 4)
        .kv_u32("qwen3.attention.key_length", 128)
        .kv_u32("qwen3.attention.value_length", 128)
        .tensor_f16("blk.0.attn_q.weight", 512 * 1024)
        .tensor_f16("blk.1.attn_q.weight", 512 * 1024)
        .tensor_f16("output.weight", 2 * 512 * 1024)
        .tensor_f16("token_embd.weight", 4 * 512 * 1024)
        .into_in_memory_fs(path);

    let mut svc = common::minimal_llama_service("demo", 0);
    common::set_model_path(&mut svc, path);
    let inputs = estimator::EstimatorInputs::from_service(&svc).unwrap();
    let est = estimator::estimate_from_path(&fs, &inputs).unwrap();
    assert!(est.weights_bytes > 0, "weights_bytes must be positive");
    assert_eq!(est.architecture, "qwen3");
    assert!(est.kv_per_token > 0, "kv_per_token must be positive");
}
