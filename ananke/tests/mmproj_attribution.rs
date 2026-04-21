//! Integration test: when a llama-cpp service has an mmproj path set, the
//! estimator adds the mmproj file's tensor bytes to `weights_bytes`.
#![cfg(feature = "test-fakes")]

mod common;

use std::path::{Path, PathBuf};

use ananke::{config::TemplateConfig, estimator, system::InMemoryFs};
use common::synth_gguf;

fn svc_with_mmproj(model: PathBuf, mmproj: Option<PathBuf>) -> ananke::config::ServiceConfig {
    let mut svc = common::minimal_llama_service("mmproj-svc", 0);
    let TemplateConfig::LlamaCpp(lc) = &mut svc.template_config else {
        unreachable!();
    };
    lc.model = model;
    lc.mmproj = mmproj;
    svc
}

#[test]
fn mmproj_bytes_included_in_weights_estimate() {
    let main_path = Path::new("/fake/main.gguf");
    let mmproj_path = Path::new("/fake/mmproj.gguf");

    let main_bytes = synth_gguf::Builder::new()
        .kv_string("general.architecture", "qwen3")
        .kv_u32("qwen3.block_count", 2)
        .kv_u32("qwen3.attention.head_count_kv", 4)
        .kv_u32("qwen3.attention.key_length", 128)
        .kv_u32("qwen3.attention.value_length", 128)
        .tensor_f16("blk.0.attn_q.weight", 512 * 1024)
        .tensor_f16("blk.1.attn_q.weight", 512 * 1024)
        .build();

    // 1 MiB of F16 elements = 512 * 1024 elements * 2 bytes each.
    let mmproj_bytes = synth_gguf::Builder::new()
        .kv_string("general.architecture", "clip")
        .tensor_f16("mm.0.weight", 512 * 1024)
        .build();

    let fs = InMemoryFs::new();
    fs.insert(main_path, main_bytes);
    fs.insert(mmproj_path, mmproj_bytes);

    let svc_without_mmproj = svc_with_mmproj(main_path.to_path_buf(), None);
    let svc_with = svc_with_mmproj(main_path.to_path_buf(), Some(mmproj_path.to_path_buf()));

    let inputs_without = estimator::EstimatorInputs::from_service(&svc_without_mmproj).unwrap();
    let inputs_with = estimator::EstimatorInputs::from_service(&svc_with).unwrap();
    let est_without = estimator::estimate_from_path(&fs, &inputs_without).unwrap();
    let est_with = estimator::estimate_from_path(&fs, &inputs_with).unwrap();

    assert!(
        est_with.weights_bytes > est_without.weights_bytes,
        "estimate with mmproj ({}) must exceed estimate without ({})",
        est_with.weights_bytes,
        est_without.weights_bytes
    );
}
