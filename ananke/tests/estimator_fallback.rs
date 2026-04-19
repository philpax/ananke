//! Integration test: fallback estimator handles an unknown architecture and
//! produces a coarse estimate of at least 512 MB.

mod common;

use std::path::Path;

use ananke::estimator;
use common::synth_gguf;

#[test]
fn fallback_estimator_produces_at_least_512mb() {
    let path = Path::new("/fake/fallback.gguf");
    // "novel-arch" is not known to any specific estimator family, so the
    // fallback path applies: total_tensor_bytes × 1.15 + 512 MB.
    let fs = synth_gguf::Builder::new()
        .kv_string("general.architecture", "novel-arch")
        .tensor_f16("some.weight", 1024)
        .into_in_memory_fs(path);

    let svc = common::minimal_llama_service("demo", 0);
    let est = estimator::estimate_from_path(&fs, path, &svc).unwrap();
    assert!(
        est.weights_bytes >= 512 * 1024 * 1024,
        "fallback must reserve at least 512 MB; got {}",
        est.weights_bytes
    );
}
