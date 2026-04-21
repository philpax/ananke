//! Integration tests for the fallback estimator + its opt-in gate.
#![cfg(feature = "test-fakes")]

mod common;

use std::path::Path;

use ananke::estimator::{self, EstimatorError};
use common::synth_gguf;

/// Unknown architectures must hard-reject by default — silent fallback
/// hid glm4moe's 67× under-reservation for weeks. The error must be the
/// typed `UnknownArchitecture` variant so callers can match on it rather
/// than stringly-typed substrings.
#[test]
fn unknown_architecture_rejects_unless_opted_in() {
    let path = Path::new("/fake/fallback.gguf");
    let fs = synth_gguf::Builder::new()
        .kv_string("general.architecture", "novel-arch")
        .tensor_f16("some.weight", 1024)
        .into_in_memory_fs(path);

    let mut svc = common::minimal_llama_service("demo", 0);
    common::set_model_path(&mut svc, path);
    let inputs = estimator::EstimatorInputs::from_service(&svc).unwrap();

    match estimator::estimate_from_path(&fs, &inputs) {
        Err(EstimatorError::UnknownArchitecture { architecture }) => {
            assert_eq!(architecture.as_str(), "novel-arch");
        }
        other => panic!("expected UnknownArchitecture; got {other:?}"),
    }
}

/// With `estimation.allow_fallback = true`, an unknown architecture
/// goes through the coarse fallback estimator (formula lives in
/// `ananke::estimator::fallback`).
#[test]
fn fallback_estimator_runs_when_opted_in() {
    let path = Path::new("/fake/fallback.gguf");
    let fs = synth_gguf::Builder::new()
        .kv_string("general.architecture", "novel-arch")
        .tensor_f16("some.weight", 1024)
        .into_in_memory_fs(path);

    let mut svc = common::minimal_llama_service("demo", 0);
    common::set_model_path(&mut svc, path);
    let mut inputs = estimator::EstimatorInputs::from_service(&svc).unwrap();
    inputs.allow_fallback = true;
    let est = estimator::estimate_from_path(&fs, &inputs).unwrap();
    assert!(
        est.weights_bytes >= 512 * 1024 * 1024,
        "fallback must reserve at least 512 MB; got {}",
        est.weights_bytes
    );
}
