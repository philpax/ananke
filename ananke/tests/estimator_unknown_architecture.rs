//! What happens to a model whose architecture the estimator does not know.
#![cfg(feature = "test-fakes")]

mod common;

use std::path::Path;

use ananke::config::{AllocationMode, Template};
use ananke_estimate::{self as estimator, EstimatorError};
use ananke_gguf::{Architecture, keys};
use common::synth_gguf;

/// An architecture no family covers is refused outright. Estimating one anyway
/// reserved 400 MiB against glm4moe's real 27 GiB, so the error is a typed
/// variant callers can match rather than a plausible number nothing checks.
#[test]
fn an_unrecognised_architecture_is_refused() {
    let path = Path::new("/fake/novel.gguf");
    let fs = synth_gguf::Builder::new()
        .kv_string(keys::ARCHITECTURE, "novel-arch")
        .tensor_f16("some.weight", 1024)
        .into_in_memory_fs(path);

    let mut svc = common::minimal_llama_service("demo", 0);
    common::set_model_path(&mut svc, path);
    let inputs = ananke::config::estimator_inputs(&svc).unwrap();

    match estimator::estimate_from_path(&fs, &inputs) {
        Err(EstimatorError::UnknownArchitecture { architecture }) => {
            assert_eq!(architecture, Architecture::from("novel-arch"));
        }
        other => panic!("expected UnknownArchitecture; got {other:?}"),
    }
}

/// That refusal is only actionable if the operator can declare the reservation
/// themselves, so a llama-cpp service may now carry an allocation mode. It was
/// rejected outright for this template while the weights-only fallback existed.
#[test]
fn a_llama_cpp_service_may_declare_its_own_reservation() {
    let resolved = AllocationMode::from_parts(
        Template::LlamaCpp,
        Some("static"),
        Some(24.0),
        None,
        None,
        0,
    )
    .expect("llama-cpp accepts an explicit reservation");
    assert_eq!(
        resolved,
        AllocationMode::Static {
            reserve_mb: 24 * 1024
        }
    );
}

/// Declaring the mode without the figure it needs is a config error, not a
/// silent zero-byte reservation.
#[test]
fn a_declared_mode_without_its_reservation_is_rejected() {
    let err = AllocationMode::from_parts(Template::LlamaCpp, Some("static"), None, None, None, 0)
        .expect_err("static without reserve_gb must fail");
    assert!(
        err.to_string().contains("reserve_gb"),
        "unhelpful error: {err}"
    );
}
