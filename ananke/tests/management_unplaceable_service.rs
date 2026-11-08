//! Integration test: how `GET /api/services` reports a service that cannot be
//! placed at all.
//!
//! The `deepseek-v4-flash` case: the model is far too large for the host, so
//! every pack fails and the placement preview names no devices. Summing that
//! empty map gives `0 B`, which reads as "this service needs no memory" —
//! indistinguishable from a genuinely CPU-only service or one that has never
//! been estimated. The row must instead report the model's aggregate demand,
//! and the verdict must name the device that came up short.
#![cfg(feature = "test-fakes")]

mod common;

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use ananke::config::{PlacementPolicy, ServiceConfig, TemplateConfig};
use ananke_placement::devices::{CpuSnapshot, DeviceSnapshot, GpuSnapshot};
use ananke_system::Fs;
use axum::{body::to_bytes, http::StatusCode};
use common::{build_harness_with_snapshot, synth_gguf};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn external_gpu_pressure_is_not_reported_as_achievable_eviction() {
    let gib = 1024 * 1024 * 1024;
    let model_path = Path::new("/fake/external-pressure.gguf");
    let gguf_bytes = synth_gguf::Builder::new()
        .kv_string("general.architecture", "qwen3")
        .kv_u32("qwen3.block_count", 2)
        .kv_u32("qwen3.attention.head_count_kv", 4)
        .kv_u32("qwen3.attention.key_length", 128)
        .kv_u32("qwen3.attention.value_length", 128)
        .tensor_f16("blk.0.attn_q.weight", 128 * 1024 * 1024)
        .tensor_f16("blk.1.attn_q.weight", 128 * 1024 * 1024)
        .tensor_f16("output.weight", gib)
        .tensor_f16("token_embd.weight", 1024)
        .build();
    let mut service = oversized_service(model_path.to_path_buf());
    service.placement_policy = PlacementPolicy::GpuOnly;
    let snapshot = DeviceSnapshot {
        gpus: vec![
            GpuSnapshot {
                id: 0,
                name: "GPU 0".into(),
                total_bytes: 24 * gib,
                free_bytes: gib,
            },
            GpuSnapshot {
                id: 1,
                name: "GPU 1".into(),
                total_bytes: 24 * gib,
                free_bytes: 24 * gib,
            },
        ],
        cpu: Some(CpuSnapshot {
            total_bytes: 128 * gib,
            available_bytes: 64 * gib,
        }),
        taken_at_ms: 0,
    };
    let h = build_harness_with_snapshot(vec![service], snapshot).await;
    h.fs.write(model_path, &gguf_bytes).unwrap();
    for uri in ["/api/services", "/api/services/too-big"] {
        let app = ananke::api::management::router(h.state.clone());
        let request = axum::http::Request::builder()
            .uri(uri)
            .body(axum::body::Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let service = if uri == "/api/services" {
            &parsed["services"][0]
        } else {
            &parsed
        };
        let verdict = if uri == "/api/services" {
            &service["fit_verdict"]
        } else {
            &service["placement_preview"]["verdict"]
        };
        assert_eq!(verdict["kind"], "does_not_fit", "{uri}: {service}");
        let shortfalls = verdict["shortfalls"].as_array().unwrap();
        assert!(
            shortfalls.iter().any(|shortfall| {
                shortfall["device"] == "gpu:0"
                    && shortfall["requested_bytes"].as_u64().unwrap()
                        > shortfall["available_bytes"].as_u64().unwrap()
            }),
            "{uri}: {shortfalls:?}"
        );
    }
    h.cleanup().await;
}

fn oversized_service(model_path: PathBuf) -> ServiceConfig {
    let mut svc = common::minimal_llama_service("too-big", 0);
    // No override, so the estimator and packer actually run.
    svc.placement_override = BTreeMap::new();
    svc.placement_policy = PlacementPolicy::CpuOnly;
    let TemplateConfig::LlamaCpp(lc) = &mut svc.template_config else {
        unreachable!();
    };
    lc.model = model_path;
    svc
}

#[tokio::test(flavor = "current_thread")]
async fn unplaceable_service_reports_its_demand_and_names_the_short_device() {
    let model_path = Path::new("/fake/too-big.gguf");
    // Two layers of ~8 GiB each (f16, so elements × 2 bytes) against a host
    // with 1 GiB of RAM: nothing fits, on empty hardware or otherwise.
    let layer_elements = 4 * 1024 * 1024 * 1024;
    let gguf_bytes = synth_gguf::Builder::new()
        .kv_string("general.architecture", "qwen3")
        .kv_u32("qwen3.block_count", 2)
        .kv_u32("qwen3.attention.head_count_kv", 4)
        .kv_u32("qwen3.attention.key_length", 128)
        .kv_u32("qwen3.attention.value_length", 128)
        .tensor_f16("blk.0.attn_q.weight", layer_elements)
        .tensor_f16("blk.1.attn_q.weight", layer_elements)
        .tensor_f16("output.weight", 1024)
        .tensor_f16("token_embd.weight", 1024)
        .build();

    let snapshot = DeviceSnapshot {
        gpus: vec![],
        cpu: Some(CpuSnapshot {
            total_bytes: 1024 * 1024 * 1024,
            available_bytes: 1024 * 1024 * 1024,
        }),
        taken_at_ms: 0,
    };

    let h =
        build_harness_with_snapshot(vec![oversized_service(model_path.to_path_buf())], snapshot)
            .await;
    h.fs.write(model_path, &gguf_bytes).unwrap();

    let app = ananke::api::management::router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("GET")
        .uri("/api/services")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let svc = &parsed["services"][0];

    assert_eq!(svc["fit_verdict"]["kind"], "does_not_fit", "got {svc}");

    // Host RAM is the binding constraint — there are no GPUs at all — so `cpu`
    // is the device named, with an id that cross-references `GET /api/devices`.
    let shortfalls = svc["fit_verdict"]["shortfalls"]
        .as_array()
        .unwrap_or_else(|| panic!("expected shortfalls, got {svc}"));
    assert_eq!(shortfalls.len(), 1, "got {shortfalls:?}");
    assert_eq!(shortfalls[0]["device"], "cpu");
    let requested = shortfalls[0]["requested_bytes"].as_u64().unwrap();
    let available = shortfalls[0]["available_bytes"].as_u64().unwrap();
    assert!(
        requested > available,
        "a shortfall reports less available than requested, got {shortfalls:?}"
    );

    // An unplaceable service must not report 0 B. With no placement to sum,
    // the row falls back to the estimator's demand, which is at least the
    // ~16 GiB of weights.
    let footprint = svc["footprint_bytes"]
        .as_u64()
        .unwrap_or_else(|| panic!("expected a footprint, got {svc}"));
    assert!(
        footprint >= 16 * 1024 * 1024 * 1024,
        "the fallback reports the model's aggregate demand, got {footprint}"
    );

    // The row shows the total and the breakdown together, so they have to be the
    // same figure. They are computed once and summed rather than derived twice —
    // this is the assertion that keeps it that way, and it runs on the fallback
    // path because that is the one where the total does not come from a
    // placement and so could most easily drift from the parts.
    let devices = svc["footprint_devices"]
        .as_array()
        .unwrap_or_else(|| panic!("expected a per-device breakdown, got {svc}"));
    assert!(!devices.is_empty(), "the demand names its devices");
    let summed: u64 = devices
        .iter()
        .map(|d| {
            d["bytes"]
                .as_u64()
                .unwrap_or_else(|| panic!("a device's share is a byte count, got {d}"))
        })
        .sum();
    assert_eq!(
        summed, footprint,
        "the breakdown sums to the total, got {devices:?}"
    );
    assert!(
        devices.iter().any(|d| d["device"] == "cpu"),
        "with no GPUs the demand lands on the host, got {devices:?}"
    );

    h.cleanup().await;
}
