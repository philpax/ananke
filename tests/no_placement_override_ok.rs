//! Integration test: a service without a `placement_override` can start when
//! the model path points at a valid GGUF and the snapshot has adequate free
//! memory. The estimator + placement engine selects CPU placement.

mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;

use ananke::config::parse::RawService;
use ananke::config::{
    Filters, HealthSettings, Lifecycle, PlacementPolicy, ServiceConfig, Template,
};
use ananke::devices::{CpuSnapshot, DeviceSnapshot};
use ananke::openai_api;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::{build_harness_with_snapshot, synth_gguf};
use smol_str::SmolStr;
use tower::util::ServiceExt;

fn service_without_override(model_path: PathBuf) -> ServiceConfig {
    // No placement_override — the estimator + placer will be invoked.
    // CpuOnly so the placer maps everything to the CPU snapshot.
    ServiceConfig {
        name: SmolStr::new("no-override"),
        template: Template::LlamaCpp,
        port: 0,
        private_port: 0,
        lifecycle: Lifecycle::OnDemand,
        priority: 50,
        health: HealthSettings {
            http_path: "/health".into(),
            timeout_ms: 5_000,
            probe_interval_ms: 200,
        },
        placement_override: BTreeMap::new(),
        placement_policy: PlacementPolicy::CpuOnly,
        idle_timeout_ms: 60_000,
        warming_grace_ms: 100,
        drain_timeout_ms: 1_000,
        extended_stream_drain_ms: 1_000,
        max_request_duration_ms: 5_000,
        filters: Filters::default(),
        raw: RawService {
            name: Some(SmolStr::new("no-override")),
            template: Some(SmolStr::new("llama-cpp")),
            model: Some(model_path),
            port: Some(0),
            ..Default::default()
        },
    }
}

#[tokio::test(flavor = "current_thread")]
async fn no_placement_override_chat_succeeds() {
    // Write a synthetic GGUF so `estimate_from_path` has something to read.
    let file = synth_gguf::tempfile("no-override");
    synth_gguf::Builder::new()
        .kv_string("general.architecture", "qwen3")
        .kv_u32("qwen3.block_count", 2)
        .kv_u32("qwen3.attention.head_count_kv", 4)
        .kv_u32("qwen3.attention.key_length", 128)
        .kv_u32("qwen3.attention.value_length", 128)
        .tensor_f16("blk.0.attn_q.weight", 512 * 1024)
        .tensor_f16("blk.1.attn_q.weight", 512 * 1024)
        .tensor_f16("output.weight", 512 * 1024)
        .tensor_f16("token_embd.weight", 512 * 1024)
        .write_to(file.path());

    let model_path = file.path().to_path_buf();
    let svc = service_without_override(model_path);

    // Seed a CPU-only snapshot with plenty of free bytes so the placer succeeds.
    let snapshot = DeviceSnapshot {
        gpus: vec![],
        cpu: Some(CpuSnapshot {
            total_bytes: 64 * 1024 * 1024 * 1024,
            available_bytes: 64 * 1024 * 1024 * 1024,
        }),
        taken_at_ms: 0,
    };

    let h = build_harness_with_snapshot(vec![svc], snapshot).await;
    let app = openai_api::router(h.state.clone());

    let body = r#"{"model":"no-override","messages":[{"role":"user","content":"hi"}]}"#;
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    h.cleanup().await;
}
