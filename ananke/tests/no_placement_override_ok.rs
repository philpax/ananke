//! Integration test: a service without a `placement_override` can start when
//! the model path points at a valid GGUF and the snapshot has adequate free
//! memory. The estimator + placement engine selects CPU placement.
#![cfg(feature = "test-fakes")]

mod common;

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use ananke::{
    api::openai,
    config::{PlacementPolicy, ServiceConfig, TemplateConfig},
    devices::{CpuSnapshot, DeviceSnapshot},
    system::Fs,
};
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use common::{build_harness_with_snapshot, synth_gguf};
use tower::util::ServiceExt;

fn service_without_override(model_path: PathBuf) -> ServiceConfig {
    // No placement_override — the estimator + placer will be invoked.
    // CpuOnly so the placer maps everything to the CPU snapshot.
    let mut svc = common::minimal_llama_service("no-override", 0);
    svc.placement_override = BTreeMap::new();
    svc.placement_policy = PlacementPolicy::CpuOnly;
    let TemplateConfig::LlamaCpp(lc) = &mut svc.template_config else {
        unreachable!();
    };
    lc.model = model_path;
    svc
}

#[tokio::test(flavor = "current_thread")]
async fn no_placement_override_chat_succeeds() {
    let model_path = Path::new("/fake/no-override.gguf");
    let gguf_bytes = synth_gguf::Builder::new()
        .kv_string("general.architecture", "qwen3")
        .kv_u32("qwen3.block_count", 2)
        .kv_u32("qwen3.attention.head_count_kv", 4)
        .kv_u32("qwen3.attention.key_length", 128)
        .kv_u32("qwen3.attention.value_length", 128)
        .tensor_f16("blk.0.attn_q.weight", 512 * 1024)
        .tensor_f16("blk.1.attn_q.weight", 512 * 1024)
        .tensor_f16("output.weight", 512 * 1024)
        .tensor_f16("token_embd.weight", 512 * 1024)
        .build();

    let svc = service_without_override(model_path.to_path_buf());

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
    // Seed the GGUF into the harness's shared in-memory filesystem so the
    // supervisor's estimator call finds it when the first request comes in.
    h.fs.write(model_path, &gguf_bytes).unwrap();
    let app = openai::router(h.state.clone());

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
