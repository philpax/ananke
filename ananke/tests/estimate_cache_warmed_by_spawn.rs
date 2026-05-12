//! Integration test: the supervisor's spawn-time estimator run
//! populates the daemon-wide `EstimateCache`, so a subsequent
//! `GET /api/services/{name}` call gets a cache hit and doesn't have
//! to re-parse the GGUF.
//!
//! Proof technique: spawn a service against a synthetic GGUF, then
//! *remove the GGUF from the in-memory filesystem* and call the
//! management detail endpoint. If the cache wasn't warmed, the
//! handler would try to read the (now-missing) file and surface
//! `model_info: null` / `estimate: null`. If it was, both fields are
//! populated from the cached entry the supervisor stamped at spawn.
#![cfg(feature = "test-fakes")]

mod common;

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use ananke::{
    api::{management, openai},
    config::{PlacementPolicy, ServiceConfig, TemplateConfig},
    devices::{CpuSnapshot, DeviceSnapshot},
    system::Fs,
};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use common::{build_harness_with_snapshot, synth_gguf};
use tower::util::ServiceExt;

fn service(model_path: PathBuf) -> ServiceConfig {
    let mut svc = common::minimal_llama_service("warmth", 0);
    svc.placement_override = BTreeMap::new();
    svc.placement_policy = PlacementPolicy::CpuOnly;
    let TemplateConfig::LlamaCpp(lc) = &mut svc.template_config else {
        unreachable!();
    };
    lc.model = model_path;
    svc
}

#[tokio::test(flavor = "current_thread")]
async fn spawn_warms_estimate_cache_for_detail_endpoint() {
    let model_path = Path::new("/fake/warmth.gguf");
    let gguf_bytes = synth_gguf::Builder::new()
        .kv_string("general.architecture", "qwen3")
        .kv_string("general.name", "Warmth Test Model")
        .kv_u32("qwen3.block_count", 2)
        .kv_u32("qwen3.context_length", 8192)
        .kv_u32("qwen3.attention.head_count_kv", 4)
        .kv_u32("qwen3.attention.key_length", 128)
        .kv_u32("qwen3.attention.value_length", 128)
        .tensor_f16("blk.0.attn_q.weight", 512 * 1024)
        .tensor_f16("blk.1.attn_q.weight", 512 * 1024)
        .tensor_f16("output.weight", 512 * 1024)
        .tensor_f16("token_embd.weight", 512 * 1024)
        .build();

    let snapshot = DeviceSnapshot {
        gpus: vec![],
        cpu: Some(CpuSnapshot {
            total_bytes: 64 * 1024 * 1024 * 1024,
            available_bytes: 64 * 1024 * 1024 * 1024,
        }),
        taken_at_ms: 0,
    };

    let h = build_harness_with_snapshot(vec![service(model_path.to_path_buf())], snapshot).await;
    h.fs.write(model_path, &gguf_bytes).unwrap();

    // Step 1: drive a chat through the OpenAI router. This runs the
    // supervisor through Ensure → estimator → packer → spawn, which
    // is exactly the path that should warm the cache.
    let openai = openai::router(h.state.clone());
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"warmth","messages":[{"role":"user","content":"hi"}]}"#,
        ))
        .unwrap();
    let resp = openai.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "service must start before we can assert anything about its cache entry"
    );

    // Step 2: pull the GGUF out from under the daemon's feet. If the
    // detail handler still has to read it, the assertions below will
    // see `model_info: null` and fail.
    h.fs.remove_file(model_path).unwrap();

    // Step 3: ask the management API for the service's detail.
    let mgmt = management::router(h.state.clone());
    let req = Request::builder()
        .method("GET")
        .uri("/api/services/warmth")
        .body(Body::empty())
        .unwrap();
    let resp = mgmt.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

    assert!(
        !parsed["model_info"].is_null(),
        "model_info must be present from the cache even though the GGUF is gone: {parsed}"
    );
    assert_eq!(
        parsed["model_info"]["architecture"], "qwen3",
        "cache must carry the architecture the supervisor measured"
    );
    assert_eq!(
        parsed["model_info"]["model_name"], "Warmth Test Model",
        "cache must carry general.name read by the supervisor's pass"
    );
    assert_eq!(parsed["model_info"]["block_count"], 2);
    assert!(
        !parsed["estimate"].is_null(),
        "estimate must be present from the cache: {parsed}"
    );
    assert_eq!(parsed["estimate"]["configured_context"], 4096);

    h.cleanup().await;
}
