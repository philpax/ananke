//! Integration test: when a service has `raw.mmproj` set, the estimator adds
//! the mmproj file's tensor bytes to `weights_bytes`.

mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;

use ananke::config::parse::RawService;
use ananke::config::{
    DeviceSlot, Filters, HealthSettings, Lifecycle, PlacementPolicy, ServiceConfig, Template,
};
use ananke::estimator;
use common::synth_gguf;
use smol_str::SmolStr;

fn svc_with_mmproj(model: PathBuf, mmproj: PathBuf) -> ServiceConfig {
    ServiceConfig {
        name: SmolStr::new("mmproj-svc"),
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
        placement_override: {
            let mut m = BTreeMap::new();
            m.insert(DeviceSlot::Cpu, 100);
            m
        },
        placement_policy: PlacementPolicy::CpuOnly,
        idle_timeout_ms: 60_000,
        warming_grace_ms: 100,
        drain_timeout_ms: 1_000,
        extended_stream_drain_ms: 1_000,
        max_request_duration_ms: 5_000,
        filters: Filters::default(),
        raw: RawService {
            name: Some(SmolStr::new("mmproj-svc")),
            template: Some(SmolStr::new("llama-cpp")),
            model: Some(model),
            port: Some(0),
            mmproj: Some(mmproj),
            ..Default::default()
        },
    }
}

#[test]
fn mmproj_bytes_included_in_weights_estimate() {
    // Write the main model GGUF.
    let main_file = synth_gguf::tempfile("mmproj-main");
    synth_gguf::Builder::new()
        .kv_string("general.architecture", "qwen3")
        .kv_u32("qwen3.block_count", 2)
        .kv_u32("qwen3.attention.head_count_kv", 4)
        .kv_u32("qwen3.attention.key_length", 128)
        .kv_u32("qwen3.attention.value_length", 128)
        .tensor_f16("blk.0.attn_q.weight", 512 * 1024)
        .tensor_f16("blk.1.attn_q.weight", 512 * 1024)
        .write_to(main_file.path());

    // Write a separate mmproj GGUF with a known tensor size.
    let mmproj_file = synth_gguf::tempfile("mmproj-proj");
    // 1 MiB of F16 elements = 512 * 1024 elements * 2 bytes each.
    synth_gguf::Builder::new()
        .kv_string("general.architecture", "clip")
        .tensor_f16("mm.0.weight", 512 * 1024)
        .write_to(mmproj_file.path());

    let svc_without_mmproj = {
        let mut s = svc_with_mmproj(
            main_file.path().to_path_buf(),
            mmproj_file.path().to_path_buf(),
        );
        s.raw.mmproj = None;
        s
    };
    let svc_with = svc_with_mmproj(
        main_file.path().to_path_buf(),
        mmproj_file.path().to_path_buf(),
    );

    let est_without = estimator::estimate_from_path(main_file.path(), &svc_without_mmproj).unwrap();
    let est_with = estimator::estimate_from_path(main_file.path(), &svc_with).unwrap();

    assert!(
        est_with.weights_bytes > est_without.weights_bytes,
        "estimate with mmproj ({}) must exceed estimate without ({})",
        est_with.weights_bytes,
        est_without.weights_bytes
    );
}
