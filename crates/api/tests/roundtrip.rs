use ananke_api::{
    config::ConfigResponse,
    devices::list::{DeviceReservation, DeviceSummary},
    internal::{event::Event, log_line::LogLine},
    oneshot::create::{OneshotAllocation, OneshotDevices, OneshotRequest, OneshotResponse},
    services::{
        command::{EnvVar, LaunchCommand, LaunchCommandSource},
        detail::ServiceDetail,
        disable::DisableResponse,
        enable::EnableResponse,
        list::{DeviceFootprint, ServiceSummary},
        logs::LogsResponse,
        start::StartResponse,
        stop::StopResponse,
    },
    shared::errors::{ApiErrorBody, ApiErrorCodeSlug, ApiErrorKind},
};
use pretty_assertions::assert_eq;
use smol_str::SmolStr;

fn roundtrip<T>(value: T) -> T
where
    T: serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    let json = serde_json::to_string(&value).expect("serialize");
    serde_json::from_str(&json).expect("deserialize")
}

#[test]
fn service_summary_roundtrips() {
    let mut ananke_metadata = ananke_api::shared::metadata::AnankeMetadata::new();
    ananke_metadata.insert("tags".into(), serde_json::json!(["general", "chat"]));
    ananke_metadata.insert("discord_visible".into(), serde_json::json!(true));
    let v = ServiceSummary {
        name: "demo".into(),
        state: "running".into(),
        lifecycle: "persistent".into(),
        priority: 50,
        port: 11435,
        openai_compat: true,
        web_ui_url: Some("https://models.example/demo/".into()),
        run_id: Some(1),
        pid: Some(1234),
        inflight_count: 0,
        elastic_borrower: None,
        has_mmproj: Some(true),
        modality: ananke_api::shared::modality::Modality::Chat,
        ananke_metadata,
        fit_verdict: None,
        footprint_bytes: Some(7_516_192_768),
        // Populated rather than empty: the field is elided from JSON when it is,
        // so an empty vec would round-trip through a document that never carried
        // it and prove nothing about the shape.
        footprint_devices: vec![
            DeviceFootprint {
                device: "gpu:0".into(),
                bytes: 4_294_967_296,
            },
            DeviceFootprint {
                device: "cpu".into(),
                bytes: 3_221_225_472,
            },
        ],
        last_used_ms: None,
    };
    assert_eq!(v.clone(), roundtrip(v));
}

#[test]
fn start_response_tagged_union() {
    let v = StartResponse::Unavailable {
        error: ApiErrorBody {
            code: ApiErrorCodeSlug::InsufficientCapacity,
            message: "no fit".into(),
            kind: ApiErrorKind::ServerError,
        },
    };
    let json = serde_json::to_value(&v).unwrap();
    assert_eq!(
        json,
        serde_json::json!({
            "status": "unavailable",
            "error": {
                "code": "insufficient_capacity",
                "message": "no fit",
                "type": "server_error",
            }
        })
    );
}

#[test]
fn event_state_changed_tag() {
    let v = Event::StateChanged {
        service: SmolStr::new("demo"),
        from: "idle".into(),
        to: "starting".into(),
        at_ms: 1,
    };
    let json = serde_json::to_value(&v).unwrap();
    assert_eq!(json["type"], "state_changed");
    assert_eq!(json["service"], "demo");
}

#[test]
fn oneshot_request_optional_fields_omitted() {
    let v = OneshotRequest {
        name: None,
        template: "command".into(),
        command: Some(vec!["python".into(), "batch.py".into()]),
        workdir: None,
        allocation: OneshotAllocation {
            mode: Some("static".into()),
            reserve_gb: Some(16.0),
            min_reserve_gb: None,
            max_reserve_gb: None,
        },
        devices: Some(OneshotDevices {
            placement: Some("gpu-only".into()),
        }),
        priority: Some(40),
        ttl: Some("2h".into()),
        port: None,
        health: None,
        metadata: Default::default(),
    };
    let json = serde_json::to_value(&v).unwrap();
    let obj = json.as_object().unwrap();
    assert!(!obj.contains_key("name"));
    assert!(!obj.contains_key("workdir"));
    assert!(!obj.contains_key("port"));
    assert!(!obj.contains_key("health"));
    assert!(!obj.contains_key("metadata"));
}

#[test]
fn logs_response_roundtrips() {
    let v = LogsResponse {
        logs: vec![LogLine {
            timestamp_ms: 1,
            stream: "stdout".into(),
            line: "hello".into(),
            run_id: 1,
            seq: 1,
        }],
        next_cursor: None,
    };
    assert_eq!(v.clone(), roundtrip(v));
}

#[test]
fn config_response_roundtrips() {
    let v = ConfigResponse {
        content: "[daemon]\n".into(),
        hash: "abc".into(),
        writable: true,
    };
    assert_eq!(v.clone(), roundtrip(v));
}

#[test]
fn device_summary_roundtrips() {
    let v = DeviceSummary {
        id: "gpu:0".into(),
        name: "RTX 3090".into(),
        total_bytes: 1 << 34,
        free_bytes: 1 << 33,
        reservations: vec![DeviceReservation {
            service: "demo".into(),
            bytes: 1 << 30,
            elastic: false,
        }],
    };
    assert_eq!(v.clone(), roundtrip(v));
}

#[test]
fn service_detail_roundtrips() {
    let v = ServiceDetail {
        name: "demo".into(),
        state: "idle".into(),
        lifecycle: "persistent".into(),
        priority: 50,
        port: 11435,
        private_port: 40000,
        template: "llamacpp".into(),
        placement_override: Default::default(),
        idle_timeout_ms: 600_000,
        run_id: None,
        pid: None,
        recent_logs: vec![],
        recent_restarts: vec![],
        rolling_mean: None,
        rolling_mean_host: None,
        rolling_samples_host: 0,
        rolling_samples: 0,
        observed_peak_bytes: 0,
        elastic_borrower: None,
        model_info: None,
        estimate: None,
        placement_preview: None,
        current_allocation: Default::default(),
        modality: ananke_api::shared::modality::Modality::Chat,
        ananke_metadata: ananke_api::shared::metadata::AnankeMetadata::new(),
        last_used_ms: None,
        runtime: Some(ananke_api::services::detail::RuntimeInfo {
            kind: "ik-llama".into(),
            ik: Some(ananke_api::services::detail::IkParams {
                mla: Some(1),
                dsa: true,
                attn_max_batch: Some(512),
                runtime_repack: false,
            }),
        }),
        serving: Some(ananke_api::services::detail::ServingConfig {
            binary: "/bin/ik-llama-server".into(),
            cache_type_k: "f16".into(),
            cache_type_v: "f16".into(),
            flash_attn: false,
            parallel: 2,
            kv_unified: false,
            effective_context_per_slot: Some(65536),
            spec_type: None,
            draft_model: None,
            expert_offload: "off".into(),
            batch_size: Some(2048),
            ubatch_size: Some(2048),
            threads: Some(24),
            threads_batch: None,
            numa: None,
            mmap: false,
            mlock: false,
        }),
        container: Some(ananke_api::services::detail::ContainerDetail {
            runtime: "docker".into(),
            image: "ghcr.io/ggml-org/llama.cpp:server-cuda".into(),
            network: "host".into(),
            container_id: Some("abc123".into()),
            container_name: Some("ananke-demo-1".into()),
        }),
    };
    assert_eq!(v.clone(), roundtrip(v));
}

#[test]
fn service_detail_roundtrips_mainline() {
    let v = ServiceDetail {
        name: "demo".into(),
        state: "idle".into(),
        lifecycle: "persistent".into(),
        priority: 50,
        port: 11435,
        private_port: 40000,
        template: "llamacpp".into(),
        placement_override: Default::default(),
        idle_timeout_ms: 600_000,
        run_id: None,
        pid: None,
        recent_logs: vec![],
        recent_restarts: vec![],
        rolling_mean: None,
        rolling_mean_host: None,
        rolling_samples_host: 0,
        rolling_samples: 0,
        observed_peak_bytes: 0,
        elastic_borrower: None,
        model_info: None,
        estimate: None,
        placement_preview: None,
        current_allocation: Default::default(),
        modality: ananke_api::shared::modality::Modality::Chat,
        ananke_metadata: ananke_api::shared::metadata::AnankeMetadata::new(),
        last_used_ms: None,
        runtime: Some(ananke_api::services::detail::RuntimeInfo {
            kind: "llama-cpp".into(),
            ik: None,
        }),
        serving: None,
        container: None,
    };
    assert_eq!(v.clone(), roundtrip(v));
}

#[test]
fn service_detail_roundtrips_no_runtime() {
    let v = ServiceDetail {
        name: "demo".into(),
        state: "idle".into(),
        lifecycle: "persistent".into(),
        priority: 50,
        port: 11435,
        private_port: 40000,
        template: "command".into(),
        placement_override: Default::default(),
        idle_timeout_ms: 600_000,
        run_id: None,
        pid: None,
        recent_logs: vec![],
        recent_restarts: vec![],
        rolling_mean: None,
        rolling_mean_host: None,
        rolling_samples_host: 0,
        rolling_samples: 0,
        observed_peak_bytes: 0,
        elastic_borrower: None,
        model_info: None,
        estimate: None,
        placement_preview: None,
        current_allocation: Default::default(),
        modality: ananke_api::shared::modality::Modality::Chat,
        ananke_metadata: ananke_api::shared::metadata::AnankeMetadata::new(),
        last_used_ms: None,
        runtime: None,
        serving: None,
        container: Some(ananke_api::services::detail::ContainerDetail {
            runtime: "podman".into(),
            image: "ninfer:local".into(),
            network: "host".into(),
            container_id: None,
            container_name: None,
        }),
    };
    assert_eq!(v.clone(), roundtrip(v));
}

#[test]
fn oneshot_response_roundtrips() {
    let v = OneshotResponse {
        id: "oneshot_01H".into(),
        name: "sd-batch".into(),
        port: 18001,
        logs_url: "/api/oneshot/oneshot_01H/logs/stream".into(),
    };
    assert_eq!(v.clone(), roundtrip(v));
}

#[test]
fn stop_response_tagged_union() {
    let v = StopResponse::Drained;
    let json = serde_json::to_value(&v).unwrap();
    assert_eq!(json, serde_json::json!({"status": "drained"}));
}

#[test]
fn enable_response_tagged_union() {
    let v = EnableResponse::NotDisabled;
    let json = serde_json::to_value(&v).unwrap();
    assert_eq!(json, serde_json::json!({"status": "not_disabled"}));
}

#[test]
fn disable_response_tagged_union() {
    let v = DisableResponse::Disabled;
    let json = serde_json::to_value(&v).unwrap();
    assert_eq!(json, serde_json::json!({"status": "disabled"}));
}

#[test]
fn error_slug_serde_roundtrips() {
    // The 15 named variants serialise to their snake_case slugs.
    let cases = [
        (ApiErrorCodeSlug::ModelNotFound, "model_not_found"),
        (ApiErrorCodeSlug::ServiceNotFound, "service_not_found"),
        (ApiErrorCodeSlug::ServiceDisabled, "service_disabled"),
        (ApiErrorCodeSlug::StartQueueFull, "start_queue_full"),
        (ApiErrorCodeSlug::StartFailed, "start_failed"),
        (
            ApiErrorCodeSlug::InsufficientCapacity,
            "insufficient_capacity",
        ),
        (ApiErrorCodeSlug::ServiceBlocked, "service_blocked"),
        (
            ApiErrorCodeSlug::UpstreamUnavailable,
            "upstream_unavailable",
        ),
        (ApiErrorCodeSlug::ProxyInternal, "proxy_internal"),
        (ApiErrorCodeSlug::NotImplemented, "not_implemented"),
        (ApiErrorCodeSlug::InvalidCursor, "invalid_cursor"),
        (ApiErrorCodeSlug::IfMatchRequired, "if_match_required"),
        (ApiErrorCodeSlug::HashMismatch, "hash_mismatch"),
        (ApiErrorCodeSlug::PersistFailed, "persist_failed"),
    ];
    for (variant, expected) in cases {
        let json = serde_json::to_string(&variant).unwrap();
        assert_eq!(json, format!("\"{expected}\""));
        let back: ApiErrorCodeSlug = serde_json::from_str(&json).unwrap();
        assert_eq!(back, variant);
    }
}

#[test]
fn error_slug_invalid_request_renamed() {
    // InvalidRequest serialises as "invalid_request_error" (not
    // "invalid_request") to match OpenAI's error-type taxonomy and
    // preserve wire compatibility with the daemon's existing slug.
    let json = serde_json::to_string(&ApiErrorCodeSlug::InvalidRequest).unwrap();
    assert_eq!(json, "\"invalid_request_error\"");
    let back: ApiErrorCodeSlug = serde_json::from_str(&json).unwrap();
    assert_eq!(back, ApiErrorCodeSlug::InvalidRequest);
}

#[test]
fn error_slug_other_fallback() {
    // An unknown slug deserialises to `Other` so clients don't break
    // when the daemon adds a new code before they're updated.
    let back: ApiErrorCodeSlug = serde_json::from_str("\"totally_new_error\"").unwrap();
    assert_eq!(back, ApiErrorCodeSlug::Other);
}

#[test]
fn process_launch_command_json_is_byte_identical() {
    // The process variant of `LaunchCommand` must keep producing exactly
    // the four pre-container fields so existing consumers see no change.
    let v = LaunchCommand {
        source: LaunchCommandSource::Preview,
        argv: vec!["llama-server".into(), "-m".into(), "/models/x.gguf".into()],
        env: vec![EnvVar {
            key: "CUDA_VISIBLE_DEVICES".into(),
            value: "0".into(),
        }],
        env_inherit: true,
        container: None,
    };
    let json = serde_json::to_value(&v).unwrap();
    assert_eq!(
        json,
        serde_json::json!({
            "source": "preview",
            "argv": ["llama-server", "-m", "/models/x.gguf"],
            "env": [{"key": "CUDA_VISIBLE_DEVICES", "value": "0"}],
            "env_inherit": true
        })
    );
    assert_eq!(v.clone(), roundtrip(v));
}

#[test]
fn launch_preview_wire_union_roundtrips() {
    let v = LaunchCommand {
        source: LaunchCommandSource::Running,
        argv: vec!["--model".into(), "nvidia/diffusiongemma".into()],
        env: vec![],
        env_inherit: false,
        container: Some(ananke_api::services::command::LaunchContainer {
            runtime: "docker".into(),
            image: "vllm/vllm-openai:v0.26.0".into(),
            name_pattern: "ananke-diffusiongemma-<run-id>".into(),
            argv: vec!["--model".into(), "nvidia/diffusiongemma".into()],
            env: vec![],
            env_passthrough: vec!["HF_TOKEN".into()],
            mounts: vec![ananke_api::services::command::LaunchMount {
                source: "/cache".into(),
                target: "/root/.cache".into(),
                read_only: false,
            }],
            network: "bridge".into(),
            publication: Some("127.0.0.1:40000:8000".into()),
            ipc: "host".into(),
            gpu_devices: vec!["nvidia.com/gpu=0".into()],
            create_argv: vec![
                "docker".into(),
                "create".into(),
                "--name".into(),
                "ananke-diffusiongemma-0".into(),
                "vllm/vllm-openai:v0.26.0".into(),
                "--model".into(),
                "nvidia/diffusiongemma".into(),
            ],
        }),
    };
    assert_eq!(v.clone(), roundtrip(v));

    // The union is discriminated by the presence of `container`, so the
    // process variant must not gain the key even after a container variant
    // has been serialised in the same process.
    let process = LaunchCommand {
        source: LaunchCommandSource::Preview,
        argv: vec!["llama-server".into()],
        env: vec![],
        env_inherit: true,
        container: None,
    };
    let json = serde_json::to_value(&process).unwrap();
    assert!(
        json.get("container").is_none(),
        "the process variant must stay byte-identical: {json}"
    );
    assert_eq!(process.clone(), roundtrip(process));
}

#[test]
fn container_preview_redacts_passthrough_secrets() {
    // `env_passthrough` carries variable names only. A resolved value here
    // would leak a token into the management API and into any UI that
    // renders a copy-pasteable command.
    let secret = "hf_thisisnotarealtokenbutlooksenough";
    // SAFETY: single-threaded test setup; the variable is read only through
    // the assertion below.
    unsafe { std::env::set_var("HF_TOKEN", secret) };

    let v = LaunchCommand {
        source: LaunchCommandSource::Preview,
        argv: vec!["--model".into(), "x".into()],
        env: vec![],
        env_inherit: false,
        container: Some(ananke_api::services::command::LaunchContainer {
            runtime: "docker".into(),
            image: "vllm/vllm-openai:v0.26.0".into(),
            name_pattern: "ananke-vllm-<run-id>".into(),
            argv: vec!["--model".into(), "x".into()],
            env: vec![],
            env_passthrough: vec!["HF_TOKEN".into()],
            mounts: vec![],
            network: "host".into(),
            publication: None,
            ipc: "private".into(),
            gpu_devices: vec![],
            create_argv: vec![
                "docker".into(),
                "create".into(),
                "-e".into(),
                // The create argv passes the name alone; the runtime reads
                // the value from the daemon's own environment.
                "HF_TOKEN".into(),
                "vllm/vllm-openai:v0.26.0".into(),
            ],
        }),
    };

    let json = serde_json::to_string(&v).unwrap();
    assert!(json.contains("HF_TOKEN"), "the name is shown");
    assert!(
        !json.contains(secret),
        "a passthrough value must never reach the wire: {json}"
    );
    assert_eq!(v.clone(), roundtrip(v));
}

#[test]
fn service_detail_container_wire_roundtrips() {
    use ananke_api::services::detail::ContainerDetail;

    // Running: the live identity is present.
    let running = ContainerDetail {
        runtime: "podman".into(),
        image: "ghcr.io/ggml-org/llama.cpp:server-cuda".into(),
        network: "host".into(),
        container_id: Some("abc123".into()),
        container_name: Some("ananke-muse-glimmer-42".into()),
    };
    assert_eq!(running.clone(), roundtrip(running.clone()));
    let json = serde_json::to_value(&running).unwrap();
    assert_eq!(json["runtime"], "podman");
    assert_eq!(json["container_id"], "abc123");

    // Idle: the identity fields are absent rather than null, so a client
    // can tell "not running" from "running with no id".
    let idle = ContainerDetail {
        container_id: None,
        container_name: None,
        ..running
    };
    let json = serde_json::to_value(&idle).unwrap();
    assert!(json.get("container_id").is_none());
    assert!(json.get("container_name").is_none());
    assert_eq!(idle.clone(), roundtrip(idle));
}

#[test]
fn error_kind_serde() {
    assert_eq!(
        serde_json::to_string(&ApiErrorKind::ServerError).unwrap(),
        "\"server_error\""
    );
    let back: ApiErrorKind = serde_json::from_str("\"invalid_request_error\"").unwrap();
    assert_eq!(back, ApiErrorKind::InvalidRequestError);
    let back: ApiErrorKind = serde_json::from_str("\"server_error\"").unwrap();
    assert_eq!(back, ApiErrorKind::ServerError);
    // Forward-compat fallback.
    let back: ApiErrorKind = serde_json::from_str("\"unknown_kind\"").unwrap();
    assert_eq!(back, ApiErrorKind::Other);
}

#[test]
fn error_slug_display_matches_serialisation() {
    // Display must yield the bare slug string (no quotes) so
    // anankectl's `println!("{}", error.code)` keeps working.
    assert_eq!(
        ApiErrorCodeSlug::InsufficientCapacity.to_string(),
        "insufficient_capacity"
    );
    assert_eq!(
        ApiErrorCodeSlug::InvalidRequest.to_string(),
        "invalid_request_error"
    );
    assert_eq!(ApiErrorKind::ServerError.to_string(), "server_error");
}
