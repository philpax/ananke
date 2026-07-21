use ananke_api::{
    config::ConfigResponse,
    devices::list::{DeviceReservation, DeviceSummary},
    internal::{event::Event, log_line::LogLine},
    oneshot::create::{OneshotAllocation, OneshotDevices, OneshotRequest, OneshotResponse},
    services::{
        detail::ServiceDetail, disable::DisableResponse, enable::EnableResponse,
        list::ServiceSummary, logs::LogsResponse, start::StartResponse, stop::StopResponse,
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
        run_id: Some(1),
        pid: Some(1234),
        inflight_count: 0,
        elastic_borrower: None,
        has_mmproj: Some(true),
        modality: ananke_api::shared::modality::Modality::Chat,
        ananke_metadata,
        fit_verdict: None,
        vram_bytes: None,
        last_used_ms: None,
    };
    assert_eq!(v.clone(), roundtrip(v));
}

#[test]
fn start_response_tagged_union() {
    let v = StartResponse::Unavailable {
        error: ApiErrorBody {
            code: ApiErrorCodeSlug::InsufficientVram,
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
                "code": "insufficient_vram",
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
            vram_gb: Some(16.0),
            min_vram_gb: None,
            max_vram_gb: None,
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
        rolling_mean: None,
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
                fit: true,
                attn_max_batch: Some(512),
                runtime_repack: false,
                fit_margins_mib: vec![5120, 5120],
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
        (ApiErrorCodeSlug::InsufficientVram, "insufficient_vram"),
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
fn error_kind_serde() {
    assert_eq!(
        serde_json::to_string(&ApiErrorKind::InvalidRequestError).unwrap(),
        "\"invalid_request_error\""
    );
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
        ApiErrorCodeSlug::InsufficientVram.to_string(),
        "insufficient_vram"
    );
    assert_eq!(
        ApiErrorCodeSlug::InvalidRequest.to_string(),
        "invalid_request_error"
    );
    assert_eq!(ApiErrorKind::ServerError.to_string(), "server_error");
}
