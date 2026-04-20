use ananke_api::{
    ConfigResponse, DeviceReservation, DeviceSummary, DisableResponse, EnableResponse, Event,
    LogLine, LogsResponse, OneshotRequest, OneshotResponse, ServiceDetail, ServiceSummary,
    StartResponse, StopResponse,
    oneshot::{OneshotAllocation, OneshotDevices},
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
    let mut ananke_metadata = ananke_api::AnankeMetadata::new();
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
        elastic_borrower: None,
        ananke_metadata,
    };
    assert_eq!(v.clone(), roundtrip(v));
}

#[test]
fn start_response_tagged_union() {
    let v = StartResponse::Unavailable {
        reason: "no fit".into(),
    };
    let json = serde_json::to_value(&v).unwrap();
    assert_eq!(
        json,
        serde_json::json!({"status": "unavailable", "reason": "no fit"})
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
        metadata: Default::default(),
    };
    let json = serde_json::to_value(&v).unwrap();
    let obj = json.as_object().unwrap();
    assert!(!obj.contains_key("name"));
    assert!(!obj.contains_key("workdir"));
    assert!(!obj.contains_key("port"));
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
        ananke_metadata: ananke_api::AnankeMetadata::new(),
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
