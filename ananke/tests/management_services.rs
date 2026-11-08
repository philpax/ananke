//! Integration tests for the management API's service endpoints.
#![cfg(feature = "test-fakes")]

mod common;

use ananke::config::SplitMode;
use ananke_config::validate::{
    OpenAiProxyConfig, TemplateConfig, test_fixtures::minimal_command_service,
};
use axum::{body::to_bytes, http::StatusCode};
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn api_services_lists_registered() {
    let h = build_harness(vec![
        minimal_llama_service("alpha", 0),
        minimal_llama_service("beta", 0),
    ])
    .await;
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
    assert!(parsed["services"].is_array());
    let names: Vec<String> = parsed["services"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["name"].as_str().unwrap().to_string())
        .collect();
    assert!(names.contains(&"alpha".to_string()));
    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn api_service_detail_by_name() {
    let h = build_harness(vec![minimal_llama_service("alpha", 12345)]).await;
    let app = ananke::api::management::router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("GET")
        .uri("/api/services/alpha")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["name"], "alpha");
    assert_eq!(parsed["port"], 12345);
    h.cleanup().await;
}

async fn get_json(h: &common::TestHarness, uri: &str) -> serde_json::Value {
    let app = ananke::api::management::router(h.state.clone());
    let req = axum::http::Request::builder()
        .uri(uri)
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test(flavor = "current_thread")]
async fn detail_reports_a_tensor_split_with_its_weights() {
    let mut svc = minimal_llama_service("split", 0);
    svc.gpu_allow = vec![0, 1];
    svc.split_mode = SplitMode::Tensor;
    svc.tensor_split_weights = Some(vec![1.0, 2.0]);
    let h = build_harness(vec![svc]).await;
    let detail = get_json(&h, "/api/services/split").await;
    let placement = &detail["placement"];
    assert_eq!(placement["gpu_allow"], serde_json::json!([0, 1]));
    assert_eq!(placement["split"]["mode"], "tensor");
    assert_eq!(
        placement["split"]["tensor_split_weights"],
        serde_json::json!([1.0, 2.0])
    );
    assert_eq!(placement["allocation"]["kind"], "estimated");
    assert!(detail.get("command").is_none());
    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn openai_compat_and_command_settings_reach_the_wire() {
    let mut proxied = minimal_command_service("proxied", vec!["serve".into()]);
    proxied.openai_compat = true;
    proxied.description = Some("An image model.".into());
    if let TemplateConfig::Command(cmd) = &mut proxied.template_config {
        cmd.openai_proxy = Some(OpenAiProxyConfig {
            upstream_model: "upstream-name".into(),
        });
    }
    let plain = minimal_command_service("plain", vec!["serve".into()]);
    let h = build_harness(vec![proxied, plain]).await;

    let list = get_json(&h, "/api/services").await;
    let row = |name: &str| {
        list["services"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["name"] == name)
            .unwrap()
            .clone()
    };
    assert_eq!(row("proxied")["openai_compat"], true);
    assert_eq!(row("proxied")["description"], "An image model.");
    assert_eq!(row("plain")["openai_compat"], false);
    assert!(row("plain").get("description").is_none());

    let detail = get_json(&h, "/api/services/proxied").await;
    assert_eq!(detail["command"]["upstream_model"], "upstream-name");
    assert_eq!(detail["command"]["private_port_fixed"], false);
    assert!(detail["placement"].get("split").is_none());
    h.cleanup().await;
}
