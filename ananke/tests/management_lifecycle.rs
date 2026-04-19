//! Integration tests for the management lifecycle endpoints.

mod common;

use ananke_api::{DisableResponse, EnableResponse, StartResponse};
use axum::{body::to_bytes, http::StatusCode};
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn start_on_idle_service_returns_accepted() {
    let h = build_harness(vec![minimal_llama_service("demo", 0)]).await;
    let app = ananke::api::management::router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/api/services/demo/start")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    // The service is on-demand so `start` either queues or reports running.
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: StartResponse = serde_json::from_slice(&bytes).unwrap();
    assert!(matches!(
        parsed,
        StartResponse::AlreadyRunning
            | StartResponse::QueueFull
            | StartResponse::Unavailable { .. }
    ));
    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn disable_then_enable_roundtrip() {
    let h = build_harness(vec![minimal_llama_service("demo", 0)]).await;
    let app = ananke::api::management::router(h.state.clone());

    // Disable the service.
    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/api/services/demo/disable")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: DisableResponse = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed, DisableResponse::Disabled);

    // Enable the service.
    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/api/services/demo/enable")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: EnableResponse = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed, EnableResponse::Enabled);

    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn start_on_missing_service_returns_404() {
    let h = build_harness(vec![]).await;
    let app = ananke::api::management::router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/api/services/ghost/start")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn stop_on_missing_service_returns_404() {
    let h = build_harness(vec![]).await;
    let app = ananke::api::management::router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/api/services/ghost/stop")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn disable_already_disabled_returns_already_disabled() {
    let h = build_harness(vec![minimal_llama_service("demo", 0)]).await;
    let app = ananke::api::management::router(h.state.clone());

    // Disable once.
    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/api/services/demo/disable")
        .body(axum::body::Body::empty())
        .unwrap();
    let _resp = app.clone().oneshot(req).await.unwrap();

    // Disable again.
    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/api/services/demo/disable")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: DisableResponse = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed, DisableResponse::AlreadyDisabled);

    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn enable_already_enabled_returns_already_enabled() {
    let h = build_harness(vec![minimal_llama_service("demo", 0)]).await;
    let app = ananke::api::management::router(h.state.clone());

    // Enable on an idle (non-disabled) service.
    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/api/services/demo/enable")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: EnableResponse = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed, EnableResponse::AlreadyEnabled);

    h.cleanup().await;
}
