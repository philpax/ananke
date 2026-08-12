//! Integration tests for the GET/PUT /api/config and POST /api/config/validate endpoints.
#![cfg(feature = "test-fakes")]

mod common;

use ananke::api::management::router;
use ananke_api::config::{
    ConfigResponse,
    validate::{
        ConfigValidateRequest, ConfigValidateResponse, ValidationContext, ValidationErrorCode,
    },
};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header::IF_MATCH},
};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn get_config_returns_content_and_hash() {
    let h = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let app = router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("GET")
        .uri("/api/config")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: ConfigResponse = serde_json::from_slice(&bytes).unwrap();
    // The in-memory config manager starts with an empty raw string, so the
    // hash must still be present (it is the hash of the empty string).
    assert!(!parsed.hash.is_empty());
    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn put_without_if_match_is_428() {
    let h = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let app = router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("PUT")
        .uri("/api/config")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::PRECONDITION_REQUIRED);
    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn put_with_wrong_hash_is_412() {
    let h = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let app = router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("PUT")
        .uri("/api/config")
        .header(IF_MATCH, "\"wrong\"")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::PRECONDITION_FAILED);
    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn post_validate_returns_parser_span() {
    let h = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let app = router(h.state.clone());
    let request = ConfigValidateRequest {
        content: "[daemon]\nshutdown_timeout = \"unterminated\n".into(),
    };
    let req = Request::builder()
        .method("POST")
        .uri("/api/config/validate")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&request).unwrap()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: ConfigValidateResponse = serde_json::from_slice(&bytes).unwrap();
    assert!(!parsed.valid);
    assert_eq!(parsed.errors.len(), 1);
    let error = &parsed.errors[0];
    assert_eq!(error.code, ValidationErrorCode::Parse);
    assert!(error.location.is_some());
    assert!(error.line.is_some());
    assert!(error.column.is_some());
    assert!(matches!(error.context, ValidationContext::Parse { .. }));
    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn post_validate_returns_multiple_typed_diagnostics() {
    let h = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let app = router(h.state.clone());
    let request = ConfigValidateRequest {
        content: "[daemon]\nshutdown_timeout = \"bogus\"\nmanagement_listen = \"not-an-address\"\n"
            .into(),
    };
    let req = Request::builder()
        .method("POST")
        .uri("/api/config/validate")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&request).unwrap()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: ConfigValidateResponse = serde_json::from_slice(&bytes).unwrap();
    assert!(!parsed.valid);
    assert_eq!(
        parsed
            .errors
            .iter()
            .map(|error| error.code)
            .collect::<Vec<_>>(),
        vec![
            ValidationErrorCode::DurationInvalid,
            ValidationErrorCode::ValueInvalid
        ]
    );
    assert!(parsed.errors.iter().all(|error| error.location.is_none()));
    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn put_returns_validation_errors_without_preflight() {
    let h = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let (raw, hash) = h.state.config.raw();
    assert!(raw.is_empty());
    let app = router(h.state.clone());
    let body = "[daemon]\nshutdown_timeout = \"bogus\"\nmanagement_listen = \"not-an-address\"\n";
    let req = Request::builder()
        .method("PUT")
        .uri("/api/config")
        .header(IF_MATCH, format!("\"{hash}\""))
        .body(Body::from(body))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: ConfigValidateResponse = serde_json::from_slice(&bytes).unwrap();
    assert!(!parsed.valid);
    assert_eq!(parsed.errors.len(), 2);
    assert_eq!(parsed.errors[0].code, ValidationErrorCode::DurationInvalid);
    assert_eq!(parsed.errors[1].code, ValidationErrorCode::ValueInvalid);
    h.cleanup().await;
}
