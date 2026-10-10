//! Integration checks for feature-gated dashboard assets and same-origin chat.
#![cfg(feature = "test-fakes")]

mod common;

use axum::http::StatusCode;
use tokio::net::TcpListener;

#[tokio::test]
async fn management_serves_native_api_with_feature_dependent_assets() {
    let mut service = common::minimal_llama_service("demo", 0);
    service.metadata.insert(
        "web_ui_url".into(),
        serde_json::json!("https://models.example/demo/"),
    );
    let harness = common::build_harness(vec![service]).await;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let router = ananke::api::management::router(harness.state.clone());
    let server = tokio::spawn(async move { axum::serve(listener, router).await });
    let client = reqwest::Client::new();
    let models: serde_json::Value = client
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(models["data"][0]["id"], "demo");
    let services: serde_json::Value = client
        .get(format!("{base}/api/services"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(services["services"][0]["openai_compat"], true);
    assert_eq!(
        services["services"][0]["web_ui_url"],
        "https://models.example/demo/"
    );
    for path in [
        "/api/missing",
        "/v1/missing",
        "/assets/missing.js",
        "/fonts/missing.woff2",
    ] {
        assert_eq!(
            client
                .get(format!("{base}{path}"))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::NOT_FOUND,
            "{path}"
        );
    }
    let index = client.get(format!("{base}/")).send().await.unwrap();
    if cfg!(feature = "frontend") {
        assert_eq!(index.status(), StatusCode::OK);
        assert!(
            index.headers()["content-type"]
                .to_str()
                .unwrap()
                .starts_with("text/html")
        );
        let html = index.text().await.unwrap();
        assert!(html.contains("<title>Ananke</title>"));
        let script = html
            .split("src=\"")
            .nth(1)
            .unwrap()
            .split('"')
            .next()
            .unwrap();
        let asset = client.get(format!("{base}{script}")).send().await.unwrap();
        assert_eq!(asset.status(), StatusCode::OK);
        assert!(asset.bytes().await.unwrap().len() > 1000);
        for path in [
            "/chat",
            "/services/demo",
            "/fonts/WorkSans-Regular.ttf",
            "/fonts/Iosevka-Extended.woff2",
        ] {
            assert_eq!(
                client
                    .get(format!("{base}{path}"))
                    .send()
                    .await
                    .unwrap()
                    .status(),
                StatusCode::OK,
                "{path}"
            );
        }
    } else {
        assert_eq!(index.status(), StatusCode::NOT_FOUND);
    }
    server.abort();
    harness.shutdown().await;
}
