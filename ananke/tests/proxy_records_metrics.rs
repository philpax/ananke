//! End-to-end test: the per-service reverse proxy records per-request token
//! metrics for the token-generating endpoints and skips everything else.
#![cfg(feature = "test-fakes")]

mod common;

use std::{
    net::SocketAddr,
    sync::{Arc, atomic::AtomicU64},
    time::Duration,
};

use ananke::{api::proxy, db::Database};
use futures::future::BoxFuture;
use tokio::sync::watch;

/// Sum of `request_count` across every bucket for a service — the number of
/// recorded metric rows regardless of how the query buckets them in time.
async fn recorded_metrics(db: &Database, service_id: i64) -> i64 {
    let now = ananke::tracking::now_unix_ms();
    db.query_request_metrics(Some(service_id), 0, now + 60_000, 1_000)
        .await
        .expect("query request metrics")
        .iter()
        .map(|b| b.request_count)
        .sum()
}

#[tokio::test]
async fn per_service_proxy_records_token_endpoints_only() {
    let echo_port = common::free_port();
    let echo_addr: SocketAddr = format!("127.0.0.1:{echo_port}").parse().unwrap();
    let (echo_shutdown_tx, echo_shutdown_rx) = watch::channel(false);
    let echo_state = common::echo_server::EchoState::default();
    tokio::spawn(common::echo_server::serve(
        echo_addr,
        echo_state,
        echo_shutdown_rx,
    ));
    let _echo_shutdown = echo_shutdown_tx;

    let db = Database::open_in_memory().await.unwrap();
    let service_id = db
        .upsert_service("demo", ananke::tracking::now_unix_ms())
        .await
        .unwrap();

    let metrics =
        proxy::ProxyMetrics::new(db.clone(), service_id, "demo".into(), Arc::new(|| Some(7)));

    let proxy_port = common::free_port();
    let proxy_addr: SocketAddr = format!("127.0.0.1:{proxy_port}").parse().unwrap();
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    // A no-op `before_request` (always proceed) and activity ping keep the
    // test focused on the metrics wrap without a supervisor behind it.
    let before_request: Arc<
        dyn Fn() -> BoxFuture<'static, Option<proxy::ProxyError>> + Send + Sync,
    > = Arc::new(|| Box::pin(async { None }));
    let inflight = Arc::new(AtomicU64::new(0));
    let activity_ping: Arc<dyn Fn() + Send + Sync> = Arc::new(|| {});

    tokio::spawn(proxy::serve_with_activity(
        proxy_addr,
        echo_port,
        shutdown_rx,
        before_request,
        inflight,
        activity_ping,
        Some(metrics),
    ));

    tokio::time::sleep(Duration::from_millis(50)).await;

    let client = reqwest::Client::new();

    // A `/health` request must not record a metric — it is not a
    // token-generating endpoint.
    let resp = client
        .get(format!("http://127.0.0.1:{proxy_port}/health"))
        .send()
        .await
        .expect("health request");
    assert_eq!(resp.status(), 200);
    let _ = resp.bytes().await;

    // A `/v1/chat/completions` request must record exactly one metric.
    let resp = client
        .post(format!("http://127.0.0.1:{proxy_port}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "demo", "messages": []}))
        .send()
        .await
        .expect("chat request");
    assert_eq!(resp.status(), 200);
    // Drain the body so the `MetricsBody` reaches end-of-stream and records.
    let _ = resp.bytes().await;

    // The metric is written by a spawned task after the body ends; poll until
    // it lands (or fail the assertion after a bounded wait).
    let mut recorded = 0;
    for _ in 0..50 {
        recorded = recorded_metrics(&db, service_id).await;
        if recorded >= 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        recorded, 1,
        "expected exactly one metric (chat only, not health)"
    );

    let _ = shutdown_tx.send(true);
}
