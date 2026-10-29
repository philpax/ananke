//! Integration tests for manual reservations: placing one evicts whatever
//! services are in the way, holds its memory against later starts, survives
//! a reload from the database, and releases the memory when removed.
#![cfg(feature = "test-fakes")]

mod common;

use ananke::{
    api::{management, openai},
    config::DeviceSlot,
};
use ananke_api::reservations::{CreateReservationResponse, ReservationsResponse};
use ananke_placement::devices::{CpuSnapshot, DeviceSnapshot};
use ananke_proxy::InflightGuard;
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use common::{build_harness, minimal_llama_service};
use smol_str::SmolStr;
use tower::util::ServiceExt;

const GIB: u64 = 1024 * 1024 * 1024;

fn cpu_snapshot(gib: u64) -> DeviceSnapshot {
    DeviceSnapshot {
        gpus: Vec::new(),
        cpu: Some(CpuSnapshot {
            total_bytes: gib * GIB,
            available_bytes: gib * GIB,
        }),
        taken_at_ms: 0,
    }
}

/// A CPU service pledging `gib` GiB.
fn service(name: &str, gib: u64) -> ananke::config::ServiceConfig {
    let mut svc = minimal_llama_service(name, 0);
    svc.placement_override.clear();
    svc.placement_override.insert(DeviceSlot::Cpu, gib * 1024);
    svc
}

async fn send(
    app: axum::Router,
    method: &str,
    uri: &str,
    body: Option<String>,
) -> (StatusCode, String) {
    let mut req = Request::builder().method(method).uri(uri);
    if body.is_some() {
        req = req.header("content-type", "application/json");
    }
    let resp = app
        .oneshot(
            req.body(body.map(Body::from).unwrap_or_else(Body::empty))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap_or_default();
    (status, String::from_utf8_lossy(&bytes).to_string())
}

async fn chat(state: &ananke::daemon::app_state::AppState, model: &str) -> (StatusCode, String) {
    send(
        openai::router(state.clone()),
        "POST",
        "/v1/chat/completions",
        Some(format!(r#"{{"model":"{model}","messages":[]}}"#)),
    )
    .await
}

async fn reserve(state: &ananke::daemon::app_state::AppState, body: &str) -> (StatusCode, String) {
    send(
        management::router(state.clone()),
        "POST",
        "/api/reservations",
        Some(body.to_string()),
    )
    .await
}

#[tokio::test(flavor = "multi_thread")]
async fn reservation_evicts_idle_service_and_holds_its_memory_until_removed() {
    let h = build_harness(vec![service("alpha", 10)]).await;
    *h.state.snapshot.write() = cpu_snapshot(15);

    let (st, body) = chat(&h.state, "alpha").await;
    assert_eq!(st, StatusCode::OK, "alpha should load: {body}");

    // 15 GiB with alpha's 10 pledged leaves 5; 8 needs alpha gone. Alpha
    // is running with nothing in flight, so it counts as idle.
    let (st, body) = reserve(
        &h.state,
        &format!(r#"{{"devices":{{"cpu":{}}},"label":"train"}}"#, 8 * GIB),
    )
    .await;
    assert_eq!(st, StatusCode::CREATED, "{body}");
    let created: CreateReservationResponse = serde_json::from_str(&body).unwrap();
    assert_eq!(created.evicted, vec!["alpha".to_string()]);
    assert_eq!(created.reservation.label.as_deref(), Some("train"));
    assert_eq!(created.reservation.devices["cpu"], 8 * GIB);
    assert!(
        h.state.allocations.lock().get("alpha").is_none(),
        "the response waits for the evicted service to stop"
    );

    // Alpha cannot come back while the reservation holds the memory, and
    // cannot evict it.
    let (st, body) = chat(&h.state, "alpha").await;
    assert_ne!(
        st,
        StatusCode::OK,
        "alpha must not fit beside the reservation: {body}"
    );

    let (st, body) = send(
        management::router(h.state.clone()),
        "GET",
        "/api/reservations",
        None,
    )
    .await;
    assert_eq!(st, StatusCode::OK);
    let listed: ReservationsResponse = serde_json::from_str(&body).unwrap();
    assert_eq!(listed.manual, vec![created.reservation.clone()]);
    assert!(listed.services.is_empty(), "{listed:?}");

    let (st, body) = send(
        management::router(h.state.clone()),
        "GET",
        "/api/devices",
        None,
    )
    .await;
    assert_eq!(st, StatusCode::OK);
    assert!(
        body.contains(&format!(r#""reservation":{}"#, created.reservation.id)),
        "{body}"
    );

    let uri = format!("/api/reservations/{}", created.reservation.id);
    let (st, _) = send(management::router(h.state.clone()), "DELETE", &uri, None).await;
    assert_eq!(st, StatusCode::NO_CONTENT);
    let (st, _) = send(management::router(h.state.clone()), "DELETE", &uri, None).await;
    assert_eq!(st, StatusCode::NOT_FOUND);

    let (st, body) = chat(&h.state, "alpha").await;
    assert_eq!(
        st,
        StatusCode::OK,
        "alpha should load once the memory is released: {body}"
    );

    h.cleanup().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn busy_and_persistent_services_at_top_priority_are_evicted() {
    let mut alpha = service("alpha", 10);
    alpha.priority = 255;
    let mut beta = service("beta", 10);
    beta.priority = 255;
    beta.lifecycle = ananke::config::Lifecycle::Persistent;
    let h = build_harness(vec![alpha, beta]).await;
    *h.state.snapshot.write() = cpu_snapshot(30);
    for name in ["alpha", "beta"] {
        let (st, body) = chat(&h.state, name).await;
        assert_eq!(st, StatusCode::OK, "{name}: {body}");
    }

    let busy = InflightGuard::new(h.state.inflight.counter(&SmolStr::new("alpha")));
    let (st, body) = reserve(
        &h.state,
        &format!(r#"{{"devices":{{"cpu":{}}},"wait":false}}"#, 25 * GIB),
    )
    .await;
    assert_eq!(st, StatusCode::CREATED, "{body}");
    let created: CreateReservationResponse = serde_json::from_str(&body).unwrap();
    let mut evicted = created.evicted.clone();
    evicted.sort();
    assert_eq!(evicted, vec!["alpha".to_string(), "beta".to_string()]);
    drop(busy);

    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        while ["alpha", "beta"]
            .iter()
            .any(|name| h.state.allocations.lock().contains_key(*name))
        {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("both evicted services stop");

    h.cleanup().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn reservation_that_fits_evicts_nothing_and_requests_are_validated() {
    let h = build_harness(vec![service("alpha", 4)]).await;
    *h.state.snapshot.write() = cpu_snapshot(15);
    let (st, _) = chat(&h.state, "alpha").await;
    assert_eq!(st, StatusCode::OK);

    let (st, body) = reserve(&h.state, &format!(r#"{{"devices":{{"cpu":{}}}}}"#, 8 * GIB)).await;
    assert_eq!(st, StatusCode::CREATED, "{body}");
    let created: CreateReservationResponse = serde_json::from_str(&body).unwrap();
    assert!(created.evicted.is_empty());
    assert!(h.state.allocations.lock().get("alpha").is_some());

    for bad in [
        r#"{"devices":{}}"#.to_string(),
        r#"{"devices":{"gpu:0":1}}"#.to_string(),
        r#"{"devices":{"tpu":1}}"#.to_string(),
        r#"{"devices":{"cpu":0}}"#.to_string(),
        format!(r#"{{"devices":{{"cpu":{}}}}}"#, 16 * GIB),
        r#"{"devices":{"cpu":1},"ttl":"1h"}"#.to_string(),
        r#"{"devices":{"cpu":1},"priority":50}"#.to_string(),
        "not json".to_string(),
    ] {
        let (st, body) = reserve(&h.state, &bad).await;
        assert_eq!(st, StatusCode::BAD_REQUEST, "{bad}: {body}");
    }

    h.cleanup().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn stored_reservations_are_restored_into_the_allocation_table() {
    let h = build_harness(vec![]).await;
    *h.state.snapshot.write() = cpu_snapshot(15);
    let (st, body) = reserve(&h.state, &format!(r#"{{"devices":{{"cpu":{}}}}}"#, 3 * GIB)).await;
    assert_eq!(st, StatusCode::CREATED, "{body}");
    let created: CreateReservationResponse = serde_json::from_str(&body).unwrap();

    // A daemon restart starts from an empty table and store; boot reloads
    // both from the database.
    let mut state = h.state.clone();
    state.reservations = ananke::reservations::ReservationStore::default();
    state.allocations = std::sync::Arc::new(parking_lot::Mutex::new(Default::default()));
    ananke::reservations::load(&state).await.unwrap();
    assert_eq!(state.reservations.list(), vec![created.reservation.clone()]);
    let key = SmolStr::new(format!("reservation:{}", created.reservation.id));
    assert_eq!(
        state
            .allocations
            .lock()
            .get(&key)
            .and_then(|row| row.get(&DeviceSlot::Cpu))
            .copied(),
        Some(3 * 1024)
    );

    h.cleanup().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn without_wait_the_reply_comes_before_the_evicted_service_stops() {
    let h = build_harness(vec![service("alpha", 10)]).await;
    *h.state.snapshot.write() = cpu_snapshot(15);
    let (st, body) = chat(&h.state, "alpha").await;
    assert_eq!(st, StatusCode::OK, "{body}");

    let (st, body) = reserve(
        &h.state,
        &format!(r#"{{"devices":{{"cpu":{}}},"wait":false}}"#, 8 * GIB),
    )
    .await;
    assert_eq!(st, StatusCode::CREATED, "{body}");
    let created: CreateReservationResponse = serde_json::from_str(&body).unwrap();
    assert_eq!(created.evicted, vec!["alpha".to_string()]);

    // The reservation holds its memory at once; alpha drains behind it.
    let key = SmolStr::new(format!("reservation:{}", created.reservation.id));
    assert!(h.state.allocations.lock().contains_key(&key));
    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        while h.state.allocations.lock().contains_key("alpha") {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("alpha stops after the reply");

    h.cleanup().await;
}
