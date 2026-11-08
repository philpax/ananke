//! Integration tests for the `/api/live` subscription WebSocket.
//!
//! Every wait is on a frame arriving under a deadline, never on a sleep: a
//! snapshot the daemon fails to push fails the test rather than being
//! papered over by a later poll.
#![cfg(feature = "test-fakes")]

mod common;

use std::{
    collections::BTreeMap,
    sync::atomic::Ordering,
    time::{Duration, Instant},
};

use ananke::{
    api::openai,
    config::{DeviceSlot, EffectiveConfig, PlacementPolicy, ServiceConfig},
};
use ananke_api::{
    events::Event,
    internal::fit_verdict::FitVerdict,
    live::{LiveData, LiveMessage, LiveRequest, LiveResource},
    services::list::ServiceSummary,
    shared::{ServiceStateKind, errors::ApiErrorCodeSlug},
};
use ananke_placement::devices::{CpuSnapshot, DeviceSnapshot, GpuSnapshot};
use ananke_proxy::InflightGuard;
use axum::{body::Body, http::Request};
use common::{
    TestHarness, build_harness, build_harness_with_snapshot,
    live::{DEADLINE, Socket, connect, next_message, send},
    minimal_llama_service,
};
use futures::SinkExt;
use smol_str::SmolStr;
use tokio_tungstenite::{
    connect_async,
    tungstenite::{self, Message, client::IntoClientRequest},
};
use tower::util::ServiceExt;

const GIB: u64 = 1024 * 1024 * 1024;

async fn subscribe(socket: &mut Socket, id: &str, resource: LiveResource) {
    send(
        socket,
        LiveRequest::Subscribe {
            id: id.into(),
            resource,
        },
    )
    .await;
}

/// Read snapshots for `id` until one satisfies `accept`, failing at the
/// deadline. Earlier snapshots are intermediate states and are skipped.
async fn snapshot_until(
    socket: &mut Socket,
    id: &str,
    what: &str,
    accept: impl Fn(&LiveData) -> bool,
) -> LiveData {
    let deadline = Instant::now() + DEADLINE;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match next_message(socket, left).await {
            Some(LiveMessage::Snapshot { id: got, data }) if got == id && accept(&data) => {
                return data;
            }
            Some(_) => {}
            None => panic!("no `{id}` snapshot where {what} within {DEADLINE:?}"),
        }
    }
}

fn service<'a>(data: &'a LiveData, name: &str) -> &'a ServiceSummary {
    match data {
        LiveData::Services(r) => r.services.iter().find(|s| s.name == name).unwrap(),
        other => panic!("expected a services snapshot, got {other:?}"),
    }
}

fn error_frame(message: Option<LiveMessage>) -> (Option<String>, u16, ApiErrorCodeSlug) {
    match message {
        Some(LiveMessage::Error { id, status, error }) => (id, status, error.error.code),
        other => panic!("expected an error frame, got {other:?}"),
    }
}

/// #45, symptom 1: a finished request must clear the in-flight count
/// without the client reading anything again.
#[tokio::test(flavor = "multi_thread")]
async fn inflight_clears_without_another_read() {
    let h = build_harness(vec![minimal_llama_service("demo", 0)]).await;
    let mut socket = connect(&h).await;
    subscribe(&mut socket, "s", LiveResource::Services).await;
    snapshot_until(&mut socket, "s", "demo is idle", |d| {
        service(d, "demo").inflight_count == 0
    })
    .await;

    let guard = InflightGuard::new(h.state.inflight.counter(&SmolStr::new("demo")));
    snapshot_until(&mut socket, "s", "one request is in flight", |d| {
        service(d, "demo").inflight_count == 1
    })
    .await;
    drop(guard);
    snapshot_until(&mut socket, "s", "the request has cleared", |d| {
        service(d, "demo").inflight_count == 0
    })
    .await;

    // The same through the real OpenAI path: a hanging stream holds its
    // guard in the response body until the body is dropped.
    h.echo_state.hang.store(true, Ordering::Relaxed);
    let req = Request::builder()
        .method("POST")
        .uri("/completion")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"demo","prompt":"hi","stream":true}"#,
        ))
        .unwrap();
    let resp = openai::router(h.state.clone()).oneshot(req).await.unwrap();
    snapshot_until(&mut socket, "s", "the proxied request is in flight", |d| {
        service(d, "demo").inflight_count == 1
    })
    .await;
    drop(resp);
    snapshot_until(&mut socket, "s", "the proxied request has cleared", |d| {
        service(d, "demo").inflight_count == 0
    })
    .await;

    h.cleanup().await;
}

fn gpu_service(name: &str) -> ServiceConfig {
    let mut svc = minimal_llama_service(name, 0);
    svc.placement_policy = PlacementPolicy::GpuOnly;
    svc.placement_override = BTreeMap::from([(DeviceSlot::Gpu(0), 16 * 1024)]);
    svc
}

fn one_gpu(free_bytes: u64) -> DeviceSnapshot {
    DeviceSnapshot {
        gpus: vec![GpuSnapshot {
            id: 0,
            name: "test gpu".into(),
            total_bytes: 24 * GIB,
            free_bytes,
        }],
        cpu: Some(CpuSnapshot {
            total_bytes: 64 * GIB,
            available_bytes: 64 * GIB,
        }),
        taken_at_ms: 0,
    }
}

async fn post(h: &TestHarness, uri: &str) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    let resp = ananke::api::management::router(h.state.clone())
        .oneshot(req)
        .await
        .unwrap();
    assert!(resp.status().is_success(), "{uri}: {}", resp.status());
}

/// #45, symptom 2: stopping a service releases its pledge at once, but the
/// freed VRAM only shows on a later sampler tick. That tick alone must
/// flip the peer's verdict and the device's free bytes.
#[tokio::test(flavor = "multi_thread")]
async fn freed_capacity_flips_peer_verdict() {
    let h =
        build_harness_with_snapshot(vec![gpu_service("a"), gpu_service("b")], one_gpu(24 * GIB))
            .await;
    let mut services = connect(&h).await;
    subscribe(&mut services, "s", LiveResource::Services).await;
    let mut devices = connect(&h).await;
    subscribe(&mut devices, "d", LiveResource::Devices).await;

    post(&h, "/api/services/a/start").await;
    // a's pledge alone is enough to push b out.
    snapshot_until(
        &mut services,
        "s",
        "a is running and b no longer fits",
        |d| service(d, "a").pid.is_some() && service(d, "b").fit_verdict != Some(FitVerdict::Fits),
    )
    .await;
    // The sampler then sees a's memory in use.
    *h.state.snapshot.write() = one_gpu(8 * GIB);
    snapshot_until(&mut devices, "d", "gpu:0 shows a's memory in use", |d| {
        matches!(d, LiveData::Devices(list)
            if list.iter().any(|g| g.id == "gpu:0" && g.free_bytes == 8 * GIB))
    })
    .await;

    post(&h, "/api/services/a/stop").await;
    snapshot_until(&mut services, "s", "a has stopped and b still waits", |d| {
        let (a, b) = (service(d, "a"), service(d, "b"));
        a.pid.is_none() && b.fit_verdict != Some(FitVerdict::Fits)
    })
    .await;

    // The next sampler tick shows the memory back. Nothing else happens.
    *h.state.snapshot.write() = one_gpu(24 * GIB);
    snapshot_until(&mut services, "s", "b fits again", |d| {
        service(d, "b").fit_verdict == Some(FitVerdict::Fits)
    })
    .await;
    snapshot_until(&mut devices, "d", "gpu:0 shows the freed memory", |d| {
        matches!(d, LiveData::Devices(list)
            if list.iter().any(|g| g.id == "gpu:0" && g.free_bytes == 24 * GIB))
    })
    .await;

    h.cleanup().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn identical_snapshots_are_not_resent() {
    let h = build_harness(vec![minimal_llama_service("demo", 0)]).await;
    let mut socket = connect(&h).await;
    subscribe(&mut socket, "d", LiveResource::Devices).await;
    snapshot_until(&mut socket, "d", "the first snapshot arrives", |_| true).await;

    let same = h.state.snapshot.read().clone();
    *h.state.snapshot.write() = same.clone();
    assert!(
        next_message(&mut socket, Duration::from_millis(500))
            .await
            .is_none(),
        "an unchanged body must not be resent"
    );

    // The trigger was live all along: a real change still arrives.
    let mut changed = same;
    if let Some(cpu) = changed.cpu.as_mut() {
        cpu.available_bytes -= GIB;
    }
    *h.state.snapshot.write() = changed;
    snapshot_until(&mut socket, "d", "the changed body arrives", |_| true).await;

    h.cleanup().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn pid_cleared_after_stop() {
    let h = build_harness(vec![minimal_llama_service("demo", 0)]).await;
    let mut socket = connect(&h).await;
    subscribe(&mut socket, "s", LiveResource::Services).await;

    post(&h, "/api/services/demo/start").await;
    snapshot_until(&mut socket, "s", "demo has a pid", |d| {
        service(d, "demo").pid.is_some()
    })
    .await;
    post(&h, "/api/services/demo/stop").await;
    let last = snapshot_until(&mut socket, "s", "demo's pid has cleared", |d| {
        let s = service(d, "demo");
        s.pid.is_none() && s.run_id.is_none()
    })
    .await;
    assert_eq!(service(&last, "demo").state, ServiceStateKind::Idle);

    h.cleanup().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn service_subscription_reports_not_found_then_recovers() {
    let alpha = minimal_llama_service("alpha", 0);
    let h = build_harness(vec![alpha.clone()]).await;
    let mut socket = connect(&h).await;
    subscribe(
        &mut socket,
        "ghost",
        LiveResource::Service {
            name: "ghost".into(),
        },
    )
    .await;
    assert_eq!(
        error_frame(next_message(&mut socket, DEADLINE).await),
        (Some("ghost".into()), 404, ApiErrorCodeSlug::ServiceNotFound)
    );

    let mut ghost = minimal_llama_service("ghost", 0);
    ghost.private_port = alpha.private_port;
    let daemon = h.state.config.effective().daemon.clone();
    h.state.config.swap_effective_for_test(EffectiveConfig {
        daemon,
        services: vec![alpha, ghost],
    });
    h.state.events.publish(Event::ConfigReloaded {
        at_ms: 0,
        changed_services: vec![SmolStr::new("ghost")],
    });
    let detail = snapshot_until(&mut socket, "ghost", "ghost exists", |_| true).await;
    assert!(matches!(detail, LiveData::Service(d) if d.name == "ghost"));

    h.cleanup().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn protocol_errors_and_cancel() {
    let h = build_harness(vec![minimal_llama_service("demo", 0)]).await;
    let mut socket = connect(&h).await;

    socket.send(Message::Text("not json".into())).await.unwrap();
    assert_eq!(
        error_frame(next_message(&mut socket, DEADLINE).await),
        (None, 400, ApiErrorCodeSlug::InvalidLiveRequest)
    );

    subscribe(&mut socket, "info", LiveResource::Info).await;
    snapshot_until(&mut socket, "info", "the first info snapshot", |_| true).await;
    subscribe(&mut socket, "info", LiveResource::Info).await;
    assert_eq!(
        error_frame(next_message(&mut socket, DEADLINE).await),
        (
            Some("info".into()),
            409,
            ApiErrorCodeSlug::DuplicateSubscriptionId
        )
    );

    // Fill the connection to its limit with info subscriptions (one already
    // held); each answers with one snapshot.
    for i in 1..64 {
        subscribe(&mut socket, &format!("i{i}"), LiveResource::Info).await;
    }
    for _ in 1..64 {
        assert!(matches!(
            next_message(&mut socket, DEADLINE).await,
            Some(LiveMessage::Snapshot { .. })
        ));
    }
    subscribe(&mut socket, "one-too-many", LiveResource::Info).await;
    assert_eq!(
        error_frame(next_message(&mut socket, DEADLINE).await),
        (
            Some("one-too-many".into()),
            429,
            ApiErrorCodeSlug::SubscriptionLimit
        )
    );

    // Cancel every info subscription; a config reload then pushes nothing.
    send(&mut socket, LiveRequest::Cancel { id: "info".into() }).await;
    for i in 1..64 {
        send(
            &mut socket,
            LiveRequest::Cancel {
                id: format!("i{i}"),
            },
        )
        .await;
    }
    // Frames are handled in order, so the probe's snapshot proves every
    // cancel above was processed before the reload below.
    subscribe(&mut socket, "probe", LiveResource::Devices).await;
    snapshot_until(&mut socket, "probe", "the probe snapshot", |_| true).await;

    // Change the info body; a live info subscription would push it.
    let effective = h.state.config.effective();
    h.state.config.swap_effective_for_test(EffectiveConfig {
        daemon: ananke::config::DaemonSettings {
            openai_listen: "127.0.0.1:1".into(),
            ..effective.daemon.clone()
        },
        services: effective.services.clone(),
    });
    h.state.events.publish(Event::ConfigReloaded {
        at_ms: 0,
        changed_services: vec![],
    });
    let deadline = Instant::now() + Duration::from_millis(500);
    while let Some(message) = next_message(
        &mut socket,
        deadline.saturating_duration_since(Instant::now()),
    )
    .await
    {
        if let LiveMessage::Snapshot { id, .. } | LiveMessage::Error { id: Some(id), .. } = &message
        {
            assert_eq!(id, "probe", "a cancelled subscription pushed: {message:?}");
        }
    }

    h.cleanup().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn cross_site_origin_is_rejected() {
    let h = build_harness(vec![]).await;
    let addr = h.spawn_management_server().await;
    let mut request = format!("ws://{addr}/api/live")
        .into_client_request()
        .unwrap();
    request
        .headers_mut()
        .insert("origin", "http://evil.example".parse().unwrap());
    match connect_async(request).await {
        Err(tungstenite::Error::Http(resp)) => assert_eq!(resp.status(), 403),
        other => panic!("expected a 403 handshake, got {other:?}"),
    }

    let mut request = format!("ws://{addr}/api/live")
        .into_client_request()
        .unwrap();
    request
        .headers_mut()
        .insert("origin", format!("http://{addr}").parse().unwrap());
    assert!(connect_async(request).await.is_ok());

    h.cleanup().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn missing_origin_is_allowed() {
    let h = build_harness(vec![]).await;
    let mut socket = connect(&h).await;
    subscribe(&mut socket, "info", LiveResource::Info).await;
    snapshot_until(&mut socket, "info", "an info snapshot", |_| true).await;
    h.cleanup().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn events_history_backfills_on_connect() {
    let h = build_harness(vec![minimal_llama_service("demo", 0)]).await;
    let since = h.state.events.history().since_ms;
    let before = Event::ConfigReloaded {
        at_ms: since + 1,
        changed_services: vec![SmolStr::new("demo")],
    };
    h.state.events.publish(before.clone());

    let mut socket = connect(&h).await;
    subscribe(&mut socket, "e", LiveResource::Events).await;
    let first = snapshot_until(&mut socket, "e", "the backfill arrives", |_| true).await;
    let LiveData::Events(history) = first else {
        panic!("expected an events snapshot, got {first:?}");
    };
    assert_eq!(history.since_ms, since);
    assert!(history.events.contains(&before));

    let after = Event::ConfigReloaded {
        at_ms: since + 2,
        changed_services: vec![],
    };
    h.state.events.publish(after.clone());
    snapshot_until(
        &mut socket,
        "e",
        "the new event arrives",
        |d| matches!(d, LiveData::Events(h) if h.events.last() == Some(&after)),
    )
    .await;

    h.cleanup().await;
}
