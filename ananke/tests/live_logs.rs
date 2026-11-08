//! Integration tests for `logs` follows on `/api/live`.
//!
//! Lines go in through the batcher with an explicit flush, so every line is
//! committed and broadcast before the test waits for it, and every wait is a
//! frame under a deadline.
#![cfg(feature = "test-fakes")]

mod common;

use std::time::Instant;

use ananke::config::EffectiveConfig;
use ananke_api::{
    events::Event,
    internal::log_line::{LogLine, LogPosition, LogStream},
    live::{LiveData, LiveFeed, LiveMessage, LiveRequest, LiveResource, LogFeed},
    services::LogsResponse,
    shared::errors::ApiErrorCodeSlug,
};
use ananke_db::logs::{LogLine as PushLine, Stream};
use axum::{
    body::{Body, to_bytes},
    http::Request,
};
use common::{
    TestHarness, build_harness,
    live::{DEADLINE, Socket, connect, next_message, send},
    minimal_llama_service,
};
use futures::SinkExt;
use smol_str::SmolStr;
use tokio_tungstenite::tungstenite::Message;
use tower::util::ServiceExt;

async fn service_id(h: &TestHarness, name: &str) -> i64 {
    h.state.db.upsert_service(name, 0).await.unwrap()
}

/// Push `lines` as `(run_id, stream, text)` and commit them as one flush.
async fn commit(h: &TestHarness, service_id: i64, lines: &[(i64, Stream, &str)]) {
    for (run_id, stream, text) in lines {
        h.state.batcher.push(PushLine {
            service_id,
            run_id: *run_id,
            timestamp_ms: 0,
            stream: *stream,
            line: (*text).to_string(),
        });
    }
    h.state.batcher.flush().await;
}

async fn commit_stdout(h: &TestHarness, service_id: i64, run_id: i64, texts: &[&str]) {
    let lines: Vec<_> = texts.iter().map(|t| (run_id, Stream::Stdout, *t)).collect();
    commit(h, service_id, &lines).await;
}

fn feed(service: &str, after: Option<LogPosition>) -> LogFeed {
    LogFeed {
        service: service.into(),
        after,
        run: None,
        stream: None,
    }
}

async fn follow(socket: &mut Socket, id: &str, feed: LogFeed) {
    send(
        socket,
        LiveRequest::Follow {
            id: id.into(),
            feed: LiveFeed::Logs(feed),
        },
    )
    .await;
}

/// Everything a follow sent before its next `log_caught_up`.
#[derive(Debug, Default)]
struct Sync {
    lines: Vec<LogLine>,
    gaps: Vec<(Option<LogPosition>, LogPosition, u64)>,
}

async fn until_caught_up(socket: &mut Socket, id: &str) -> Sync {
    let mut sync = Sync::default();
    let deadline = Instant::now() + DEADLINE;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match next_message(socket, left).await {
            Some(LiveMessage::LogLines { id: got, lines }) if got == id => {
                sync.lines.extend(lines);
            }
            Some(LiveMessage::LogGap {
                id: got,
                after,
                before,
                skipped,
            }) if got == id => sync.gaps.push((after, before, skipped)),
            Some(LiveMessage::LogCaughtUp { id: got }) if got == id => return sync,
            Some(other) => panic!("unexpected frame while catching up: {other:?}"),
            None => panic!("`{id}` did not catch up within {DEADLINE:?}"),
        }
    }
}

/// Collect `log_lines` for `id` until `n` lines have arrived, counting the
/// `log_caught_up` frames seen on the way.
async fn lines(socket: &mut Socket, id: &str, n: usize) -> (Vec<LogLine>, usize) {
    let mut out = Vec::new();
    let mut caught_up = 0;
    let deadline = Instant::now() + DEADLINE;
    while out.len() < n {
        let left = deadline.saturating_duration_since(Instant::now());
        match next_message(socket, left).await {
            Some(LiveMessage::LogLines { id: got, lines }) if got == id => out.extend(lines),
            Some(LiveMessage::LogCaughtUp { id: got }) if got == id => caught_up += 1,
            Some(LiveMessage::Snapshot { .. }) => {}
            Some(other) => panic!("unexpected frame: {other:?}"),
            None => panic!("only {} of {n} lines within {DEADLINE:?}", out.len()),
        }
    }
    (out, caught_up)
}

fn texts(lines: &[LogLine]) -> Vec<&str> {
    lines.iter().map(|l| l.line.as_str()).collect()
}

async fn get_logs(h: &TestHarness, uri: &str) -> LogsResponse {
    let req = Request::builder().uri(uri).body(Body::empty()).unwrap();
    let resp = ananke::api::management::router(h.state.clone())
        .oneshot(req)
        .await
        .unwrap();
    assert!(resp.status().is_success(), "{uri}: {}", resp.status());
    let bytes = to_bytes(resp.into_body(), 64 * 1024 * 1024).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// Silence for a moment: no frame for `id` arrives.
async fn assert_quiet(socket: &mut Socket, id: &str) {
    let deadline = Instant::now() + std::time::Duration::from_millis(400);
    while let Some(message) =
        next_message(socket, deadline.saturating_duration_since(Instant::now())).await
    {
        if let LiveMessage::LogLines { id: got, .. } = &message {
            assert_ne!(got, id, "a frame arrived: {message:?}");
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn backfill_then_follow_without_gaps_or_duplicates() {
    let h = build_harness(vec![minimal_llama_service("demo", 0)]).await;
    let sid = service_id(&h, "demo").await;
    commit_stdout(&h, sid, 1, &["1", "2", "3"]).await;
    let page = get_logs(&h, "/api/services/demo/logs").await;
    assert_eq!(page.logs.len(), 3);
    let after = page.follow_after;
    assert_eq!(after, Some(LogPosition { run_id: 1, seq: 3 }));

    // Written between the HTTP read and the follow: the race the position closes.
    commit_stdout(&h, sid, 1, &["4", "5", "6"]).await;
    let mut socket = connect(&h).await;
    follow(&mut socket, "l", feed("demo", after)).await;
    let sync = until_caught_up(&mut socket, "l").await;
    assert_eq!(texts(&sync.lines), ["4", "5", "6"]);
    assert!(sync.gaps.is_empty());

    commit_stdout(&h, sid, 1, &["7"]).await;
    let (live, _) = lines(&mut socket, "l", 1).await;
    assert_eq!(texts(&live), ["7"]);
    assert_quiet(&mut socket, "l").await;

    h.cleanup().await;
}

/// A follower that stops reading falls behind the broadcast; it then rereads
/// the store and sends every line exactly once, in order, with no gap.
#[tokio::test(flavor = "multi_thread")]
async fn lag_resyncs_from_the_database() {
    let h = build_harness(vec![minimal_llama_service("demo", 0)]).await;
    let sid = service_id(&h, "demo").await;
    let mut socket = connect(&h).await;
    follow(&mut socket, "l", feed("demo", None)).await;
    until_caught_up(&mut socket, "l").await;

    // Large lines fill the socket buffers and the session outbox, so the
    // follower blocks while far more flushes than the channel holds land.
    const FLUSHES: usize = 400;
    let padding = "x".repeat(64 * 1024);
    for i in 0..FLUSHES {
        let text = format!("{i} {padding}");
        commit_stdout(&h, sid, 1, &[&text]).await;
    }

    let (got, caught_up) = lines(&mut socket, "l", FLUSHES).await;
    let seqs: Vec<i64> = got.iter().map(|l| l.seq).collect();
    assert_eq!(seqs, (1..=FLUSHES as i64).collect::<Vec<_>>());
    assert!(
        caught_up >= 1,
        "the follower never fell behind and resynced"
    );
    assert_quiet(&mut socket, "l").await;

    h.cleanup().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn backfill_over_the_cap_sends_a_gap() {
    let h = build_harness(vec![minimal_llama_service("demo", 0)]).await;
    let sid = service_id(&h, "demo").await;
    let texts_in: Vec<String> = (1..=1500).map(|i| i.to_string()).collect();
    for chunk in texts_in.chunks(100) {
        let refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
        commit_stdout(&h, sid, 7, &refs).await;
    }

    let mut socket = connect(&h).await;
    follow(&mut socket, "l", feed("demo", None)).await;
    let sync = until_caught_up(&mut socket, "l").await;
    assert_eq!(
        sync.gaps,
        [(
            None,
            LogPosition {
                run_id: 7,
                seq: 501
            },
            500
        )]
    );
    assert_eq!(sync.lines.len(), 1000);
    assert_eq!(sync.lines[0].seq, 501);
    assert_eq!(sync.lines[999].seq, 1500);

    h.cleanup().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn filters_apply_to_backfill_and_follow() {
    let h = build_harness(vec![minimal_llama_service("demo", 0)]).await;
    let sid = service_id(&h, "demo").await;
    commit(
        &h,
        sid,
        &[
            (1, Stream::Stderr, "run 1 err"),
            (2, Stream::Stdout, "run 2 out"),
            (2, Stream::Stderr, "run 2 err a"),
        ],
    )
    .await;

    let mut socket = connect(&h).await;
    follow(
        &mut socket,
        "l",
        LogFeed {
            service: "demo".into(),
            after: None,
            run: Some(2),
            stream: Some(LogStream::Stderr),
        },
    )
    .await;
    let sync = until_caught_up(&mut socket, "l").await;
    assert_eq!(texts(&sync.lines), ["run 2 err a"]);

    commit(
        &h,
        sid,
        &[
            (2, Stream::Stdout, "run 2 out b"),
            (1, Stream::Stderr, "run 1 err b"),
            (2, Stream::Stderr, "run 2 err b"),
        ],
    )
    .await;
    let (live, _) = lines(&mut socket, "l", 1).await;
    assert_eq!(texts(&live), ["run 2 err b"]);
    assert_quiet(&mut socket, "l").await;

    h.cleanup().await;
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

/// Wait for a `services` snapshot in which demo's run id satisfies `accept`.
async fn demo_run(socket: &mut Socket, accept: impl Fn(Option<i64>) -> bool) -> Option<i64> {
    let deadline = Instant::now() + DEADLINE;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match next_message(socket, left).await {
            Some(LiveMessage::Snapshot {
                data: LiveData::Services(r),
                ..
            }) => {
                let run = r.services.iter().find(|s| s.name == "demo").unwrap().run_id;
                if accept(run) {
                    return run;
                }
            }
            Some(_) => {}
            None => panic!("demo's run id did not settle within {DEADLINE:?}"),
        }
    }
}

/// A restart mid-follow: the new run's lines arrive on the same follow, and
/// its id sorts after the old run's.
#[tokio::test(flavor = "multi_thread")]
async fn a_restart_mid_follow_delivers_the_new_run() {
    let h = build_harness(vec![minimal_llama_service("demo", 0)]).await;
    let sid = service_id(&h, "demo").await;
    let mut runs = connect(&h).await;
    send(
        &mut runs,
        LiveRequest::Subscribe {
            id: "s".into(),
            resource: LiveResource::Services,
        },
    )
    .await;
    let mut socket = connect(&h).await;
    follow(&mut socket, "l", feed("demo", None)).await;
    until_caught_up(&mut socket, "l").await;

    post(&h, "/api/services/demo/start").await;
    let first = demo_run(&mut runs, |r| r.is_some()).await.unwrap();
    commit_stdout(&h, sid, first, &["first run"]).await;

    post(&h, "/api/services/demo/stop").await;
    demo_run(&mut runs, |r| r.is_none()).await;
    post(&h, "/api/services/demo/start").await;
    let second = demo_run(&mut runs, |r| r.is_some()).await.unwrap();
    assert!(
        second > first,
        "run ids must increase: {first} then {second}"
    );
    commit_stdout(&h, sid, second, &["second run"]).await;

    let (got, _) = lines(&mut socket, "l", 2).await;
    let positions: Vec<_> = got.iter().map(|l| (l.run_id, l.line.as_str())).collect();
    assert_eq!(positions, [(first, "first run"), (second, "second run")]);

    h.cleanup().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn cancel_stops_lines() {
    let h = build_harness(vec![minimal_llama_service("demo", 0)]).await;
    let sid = service_id(&h, "demo").await;
    let mut socket = connect(&h).await;
    follow(&mut socket, "l", feed("demo", None)).await;
    until_caught_up(&mut socket, "l").await;
    send(&mut socket, LiveRequest::Cancel { id: "l".into() }).await;
    // Frames are handled in order: the probe's answer proves the cancel was.
    follow(&mut socket, "probe", feed("demo", None)).await;
    until_caught_up(&mut socket, "probe").await;

    commit_stdout(&h, sid, 1, &["after cancel"]).await;
    let (got, _) = lines(&mut socket, "probe", 1).await;
    assert_eq!(texts(&got), ["after cancel"]);
    assert_quiet(&mut socket, "l").await;

    h.cleanup().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn unknown_service_errors_then_recovers_on_reload() {
    let alpha = minimal_llama_service("alpha", 0);
    let h = build_harness(vec![alpha.clone()]).await;
    let mut socket = connect(&h).await;
    follow(&mut socket, "ghost", feed("ghost", None)).await;
    match next_message(&mut socket, DEADLINE).await {
        Some(LiveMessage::Error { id, status, error }) => {
            assert_eq!(id.as_deref(), Some("ghost"));
            assert_eq!(status, 404);
            assert_eq!(error.error.code, ApiErrorCodeSlug::ServiceNotFound);
        }
        other => panic!("expected a 404, got {other:?}"),
    }

    let mut ghost = minimal_llama_service("ghost", 0);
    ghost.private_port = alpha.private_port;
    let daemon = h.state.config.effective().daemon.clone();
    h.state.config.swap_effective_for_test(EffectiveConfig {
        daemon,
        services: vec![alpha, ghost],
    });
    // Published before the service is provisioned, as a real reload is.
    h.state.events.publish(Event::ConfigReloaded {
        at_ms: 0,
        changed_services: vec![SmolStr::new("ghost")],
    });
    let sid = service_id(&h, "ghost").await;
    until_caught_up(&mut socket, "ghost").await;
    commit_stdout(&h, sid, 1, &["boo"]).await;
    let (got, _) = lines(&mut socket, "ghost", 1).await;
    assert_eq!(texts(&got), ["boo"]);

    h.cleanup().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn bad_stream_filter_is_invalid() {
    let h = build_harness(vec![minimal_llama_service("demo", 0)]).await;
    let mut socket = connect(&h).await;
    let text =
        r#"{"type":"follow","id":"l","feed":{"kind":"logs","service":"demo","stream":"stdin"}}"#;
    socket.send(Message::Text(text.into())).await.unwrap();
    match next_message(&mut socket, DEADLINE).await {
        Some(LiveMessage::Error { id, status, error }) => {
            assert_eq!(id, None);
            assert_eq!(status, 400);
            assert_eq!(error.error.code, ApiErrorCodeSlug::InvalidLiveRequest);
        }
        other => panic!("expected a 400, got {other:?}"),
    }
    h.cleanup().await;
}
