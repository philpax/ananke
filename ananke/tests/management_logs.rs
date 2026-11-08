//! Integration tests for the GET /api/services/{name}/logs endpoint.
#![cfg(feature = "test-fakes")]

mod common;

use ananke_api::{internal::log_line::LogPosition, services::LogsResponse};
use ananke_db::models::ServiceLog;
use axum::{body::to_bytes, http::StatusCode};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn returns_empty_for_idle_service() {
    let h = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let app = ananke::api::management::router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("GET")
        .uri("/api/services/demo/logs")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: LogsResponse = serde_json::from_slice(&bytes).unwrap();
    assert!(parsed.logs.is_empty());
    assert!(parsed.next_cursor.is_none());
    assert!(parsed.follow_after.is_none());
    h.cleanup().await;
}

/// `follow_after` is the newest line of the whole log, regardless of the
/// page's filters, and is withheld when rows newer than the page exist.
#[tokio::test(flavor = "current_thread")]
async fn follow_after_marks_the_newest_line() {
    let h = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let service_id = h
        .state
        .db
        .resolve_service_id("demo")
        .await
        .unwrap()
        .unwrap();
    let row = |run_id: i64, seq: i64, stream: &str| ServiceLog {
        service_id,
        run_id,
        timestamp_ms: 1_000 * run_id + seq,
        seq,
        stream: stream.into(),
        line: format!("{run_id}:{seq}"),
    };
    h.state
        .db
        .insert_log_batch(&[
            row(1, 1, "stdout"),
            row(1, 2, "stderr"),
            row(2, 1, "stdout"),
            row(2, 2, "stderr"),
        ])
        .await
        .unwrap();
    let newest = Some(LogPosition { run_id: 2, seq: 2 });

    let page = get_page(&h, "/api/services/demo/logs").await;
    assert_eq!(page.follow_after, newest);

    let filtered = get_page(&h, "/api/services/demo/logs?run=1&stream=stdout").await;
    assert_eq!(filtered.logs.len(), 1);
    assert_eq!(filtered.follow_after, newest);

    let bounded = get_page(&h, "/api/services/demo/logs?until=1500").await;
    assert_eq!(bounded.follow_after, None);

    let first = get_page(&h, "/api/services/demo/logs?limit=1").await;
    let cursor = first.next_cursor.unwrap();
    let older = get_page(
        &h,
        &format!(
            "/api/services/demo/logs?limit=1&before={}",
            encode_query(&cursor)
        ),
    )
    .await;
    assert_eq!(older.follow_after, None);
    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn rejects_malformed_cursor() {
    let h = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let app = ananke::api::management::router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("GET")
        .uri("/api/services/demo/logs?before=notbase64!!!")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    h.cleanup().await;
}

/// Pages are ordered by `(run_id, seq)`, which is what the cursor compares,
/// so paging through mixed data neither skips nor repeats a row. The old run
/// carries a 31-bit masked id from before run ids were monotonic, and its
/// timestamps are later than the new run's, as after a clock step.
#[tokio::test(flavor = "current_thread")]
async fn pages_are_ordered_by_position() {
    let h = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let service_id = h
        .state
        .db
        .resolve_service_id("demo")
        .await
        .unwrap()
        .unwrap();
    let old_run = 0x7FFF_0000;
    let new_run = 1_761_000_000_000;
    let row = |run_id: i64, seq: i64, timestamp_ms: i64| ServiceLog {
        service_id,
        run_id,
        timestamp_ms,
        seq,
        stream: "stdout".into(),
        line: format!("{run_id}:{seq}"),
    };
    h.state
        .db
        .insert_log_batch(&[
            row(old_run, 1, 5_000),
            row(old_run, 2, 5_000),
            row(old_run, 3, 6_000),
            row(new_run, 1, 1_000),
            row(new_run, 2, 1_000),
        ])
        .await
        .unwrap();

    let mut seen = Vec::new();
    let mut cursor: Option<String> = None;
    loop {
        let uri = match &cursor {
            Some(c) => format!(
                "/api/services/demo/logs?limit=2&before={}",
                c.replace('+', "%2B")
                    .replace('/', "%2F")
                    .replace('=', "%3D")
            ),
            None => "/api/services/demo/logs?limit=2".to_string(),
        };
        let page = get_page(&h, &uri).await;
        seen.extend(page.logs.iter().map(|l| (l.run_id, l.seq)));
        match page.next_cursor {
            Some(c) => cursor = Some(c),
            None => break,
        }
    }
    assert_eq!(
        seen,
        vec![
            (new_run, 2),
            (new_run, 1),
            (old_run, 3),
            (old_run, 2),
            (old_run, 1),
        ]
    );
    h.cleanup().await;
}

async fn get_page(h: &common::TestHarness, uri: &str) -> LogsResponse {
    let app = ananke::api::management::router(h.state.clone());
    let req = axum::http::Request::builder()
        .method("GET")
        .uri(uri)
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// Percent-encode the characters base64 uses that a query string does not
/// carry literally.
fn encode_query(cursor: &str) -> String {
    cursor
        .replace('+', "%2B")
        .replace('/', "%2F")
        .replace('=', "%3D")
}
