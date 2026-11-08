//! `GET /api/services/{name}/logs?since&until&run&limit&stream&before`
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use ananke_api::{
    internal::log_line::{LogLine, LogPosition},
    services::logs::{LogsQuery, LogsResponse},
    shared::errors::ApiError,
};
use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use base64::{Engine, engine::general_purpose::STANDARD as B64};
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

use crate::{api::errors::ApiErrorCode, daemon::app_state::AppState};

const DEFAULT_LIMIT: u32 = 200;
const MAX_LIMIT: u32 = 1000;

#[derive(Serialize, Deserialize)]
struct Cursor {
    run_id: i64,
    seq: i64,
}

#[utoipa::path(
    summary = "Get service logs (paginated)",
    get,
    path = "/api/services/{name}/logs",
    params(
        ("name" = String, Path, description = "Service name"),
        ("since" = Option<i64>, Query, description = "Earliest timestamp_ms, inclusive"),
        ("until" = Option<i64>, Query, description = "Latest timestamp_ms, inclusive"),
        ("run" = Option<i64>, Query, description = "Restrict to one run_id"),
        ("stream" = Option<String>, Query, description = "\"stdout\", \"stderr\", or \"combined\""),
        ("limit" = Option<u32>, Query, description = "Max rows to return (≤1000, default 200)"),
        ("before" = Option<String>, Query, description = "Opaque cursor from a prior response"),
    ),
    responses(
        (status = 200, body = LogsResponse),
        (status = 400, body = ApiError, description = "invalid_cursor"),
        (status = 404, body = ApiError, description = "service_not_found")
    )
)]
pub async fn get_logs(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Query(q): Query<LogsQuery>,
) -> Response {
    // Resolve service name → service_id via the existing `services` model.
    let service_id = match resolve_service_id(&state, &name).await {
        Some(id) => id,
        None => {
            return ApiErrorCode::ServiceNotFound {
                name: SmolStr::new(&name),
            }
            .into_response();
        }
    };

    let limit = q.limit.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT) as usize;
    let cursor = match q.before.as_deref().map(decode_cursor) {
        Some(Ok(c)) => Some(c),
        Some(Err(_)) => {
            return ApiErrorCode::InvalidCursor.into_response();
        }
        None => None,
    };

    let mut rows = state
        .db
        .fetch_service_logs(service_id)
        .await
        .unwrap_or_default();

    // Read from the same rows the page is cut from, so every line on the
    // page is at or below it. Rows past `until` or above `before` are newer
    // than the page, so a follow from the newest line would skip them.
    let follow_after = if q.before.is_none() && q.until.is_none() {
        rows.iter()
            .map(|r| LogPosition {
                run_id: r.run_id,
                seq: r.seq,
            })
            .max()
    } else {
        None
    };

    // Apply filters in memory. The retention cap bounds this at 50k lines
    // per service.
    rows.retain(|r| {
        q.since.is_none_or(|s| r.timestamp_ms >= s)
            && q.until.is_none_or(|u| r.timestamp_ms <= u)
            && q.run.is_none_or(|run| r.run_id == run)
            && q.stream.as_deref().is_none_or(|s| r.stream == s)
            && cursor
                .as_ref()
                .is_none_or(|c| (r.run_id, r.seq) < (c.run_id, c.seq))
    });

    // Newest first by position, the same order the cursor compares in, so a
    // page never skips or repeats a row. Timestamps are not unique and can
    // step backwards with the wall clock; positions are neither.
    rows.sort_by_key(|r| std::cmp::Reverse((r.run_id, r.seq)));

    let truncated = rows.len() > limit;
    rows.truncate(limit);

    let next_cursor = if truncated {
        rows.last().map(|r| {
            encode_cursor(&Cursor {
                run_id: r.run_id,
                seq: r.seq,
            })
        })
    } else {
        None
    };

    let logs: Vec<LogLine> = rows
        .into_iter()
        .map(|r| LogLine {
            timestamp_ms: r.timestamp_ms,
            stream: r.stream,
            line: r.line,
            run_id: r.run_id,
            seq: r.seq,
        })
        .collect();

    (
        StatusCode::OK,
        Json(LogsResponse {
            logs,
            next_cursor,
            follow_after,
        }),
    )
        .into_response()
}

async fn resolve_service_id(state: &AppState, name: &str) -> Option<i64> {
    state.db.resolve_service_id(name).await.ok().flatten()
}

fn encode_cursor(c: &Cursor) -> String {
    // Invariant: `Cursor` is two `i64` fields, which serialise without error.
    let json = serde_json::to_string(c).unwrap_or_else(|_| unreachable!("cursor is plain ints"));
    B64.encode(json)
}

fn decode_cursor(s: &str) -> Result<Cursor, ()> {
    let bytes = B64.decode(s).map_err(|_| ())?;
    serde_json::from_slice(&bytes).map_err(|_| ())
}
