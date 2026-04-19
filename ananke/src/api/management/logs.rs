//! `GET /api/services/{name}/logs?since&until&run&limit&stream&before`

use ananke_api::{ApiError, LogLine, LogsResponse};
use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use base64::{Engine, engine::general_purpose::STANDARD as B64};
use serde::{Deserialize, Serialize};

use crate::{daemon::app_state::AppState, db::models::ServiceLog};

const DEFAULT_LIMIT: u32 = 200;
const MAX_LIMIT: u32 = 1000;

#[derive(Debug, Deserialize)]
pub struct LogsQuery {
    pub since: Option<i64>,
    pub until: Option<i64>,
    pub run: Option<i64>,
    pub stream: Option<String>,
    pub limit: Option<u32>,
    pub before: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct Cursor {
    run_id: i64,
    seq: i64,
}

pub async fn get_logs(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Query(q): Query<LogsQuery>,
) -> Response {
    // Resolve service name → service_id via the existing `services` model.
    let service_id = match resolve_service_id(&state, &name).await {
        Some(id) => id,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(ApiError::new(
                    "service_not_found",
                    format!("service `{name}` not found"),
                )),
            )
                .into_response();
        }
    };

    let limit = q.limit.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT) as usize;
    let cursor = match q.before.as_deref().map(decode_cursor) {
        Some(Ok(c)) => Some(c),
        Some(Err(_)) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ApiError::new("invalid_cursor", "malformed `before` cursor")),
            )
                .into_response();
        }
        None => None,
    };

    let mut db = state.db.handle();
    let mut rows: Vec<ServiceLog> =
        ServiceLog::filter(ServiceLog::fields().service_id().eq(service_id))
            .exec(&mut db)
            .await
            .unwrap_or_default();

    // Apply filters in memory. We keep the toasty filter narrow because
    // toasty's DSL does not yet compose the full predicate we need; at
    // worst we hydrate the whole service's log buffer (bounded by retention
    // to 50k lines) and filter.
    rows.retain(|r| {
        q.since.is_none_or(|s| r.timestamp_ms >= s)
            && q.until.is_none_or(|u| r.timestamp_ms <= u)
            && q.run.is_none_or(|run| r.run_id == run)
            && q.stream.as_deref().is_none_or(|s| r.stream == s)
            && cursor
                .as_ref()
                .is_none_or(|c| (r.run_id, r.seq) < (c.run_id, c.seq))
    });

    // Newest-first sort by (timestamp_ms DESC, seq DESC).
    rows.sort_by(|a, b| b.timestamp_ms.cmp(&a.timestamp_ms).then(b.seq.cmp(&a.seq)));

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

    (StatusCode::OK, Json(LogsResponse { logs, next_cursor })).into_response()
}

async fn resolve_service_id(state: &AppState, name: &str) -> Option<i64> {
    use crate::db::models::Service;
    let mut handle = state.db.handle();
    Service::filter_by_name(name.to_string())
        .first()
        .exec(&mut handle)
        .await
        .ok()
        .flatten()
        .map(|s| s.service_id as i64)
}

fn encode_cursor(c: &Cursor) -> String {
    let json = serde_json::to_string(c).expect("cursor serialise");
    B64.encode(json)
}

fn decode_cursor(s: &str) -> Result<Cursor, ()> {
    let bytes = B64.decode(s).map_err(|_| ())?;
    serde_json::from_slice(&bytes).map_err(|_| ())
}
