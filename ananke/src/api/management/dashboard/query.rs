//! Maps dashboard operations to the existing handlers, preserving their validation and errors.

use ananke_api::dashboard::{DashboardAction, DashboardQuery, DashboardRange, DashboardRequest};
use axum::{
    Router,
    body::Body,
    http::{Method, Request},
};
use serde_json::{Value, json};
use tower::ServiceExt;

pub fn read(query: &DashboardQuery) -> Result<Request<Body>, String> {
    let (path, window, extra) = match query {
        DashboardQuery::Services => ("/api/services".into(), None, Vec::new()),
        DashboardQuery::Service { name } => {
            (format!("/api/services/{}", segment(name)), None, Vec::new())
        }
        DashboardQuery::Devices => ("/api/devices".into(), None, Vec::new()),
        DashboardQuery::Info => ("/api/info".into(), None, Vec::new()),
        DashboardQuery::Models => ("/v1/models".into(), None, Vec::new()),
        DashboardQuery::Config => ("/api/config".into(), None, Vec::new()),
        DashboardQuery::Events => return Err("dashboard: events use the capture buffer".into()),
        DashboardQuery::Metrics {
            range,
            service,
            bucket,
        } => (
            "/api/metrics".into(),
            Some(range),
            vec![
                ("service", service.clone()),
                ("bucket", Some(bucket.clone())),
            ],
        ),
        DashboardQuery::Restarts { range, service } => (
            "/api/restarts".into(),
            Some(range),
            vec![("service", service.clone())],
        ),
        DashboardQuery::Samples { range } => {
            ("/api/devices/samples".into(), Some(range), Vec::new())
        }
        DashboardQuery::Logs {
            name,
            range,
            before,
        } => (
            format!("/api/services/{}/logs", segment(name)),
            Some(range),
            vec![("before", before.clone()), ("limit", Some("200".into()))],
        ),
    };
    let mut url = reqwest::Url::parse(&format!("http://dashboard{path}"))
        .map_err(|e| format!("dashboard: build read URL: {e}"))?;
    if let Some(range) = window {
        let (since, until) = bounds(range)?;
        url.query_pairs_mut()
            .append_pair("since", &since.to_string())
            .append_pair("until", &until.to_string());
    }
    for (key, value) in extra {
        if let Some(value) = value {
            url.query_pairs_mut().append_pair(key, &value);
        }
    }
    let uri = format!(
        "{}{}",
        url.path(),
        url.query().map(|q| format!("?{q}")).unwrap_or_default()
    );
    Request::builder()
        .uri(uri)
        .body(Body::empty())
        .map_err(|e| format!("dashboard: build read: {e}"))
}

pub fn command(request: &DashboardRequest) -> Result<Request<Body>, String> {
    let (method, path, content_type, body, hash) = match request {
        DashboardRequest::Action { name, action, .. } => {
            let action = match action {
                DashboardAction::Start => "start",
                DashboardAction::Stop => "stop",
                DashboardAction::Restart => "restart",
                DashboardAction::Enable => "enable",
                DashboardAction::Disable => "disable",
            };
            (
                Method::POST,
                format!("/api/services/{}/{action}", segment(name)),
                "application/json",
                String::new(),
                None,
            )
        }
        DashboardRequest::Validate { content, .. } => (
            Method::POST,
            "/api/config/validate".into(),
            "application/json",
            json!({"content": content}).to_string(),
            None,
        ),
        DashboardRequest::Save { content, hash, .. } => (
            Method::PUT,
            "/api/config".into(),
            "text/plain",
            content.clone(),
            Some(hash),
        ),
        DashboardRequest::Chat { body, .. } => {
            let mut body =
                serde_json::to_value(body).map_err(|e| format!("dashboard: encode chat: {e}"))?;
            body["stream"] = json!(true);
            (
                Method::POST,
                "/v1/chat/completions".into(),
                "application/json",
                body.to_string(),
                None,
            )
        }
        _ => return Err("dashboard: expected a command".into()),
    };
    let mut builder = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", content_type);
    if let Some(hash) = hash {
        builder = builder.header("if-match", format!("\"{hash}\""));
    }
    builder
        .body(Body::from(body))
        .map_err(|e| format!("dashboard: build command: {e}"))
}

pub async fn response(
    router: Router,
    request: Request<Body>,
) -> Result<axum::response::Response, String> {
    router.oneshot(request).await.map_err(|e| match e {})
}

pub async fn json_response(response: axum::response::Response) -> Result<(u16, Value), String> {
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), 32 * 1024 * 1024)
        .await
        .map_err(|e| format!("dashboard: read response: {e}"))?;
    let body = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| json!({ "error": String::from_utf8_lossy(&bytes) }))
    };
    Ok((status, body))
}

fn segment(name: &str) -> String {
    name.bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
                (b as char).to_string()
            } else {
                format!("%{b:02X}")
            }
        })
        .collect()
}

fn bounds(range: &DashboardRange) -> Result<(i64, i64), String> {
    match range.live_ms {
        Some(duration) if duration > 0 && duration <= i64::MAX as u64 => {
            let now = ananke_time::now_unix_ms();
            Ok((now.saturating_sub(duration as i64), now))
        }
        Some(_) => Err("dashboard: invalid rolling window duration".into()),
        None => Ok((range.since, range.until)),
    }
}
