//! `GET /metrics` — Prometheus text-format endpoint for external scrapers.
//!
//! Exposes current-state gauges and counters derived from the daemon's
//! in-memory tables and the request_metrics database. Prometheus scrapes
//! this at its own cadence and stores history; the JSON `/api/metrics`
//! endpoint serves ananke's own historical data for the dashboard charts.

use std::fmt::Write as _;

use axum::{
    Router,
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use tracing::warn;

use crate::{daemon::app_state::AppState, supervise::state::ServiceState};

/// Numeric state codes for the `ananke_service_state` gauge. Prometheus-specific
/// — the numbers have no meaning outside the `/metrics` endpoint.
fn state_code(state: &ServiceState) -> i64 {
    match state {
        ServiceState::Idle => 0,
        ServiceState::Starting => 1,
        ServiceState::Running => 2,
        ServiceState::Draining => 3,
        ServiceState::Stopped => 4,
        ServiceState::Evicted => 5,
        ServiceState::Failed { .. } => 6,
        ServiceState::Disabled { .. } => 7,
    }
}

// --- Metric data model ---

/// One labelled sample within a metric family.
struct Sample {
    labels: Vec<(&'static str, String)>,
    value: f64,
}

/// A Prometheus metric family: a name, help text, type, and one or more
/// labelled samples. Collected first, then formatted in a single pass.
struct MetricFamily {
    name: &'static str,
    help: &'static str,
    metric_type: &'static str,
    samples: Vec<Sample>,
}

impl MetricFamily {
    fn new(name: &'static str, help: &'static str, metric_type: &'static str) -> Self {
        Self {
            name,
            help,
            metric_type,
            samples: Vec::new(),
        }
    }

    fn sample(&mut self, labels: Vec<(&'static str, String)>, value: f64) {
        self.samples.push(Sample { labels, value });
    }
}

/// Format a vector of metric families as Prometheus text format.
fn format_prometheus(families: &[MetricFamily]) -> String {
    let mut out = String::with_capacity(4096);
    for fam in families {
        let _ = writeln!(out, "# HELP {} {}", fam.name, fam.help);
        let _ = writeln!(out, "# TYPE {} {}", fam.name, fam.metric_type);
        for s in &fam.samples {
            if s.labels.is_empty() {
                let _ = writeln!(out, "{} {}", fam.name, s.value);
            } else {
                let labels: Vec<String> = s
                    .labels
                    .iter()
                    .map(|(k, v)| format!("{k}=\"{v}\""))
                    .collect();
                let _ = writeln!(out, "{}{{{}}} {}", fam.name, labels.join(","), s.value);
            }
        }
    }
    out
}

// --- Collection ---

/// Collect all Prometheus metrics from the daemon state.
async fn collect_metrics(state: &AppState) -> Vec<MetricFamily> {
    let eff = state.config.effective();
    let now = crate::tracking::now_unix_ms();
    let since = now - 7 * 24 * 60 * 60 * 1000;
    let big_bucket = 7 * 24 * 60 * 60 * 1000;

    // --- Counters: requests and tokens per service ---
    let mut requests = MetricFamily::new(
        "ananke_requests_total",
        "Total number of requests proxied.",
        "counter",
    );
    let mut tokens = MetricFamily::new("ananke_tokens_total", "Total tokens processed.", "counter");

    for svc in &eff.services {
        let sid = state.db.resolve_service_id(&svc.name).await.ok().flatten();
        let (req_total, prompt_total, completion_total) = if let Some(sid) = sid {
            match state
                .db
                .query_request_metrics(Some(sid), since, now, big_bucket)
                .await
            {
                Ok(buckets) => {
                    let req: i64 = buckets.iter().map(|b| b.request_count).sum();
                    let prompt: i64 = buckets.iter().map(|b| b.prompt_tokens).sum();
                    let completion: i64 = buckets.iter().map(|b| b.completion_tokens).sum();
                    (req, prompt, completion)
                }
                Err(e) => {
                    warn!(error = %e, service = %svc.name, "prometheus: query metrics failed");
                    (0, 0, 0)
                }
            }
        } else {
            (0, 0, 0)
        };
        requests.sample(vec![("service", svc.name.to_string())], req_total as f64);
        tokens.sample(
            vec![("service", svc.name.to_string()), ("type", "prompt".into())],
            prompt_total as f64,
        );
        tokens.sample(
            vec![
                ("service", svc.name.to_string()),
                ("type", "completion".into()),
            ],
            completion_total as f64,
        );
    }

    // --- Gauges: in-flight requests ---
    let mut inflight = MetricFamily::new(
        "ananke_inflight_requests",
        "Current number of in-flight requests.",
        "gauge",
    );
    for svc in &eff.services {
        inflight.sample(
            vec![("service", svc.name.to_string())],
            state.inflight.current(&svc.name) as f64,
        );
    }

    // --- Gauges: device memory (GPU VRAM + CPU RAM) ---
    let snap = state.snapshot.read().clone();
    let mut mem_total = MetricFamily::new(
        "ananke_memory_bytes",
        "Total memory capacity in bytes.",
        "gauge",
    );
    let mut mem_free =
        MetricFamily::new("ananke_memory_free_bytes", "Free memory in bytes.", "gauge");
    let mut mem_used =
        MetricFamily::new("ananke_memory_used_bytes", "Used memory in bytes.", "gauge");
    for g in &snap.gpus {
        let device = format!("gpu:{}", g.id);
        let used = g.total_bytes.saturating_sub(g.free_bytes);
        mem_total.sample(vec![("device", device.clone())], g.total_bytes as f64);
        mem_free.sample(vec![("device", device.clone())], g.free_bytes as f64);
        mem_used.sample(vec![("device", device)], used as f64);
    }
    if let Some(c) = &snap.cpu {
        let used = c.total_bytes.saturating_sub(c.available_bytes);
        mem_total.sample(vec![("device", "cpu".into())], c.total_bytes as f64);
        mem_free.sample(vec![("device", "cpu".into())], c.available_bytes as f64);
        mem_used.sample(vec![("device", "cpu".into())], used as f64);
    }

    // --- Gauges: service state ---
    let mut svc_state = MetricFamily::new(
        "ananke_service_state",
        "Numeric service state (0=idle, 1=starting, 2=running, 3=draining, 4=stopped, 5=evicted, 6=failed, 7=disabled, 8=unknown).",
        "gauge",
    );
    for svc in &eff.services {
        let handle = state.registry.get(&svc.name);
        let code = handle
            .as_ref()
            .map(|h| state_code(&h.peek().state))
            .unwrap_or(8);
        svc_state.sample(vec![("service", svc.name.to_string())], code as f64);
    }

    vec![
        requests, tokens, inflight, mem_total, mem_free, mem_used, svc_state,
    ]
}

// --- Handler ---

#[utoipa::path(
    get,
    path = "/metrics",
    responses((status = 200, description = "Prometheus text-format metrics"))
)]
pub async fn get_prometheus_metrics(State(state): State<AppState>) -> Response {
    let families = collect_metrics(&state).await;
    let body = format_prometheus(&families);
    (
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
        .into_response()
}

pub fn register(router: Router, state: AppState) -> Router {
    let mgmt: Router = Router::new()
        .route("/metrics", get(get_prometheus_metrics))
        .with_state(state);
    router.merge(mgmt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_prometheus_single_family() {
        let families = vec![{
            let mut f = MetricFamily::new("test_metric", "A test.", "counter");
            f.sample(vec![("svc", "alpha".into())], 42.0);
            f.sample(vec![("svc", "beta".into())], 7.0);
            f
        }];
        let out = format_prometheus(&families);
        assert!(out.contains("# HELP test_metric A test."));
        assert!(out.contains("# TYPE test_metric counter"));
        assert!(out.contains(r#"test_metric{svc="alpha"} 42"#));
        assert!(out.contains(r#"test_metric{svc="beta"} 7"#));
    }

    #[test]
    fn format_prometheus_unlabelled_sample() {
        let families = vec![{
            let mut f = MetricFamily::new("up", "Is up.", "gauge");
            f.sample(vec![], 1.0);
            f
        }];
        let out = format_prometheus(&families);
        assert!(out.contains("up 1\n"));
    }

    #[test]
    fn format_prometheus_multi_label() {
        let families = vec![{
            let mut f = MetricFamily::new("tokens", "Tokens.", "counter");
            f.sample(
                vec![("service", "demo".into()), ("type", "prompt".into())],
                100.0,
            );
            f
        }];
        let out = format_prometheus(&families);
        assert!(out.contains(r#"tokens{service="demo",type="prompt"} 100"#));
    }
}
