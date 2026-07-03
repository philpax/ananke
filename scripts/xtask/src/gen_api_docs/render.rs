//! Markdown rendering from OpenAPI spec data.
//!
//! Produces the endpoint summary table, per-endpoint detail sections
//! (parameters, request/response TypeScript types, response status table),
//! and the error code slug table. Prose fragments are injected by the
//! parent `mod.rs` orchestration.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use super::ts::{schema_ref_name, ts_type};

/// Render the full `docs/api.md` document from an OpenAPI spec.
pub fn render_markdown(spec: &Value, prose: &ProseFragments) -> String {
    let mut out = String::with_capacity(96 * 1024);

    // Title + overview prose.
    out.push_str("# Ananke API\n\n");
    out.push_str("## Overview\n\n");
    out.push_str(prose.overview.trim_end());
    out.push_str("\n\n");

    // Endpoint summary table.
    out.push_str("## Endpoints\n\n");
    render_endpoint_table(&mut out, spec);
    out.push('\n');

    // OpenAI API section.
    out.push_str("## OpenAI-compatible API\n\n");
    out.push_str(prose.openai_api.trim_end());
    out.push_str("\n\n");
    render_endpoint_group(&mut out, spec, "/v1/", 3);
    out.push('\n');

    // Management API section.
    out.push_str("## Management API\n\n");
    out.push_str(prose.management_api.trim_end());
    out.push_str("\n\n");

    // Group management endpoints by resource.
    render_management_groups(&mut out, spec);
    out.push('\n');

    // Per-service reverse proxy.
    out.push_str("## Per-service reverse proxy\n\n");
    out.push_str(prose.per_service_proxy.trim_end());
    out.push_str("\n\n");

    // Service states.
    out.push_str("## Service states\n\n");
    out.push_str(prose.service_states.trim_end());
    out.push_str("\n\n");

    // Events WebSocket.
    out.push_str("## WebSocket: /api/events\n\n");
    out.push_str(prose.events_ws.trim_end());
    out.push_str("\n\n");

    // Logs WebSocket.
    out.push_str("## WebSocket: /api/services/{name}/logs/stream\n\n");
    out.push_str(prose.logs_ws.trim_end());
    out.push_str("\n\n");

    // Errors.
    out.push_str("## Error codes\n\n");
    out.push_str(prose.errors.trim_end());
    out.push_str("\n\n");
    render_error_code_table(&mut out, spec);
    out.push('\n');

    // Prometheus.
    out.push_str("## Prometheus metrics\n\n");
    out.push_str(prose.prometheus.trim_end());
    out.push_str("\n\n");

    // OpenAPI spec endpoint.
    out.push_str("## OpenAPI specification\n\n");
    out.push_str(prose.openapi_spec.trim_end());
    out.push_str("\n\n");

    // Frontend.
    out.push_str("## Embedded frontend\n\n");
    out.push_str(prose.frontend.trim_end());
    out.push_str("\n\n");

    // Collapse any triple+ newlines to double.
    while out.contains("\n\n\n") {
        out = out.replace("\n\n\n", "\n\n");
    }
    out
}

/// Prose fragments transcluded at compile time by `mod.rs`.
pub struct ProseFragments {
    pub overview: &'static str,
    pub openai_api: &'static str,
    pub management_api: &'static str,
    pub per_service_proxy: &'static str,
    pub service_states: &'static str,
    pub events_ws: &'static str,
    pub logs_ws: &'static str,
    pub errors: &'static str,
    pub prometheus: &'static str,
    pub openapi_spec: &'static str,
    pub frontend: &'static str,
}

/// Render a summary table of all endpoints (one row per path+method).
fn render_endpoint_table(out: &mut String, spec: &Value) {
    out.push_str("| Method | Path | Description |\n");
    out.push_str("| --- | --- | --- |\n");
    let Some(paths) = spec.get("paths").and_then(|p| p.as_object()) else {
        return;
    };
    let mut sorted: Vec<_> = paths.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    for (path, item) in sorted {
        for method in ["get", "post", "put", "delete", "patch"] {
            if let Some(op) = item.get(method) {
                let desc = endpoint_description(op);
                out.push_str(&format!(
                    "| `{}` | `{}` | {} |\n",
                    method.to_uppercase(),
                    path,
                    desc.replace('\n', " ")
                ));
            }
        }
    }
}

/// Extract a human-readable description from an operation.
fn endpoint_description(op: &Value) -> String {
    if let Some(s) = op.get("summary").and_then(|s| s.as_str()) {
        if !s.is_empty() {
            return s.to_string();
        }
    }
    if let Some(s) = op.get("description").and_then(|d| d.as_str()) {
        if !s.is_empty() {
            return s.to_string();
        }
    }
    String::new()
}

/// Group management API endpoints by resource prefix and render each group.
fn render_management_groups(out: &mut String, spec: &Value) {
    let Some(paths) = spec.get("paths").and_then(|p| p.as_object()) else {
        return;
    };

    let mut groups: BTreeMap<String, Vec<(&String, &Value)>> = BTreeMap::new();
    for (path, item) in paths {
        if !path.starts_with("/api/") {
            continue;
        }
        let group = if path.starts_with("/api/services") {
            "Services"
        } else if path.starts_with("/api/config") {
            "Config"
        } else if path.starts_with("/api/devices") {
            "Devices"
        } else if path.starts_with("/api/metrics") {
            "Metrics"
        } else if path.starts_with("/api/oneshot") {
            "Oneshot"
        } else if path.starts_with("/api/info") {
            "Info"
        } else {
            "Other"
        };
        groups
            .entry(group.to_string())
            .or_default()
            .push((path, item));
    }

    for (group, items) in &groups {
        out.push_str(&format!("### {}\n\n", group));
        let mut sorted = items.clone();
        sorted.sort_by(|a, b| a.0.cmp(b.0));
        for (path, item) in &sorted {
            for method in ["get", "post", "put", "delete", "patch"] {
                let Some(op) = item.get(method) else { continue };
                render_endpoint_detail(out, method, path, op, spec, 4);
            }
        }
    }
}

/// Render endpoints matching a prefix, flat (no sub-grouping).
fn render_endpoint_group(out: &mut String, spec: &Value, prefix: &str, heading_level: usize) {
    let Some(paths) = spec.get("paths").and_then(|p| p.as_object()) else {
        return;
    };
    let mut sorted: Vec<_> = paths
        .iter()
        .filter(|(path, _)| path.starts_with(prefix))
        .collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    for (path, item) in sorted {
        for method in ["get", "post", "put", "delete", "patch"] {
            let Some(op) = item.get(method) else { continue };
            render_endpoint_detail(out, method, path, op, spec, heading_level);
        }
    }
}

/// Render a single endpoint's detail section.
///
/// Shows the parameters table, request body TypeScript type, response
/// status table, and success response TypeScript type. All schemas are
/// expanded inline in the TS blocks.
fn render_endpoint_detail(
    out: &mut String,
    method: &str,
    path: &str,
    op: &Value,
    spec: &Value,
    heading_level: usize,
) {
    let prefix = "#".repeat(heading_level);
    out.push_str(&format!(
        "{} {} {}\n\n",
        prefix,
        method.to_uppercase(),
        path
    ));

    let desc = endpoint_description(op);
    if !desc.is_empty() {
        out.push_str(&desc);
        out.push_str("\n\n");
    }

    // Parameters.
    if let Some(params) = op.get("parameters").and_then(|p| p.as_array()) {
        if !params.is_empty() {
            out.push_str("| Parameter | In | Required | Description |\n");
            out.push_str("| --- | --- | --- | --- |\n");
            for param in params {
                let name = param.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let location = param.get("in").and_then(|i| i.as_str()).unwrap_or("");
                let required = param
                    .get("required")
                    .and_then(|r| r.as_bool())
                    .unwrap_or(false);
                let desc = param
                    .get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or("");
                out.push_str(&format!(
                    "| `{}` | {} | {} | {} |\n",
                    name,
                    location,
                    if required { "yes" } else { "no" },
                    desc
                ));
            }
            out.push('\n');
        }
    }

    // Request body — TypeScript type.
    if let Some(rb) = op.get("requestBody") {
        if let Some(schema) = rb
            .get("content")
            .and_then(|c| c.get("application/json"))
            .and_then(|j| j.get("schema"))
        {
            out.push_str("**Request body**:\n\n");
            let mut visited = BTreeSet::new();
            let ts = ts_type(schema, spec, &mut visited, 0);
            out.push_str("```typescript\n");
            out.push_str(&ts);
            out.push_str("\n```\n\n");
        }
    }

    // Responses — status table.
    if let Some(responses) = op.get("responses").and_then(|r| r.as_object()) {
        out.push_str("| Status | Description | Body |\n");
        out.push_str("| --- | --- | --- |\n");
        let mut codes: Vec<_> = responses.iter().collect();
        codes.sort_by(|a, b| a.0.cmp(b.0));

        let mut example_schema: Option<&Value> = None;
        let mut example_code: &str = "";

        for (code, resp) in &codes {
            let desc = resp
                .get("description")
                .and_then(|d| d.as_str())
                .unwrap_or("");
            let body = resp
                .get("content")
                .and_then(|c| c.get("application/json"))
                .and_then(|j| j.get("schema"))
                .map(|s| format!("`{}`", schema_ref_name(s)))
                .unwrap_or("—".to_string());
            out.push_str(&format!("| {} | {} | {} |\n", code, desc, body));

            if example_schema.is_none() && code.starts_with('2') && code.as_str() != "204" {
                if let Some(s) = resp
                    .get("content")
                    .and_then(|c| c.get("application/json"))
                    .and_then(|j| j.get("schema"))
                {
                    example_schema = Some(s);
                    example_code = code;
                }
            }
        }
        out.push('\n');

        // TypeScript type for the success response.
        if let Some(schema) = example_schema {
            out.push_str(&format!("**Response ({})**:\n\n", example_code));
            let mut visited = BTreeSet::new();
            let ts = ts_type(schema, spec, &mut visited, 0);
            if ts.contains('\n') || ts.contains(' ') || ts.contains('{') {
                out.push_str("```typescript\n");
                out.push_str(&ts);
                out.push_str("\n```\n\n");
            }
        }
    }
}

/// Render a table of error code slugs from the `ApiErrorCodeSlug` schema.
fn render_error_code_table(out: &mut String, spec: &Value) {
    let Some(enums) = spec
        .get("components")
        .and_then(|c| c.get("schemas"))
        .and_then(|s| s.get("ApiErrorCodeSlug"))
        .and_then(|s| s.get("enum"))
        .and_then(|e| e.as_array())
    else {
        return;
    };
    out.push_str("| Slug | Description |\n");
    out.push_str("| --- | --- |\n");
    for v in enums {
        let slug = v.as_str().unwrap_or("");
        let desc = match slug {
            "model_not_found" => "Client referenced a model name that isn't configured.",
            "service_not_found" => {
                "Management caller referenced a service that isn't in the config."
            }
            "service_disabled" => "Service is administratively disabled.",
            "start_queue_full" => "Supervisor's start queue saturated.",
            "start_failed" => "Spawn or health-probe failure during ensure.",
            "insufficient_vram" => "Packer couldn't lay out the model on available devices.",
            "service_blocked" => "Queued behind a busy non-elastic peer.",
            "upstream_unavailable" => "Upstream child rejected the wire or never replied.",
            "proxy_internal" => "Bug inside the proxy itself (URI parse, header build, etc.).",
            "not_implemented" => "OpenAI endpoint the daemon hasn't implemented.",
            "invalid_request_error" => {
                "Client request was malformed (bad JSON, missing field, etc.)."
            }
            "invalid_cursor" => "Log-paging cursor failed to decode.",
            "if_match_required" => "Config PUT arrived without an If-Match header.",
            "hash_mismatch" => "Config PUT's If-Match didn't match the current hash.",
            "persist_failed" => "Config write failed at the IO layer.",
            "other" => "Forward-compatibility fallback for unknown codes.",
            _ => "",
        };
        out.push_str(&format!("| `{}` | {} |\n", slug, desc));
    }
    out.push('\n');
}
