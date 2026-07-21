//! Config-default constants and the documentation descriptor table.
//!
//! The `DEFAULT_*` constants are the single source of truth for every
//! default the daemon applies when a config field is omitted. The
//! descriptor table (`all_sections`) is what the `gen-config-docs` xtask
//! renders into `docs/configuration.md`; because it references these
//! constants directly, changing a constant changes the generated doc and
//! trips `--check` in CI.

use serde::Serialize;

use crate::defaults::{MANAGEMENT_LISTEN, OPENAI_LISTEN};

// ── moved constants ──────────────────────────────────────────────────────

/// Default idle-before-drain timeout for on-demand services (10 minutes).
pub const DEFAULT_IDLE_TIMEOUT_MS: u64 = 600_000;

/// Default OpenAI request body limit (64 MiB). Generous so multi-megabyte
/// base64 vision payloads pass; axum's own default is only 2 MiB, which
/// rejects most real images with `413 Payload Too Large`.
pub const DEFAULT_OPENAI_MAX_BODY_MB: u64 = 64;

/// [`DEFAULT_OPENAI_MAX_BODY_MB`] expressed in bytes, for the contexts that
/// want a byte count directly (e.g. the `DaemonSettings` default).
pub const DEFAULT_OPENAI_MAX_BODY_BYTES: usize = DEFAULT_OPENAI_MAX_BODY_MB as usize * 1024 * 1024;

/// Default cadence for the health-probe loop (5 seconds).
pub const DEFAULT_HEALTH_PROBE_INTERVAL_MS: u64 = 5_000;

/// Default per-probe timeout for health checks (3 minutes).
pub const DEFAULT_HEALTH_TIMEOUT_MS: u64 = 180_000;

/// Default drain timeout before the supervisor escalates to SIGKILL (30 seconds).
pub const DEFAULT_DRAIN_TIMEOUT_MS: u64 = 30_000;

/// Default extra grace granted to in-flight streaming requests during drain
/// (30 seconds).
pub const DEFAULT_EXTENDED_STREAM_DRAIN_MS: u64 = 30_000;

/// Default cap on the wall-clock duration of a single proxied request
/// (10 minutes).
pub const DEFAULT_MAX_REQUEST_DURATION_MS: u64 = 600_000;

/// Default service scheduling priority (higher wins eviction contests).
pub const DEFAULT_SERVICE_PRIORITY: u8 = 50;

/// Default minimum runtime a borrower must accumulate before the balloon
/// resolver may fast-kill it (1 minute).
pub const DEFAULT_MIN_BORROWER_RUNTIME_MS: u64 = 60_000;

/// Default rolling window for the auto-restart error-rate watchdog (2
/// minutes). Validated against production data: a service that wedges into
/// 100 % 5xx is caught ~60 s after the first error at typical traffic.
pub const DEFAULT_AUTO_RESTART_WINDOW_MS: u64 = 120_000;

/// Default error-rate threshold (fraction of the window) that trips the
/// watchdog.
pub const DEFAULT_AUTO_RESTART_MAX_ERROR_RATE: f64 = 0.5;

/// Default minimum request count in the window before the ratio is trusted.
/// Never fired across 8.5 hours of healthy production traffic; fired within
/// a minute of a real wedge.
pub const DEFAULT_AUTO_RESTART_MIN_REQUESTS: u32 = 20;

/// Default cadence at which the watchdog polls the metrics store (30 s).
pub const DEFAULT_AUTO_RESTART_POLL_INTERVAL_MS: u64 = 30_000;

/// Default anti-flap cooldown: a fresh run must live this long before
/// another auto-restart may fire (5 minutes).
pub const DEFAULT_AUTO_RESTART_MIN_UPTIME_MS: u64 = 300_000;

/// Default number of auto-restarts tolerated within
/// [`DEFAULT_AUTO_RESTART_FLAP_WINDOW_MS`] before the service is disabled.
pub const DEFAULT_AUTO_RESTART_MAX_RESTARTS: u32 = 3;

/// Default sliding window over which [`DEFAULT_AUTO_RESTART_MAX_RESTARTS`]
/// is counted (30 minutes).
pub const DEFAULT_AUTO_RESTART_FLAP_WINDOW_MS: u64 = 1_800_000;

/// Default time-to-first-token stall timeout for the auto-restart stall
/// watchdog (5 minutes). A proxied request that produces no response token
/// within this window is treated as an upstream wedge and triggers a restart.
/// Deliberately generous — healthy prefill (even image inference) reaches the
/// first token in seconds, so a full five minutes of silence is unambiguous.
pub const DEFAULT_AUTO_RESTART_TTFT_STALL_MS: u64 = 300_000;

/// Default generation-stall timeout for the auto-restart watchdog (5
/// minutes). While at least one request is in flight, the child's Prometheus
/// progress counters (prompt + predicted tokens) must advance within this
/// window or the run is treated as wedged and restarted. Matches the TTFT
/// stall default: healthy prefill and decode both advance the counters every
/// batch, so five minutes of flat counters under load is unambiguous.
pub const DEFAULT_AUTO_RESTART_GENERATION_STALL_MS: u64 = 300_000;

/// Default cadence at which the generation-stall watchdog polls the child's
/// `/metrics` endpoint (30 s).
pub const DEFAULT_AUTO_RESTART_GENERATION_STALL_POLL_MS: u64 = 30_000;

/// Default concurrency cap on pending start requests waiting for the same
/// supervisor to finish starting before they are rejected with `QueueFull`.
pub const DEFAULT_START_QUEUE_DEPTH: usize = 10;

/// Inclusive lower bound of the default loopback port range handed out to
/// llama-server children for their private listener.
pub const DEFAULT_PRIVATE_PORT_START: u16 = 40_000;

/// Inclusive upper bound of the default private-listener port range.
/// Override (together with [`DEFAULT_PRIVATE_PORT_START`]) when another
/// process on the host occupies the default window.
pub const DEFAULT_PRIVATE_PORT_END: u16 = 59_999;

// ── duration formatting helper ───────────────────────────────────────────

/// Convert a millisecond constant to the human-readable duration string
/// used in docs (`600_000` → `"10m"`, `30_000` → `"30s"`).
///
/// Picks the largest clean unit (seconds, minutes, or hours) when the value
/// divides evenly; otherwise falls back to the raw `{n}ms` form so the
/// doc never lies about a non-clean value.
pub fn fmt_duration_ms(ms: u64) -> String {
    if ms == 0 {
        return "0s".to_string();
    }
    if ms.is_multiple_of(3_600_000) {
        return format!("{}h", ms / 3_600_000);
    }
    if ms.is_multiple_of(60_000) {
        return format!("{}m", ms / 60_000);
    }
    // Seconds only for sub-minute values, so non-clean multiples of 1000
    // (e.g. 90 000 ms = 1.5 min) fall through to the raw-ms form.
    if ms < 60_000 && ms.is_multiple_of(1_000) {
        return format!("{}s", ms / 1_000);
    }
    format!("{ms}ms")
}

// ── descriptor structs ──────────────────────────────────────────────────

/// A documentation section containing a field-reference table.
#[derive(Debug, Serialize)]
pub struct SectionDoc {
    /// Anchor id used for intra-doc links (e.g. "daemon", "openai_api").
    pub id: &'static str,
    /// Heading text for the section (e.g. "Daemon Settings").
    pub title: &'static str,
    /// Field descriptors rendered as table rows.
    pub fields: Vec<FieldDoc>,
}

/// A single field's documentation rendered as a table row.
#[derive(Debug, Serialize)]
pub struct FieldDoc {
    /// TOML field name (e.g. "management_listen").
    pub name: &'static str,
    /// Type string shown in the Type column (e.g. "string", "duration string", "u16").
    pub ty: &'static str,
    /// Rendered default value (e.g. "127.0.0.1:7071", "10m", "50").
    pub default: String,
    /// One-line description shown in the Description column. Accepts a
    /// computed `String` so fields with an enum vocabulary can render the
    /// accepted-value list straight from [`crate::flags`].
    pub description: String,
}

/// Convenience constructor for a `FieldDoc`.
fn field(
    name: &'static str,
    ty: &'static str,
    default: impl Into<String>,
    description: impl Into<String>,
) -> FieldDoc {
    FieldDoc {
        name,
        ty,
        default: default.into(),
        description: description.into(),
    }
}

/// Render an enum vocabulary from [`crate::flags`] as a backtick-quoted,
/// comma-separated list for a field's Description column, e.g.
/// `` `"layer"`, `"row"`, `"tensor"` ``. Keeps the accepted-value list in
/// the docs sourced from the same constants the daemon validates against.
fn code_values(values: &[&str]) -> String {
    values
        .iter()
        .map(|v| format!("`\"{v}\"`"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Wrap a value in backticks for the Default column.
fn bt(v: impl std::fmt::Display) -> String {
    format!("`{v}`")
}

/// Wrap a duration constant in backticks.
fn bt_dur(ms: u64) -> String {
    bt(fmt_duration_ms(ms))
}

/// Return every field-reference table in `docs/configuration.md`.
///
/// The descriptor table is hand-maintained: adding a new config field
/// requires adding an entry here for it to appear in the generated docs.
/// CI's `--check` catches default-value drift (a constant change) but not
/// a missing entry — code review is the backstop for the latter.
pub fn all_sections() -> Vec<SectionDoc> {
    vec![
        SectionDoc {
            id: "daemon",
            title: "Daemon Settings",
            fields: vec![
                field(
                    "management_listen",
                    "string",
                    bt(MANAGEMENT_LISTEN),
                    "Bind address for the management API. Non-loopback requires `allow_external_management = true`.",
                ),
                field(
                    "allow_external_management",
                    "bool",
                    "`false`",
                    "Must be `true` when `management_listen` is non-loopback.",
                ),
                field(
                    "allow_external_services",
                    "bool",
                    "`false`",
                    "Bind per-service reverse proxies on `0.0.0.0` instead of `127.0.0.1`. Controls only the per-service proxies, not the OpenAI multiplexer (which honours `openai_api.listen`).",
                ),
                field(
                    "data_dir",
                    "path",
                    "`$XDG_DATA_HOME/ananke` (or `~/.local/share/ananke`)",
                    "Directory for the SQLite database and runtime state.",
                ),
                field(
                    "shutdown_timeout",
                    "duration string",
                    "`120s`",
                    "Max time to wait for services to drain on daemon shutdown.",
                ),
                field(
                    "private_port_start",
                    "u16",
                    bt(DEFAULT_PRIVATE_PORT_START),
                    "Inclusive lower bound of the loopback port range handed to llama-server children for their private listener.",
                ),
                field(
                    "private_port_end",
                    "u16",
                    bt(DEFAULT_PRIVATE_PORT_END),
                    "Inclusive upper bound of the private-listener port range. Override when another process occupies the default window.",
                ),
                field(
                    "llama_server",
                    "path",
                    "`llama-server` (from `$PATH`)",
                    "Default llama-server executable for every llama-cpp service. Overridable per-service.",
                ),
            ],
        },
        SectionDoc {
            id: "openai_api",
            title: "OpenAI API Settings",
            fields: vec![
                field(
                    "listen",
                    "string",
                    bt(OPENAI_LISTEN),
                    "Bind address for the OpenAI-compatible API.",
                ),
                field(
                    "enabled",
                    "bool",
                    "`true`",
                    "Set to `false` to disable the OpenAI API entirely.",
                ),
                field(
                    "max_request_duration",
                    "duration string",
                    bt_dur(DEFAULT_MAX_REQUEST_DURATION_MS),
                    "Max wall-clock duration per proxied request.",
                ),
                field(
                    "allow_cors",
                    "bool",
                    "`true`",
                    "Allow cross-origin requests from browsers. Set to `false` to block browser-based access.",
                ),
                field(
                    "max_body_mb",
                    "u64",
                    bt(DEFAULT_OPENAI_MAX_BODY_MB),
                    "Max request body size in MiB. Raise for large or many images (vision payloads are base64-encoded).",
                ),
            ],
        },
        SectionDoc {
            id: "defaults",
            title: "Global Defaults",
            fields: vec![
                field(
                    "idle_timeout",
                    "duration string",
                    bt_dur(DEFAULT_IDLE_TIMEOUT_MS),
                    "Default idle timeout for on-demand services.",
                ),
                field(
                    "priority",
                    "u8",
                    bt(DEFAULT_SERVICE_PRIORITY),
                    "Default eviction priority (higher wins eviction contests).",
                ),
                field(
                    "start_queue_depth",
                    "u32",
                    bt(DEFAULT_START_QUEUE_DEPTH),
                    "Default concurrency cap on pending start requests waiting for the same supervisor before they are rejected with `QueueFull`.",
                ),
            ],
        },
        SectionDoc {
            id: "devices",
            title: "Device Configuration",
            fields: vec![
                field(
                    "gpu_ids",
                    "array of u32",
                    "all visible GPUs",
                    "Only probe these GPUs.",
                ),
                field(
                    "default_gpu_reserved_mb",
                    "u64",
                    "`0`",
                    "VRAM (MiB) kept free on every GPU that lacks a `gpu_reserved_mb` entry.",
                ),
                field(
                    "gpu_reserved_mb",
                    "map string → u64",
                    "empty",
                    "Per-GPU VRAM reserve (MiB), keyed by GPU id string.",
                ),
                field(
                    "cpu.enabled",
                    "bool",
                    "`true`",
                    "Allow CPU placement for services.",
                ),
                field(
                    "cpu.reserved_gb",
                    "u64",
                    "`0`",
                    "Host RAM (GiB) the daemon keeps free. Bounds how much expert weight a hybrid MoE service may offload to the CPU; a placement that would exceed it is rejected.",
                ),
            ],
        },
        SectionDoc {
            id: "service_common",
            title: "Common Fields",
            fields: vec![
                field("name", "string", "*required*", "Unique service identifier."),
                field(
                    "template",
                    "string",
                    "*required*",
                    "`\"llama-cpp\"` or `\"command\"`.",
                ),
                field(
                    "port",
                    "u16",
                    "*required*",
                    "Public-facing port for the service's reverse proxy.",
                ),
                field(
                    "lifecycle",
                    "string",
                    "`\"on_demand\"`",
                    "`\"on_demand\"` or `\"persistent\"` (see [Lifecycle](#lifecycle)).",
                ),
                field(
                    "priority",
                    "u8",
                    format!("{} (or `[defaults]` value)", bt(DEFAULT_SERVICE_PRIORITY)),
                    "Eviction priority; higher wins eviction contests.",
                ),
                field(
                    "idle_timeout",
                    "duration string",
                    format!(
                        "{} (or `[defaults]` value)",
                        bt_dur(DEFAULT_IDLE_TIMEOUT_MS)
                    ),
                    "Idle timeout for on-demand services.",
                ),
                field(
                    "description",
                    "string",
                    "none",
                    "Human-readable description exposed through `/v1/models` and `/api/services`.",
                ),
                field(
                    "modality",
                    "string",
                    "`\"chat\"`",
                    "`\"chat\"` or `\"embedding\"` (see [Embedding Services](#embedding-services)). On `llama-cpp` services, `\"embedding\"` also passes `--embeddings` to llama-server. Any other string is a hard config error.",
                ),
                field(
                    "extra_args",
                    "array of string",
                    "none",
                    "Extra argv appended to the service's launch command.",
                ),
                field(
                    "extra_args_append",
                    "array of string",
                    "none",
                    "Extra argv appended to the inherited list (use with `extends`; concatenated with parent's list).",
                ),
                field(
                    "env",
                    "map string → string",
                    "none",
                    "Environment variables set on the spawned process. Accepts `{port}`, `{gpu_ids}`, `{vram_mb}`, `{model}`, `{name}` placeholders.",
                ),
                field(
                    "env_inherit",
                    "bool",
                    "`true`",
                    "Whether the child process inherits the daemon's environment (`$PATH`, `$HOME`, locale, …). Per-service `env` entries override individual inherited keys. Set `false` to start with a clean environment containing only the variables in `env` plus `CUDA_VISIBLE_DEVICES`.",
                ),
                field(
                    "drain_timeout",
                    "duration string",
                    bt_dur(DEFAULT_DRAIN_TIMEOUT_MS),
                    "Drain timeout before the supervisor escalates to SIGKILL.",
                ),
                field(
                    "extended_stream_drain",
                    "duration string",
                    bt_dur(DEFAULT_EXTENDED_STREAM_DRAIN_MS),
                    "Extra grace granted to in-flight streaming requests during drain.",
                ),
                field(
                    "max_request_duration",
                    "duration string",
                    bt_dur(DEFAULT_MAX_REQUEST_DURATION_MS),
                    "Cap on wall-clock duration of a single proxied request.",
                ),
                field(
                    "start_queue_depth",
                    "u32",
                    format!("{} (or `[defaults]` value)", bt(DEFAULT_START_QUEUE_DEPTH)),
                    "Concurrency cap on pending start requests before `QueueFull` rejection.",
                ),
                field(
                    "extends",
                    "string",
                    "none",
                    "Name of a parent service to inherit from. See [Service Inheritance](#service-inheritance).",
                ),
                field(
                    "migrate_from",
                    "string",
                    "none",
                    "Old service name to preserve database history from. See [Service Migration](#service-migration).",
                ),
            ],
        },
        SectionDoc {
            id: "service_devices",
            title: "Placement",
            fields: vec![
                field(
                    "placement",
                    "string",
                    "`\"gpu-only\"`",
                    "Placement policy (see below).",
                ),
                field(
                    "gpu_allow",
                    "array of u32",
                    "all `[devices]` GPUs",
                    "Restrict the service to these GPU ids.",
                ),
                field(
                    "gpu_headroom_mb",
                    "u64",
                    "`0`",
                    "Extra per-GPU VRAM (MiB) to keep free when placing *this* service, added on top of the global `[devices]` reserve. Lets a single model be packed more conservatively without bypassing the estimator.",
                ),
                field(
                    "placement_override",
                    "map string → u64",
                    "none",
                    "Hand-pin VRAM (MiB) per device slot. Keys: `\"cpu\"` or `\"gpu:N\"`. Overrides the estimator's per-slot distribution. Must be non-empty if present; zero values and `cpu` keys under `gpu-only` are rejected.",
                ),
                field(
                    "split",
                    "string",
                    "`\"layer\"`",
                    format!(
                        "Multi-GPU split mode for llama.cpp services: {}. Maps to llama.cpp's `--split-mode`. See [Multi-GPU split modes](#multi-gpu-split-modes) for constraints.",
                        code_values(crate::flags::split_mode::ALL)
                    ),
                ),
                field(
                    "tensor_split_weights",
                    "array of f32",
                    "none",
                    "Optional per-GPU weights for the `--tensor-split` ratio in sharded (`row`/`tensor`) modes. One positive weight per allowed GPU, in ascending GPU-id order. Unset keeps the historical equal `1,1,...` split. Use this for heterogeneous GPUs (e.g. weight by relative memory bandwidth). Weights are meaningful to four decimal places; additional precision is rounded when converting to the integer `--tensor-split` ratio. See [Multi-GPU split modes](#multi-gpu-split-modes).",
                ),
            ],
        },
        SectionDoc {
            id: "service_health",
            title: "Health Checks",
            fields: vec![
                field(
                    "http",
                    "string",
                    "`/v1/models`",
                    "HTTP path to probe for readiness. Set to `\"\"` (empty string) to disable the health check entirely - the service transitions to Running immediately after spawn, with no readiness probe.",
                ),
                field(
                    "timeout",
                    "duration string",
                    bt_dur(DEFAULT_HEALTH_TIMEOUT_MS),
                    "Per-probe timeout before a health check fails.",
                ),
                field(
                    "probe_interval",
                    "duration string",
                    bt_dur(DEFAULT_HEALTH_PROBE_INTERVAL_MS),
                    "Cadence between health probes.",
                ),
            ],
        },
        SectionDoc {
            id: "service_allocation",
            title: "Resource Allocation",
            fields: vec![
                field(
                    "mode",
                    "string",
                    "*required* (command only)",
                    "`\"static\"` or `\"dynamic\"`. Rejected for llama-cpp services. Applies to `command` services only.",
                ),
                field(
                    "vram_gb",
                    "f32",
                    "none",
                    "`static` only. VRAM to reserve, in GiB. Required for `static`.",
                ),
                field(
                    "min_vram_gb",
                    "f32",
                    "none",
                    "`dynamic` only. Minimum VRAM in GiB. Required for `dynamic`.",
                ),
                field(
                    "max_vram_gb",
                    "f32",
                    "none",
                    "`dynamic` only. Maximum VRAM in GiB. Required for `dynamic`; must be > `min_vram_gb`.",
                ),
                field(
                    "min_borrower_runtime",
                    "duration string",
                    bt_dur(DEFAULT_MIN_BORROWER_RUNTIME_MS),
                    "`dynamic` only. Balloon resolver grace period: minimum runtime a borrower must accumulate before it may be fast-killed.",
                ),
            ],
        },
        SectionDoc {
            id: "service_filters",
            title: "Request Filters",
            fields: vec![
                field(
                    "strip_params",
                    "array of string",
                    "none",
                    "JSON keys to remove from the request body before forwarding.",
                ),
                field(
                    "set_params",
                    "map string → toml value",
                    "none",
                    "JSON key/value pairs to set on the request body before forwarding.",
                ),
            ],
        },
        SectionDoc {
            id: "service_tracking",
            title: "Tracking",
            fields: vec![field(
                "cgroup_parent",
                "string",
                "none",
                "Cgroup v2 path under which the service's actual workload pids live. Used by services whose workload runs in a container and is therefore reparented out of the daemon's process tree, so descendant-pid attribution can't reach it. Pids whose `/proc/<pid>/cgroup` path equals this value or sits inside its subtree are summed into the service's observed peak. Must be an absolute cgroup path (no trailing slash).",
            )],
        },
        SectionDoc {
            id: "service_auto_restart",
            title: "Auto-restart",
            fields: vec![
                field(
                    "error_rate",
                    "table | `false`",
                    "on, with the defaults below",
                    "Error-rate watchdog. `false` disables it; a table enables it and overrides individual thresholds.",
                ),
                field(
                    "periodic",
                    "table | `false`",
                    "off",
                    "Periodic restart. Absent or `false` disables it; a table (with an `interval`) enables it.",
                ),
                field(
                    "ttft_stall",
                    "table | `false`",
                    "on, with the defaults below",
                    "Time-to-first-token stall watchdog. `false` disables it; a table enables it and overrides the timeout. Catches a wedged child that accepts a streaming request but never emits a frame — a failure the error-rate watchdog cannot see, because the request never completes. Restarts only when the whole service has gone silent, so it never fights healthy concurrent traffic.",
                ),
                field(
                    "generation_stall",
                    "table | bool",
                    "on for `llama-cpp` services, off for `command` services",
                    "Generation-stall watchdog. Polls the child's `/metrics` progress counters and restarts when they stay flat while requests are in flight — the wedge `ttft_stall` cannot see, because non-streaming requests give the proxy nothing to watch. Needs the child's `--metrics` endpoint; see the generation-stall trigger section below.",
                ),
                field(
                    "min_uptime",
                    "duration string",
                    bt_dur(DEFAULT_AUTO_RESTART_MIN_UPTIME_MS),
                    "Minimum uptime a fresh run must reach before an error-rate or generation-stall restart may fire — the anti-flap cooldown.",
                ),
                field(
                    "max_restarts",
                    "u32",
                    bt(DEFAULT_AUTO_RESTART_MAX_RESTARTS),
                    "Error-rate and stall restarts tolerated within `flap_window` before the service is disabled with reason `auto_restart_loop` instead of restarted again. Periodic restarts are intentional and do not count toward this cap.",
                ),
                field(
                    "flap_window",
                    "duration string",
                    bt_dur(DEFAULT_AUTO_RESTART_FLAP_WINDOW_MS),
                    "Sliding window over which `max_restarts` is counted.",
                ),
            ],
        },
        SectionDoc {
            id: "service_auto_restart_error_rate",
            title: "Error-rate trigger",
            fields: vec![
                field(
                    "window",
                    "duration string",
                    bt_dur(DEFAULT_AUTO_RESTART_WINDOW_MS),
                    "Rolling window over which the error rate is measured. Scoped to the current run, so a fresh process starts from zero.",
                ),
                field(
                    "max_error_rate",
                    "float (0.0–1.0]",
                    bt(DEFAULT_AUTO_RESTART_MAX_ERROR_RATE),
                    "Fraction of requests in the window that must be errors to trigger.",
                ),
                field(
                    "min_requests",
                    "u32",
                    bt(DEFAULT_AUTO_RESTART_MIN_REQUESTS),
                    "Minimum request count in the window before the ratio is trusted — stops a 2-of-2-failed service from restarting.",
                ),
                field(
                    "poll_interval",
                    "duration string",
                    bt_dur(DEFAULT_AUTO_RESTART_POLL_INTERVAL_MS),
                    "How often the watchdog queries the metrics store.",
                ),
                field(
                    "error_statuses",
                    "`\"5xx\"` | `\"4xx+5xx\"`",
                    "`5xx`",
                    "Which HTTP statuses count as errors. `5xx` (server errors only) is the default because a 4xx is usually the client's fault, not the service's. `4xx+5xx` counts any status ≥ 400.",
                ),
            ],
        },
        SectionDoc {
            id: "service_auto_restart_periodic",
            title: "Periodic trigger",
            fields: vec![
                field(
                    "interval",
                    "duration string",
                    "required",
                    "How long a run may live before a periodic restart is due, measured from when it entered `Running`.",
                ),
                field(
                    "mode",
                    "`\"immediate\"` | `\"on-idle\"` | `\"on-request\"`",
                    "`on-request`",
                    "How the restart is timed once the interval elapses. `immediate` drains and respawns at once (interrupting in-flight traffic gracefully). `on-idle` waits for a quiet window with no in-flight requests, then restarts — zero disruption, but may never fire under continuous load. `on-request` marks the run stale and lets the next request drive the restart, blocking that request on the fresh process; it guarantees the restart happens even under continuous load.",
                ),
            ],
        },
        SectionDoc {
            id: "service_auto_restart_ttft_stall",
            title: "Stall trigger",
            fields: vec![field(
                "timeout",
                "duration string",
                bt_dur(DEFAULT_AUTO_RESTART_TTFT_STALL_MS),
                "How long a streaming request may stay in-flight with no response frame before the service is restarted. A restart fires only if the *whole service* produced no frame in that window — a request merely queued behind a healthy generation does not trip it. Only streaming requests are watched (non-streaming and embeddings are bounded by `max_request_duration` instead). Does not gate on `min_uptime`; the flap cap still applies.",
            )],
        },
        SectionDoc {
            id: "service_auto_restart_generation_stall",
            title: "Generation-stall trigger",
            fields: vec![
                field(
                    "timeout",
                    "duration string",
                    bt_dur(DEFAULT_AUTO_RESTART_GENERATION_STALL_MS),
                    "How long the child's `/metrics` progress counters may stay flat, with at least one request in flight, before the service is restarted. Healthy prefill and decode both advance the counters every batch, so the default is unambiguous under load. An idle service (nothing in flight) never trips it.",
                ),
                field(
                    "poll_interval",
                    "duration string",
                    bt_dur(DEFAULT_AUTO_RESTART_GENERATION_STALL_POLL_MS),
                    "How often the child's `/metrics` endpoint is polled.",
                ),
            ],
        },
        SectionDoc {
            id: "llama_cpp",
            title: "llama-cpp field reference",
            fields: vec![
                field(
                    "model",
                    "path",
                    "*required*",
                    "Path to the GGUF model file.",
                ),
                field(
                    "mmproj",
                    "path",
                    "none",
                    "Path to an optional vision projector GGUF. Services with an `mmproj` render a purple `vision` badge.",
                ),
                field(
                    "context",
                    "u32",
                    "`4096` (estimator default)",
                    "Context window size. If unset, a warning is logged and the estimator defaults to 4096 tokens.",
                ),
                field(
                    "n_gpu_layers",
                    "i32",
                    "`-1`",
                    "Number of layers to offload to GPU. `-1` (default) offloads all layers. Must be `0` under `placement = \"cpu-only\"`.",
                ),
                field(
                    "expert_offload",
                    "string or u32",
                    "`\"off\"`",
                    "MoE expert-offload policy (see [MoE Expert Offload](#moe-expert-offload)).",
                ),
                field(
                    "flash_attn",
                    "bool",
                    "`false`",
                    "Enable flash attention. Required for quantised KV cache types (`cache_type_k`/`cache_type_v` other than `f16`).",
                ),
                field(
                    "cache_type_k",
                    "string",
                    "`f16`",
                    "KV cache type for keys. Non-`f16` values require `flash_attn = true`.",
                ),
                field(
                    "cache_type_v",
                    "string",
                    "`f16`",
                    "KV cache type for values. Non-`f16` values require `flash_attn = true`.",
                ),
                field("mmap", "bool", "`true`", "Memory-map the model file."),
                field(
                    "mlock",
                    "bool",
                    "`false`",
                    "Lock the model in RAM (prevents swapping).",
                ),
                field(
                    "parallel",
                    "u32",
                    "`1`",
                    "Request parallelism (`-np`). With a non-unified KV this splits the context budget across slots, so each request caps at `context / parallel`.",
                ),
                field(
                    "spec_type",
                    "string",
                    "none",
                    "Speculative-decoding type passed to `--spec-type` (e.g. `\"draft-mtp\"` for multi-token prediction).",
                ),
                field(
                    "spec_draft_n_max",
                    "u32",
                    "none",
                    "Max draft tokens per step (`--spec-draft-n-max`). Only meaningful when `spec_type` is set.",
                ),
                field(
                    "draft_model",
                    "path",
                    "none",
                    "Separate draft-model GGUF for speculative decoding (`-md` / `--model-draft`). Requires `spec_type` to be set.",
                ),
                field(
                    "kv_unified",
                    "bool",
                    "`false`",
                    "Use a single unified KV cache pool shared across all parallel slots (`-kvu` / `--kv-unified`). With `parallel > 1`, idle slots lend their share to active ones; total KV footprint is unchanged.",
                ),
                field(
                    "cache_idle_slots",
                    "bool",
                    "`true`",
                    "When `false`, pass `--no-cache-idle-slots` so idle slots' prompt-cache state is dropped (a stability mitigation).",
                ),
                field(
                    "metrics",
                    "bool",
                    "`false`, but auto-enabled while the `generation_stall` watchdog is on",
                    "Expose llama-server's Prometheus `/metrics` endpoint. The generation-stall watchdog needs it and passes `--metrics` automatically while active; an explicit `metrics = false` suppresses the flag and disables that watchdog.",
                ),
                field(
                    "slots",
                    "bool",
                    "`false`",
                    "Expose the `/slots` introspection endpoint. Note: reveals prompt contents - avoid on network-reachable ports.",
                ),
                field("batch_size", "u32", "none", "Context batch size (`-b`)."),
                field("ubatch_size", "u32", "none", "Physical batch size (`-ub`)."),
                field("threads", "u32", "none", "Number of CPU threads (`-t`)."),
                field(
                    "threads_batch",
                    "u32",
                    "none",
                    "Number of CPU threads for batch processing (`-tb`).",
                ),
                field(
                    "numa",
                    "string",
                    "none",
                    format!(
                        "NUMA thread-and-memory placement strategy (`--numa`): {}. Unset leaves llama.cpp's default.",
                        code_values(crate::flags::numa::ALL)
                    ),
                ),
                field("jinja", "bool", "`false`", "Use Jinja chat templates."),
                field(
                    "chat_template_file",
                    "path",
                    "none",
                    "Path to a custom chat template file.",
                ),
                field(
                    "override_tensor",
                    "array of string",
                    "none",
                    "Manual tensor placement rules (e.g. `[ \".ffn_(up|down)_exps.=CPU\" ]`). Incompatible with sharded split modes (`row`/`tensor`).",
                ),
                field(
                    "sampling",
                    "table",
                    "none",
                    "Sampling parameters (see [Sampling](#sampling)).",
                ),
                field(
                    "estimation",
                    "table",
                    "none",
                    "Estimator overrides (see [Estimation Overrides](#estimation-overrides)).",
                ),
                field(
                    "llama_server",
                    "path",
                    "daemon's `llama_server` or `$PATH`",
                    "Per-service override of the llama-server executable. Has no effect when `launcher` is set.",
                ),
                field(
                    "launcher",
                    "array of string",
                    "none",
                    "Full argv template that replaces the default `llama-server -m <model> ...` invocation (see [Custom llama-server Binary or Wrapper](#custom-llama-server-binary-or-wrapper)).",
                ),
            ],
        },
        SectionDoc {
            id: "llama_cpp_estimation",
            title: "Estimation overrides",
            fields: vec![
                field(
                    "compute_buffer_mb",
                    "u32",
                    "none",
                    "Override the estimated compute buffer size (MiB).",
                ),
                field(
                    "safety_factor",
                    "f32",
                    "none",
                    "Multiplier applied to the estimated VRAM footprint.",
                ),
                field(
                    "allow_fallback",
                    "bool",
                    "`false`",
                    "Accept the coarse fallback estimate when the GGUF's architecture isn't recognised by any per-family estimator. Unknown architectures hard-reject at config load by default so the operator either adds the arch to the right family list or explicitly opts in here.",
                ),
            ],
        },
        SectionDoc {
            id: "llama_cpp_sampling",
            title: "Sampling",
            fields: vec![
                field("temperature", "f32", "none", "Sampling temperature."),
                field("top_p", "f32", "none", "Nucleus sampling threshold."),
                field("top_k", "u32", "none", "Top-k sampling limit."),
                field("min_p", "f32", "none", "Minimum-p sampling threshold."),
                field(
                    "repeat_penalty",
                    "f32",
                    "none",
                    "Repeat penalty applied to generated tokens.",
                ),
            ],
        },
        SectionDoc {
            id: "command",
            title: "command field reference",
            fields: vec![
                field(
                    "command",
                    "array of string",
                    "*required*",
                    "argv to execute. Accepts placeholders (see below).",
                ),
                field(
                    "workdir",
                    "path",
                    "none",
                    "Working directory for the spawned process.",
                ),
                field(
                    "allocation",
                    "table",
                    "none",
                    "VRAM allocation (see [Resource Allocation](#resource-allocation)). Required for command services.",
                ),
                field(
                    "private_port",
                    "u16",
                    "auto-assigned",
                    "Upstream port ananke's reverse proxy should forward to. When absent, ananke picks one from the daemon's private-port pool and substitutes it into `command`/`env` via the `{port}` placeholder. Set explicitly when the external service binds a fixed port (e.g. a docker container exposing 18188 on the host).",
                ),
                field(
                    "shutdown_command",
                    "array of string",
                    "none",
                    "Optional argv run at drain time after SIGTERM-then-SIGKILL completes. Useful for external services that don't stop via signal - e.g. a docker-run wrapper where SIGTERM reaches the host shell but the container needs an explicit `docker stop`. Accepts the same placeholder substitutions as `command`.",
                ),
                field(
                    "openai_proxy",
                    "table",
                    "none",
                    "Opt the service into the OpenAI-compatible multiplexer (see [OpenAI Proxy](#openai-proxy)).",
                ),
            ],
        },
        SectionDoc {
            id: "openai_proxy",
            title: "OpenAI proxy",
            fields: vec![field(
                "upstream_model",
                "string",
                "none",
                "Model name the upstream server was started with (e.g. via `--served-model-name`). ananke rewrites the JSON `model` field to this value before forwarding.",
            )],
        },
    ]
}

// ── tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fmt_duration_ms() {
        assert_eq!(fmt_duration_ms(0), "0s");
        assert_eq!(fmt_duration_ms(5_000), "5s");
        assert_eq!(fmt_duration_ms(30_000), "30s");
        assert_eq!(fmt_duration_ms(60_000), "1m");
        assert_eq!(fmt_duration_ms(600_000), "10m");
        assert_eq!(fmt_duration_ms(1_800_000), "30m");
        assert_eq!(fmt_duration_ms(3_600_000), "1h");
        assert_eq!(fmt_duration_ms(7_200_000), "2h");
        // Non-clean values fall back to raw ms.
        assert_eq!(fmt_duration_ms(5_500), "5500ms");
        assert_eq!(fmt_duration_ms(90_000), "90000ms");
    }

    #[test]
    fn test_all_sections_covered() {
        let sections = all_sections();
        assert!(!sections.is_empty(), "all_sections() must not be empty");

        let mut ids = std::collections::HashSet::new();
        for s in &sections {
            assert!(!s.id.is_empty(), "section id must not be empty");
            assert!(!s.title.is_empty(), "section title must not be empty");
            assert!(ids.insert(s.id), "duplicate section id: {}", s.id);
            assert!(!s.fields.is_empty(), "section {} has no fields", s.id);
            for f in &s.fields {
                assert!(!f.name.is_empty(), "field name empty in {}", s.id);
                assert!(!f.ty.is_empty(), "field {} type empty in {}", f.name, s.id);
                assert!(
                    !f.default.is_empty(),
                    "field {} default empty in {}",
                    f.name,
                    s.id
                );
                assert!(
                    !f.description.is_empty(),
                    "field {} description empty in {}",
                    f.name,
                    s.id
                );
            }
        }

        // Spot-check: idle_timeout default references the constant correctly.
        let defaults = sections
            .iter()
            .find(|s| s.id == "defaults")
            .expect("defaults section must exist");
        let idle = defaults
            .fields
            .iter()
            .find(|f| f.name == "idle_timeout")
            .expect("idle_timeout field must exist");
        assert_eq!(idle.default, "`10m`");
    }

    #[test]
    fn test_private_port_defaults() {
        assert_eq!(DEFAULT_PRIVATE_PORT_START, 40_000);
        assert_eq!(DEFAULT_PRIVATE_PORT_END, 59_999);
    }
}
