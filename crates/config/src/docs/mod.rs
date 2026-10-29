//! Config-default constants and the documentation descriptor table.
//!
//! The `DEFAULT_*` constants are the single source of truth for every
//! default the daemon applies when a config field is omitted. The
//! descriptor table (`all_sections`) is what the `gen-config-docs` xtask
//! renders into `docs/configuration.md`; because it references these
//! constants directly, changing a constant changes the generated doc and
//! trips `--check` in CI.
//!
//! The descriptors themselves live in per-section sibling modules (see
//! below); this module holds the `DEFAULT_*` constants, the shared
//! `SectionDoc`/`FieldDoc` types and builder helpers, and the `all_sections`
//! assembly that stitches the sibling modules' output back into the single
//! vector the xtask renders.

use serde::Serialize;

mod auto_restart;
mod command;
mod container;
mod daemon;
mod llama_cpp;
mod service;

// ── default constants ───────────────────────────────────────────────────

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

/// Default tail grace before SIGTERM after in-flight requests end (30 seconds).
pub const DEFAULT_DRAIN_TIMEOUT_MS: u64 = 30_000;

/// Default SIGTERM-to-SIGKILL grace for a running service (10 seconds).
pub const DEFAULT_STOP_GRACE_MS: u64 = 10_000;

/// Default extra grace granted to in-flight streaming requests during drain
/// (30 seconds).
pub const DEFAULT_EXTENDED_STREAM_DRAIN_MS: u64 = 30_000;

/// Default cap on the wall-clock duration of a single proxied request
/// (10 minutes).
pub const DEFAULT_MAX_REQUEST_DURATION_MS: u64 = 600_000;

/// Default service scheduling priority (higher wins eviction contests).
pub const DEFAULT_SERVICE_PRIORITY: u8 = 50;

/// Prefix reserved for manual reservations' allocation-table keys
/// (`reservation:<id>`); no service name may start with it.
pub const RESERVED_SERVICE_NAME_PREFIX: &str = "reservation:";

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

/// Default rolling window for the auto-restart spec_collapse watchdog (2
/// minutes). Calibrated against the 2026-07-24 all-NaN-logits incident:
/// healthy speculative traffic held 60–100 % per-request acceptance, while
/// the wedged run served exactly zero accepted draft tokens on every request
/// for 45+ minutes.
pub const DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_WINDOW_MS: u64 = 120_000;

/// Default minimum count of drafted tokens in the window before an
/// all-zero acceptance is trusted. Counted in tokens rather than requests:
/// long generations arrive slowly but draft thousands of tokens each, and a
/// healthy pairing accepting none of this many drafted tokens does not
/// happen. One short unlucky generation stays under the floor.
pub const DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_MIN_DRAFT_TOKENS: u64 = 200;

/// Default cadence at which the spec_collapse watchdog polls the metrics
/// store (30 s).
pub const DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_POLL_MS: u64 = 30_000;

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
pub(crate) fn field(
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
pub(crate) fn code_values(values: &[&str]) -> String {
    values
        .iter()
        .map(|v| format!("`\"{v}\"`"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Wrap a value in backticks for the Default column.
pub(crate) fn bt(v: impl std::fmt::Display) -> String {
    format!("`{v}`")
}

/// Wrap a duration constant in backticks.
pub(crate) fn bt_dur(ms: u64) -> String {
    bt(fmt_duration_ms(ms))
}

/// Return every field-reference table in `docs/configuration.md`.
///
/// The descriptor table is hand-maintained: adding a new config field
/// requires adding an entry here for it to appear in the generated docs.
/// CI's `--check` catches default-value drift (a constant change) but not
/// a missing entry — code review is the backstop for the latter.
pub fn all_sections() -> Vec<SectionDoc> {
    let mut sections = daemon::sections();
    sections.extend(service::sections());
    sections.extend(auto_restart::sections());
    sections.extend(llama_cpp::sections());
    sections.extend(command::sections());
    sections.extend(container::sections());
    sections
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
