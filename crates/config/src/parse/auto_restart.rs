//! The `[service.auto_restart]` block: the policy table itself, the
//! boolean-or-settings `Toggle` it is built from, and each trigger's
//! settings table.

use serde::Deserialize;
use smol_str::SmolStr;

/// `[service.auto_restart]` block. Opt-in self-healing for a `Running`
/// service that is alive but degraded. Two independent triggers, both
/// feeding the existing drain → respawn cycle; guardrails below bound
/// the churn.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawAutoRestart {
    /// Error-rate watchdog. Enabled by default (write `error_rate = false`
    /// to opt a service out). A table tunes the thresholds.
    pub error_rate: Option<Toggle<RawErrorRateSettings>>,
    /// Periodic restart. Disabled by default; a table with an `interval`
    /// enables it (write `periodic = false` to be explicit).
    pub periodic: Option<Toggle<RawPeriodicSettings>>,
    /// Time-to-first-token stall watchdog. Enabled by default (write
    /// `ttft_stall = false` to opt a service out). A table tunes the timeout.
    pub ttft_stall: Option<Toggle<RawTtftStallSettings>>,
    /// Generation-stall watchdog: polls the child's Prometheus `/metrics`
    /// progress counters and restarts when they stay flat with requests
    /// in flight. Enabled by default on the `llama-cpp` template (write
    /// `generation_stall = false` to opt out); disabled by default on the
    /// `command` template. A table tunes the timeout and poll interval.
    pub generation_stall: Option<Toggle<RawGenerationStallSettings>>,
    /// Speculative-decoding collapse watchdog: restarts when a run that
    /// previously accepted draft tokens stops accepting any (e.g. all-NaN
    /// logits after state corruption — a failure that still returns HTTP
    /// 200). Enabled by default on `llama-cpp` services that configure
    /// speculative decoding (`spec_type`); an explicit per-service enable
    /// without `spec_type` is a validation error. A table tunes the window
    /// and thresholds.
    pub spec_collapse: Option<Toggle<RawSpecCollapseSettings>>,
    /// Minimum uptime a fresh run must reach before another auto-restart
    /// may fire — the anti-flap cooldown.
    pub min_uptime: Option<String>,
    /// How many auto-restarts within [`Self::flap_window`] are tolerated
    /// before the service is disabled with `AutoRestartLoop` instead of
    /// restarted again.
    pub max_restarts: Option<u32>,
    /// Sliding window over which [`Self::max_restarts`] is counted.
    pub flap_window: Option<String>,
}

/// A trigger that accepts either a bare boolean toggle or a settings
/// table. `true` (or an empty table) means "enabled with defaults";
/// `false` means "disabled". A populated table both enables the trigger
/// and overrides individual thresholds.
///
/// Deserialized by hand rather than `#[serde(untagged)]` so the inner
/// settings struct's `deny_unknown_fields` is honoured — untagged enums
/// silently drop unknown keys, which would let a typo'd threshold pass.
#[derive(Debug, Clone)]
pub enum Toggle<T> {
    /// Bare boolean form: `true` enables with defaults, `false` disables.
    Enabled(bool),
    /// Settings-table form: enables the trigger and overrides thresholds.
    Settings(Box<T>),
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Toggle<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ToggleVisitor<T>(std::marker::PhantomData<T>);

        impl<'de, T: Deserialize<'de>> serde::de::Visitor<'de> for ToggleVisitor<T> {
            type Value = Toggle<T>;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a boolean or a settings table")
            }

            fn visit_bool<E>(self, v: bool) -> Result<Self::Value, E> {
                Ok(Toggle::Enabled(v))
            }

            fn visit_map<A>(self, map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let settings = T::deserialize(serde::de::value::MapAccessDeserializer::new(map))?;
                Ok(Toggle::Settings(Box::new(settings)))
            }
        }

        deserializer.deserialize_any(ToggleVisitor(std::marker::PhantomData))
    }
}

/// `[service.auto_restart.error_rate]` thresholds. Every field is optional
/// and falls back to a built-in default during validation.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawErrorRateSettings {
    /// Rolling window over which the error rate is measured.
    pub window: Option<String>,
    /// Fraction of requests in the window that must be errors to trigger
    /// (0.0–1.0).
    pub max_error_rate: Option<f64>,
    /// Minimum request count in the window before the ratio is trusted —
    /// stops a 2-of-2-failed service from restarting.
    pub min_requests: Option<u32>,
    /// How often the watchdog queries the metrics store.
    pub poll_interval: Option<String>,
    /// Which HTTP statuses count as errors: `"5xx"` (server errors only,
    /// the default) or `"4xx+5xx"` (any status ≥ 400).
    pub error_statuses: Option<SmolStr>,
}

/// `[service.auto_restart.ttft_stall]` settings. Only the timeout is tunable.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawTtftStallSettings {
    /// How long a request may stay in-flight with no response token before
    /// the service is restarted.
    pub timeout: Option<String>,
}

/// `[service.auto_restart.generation_stall]` settings.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawGenerationStallSettings {
    /// How long the child's progress counters may stay flat, with at least
    /// one request in flight, before the service is restarted.
    pub timeout: Option<String>,
    /// How often the child's `/metrics` endpoint is polled.
    pub poll_interval: Option<String>,
}

/// `[service.auto_restart.spec_collapse]` thresholds. Every field is
/// optional and falls back to a built-in default during validation.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawSpecCollapseSettings {
    /// Rolling window over which draft acceptance is measured.
    pub window: Option<String>,
    /// Minimum count of drafted tokens in the window before an all-zero
    /// acceptance is trusted — stops a couple of unlucky short generations
    /// from restarting a healthy service. Tokens rather than requests: long
    /// generations arrive slowly but draft thousands of tokens each.
    pub min_draft_tokens: Option<u64>,
    /// How often the watchdog queries the metrics store.
    pub poll_interval: Option<String>,
}

/// `[service.auto_restart.periodic]` settings.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawPeriodicSettings {
    /// How long a run may live before a periodic restart is due. Required
    /// when periodic is enabled.
    pub interval: Option<String>,
    /// When the interval elapses, how the restart is timed: `"immediate"`,
    /// `"on-idle"`, or `"on-request"` (the default).
    pub mode: Option<SmolStr>,
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::parse::parse_toml;

    #[test]
    fn parses_auto_restart_block() {
        let toml = r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435

[service.auto_restart]
min_uptime = "10m"
max_restarts = 5
flap_window = "1h"
error_rate = { window = "2m", max_error_rate = 0.6, min_requests = 30, error_statuses = "4xx+5xx" }
periodic = { interval = "6h", mode = "on-idle" }
"#;
        let cfg = parse_toml(toml).unwrap();
        let ar = cfg.services[0].common().auto_restart.as_ref().unwrap();
        assert_eq!(ar.min_uptime.as_deref(), Some("10m"));
        assert_eq!(ar.max_restarts, Some(5));
        let Some(Toggle::Settings(er)) = &ar.error_rate else {
            panic!("expected error_rate settings table");
        };
        assert_eq!(er.window.as_deref(), Some("2m"));
        assert_eq!(er.max_error_rate, Some(0.6));
        assert_eq!(er.error_statuses.as_deref(), Some("4xx+5xx"));
        let Some(Toggle::Settings(p)) = &ar.periodic else {
            panic!("expected periodic settings table");
        };
        assert_eq!(p.interval.as_deref(), Some("6h"));
        assert_eq!(p.mode.as_deref(), Some("on-idle"));
    }
    #[test]
    fn parses_error_rate_bool_toggle() {
        let toml = r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435

[service.auto_restart]
error_rate = false
"#;
        let cfg = parse_toml(toml).unwrap();
        let ar = cfg.services[0].common().auto_restart.as_ref().unwrap();
        assert!(matches!(ar.error_rate, Some(Toggle::Enabled(false))));
    }
    #[test]
    fn parses_ttft_stall_bool_toggle() {
        let toml = r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435

[service.auto_restart]
ttft_stall = false
"#;
        let cfg = parse_toml(toml).unwrap();
        let ar = cfg.services[0].common().auto_restart.as_ref().unwrap();
        assert!(matches!(ar.ttft_stall, Some(Toggle::Enabled(false))));
    }
    #[test]
    fn ttft_stall_table_rejects_unknown_field() {
        let toml = r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435

[service.auto_restart]
ttft_stall = { timeuot = "90s" }
"#;
        let err = parse_toml(toml);
        assert!(
            err.is_err(),
            "expected parse error for typo'd ttft_stall field, got {:?}",
            err.ok()
        );
    }
    #[test]
    fn parses_generation_stall_bool_toggle() {
        let toml = r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435

[service.auto_restart]
generation_stall = false
"#;
        let cfg = parse_toml(toml).unwrap();
        let ar = cfg.services[0].common().auto_restart.as_ref().unwrap();
        assert!(matches!(ar.generation_stall, Some(Toggle::Enabled(false))));
    }
    #[test]
    fn parses_generation_stall_settings_table() {
        let toml = r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435

[service.auto_restart]
generation_stall = { timeout = "2m", poll_interval = "10s" }
"#;
        let cfg = parse_toml(toml).unwrap();
        let ar = cfg.services[0].common().auto_restart.as_ref().unwrap();
        let Some(Toggle::Settings(gs)) = &ar.generation_stall else {
            panic!("expected generation_stall settings table");
        };
        assert_eq!(gs.timeout.as_deref(), Some("2m"));
        assert_eq!(gs.poll_interval.as_deref(), Some("10s"));
    }
    #[test]
    fn generation_stall_table_rejects_unknown_field() {
        let toml = r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435

[service.auto_restart]
generation_stall = { timeuot = "2m" }
"#;
        let err = parse_toml(toml);
        assert!(
            err.is_err(),
            "expected parse error for typo'd generation_stall field, got {:?}",
            err.ok()
        );
    }
    #[test]
    fn auto_restart_table_rejects_unknown_field() {
        // The custom `Toggle` deserialize must honour the settings struct's
        // deny_unknown_fields — a plain `#[serde(untagged)]` enum would
        // silently drop this typo.
        let toml = r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435

[service.auto_restart]
error_rate = { windwo = "2m" }
"#;
        let err = parse_toml(toml);
        assert!(
            err.is_err(),
            "expected parse error for typo'd error_rate field, got {:?}",
            err.ok()
        );
    }
}
