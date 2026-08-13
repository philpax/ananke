//! Resolve a service's `[auto_restart]` block — the guardrails, the trigger
//! toggles, and the periodic-restart timer — into [`AutoRestartSettings`].

use smol_str::SmolStr;

use crate::{
    docs::{
        DEFAULT_AUTO_RESTART_FLAP_WINDOW_MS, DEFAULT_AUTO_RESTART_MAX_RESTARTS,
        DEFAULT_AUTO_RESTART_MIN_UPTIME_MS,
    },
    fields,
    parse::{RawAutoRestart, RawPeriodicSettings, Toggle},
    validate::{
        AutoRestartSettings, ConfigDiagnostic, DEFAULT_AUTO_RESTART_PERIODIC_MODE,
        ErrorRateTrigger, GenerationStallTrigger, PeriodicMode, PeriodicTrigger,
        SpecCollapseTrigger, Template, TtftStallTrigger, ValidationErrorCode, parse_duration_ms,
        validate_error_rate, validate_generation_stall, validate_spec_collapse,
        validate_ttft_stall,
    },
};

/// `has_spec_type` gates the spec-collapse watchdog: only a llama-cpp
/// service with `spec_type` produces draft counts, so the trigger is
/// meaningless anywhere else. `from_service_block` distinguishes the
/// service's own `auto_restart` block from an inherited
/// `[defaults.auto_restart]`: an explicit per-service enable on a service
/// that can never honor it is a hard error, while a fleet-wide default
/// silently resolves to disabled on the services it doesn't apply to.
pub(crate) fn validate_auto_restart(
    name: &SmolStr,
    raw: Option<&RawAutoRestart>,
    template: Template,
    has_spec_type: bool,
    from_service_block: bool,
) -> Result<AutoRestartSettings, crate::validate::ConfigDiagnostic> {
    // The generation-stall watchdog defaults per template: on for llama-cpp
    // (where ananke builds the argv and can inject `--metrics`), off for
    // command services (where it cannot; explicit opt-in soft-probes instead).
    let default_generation_stall = match template {
        Template::LlamaCpp => Some(GenerationStallTrigger::default()),
        Template::Command => None,
    };
    // The spec_collapse watchdog defaults on only where it can actually
    // observe anything: a llama-cpp service with `spec_type` set. Responses
    // elsewhere carry no draft counts.
    let default_spec_collapse = has_spec_type.then(SpecCollapseTrigger::default);
    let Some(raw) = raw else {
        return Ok(AutoRestartSettings {
            generation_stall: default_generation_stall,
            spec_collapse: default_spec_collapse,
            ..AutoRestartSettings::default()
        });
    };

    let dur = |field: &str, s: &str| {
        parse_duration_ms(s).map_err(|error| {
            ConfigDiagnostic::constraint(
                ValidationErrorCode::TemplateConstraint,
                Some(name.to_string()),
                &[&format!("auto_restart.{field}")],
                error.to_string(),
            )
        })
    };

    // Error-rate watchdog is on by default; only an explicit `false` disables it.
    let error_rate = match &raw.error_rate {
        None | Some(Toggle::Enabled(true)) => Some(ErrorRateTrigger::default()),
        Some(Toggle::Enabled(false)) => None,
        Some(Toggle::Settings(s)) => Some(validate_error_rate(name, s)?),
    };

    // Periodic is off by default; a table (with an interval) enables it. A bare
    // `true` is rejected because there is no interval to restart on.
    let periodic = match &raw.periodic {
        None | Some(Toggle::Enabled(false)) => None,
        Some(Toggle::Enabled(true)) => {
            return Err(ConfigDiagnostic::constraint(
                ValidationErrorCode::TemplateConstraint,
                Some(name.to_string()),
                &[fields::auto_restart::PERIODIC],
                "auto_restart.periodic = true needs an interval; write `periodic = { interval = \"6h\" }`".to_string(),
            ));
        }
        Some(Toggle::Settings(s)) => Some(validate_periodic(name, s)?),
    };

    // Stall watchdog is on by default; only an explicit `false` disables it.
    let ttft_stall = match &raw.ttft_stall {
        None | Some(Toggle::Enabled(true)) => Some(TtftStallTrigger::default()),
        Some(Toggle::Enabled(false)) => None,
        Some(Toggle::Settings(s)) => Some(validate_ttft_stall(name, s)?),
    };

    // Generation-stall watchdog defaults per template (see above); an
    // explicit `true` or table enables it on either template.
    let generation_stall = match &raw.generation_stall {
        None => default_generation_stall,
        Some(Toggle::Enabled(true)) => Some(GenerationStallTrigger::default()),
        Some(Toggle::Enabled(false)) => None,
        Some(Toggle::Settings(s)) => Some(validate_generation_stall(name, s)?),
    };

    // Spec-collapse watchdog: an explicit enable on a service without
    // `spec_type` is rejected when it comes from the service's own block
    // (it could never fire), and silently resolved to disabled when it
    // comes from a fleet-wide default.
    let spec_collapse = match &raw.spec_collapse {
        None => default_spec_collapse,
        Some(Toggle::Enabled(false)) => None,
        Some(_) if !has_spec_type => {
            if from_service_block {
                return Err(ConfigDiagnostic::constraint(
                    ValidationErrorCode::TemplateConstraint,
                    Some(name.to_string()),
                    &[fields::auto_restart::SPEC_COLLAPSE],
                    "auto_restart.spec_collapse requires spec_type to be set (without speculative decoding, responses carry no draft counts and the watchdog can never fire)".to_string(),
                ));
            }
            None
        }
        Some(Toggle::Enabled(true)) => Some(SpecCollapseTrigger::default()),
        Some(Toggle::Settings(s)) => Some(validate_spec_collapse(name, s)?),
    };

    let min_uptime_ms = raw
        .min_uptime
        .as_deref()
        .map(|s| dur("min_uptime", s))
        .transpose()?
        .unwrap_or(DEFAULT_AUTO_RESTART_MIN_UPTIME_MS);
    let max_restarts = raw
        .max_restarts
        .unwrap_or(DEFAULT_AUTO_RESTART_MAX_RESTARTS);
    let flap_window_ms = raw
        .flap_window
        .as_deref()
        .map(|s| dur("flap_window", s))
        .transpose()?
        .unwrap_or(DEFAULT_AUTO_RESTART_FLAP_WINDOW_MS);

    Ok(AutoRestartSettings {
        error_rate,
        periodic,
        ttft_stall,
        generation_stall,
        spec_collapse,
        min_uptime_ms,
        max_restarts,
        flap_window_ms,
    })
}

pub(crate) fn validate_periodic(
    name: &SmolStr,
    s: &RawPeriodicSettings,
) -> Result<PeriodicTrigger, crate::validate::ConfigDiagnostic> {
    let interval_ms = match s.interval.as_deref() {
        Some(x) => parse_duration_ms(x).map_err(|error| {
            ConfigDiagnostic::constraint(
                ValidationErrorCode::TemplateConstraint,
                Some(name.to_string()),
                &[fields::auto_restart::PERIODIC_INTERVAL],
                error.to_string(),
            )
        })?,
        None => {
            return Err(ConfigDiagnostic::constraint(
                ValidationErrorCode::TemplateConstraint,
                Some(name.to_string()),
                &[fields::auto_restart::PERIODIC],
                "auto_restart.periodic requires an `interval`".to_string(),
            ));
        }
    };
    let mode = match s.mode.as_deref() {
        None => DEFAULT_AUTO_RESTART_PERIODIC_MODE,
        Some("immediate") => PeriodicMode::Immediate,
        Some("on-idle") => PeriodicMode::OnIdle,
        Some("on-request") => PeriodicMode::OnRequest,
        Some(other) => {
            return Err(ConfigDiagnostic::constraint(
                ValidationErrorCode::TemplateConstraint,
                Some(name.to_string()),
                &[fields::auto_restart::PERIODIC_MODE],
                format!(
                    "auto_restart.periodic.mode must be `immediate`, `on-idle`, or `on-request`, got `{value}`",
                    value = other
                ),
            ));
        }
    };
    Ok(PeriodicTrigger { interval_ms, mode })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        docs::{
            DEFAULT_AUTO_RESTART_MIN_REQUESTS, DEFAULT_AUTO_RESTART_TTFT_STALL_MS,
            DEFAULT_AUTO_RESTART_WINDOW_MS,
        },
        validate::{
            ErrorStatusClass,
            test_fixtures::{parse_and_merge, svc_with_auto_restart},
            validate,
        },
    };

    #[test]
    fn auto_restart_defaults_error_rate_on_periodic_off() {
        // No block at all → error-rate watchdog on with defaults, periodic off.
        let svc = svc_with_auto_restart("").unwrap();
        let ar = &svc.auto_restart;
        let er = ar.error_rate.as_ref().expect("error-rate on by default");
        assert_eq!(er.window_ms, DEFAULT_AUTO_RESTART_WINDOW_MS);
        assert_eq!(er.min_requests, DEFAULT_AUTO_RESTART_MIN_REQUESTS);
        assert_eq!(er.statuses, ErrorStatusClass::ServerOnly);
        assert!(ar.periodic.is_none(), "periodic off by default");
        let stall = ar.ttft_stall.as_ref().expect("stall on by default");
        assert_eq!(stall.timeout_ms, DEFAULT_AUTO_RESTART_TTFT_STALL_MS);
        assert_eq!(ar.min_uptime_ms, DEFAULT_AUTO_RESTART_MIN_UPTIME_MS);
        assert_eq!(ar.max_restarts, DEFAULT_AUTO_RESTART_MAX_RESTARTS);
    }

    #[test]
    fn auto_restart_error_rate_false_disables_it() {
        // Only error-rate is turned off; the stall watchdog stays on, so the
        // policy is still active.
        let svc = svc_with_auto_restart("auto_restart = { error_rate = false }").unwrap();
        assert!(svc.auto_restart.error_rate.is_none());
        assert!(svc.auto_restart.ttft_stall.is_some());
        assert!(svc.auto_restart.any_enabled());
    }

    #[test]
    fn auto_restart_all_triggers_false_disables_policy() {
        let svc = svc_with_auto_restart(
            "auto_restart = { error_rate = false, ttft_stall = false, generation_stall = false, spec_collapse = false }",
        )
        .unwrap();
        assert!(svc.auto_restart.error_rate.is_none());
        assert!(svc.auto_restart.ttft_stall.is_none());
        assert!(svc.auto_restart.generation_stall.is_none());
        assert!(svc.auto_restart.spec_collapse.is_none());
        assert!(!svc.auto_restart.any_enabled());
    }
    #[test]
    fn auto_restart_periodic_table_enables_with_defaults() {
        let svc = svc_with_auto_restart("auto_restart.periodic = { interval = \"6h\" }").unwrap();
        let p = svc
            .auto_restart
            .periodic
            .as_ref()
            .expect("periodic enabled");
        assert_eq!(p.interval_ms, 6 * 60 * 60 * 1000);
        assert_eq!(p.mode, PeriodicMode::OnRequest);
        // Error-rate stays on — the block only touched periodic.
        assert!(svc.auto_restart.error_rate.is_some());
    }
    #[test]
    fn auto_restart_rejects_bad_values() {
        assert!(
            svc_with_auto_restart("auto_restart.error_rate = { max_error_rate = 1.5 }").is_err()
        );
        assert!(
            svc_with_auto_restart("auto_restart.error_rate = { error_statuses = \"3xx\" }")
                .is_err()
        );
        assert!(
            svc_with_auto_restart(
                "auto_restart.periodic = { interval = \"6h\", mode = \"eager\" }"
            )
            .is_err()
        );
        // periodic without an interval is meaningless.
        assert!(svc_with_auto_restart("auto_restart.periodic = { mode = \"immediate\" }").is_err());
        assert!(svc_with_auto_restart("auto_restart.periodic = true").is_err());
    }

    #[test]
    fn auto_restart_resolves_from_defaults_whole_block() {
        // A service with no auto_restart block inherits `[defaults.auto_restart]`.
        let src = r#"
[defaults.auto_restart]
error_rate = false
periodic = { interval = "4h", mode = "immediate" }

[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "cpu-only"
"#;
        let cfg = parse_and_merge(src);
        let svc = &validate(&cfg).unwrap().services[0];
        assert!(svc.auto_restart.error_rate.is_none());
        let p = svc.auto_restart.periodic.as_ref().unwrap();
        assert_eq!(p.mode, PeriodicMode::Immediate);
        assert_eq!(p.interval_ms, 4 * 60 * 60 * 1000);
    }
}
