//! Per-trigger validation for the auto-restart watchdogs: spec collapse,
//! generation stall, TTFT stall, and error rate.

use smol_str::SmolStr;

use crate::{
    fields,
    parse::{
        RawErrorRateSettings, RawGenerationStallSettings, RawSpecCollapseSettings,
        RawTtftStallSettings,
    },
    validate::{
        ConfigDiagnostic, ErrorRateTrigger, ErrorStatusClass, GenerationStallTrigger,
        SpecCollapseTrigger, TtftStallTrigger, ValidationErrorCode, parse_duration_ms,
    },
};

pub(crate) fn validate_spec_collapse(
    name: &SmolStr,
    s: &RawSpecCollapseSettings,
) -> Result<SpecCollapseTrigger, crate::validate::ConfigDiagnostic> {
    let d = SpecCollapseTrigger::default();
    let field = |field: &str, x: &str| {
        parse_duration_ms(x).map_err(|error| {
            ConfigDiagnostic::constraint(
                ValidationErrorCode::TemplateConstraint,
                Some(name.to_string()),
                &[&format!("auto_restart.spec_collapse.{field}")],
                error.to_string(),
            )
        })
    };
    let window_ms = s
        .window
        .as_deref()
        .map(|x| field("window", x))
        .transpose()?
        .unwrap_or(d.window_ms);
    let min_draft_tokens = s.min_draft_tokens.unwrap_or(d.min_draft_tokens);
    let poll_interval_ms = s
        .poll_interval
        .as_deref()
        .map(|x| field("poll_interval", x))
        .transpose()?
        .unwrap_or(d.poll_interval_ms);
    if window_ms == 0 {
        return Err(ConfigDiagnostic::constraint(
            ValidationErrorCode::TemplateConstraint,
            Some(name.to_string()),
            &[fields::auto_restart::SPEC_COLLAPSE_WINDOW],
            "auto_restart.spec_collapse.window must be greater than zero".to_string(),
        ));
    }
    if min_draft_tokens == 0 {
        return Err(ConfigDiagnostic::constraint(
            ValidationErrorCode::TemplateConstraint,
            Some(name.to_string()),
            &[fields::auto_restart::SPEC_COLLAPSE_MIN_DRAFT_TOKENS],
            "auto_restart.spec_collapse.min_draft_tokens must be greater than zero".to_string(),
        ));
    }
    if poll_interval_ms == 0 {
        return Err(ConfigDiagnostic::constraint(
            ValidationErrorCode::TemplateConstraint,
            Some(name.to_string()),
            &[fields::auto_restart::SPEC_COLLAPSE_POLL_INTERVAL],
            "auto_restart.spec_collapse.poll_interval must be greater than zero".to_string(),
        ));
    }
    Ok(SpecCollapseTrigger {
        window_ms,
        min_draft_tokens,
        poll_interval_ms,
    })
}

pub(crate) fn validate_generation_stall(
    name: &SmolStr,
    s: &RawGenerationStallSettings,
) -> Result<GenerationStallTrigger, crate::validate::ConfigDiagnostic> {
    let d = GenerationStallTrigger::default();
    let field = |field: &str, x: &str| {
        parse_duration_ms(x).map_err(|error| {
            ConfigDiagnostic::constraint(
                ValidationErrorCode::TemplateConstraint,
                Some(name.to_string()),
                &[&format!("auto_restart.generation_stall.{field}")],
                error.to_string(),
            )
        })
    };
    let timeout_ms = s
        .timeout
        .as_deref()
        .map(|x| field("timeout", x))
        .transpose()?
        .unwrap_or(d.timeout_ms);
    let poll_interval_ms = s
        .poll_interval
        .as_deref()
        .map(|x| field("poll_interval", x))
        .transpose()?
        .unwrap_or(d.poll_interval_ms);
    if timeout_ms == 0 {
        return Err(ConfigDiagnostic::constraint(
            ValidationErrorCode::TemplateConstraint,
            Some(name.to_string()),
            &[fields::auto_restart::GENERATION_STALL_TIMEOUT],
            "auto_restart.generation_stall.timeout must be greater than zero".to_string(),
        ));
    }
    if poll_interval_ms == 0 {
        return Err(ConfigDiagnostic::constraint(
            ValidationErrorCode::TemplateConstraint,
            Some(name.to_string()),
            &[fields::auto_restart::GENERATION_STALL_POLL_INTERVAL],
            "auto_restart.generation_stall.poll_interval must be greater than zero".to_string(),
        ));
    }
    Ok(GenerationStallTrigger {
        timeout_ms,
        poll_interval_ms,
    })
}

pub(crate) fn validate_ttft_stall(
    name: &SmolStr,
    s: &RawTtftStallSettings,
) -> Result<TtftStallTrigger, crate::validate::ConfigDiagnostic> {
    let d = TtftStallTrigger::default();
    let timeout_ms = s
        .timeout
        .as_deref()
        .map(|x| {
            parse_duration_ms(x).map_err(|error| {
                ConfigDiagnostic::constraint(
                    ValidationErrorCode::TemplateConstraint,
                    Some(name.to_string()),
                    &[fields::auto_restart::TTFT_STALL_TIMEOUT],
                    error.to_string(),
                )
            })
        })
        .transpose()?
        .unwrap_or(d.timeout_ms);
    if timeout_ms == 0 {
        return Err(ConfigDiagnostic::constraint(
            ValidationErrorCode::TemplateConstraint,
            Some(name.to_string()),
            &[fields::auto_restart::TTFT_STALL_TIMEOUT],
            "auto_restart.ttft_stall.timeout must be greater than zero".to_string(),
        ));
    }
    Ok(TtftStallTrigger { timeout_ms })
}

pub(crate) fn validate_error_rate(
    name: &SmolStr,
    s: &RawErrorRateSettings,
) -> Result<ErrorRateTrigger, crate::validate::ConfigDiagnostic> {
    let d = ErrorRateTrigger::default();
    let window_ms = s
        .window
        .as_deref()
        .map(|x| {
            parse_duration_ms(x).map_err(|error| {
                ConfigDiagnostic::constraint(
                    ValidationErrorCode::TemplateConstraint,
                    Some(name.to_string()),
                    &[fields::auto_restart::ERROR_RATE_WINDOW],
                    error.to_string(),
                )
            })
        })
        .transpose()?
        .unwrap_or(d.window_ms);
    let max_error_rate = match s.max_error_rate {
        None => d.max_error_rate,
        Some(r) if r > 0.0 && r <= 1.0 => r,
        Some(r) => {
            return Err(ConfigDiagnostic::constraint(
                ValidationErrorCode::TemplateConstraint,
                Some(name.to_string()),
                &[fields::auto_restart::ERROR_RATE_MAX_ERROR_RATE],
                format!(
                    "auto_restart.error_rate.max_error_rate must be in (0.0, 1.0], got {value}",
                    value = r
                ),
            ));
        }
    };
    let poll_interval_ms = s
        .poll_interval
        .as_deref()
        .map(|x| {
            parse_duration_ms(x).map_err(|error| {
                ConfigDiagnostic::constraint(
                    ValidationErrorCode::TemplateConstraint,
                    Some(name.to_string()),
                    &[fields::auto_restart::ERROR_RATE_POLL_INTERVAL],
                    error.to_string(),
                )
            })
        })
        .transpose()?
        .unwrap_or(d.poll_interval_ms);
    let statuses = match s.error_statuses.as_deref() {
        None => d.statuses,
        Some("5xx") => ErrorStatusClass::ServerOnly,
        Some("4xx+5xx") => ErrorStatusClass::ClientAndServer,
        Some(other) => {
            return Err(ConfigDiagnostic::constraint(
                ValidationErrorCode::TemplateConstraint,
                Some(name.to_string()),
                &[fields::auto_restart::ERROR_RATE_ERROR_STATUSES],
                format!(
                    "auto_restart.error_rate.error_statuses must be `5xx` or `4xx+5xx`, got `{value}`",
                    value = other
                ),
            ));
        }
    };
    Ok(ErrorRateTrigger {
        window_ms,
        max_error_rate,
        min_requests: s.min_requests.unwrap_or(d.min_requests),
        poll_interval_ms,
        statuses,
    })
}

#[cfg(test)]
mod tests {
    use ananke_errors::ExpectedError;

    use super::*;
    use crate::{
        docs::{
            DEFAULT_AUTO_RESTART_GENERATION_STALL_MS,
            DEFAULT_AUTO_RESTART_GENERATION_STALL_POLL_MS,
            DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_MIN_DRAFT_TOKENS,
            DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_POLL_MS,
            DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_WINDOW_MS,
        },
        fields,
        validate::{
            ServiceConfig,
            test_fixtures::{
                parse_and_merge, svc_with_auto_restart, svc_with_auto_restart_diagnostics,
            },
            validate,
        },
    };

    #[test]
    fn auto_restart_ttft_stall_false_disables_it() {
        let svc = svc_with_auto_restart("auto_restart = { ttft_stall = false }").unwrap();
        assert!(svc.auto_restart.ttft_stall.is_none());
        // Error-rate stays on — the block only touched the stall trigger.
        assert!(svc.auto_restart.error_rate.is_some());
    }

    #[test]
    fn auto_restart_ttft_stall_timeout_override() {
        let svc = svc_with_auto_restart("auto_restart.ttft_stall = { timeout = \"90s\" }").unwrap();
        let stall = svc.auto_restart.ttft_stall.as_ref().unwrap();
        assert_eq!(stall.timeout_ms, 90 * 1000);
        // Error-rate stays on with defaults.
        assert!(svc.auto_restart.error_rate.is_some());
    }

    #[test]
    fn auto_restart_ttft_stall_rejects_zero_timeout() {
        assert!(svc_with_auto_restart("auto_restart.ttft_stall = { timeout = \"0s\" }").is_err());
    }
    fn command_svc_with_auto_restart(block: &str) -> Result<ServiceConfig, ExpectedError> {
        let src = format!(
            r#"
[[service]]
name = "demo"
template = "command"
command = ["/bin/true"]
port = 11435
allocation.mode = "static"
allocation.reserve_gb = 4
devices.placement = "cpu-only"
{block}
"#
        );
        let cfg = parse_and_merge(&src);
        validate(&cfg)
            .map(|ec| ec.services.into_iter().next().unwrap())
            .map_err(|report| report.into_expected_error(std::path::PathBuf::from("<config>")))
    }

    #[test]
    fn auto_restart_generation_stall_defaults_on_for_llamacpp() {
        let svc = svc_with_auto_restart("").unwrap();
        let gs = svc
            .auto_restart
            .generation_stall
            .as_ref()
            .expect("generation stall on by default for llama-cpp");
        assert_eq!(gs.timeout_ms, DEFAULT_AUTO_RESTART_GENERATION_STALL_MS);
        assert_eq!(
            gs.poll_interval_ms,
            DEFAULT_AUTO_RESTART_GENERATION_STALL_POLL_MS
        );
    }

    #[test]
    fn auto_restart_generation_stall_defaults_off_for_command() {
        // A command service's argv is not built by ananke, so `--metrics`
        // cannot be injected — the watchdog is opt-in there.
        let svc = command_svc_with_auto_restart("").unwrap();
        assert!(svc.auto_restart.generation_stall.is_none());
        // The default also applies when an auto_restart block is present but
        // silent about generation_stall.
        let svc = command_svc_with_auto_restart("auto_restart = { error_rate = false }").unwrap();
        assert!(svc.auto_restart.generation_stall.is_none());
    }

    #[test]
    fn auto_restart_generation_stall_explicit_true_enables_on_command() {
        let svc =
            command_svc_with_auto_restart("auto_restart = { generation_stall = true }").unwrap();
        assert!(svc.auto_restart.generation_stall.is_some());
    }

    #[test]
    fn auto_restart_generation_stall_false_disables_it() {
        let svc = svc_with_auto_restart("auto_restart = { generation_stall = false }").unwrap();
        assert!(svc.auto_restart.generation_stall.is_none());
        // The other default triggers stay on.
        assert!(svc.auto_restart.error_rate.is_some());
        assert!(svc.auto_restart.ttft_stall.is_some());
    }

    #[test]
    fn auto_restart_generation_stall_overrides() {
        let svc = svc_with_auto_restart(
            "auto_restart.generation_stall = { timeout = \"2m\", poll_interval = \"10s\" }",
        )
        .unwrap();
        let gs = svc.auto_restart.generation_stall.as_ref().unwrap();
        assert_eq!(gs.timeout_ms, 2 * 60 * 1000);
        assert_eq!(gs.poll_interval_ms, 10 * 1000);
    }

    #[test]
    fn auto_restart_generation_stall_rejects_zero_durations() {
        assert!(
            svc_with_auto_restart("auto_restart.generation_stall = { timeout = \"0s\" }").is_err()
        );
        assert!(
            svc_with_auto_restart("auto_restart.generation_stall = { poll_interval = \"0s\" }")
                .is_err()
        );
    }
    /// Like `svc_with_auto_restart`, but the service configures speculative
    /// decoding — the precondition for the spec-collapse watchdog.
    fn spec_svc_with_auto_restart(block: &str) -> Result<ServiceConfig, ExpectedError> {
        svc_with_auto_restart(&format!("spec_type = \"draft-mtp\"\n{block}"))
    }

    #[test]
    fn auto_restart_spec_collapse_defaults_on_only_with_spec_type() {
        let svc = spec_svc_with_auto_restart("").unwrap();
        let sc = svc
            .auto_restart
            .spec_collapse
            .as_ref()
            .expect("spec collapse on by default when spec_type is set");
        assert_eq!(sc.window_ms, DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_WINDOW_MS);
        assert_eq!(
            sc.min_draft_tokens,
            DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_MIN_DRAFT_TOKENS
        );
        assert_eq!(
            sc.poll_interval_ms,
            DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_POLL_MS
        );

        // Without spec_type, no responses carry draft counts, so the trigger
        // defaults off — on llama-cpp and command services alike.
        let svc = svc_with_auto_restart("").unwrap();
        assert!(svc.auto_restart.spec_collapse.is_none());
        let svc = command_svc_with_auto_restart("").unwrap();
        assert!(svc.auto_restart.spec_collapse.is_none());
    }

    #[test]
    fn auto_restart_spec_collapse_toggles_and_overrides() {
        let svc = spec_svc_with_auto_restart("auto_restart = { spec_collapse = false }").unwrap();
        assert!(svc.auto_restart.spec_collapse.is_none());
        // The other default triggers stay on.
        assert!(svc.auto_restart.error_rate.is_some());

        let svc = spec_svc_with_auto_restart(
            "auto_restart.spec_collapse = { window = \"5m\", min_draft_tokens = 400, poll_interval = \"10s\" }",
        )
        .unwrap();
        let sc = svc.auto_restart.spec_collapse.as_ref().unwrap();
        assert_eq!(sc.window_ms, 5 * 60 * 1000);
        assert_eq!(sc.min_draft_tokens, 400);
        assert_eq!(sc.poll_interval_ms, 10 * 1000);
    }

    #[test]
    fn auto_restart_spec_collapse_explicit_enable_requires_spec_type() {
        // An explicit per-service enable on a service that can never produce
        // draft counts is a configuration error, not a silent no-op.
        let err = svc_with_auto_restart_diagnostics("auto_restart = { spec_collapse = true }")
            .unwrap_err();
        let diag = &err.as_slice()[0];
        assert_eq!(diag.fields(), [fields::auto_restart::SPEC_COLLAPSE]);
        assert!(
            diag.to_string()
                .contains("auto_restart.spec_collapse requires spec_type")
        );
        assert!(command_svc_with_auto_restart("auto_restart = { spec_collapse = true }").is_err());
        // An explicit disable is always fine.
        assert!(svc_with_auto_restart("auto_restart = { spec_collapse = false }").is_ok());
    }

    #[test]
    fn auto_restart_spec_collapse_rejects_zero_thresholds() {
        assert!(
            spec_svc_with_auto_restart("auto_restart.spec_collapse = { window = \"0s\" }").is_err()
        );
        assert!(
            spec_svc_with_auto_restart("auto_restart.spec_collapse = { min_draft_tokens = 0 }")
                .is_err()
        );
        assert!(
            spec_svc_with_auto_restart("auto_restart.spec_collapse = { poll_interval = \"0s\" }")
                .is_err()
        );
    }
    #[test]
    fn auto_restart_error_rate_thresholds_override() {
        let svc = svc_with_auto_restart(
            "auto_restart.error_rate = { window = \"5m\", max_error_rate = 0.8, min_requests = 50, error_statuses = \"4xx+5xx\" }",
        )
        .unwrap();
        let er = svc.auto_restart.error_rate.as_ref().unwrap();
        assert_eq!(er.window_ms, 5 * 60 * 1000);
        assert_eq!(er.max_error_rate, 0.8);
        assert_eq!(er.min_requests, 50);
        assert_eq!(er.statuses, ErrorStatusClass::ClientAndServer);
    }
}
