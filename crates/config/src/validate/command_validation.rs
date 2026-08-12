//! Validate the `command`-template body of a service: the argv, the shutdown
//! command, and the optional OpenAI-compatible proxy front-end.

use std::collections::BTreeMap;

use smol_str::SmolStr;

use crate::{
    parse::RawCommandService,
    validate::{
        CommandConfig, ConfigDiagnostic, ConstraintReason, OpenAiProxyConfig, PlaceholderChecker,
        ValidationErrorCode,
    },
};

pub(crate) fn validate_command(
    name: &SmolStr,
    cmd: &RawCommandService,
    checker: &dyn PlaceholderChecker,
) -> Result<CommandConfig, crate::validate::ConfigDiagnostic> {
    let command = cmd.command.clone().ok_or_else(|| {
        ConfigDiagnostic::constraint(
            ValidationErrorCode::TemplateConstraint,
            Some(name.to_string()),
            vec!["command.command".into()],
            ConstraintReason::CommandMissingCommand,
        )
    })?;
    if command.is_empty() {
        return Err(ConfigDiagnostic::constraint(
            ValidationErrorCode::TemplateConstraint,
            Some(name.to_string()),
            vec!["command.command".into()],
            ConstraintReason::CommandEmptyCommand,
        ));
    }
    if let Some(sd) = &cmd.shutdown_command
        && sd.is_empty()
    {
        return Err(ConfigDiagnostic::constraint(
            ValidationErrorCode::TemplateConstraint,
            Some(name.to_string()),
            vec!["command.shutdown_command".into()],
            ConstraintReason::CommandEmptyShutdownCommand,
        ));
    }
    // Dry-run the placeholder substitution so typos surface now rather
    // than at spawn/drain time. Uses a synthetic context — values are
    // arbitrary but cover every placeholder the supervisor will later
    // supply, so anything the runtime will accept also passes here.
    checker.check(name, "command", &command)?;
    if let Some(sd) = &cmd.shutdown_command {
        checker.check(name, "shutdown_command", sd)?;
    }
    let openai_proxy = match &cmd.openai_proxy {
        None => None,
        Some(proxy) => {
            let upstream_model = proxy
                .upstream_model
                .as_ref()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    ConfigDiagnostic::constraint(
                        ValidationErrorCode::TemplateConstraint,
                        Some(name.to_string()),
                        vec!["command.openai_proxy.upstream_model".into()],
                        ConstraintReason::CommandUpstreamModelEmpty,
                    )
                })?
                .clone();
            Some(OpenAiProxyConfig { upstream_model })
        }
    };
    Ok(CommandConfig {
        command,
        workdir: cmd.workdir.clone(),
        shutdown_command: cmd.shutdown_command.clone(),
        private_port_override: cmd.private_port,
        openai_proxy,
    })
}

/// Returns `true` when the command service's argv or any env value
/// references `{port}`. Heuristic for warning about an auto-assigned
/// `private_port` that the child never receives.
pub(crate) fn command_uses_port_placeholder(
    cmd: &CommandConfig,
    env: Option<&BTreeMap<String, String>>,
) -> bool {
    const PLACEHOLDER: &str = "{port}";
    cmd.command.iter().any(|a| a.contains(PLACEHOLDER))
        || env
            .map(|m| m.values().any(|v| v.contains(PLACEHOLDER)))
            .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use crate::validate::{
        ConfigDiagnosticKind, ConstraintReason, ValidationErrorCode,
        test_fixtures::parse_and_merge, validate,
    };

    #[test]
    fn command_rejects_missing_command() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "comfy"
template = "command"
port = 8188
allocation.mode = "static"
allocation.reserve_gb = 6
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert_eq!(diag.code(), ValidationErrorCode::TemplateConstraint);
        assert!(matches!(
            &*diag.kind,
            ConfigDiagnosticKind::Fields {
                reason: ConstraintReason::CommandMissingCommand,
                ..
            }
        ));
    }
    #[test]
    fn non_loopback_without_flag_is_rejected() {
        let cfg = parse_and_merge(
            r#"
[daemon]
management_listen = "0.0.0.0:17777"

[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
devices.placement_override = { "gpu:0" = 18944 }
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert_eq!(diag.code(), ValidationErrorCode::ValueInvalid);
        assert!(matches!(
            &*diag.kind,
            ConfigDiagnosticKind::Fields {
                reason: ConstraintReason::DaemonNonLoopbackWithoutFlag,
                ..
            }
        ));
    }
    #[test]
    fn non_loopback_with_flag_is_accepted() {
        let cfg = parse_and_merge(
            r#"
[daemon]
management_listen = "0.0.0.0:17777"
allow_external_management = true

[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
devices.placement_override = { "gpu:0" = 18944 }
lifecycle = "persistent"
"#,
        );
        assert!(validate(&cfg).is_ok());
    }

    #[test]
    fn command_service_honours_private_port_override() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "ext"
template = "command"
command = ["/bin/true"]
port = 8500
private_port = 18188
allocation.mode = "static"
allocation.reserve_gb = 1
"#,
        );
        let eff = validate(&cfg).expect("validate");
        let svc = &eff.services[0];
        assert_eq!(svc.private_port, 18188);
    }

    #[test]
    fn command_service_rejects_empty_shutdown_command() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "ext"
template = "command"
command = ["/bin/true"]
port = 8500
shutdown_command = []
allocation.mode = "static"
allocation.reserve_gb = 1
"#,
        );
        let err = validate(&cfg).expect_err("empty shutdown_command is rejected");
        let diag = &err.as_slice()[0];
        assert_eq!(diag.code(), ValidationErrorCode::TemplateConstraint);
        assert!(matches!(
            &*diag.kind,
            ConfigDiagnosticKind::Fields {
                reason: ConstraintReason::CommandEmptyShutdownCommand,
                ..
            }
        ));
    }
    #[test]
    fn command_service_with_openai_proxy_is_listed() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "qwen3.6-27b-vllm"
template = "command"
command = ["/run/vllm.sh", "{port}"]
port = 8500
allocation.mode = "static"
allocation.reserve_gb = 1
[service.openai_proxy]
upstream_model = "qwen3.6-27b-autoround"
"#,
        );
        let eff = validate(&cfg).expect("validate");
        let svc = &eff.services[0];
        assert!(svc.openai_compat, "openai_compat should be true");
        let cmd = svc.command().expect("command template");
        let proxy = cmd.openai_proxy.as_ref().expect("openai_proxy populated");
        assert_eq!(proxy.upstream_model, "qwen3.6-27b-autoround");
    }

    #[test]
    fn command_service_rejects_empty_openai_proxy_upstream_model() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "ext"
template = "command"
command = ["/bin/true"]
port = 8500
allocation.mode = "static"
allocation.reserve_gb = 1
[service.openai_proxy]
upstream_model = ""
"#,
        );
        let err = validate(&cfg).expect_err("empty upstream_model is rejected");
        let diag = &err.as_slice()[0];
        assert_eq!(diag.code(), ValidationErrorCode::TemplateConstraint);
        assert!(matches!(
            &*diag.kind,
            ConfigDiagnosticKind::Fields {
                reason: ConstraintReason::CommandUpstreamModelEmpty,
                ..
            }
        ));
    }

    #[test]
    fn command_service_rejects_missing_openai_proxy_upstream_model() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "ext"
template = "command"
command = ["/bin/true"]
port = 8500
allocation.mode = "static"
allocation.reserve_gb = 1
[service.openai_proxy]
"#,
        );
        let err = validate(&cfg).expect_err("missing upstream_model is rejected");
        let diag = &err.as_slice()[0];
        assert_eq!(diag.code(), ValidationErrorCode::TemplateConstraint);
        assert!(matches!(
            &*diag.kind,
            ConfigDiagnosticKind::Fields {
                reason: ConstraintReason::CommandUpstreamModelEmpty,
                ..
            }
        ));
    }

    #[test]
    fn command_service_without_openai_proxy_is_hidden() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "comfy"
template = "command"
command = ["/bin/comfyui"]
port = 8188
allocation.mode = "dynamic"
allocation.min_reserve_gb = 2
allocation.max_reserve_gb = 8
"#,
        );
        let eff = validate(&cfg).expect("validate");
        let svc = &eff.services[0];
        assert!(
            !svc.openai_compat,
            "openai_compat should default to false for command services without openai_proxy"
        );
        assert!(
            svc.command()
                .expect("command template")
                .openai_proxy
                .is_none()
        );
    }

    #[test]
    fn command_service_rejects_typo_in_placeholder_uses_injected_checker() {
        // The placeholder dry-run checker is injected by the daemon; ananke-config's
        // own `validate` uses the no-op checker, so the daemon-side tests in
        // `ananke/src/config/validate/placeholders.rs` cover the rejection path.
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "ext"
template = "command"
command = ["run", "--port={prot}"]
port = 8500
allocation.mode = "static"
allocation.reserve_gb = 1
"#,
        );
        // With the no-op checker the typo is accepted — the daemon's real checker
        // (placeholders.rs tests) rejects it.
        assert!(validate(&cfg).is_ok());
    }

    #[test]
    fn command_service_rejects_typo_in_shutdown_placeholder_uses_injected_checker() {
        // Same split as the command-typo test: the shutdown-command dry-run is
        // the daemon's injected checker's job; the daemon-side placeholders.rs
        // tests assert the rejection.
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "ext"
template = "command"
command = ["run", "--port={port}"]
shutdown_command = ["stop", "{bogus}"]
port = 8500
allocation.mode = "static"
allocation.reserve_gb = 1
"#,
        );
        assert!(validate(&cfg).is_ok());
    }
}
