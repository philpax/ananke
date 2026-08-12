//! Dry-run every `{placeholder}` a service's argv can contain at validate
//! time, so a typo fails `config validate` rather than a runtime spawn.

use ananke_config::validate::{ConfigDiagnostic, PlaceholderError};
use smol_str::SmolStr;

use crate::config::validate::PlaceholderChecker;

/// The daemon's placeholder dry-run checker: validates `command`,
/// `shutdown_command`, and llama-cpp `launcher` argv at config time.
pub struct DaemonPlaceholderChecker;

impl PlaceholderChecker for DaemonPlaceholderChecker {
    fn check(&self, name: &SmolStr, field: &str, argv: &[String]) -> Result<(), ConfigDiagnostic> {
        match field {
            "launcher" => check_launcher_placeholders(name, argv),
            _ => check_placeholders(name, field, argv),
        }
    }
}

/// Resolve every `{placeholder}` in `argv` against a synthetic context
/// covering every substitution the supervisor can produce. Propagates
/// the first [`SubstituteError`] as a config error with `field` + `name`
/// context, so a typo like `{prot}` fails `config validate` rather than
/// slipping through to a runtime `StartFailure`.
pub(crate) fn check_placeholders(
    name: &SmolStr,
    field: &str,
    argv: &[String],
) -> Result<(), ConfigDiagnostic> {
    use ananke_placement::devices::{Allocation, DeviceId};
    use ananke_templates::{PlaceholderContext, substitute};
    let mut alloc_bytes = std::collections::BTreeMap::new();
    alloc_bytes.insert(DeviceId::Gpu(0), 1);
    let alloc = Allocation { bytes: alloc_bytes };
    let ctx = PlaceholderContext {
        name,
        port: 0,
        model: Some("/m/x.gguf"),
        allocation: &alloc,
        // `None` so a `{reserve_mb}` placeholder on a dynamic allocation
        // trips the `ReserveMbOnDynamic` branch at config time, not
        // later. Static allocations re-validate at spawn time against
        // the real static_reserve_mb.
        static_reserve_mb: None,
    };
    for (i, arg) in argv.iter().enumerate() {
        substitute(arg, &ctx).map_err(|e| {
            ConfigDiagnostic::placeholder(
                Some(name.to_string()),
                field,
                Some(i),
                Some(arg.clone()),
                placeholder_error(e),
            )
        })?;
    }
    Ok(())
}

fn placeholder_error(error: ananke_templates::SubstituteError) -> PlaceholderError {
    match error {
        ananke_templates::SubstituteError::ReserveMbOnDynamic => {
            PlaceholderError::ReserveMbOnDynamic
        }
        ananke_templates::SubstituteError::ReserveMbMultiDevice => {
            PlaceholderError::ReserveMbMultiDevice
        }
        ananke_templates::SubstituteError::UnknownPlaceholder(name) => {
            PlaceholderError::UnknownPlaceholder(name)
        }
        ananke_templates::SubstituteError::SplatInsideArg => PlaceholderError::SplatInsideArg,
    }
}

/// Dry-run a llama-cpp `launcher` argv at validate time. Identical
/// purpose to [`check_placeholders`] but tolerates the `{args}` splat
/// (which would otherwise be rejected by [`substitute`]). Surfaces
/// typos like `{prot}` and misuses like `--foo={args}` as config errors
/// rather than runtime `StartFailure`s.
pub(crate) fn check_launcher_placeholders(
    name: &SmolStr,
    argv: &[String],
) -> Result<(), ConfigDiagnostic> {
    use ananke_placement::devices::{Allocation, DeviceId};
    use ananke_templates::{PlaceholderContext, substitute_launcher_argv};
    let mut alloc_bytes = std::collections::BTreeMap::new();
    alloc_bytes.insert(DeviceId::Gpu(0), 1);
    let alloc = Allocation { bytes: alloc_bytes };
    let ctx = PlaceholderContext {
        name,
        port: 0,
        model: Some("/m/x.gguf"),
        allocation: &alloc,
        static_reserve_mb: None,
    };
    substitute_launcher_argv(argv, &[], &ctx).map_err(|e| {
        ConfigDiagnostic::placeholder(
            Some(name.to_string()),
            "launcher",
            None,
            None,
            placeholder_error(e),
        )
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::config::validate::{
        DaemonPlaceholderChecker, test_fixtures::parse_and_merge, validate_with_checks,
    };

    fn validate(
        cfg: &ananke_config::parse::RawConfig,
    ) -> Result<ananke_config::validate::EffectiveConfig, ananke_errors::ExpectedError> {
        validate_with_checks(cfg, &DaemonPlaceholderChecker)
            .map_err(|report| report.into_expected_error(std::path::PathBuf::from("<config>")))
    }

    #[test]
    fn launcher_accepts_well_formed_template() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
launcher = ["/opt/podman-wrap.sh", "{model}", "{args}"]
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let ec = validate(&cfg).unwrap();
        let lc = ec.services[0].llama_cpp().unwrap();
        assert_eq!(
            lc.launcher.as_deref(),
            Some(
                &[
                    "/opt/podman-wrap.sh".to_string(),
                    "{model}".into(),
                    "{args}".into()
                ][..]
            )
        );
    }

    #[test]
    fn launcher_rejects_unknown_placeholder() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
launcher = ["wrap.sh", "{model}", "{bogus}", "{args}"]
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("{bogus}") && msg.contains("launcher"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn launcher_rejects_splat_embedded_in_arg() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
launcher = ["wrap.sh", "{model}", "--foo={args}"]
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("{args}"));
    }

    #[test]
    fn launcher_rejects_empty_argv() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
launcher = []
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("launcher"));
    }

    #[test]
    fn command_service_rejects_typo_in_placeholder() {
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
        let err = validate(&cfg).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("command[1]") && msg.contains("{prot}"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn command_service_rejects_typo_in_shutdown_placeholder() {
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
        let err = validate(&cfg).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("shutdown_command[1]") && msg.contains("{bogus}"),
            "unexpected error: {err}"
        );
    }
}
