//! Configuration defaults, vocabulary, parse/merge/validate pipeline, and
//! the config manager.
//!
//! The defaults, docs descriptors, placement vocabulary, and byte-unit
//! conversions form a leaf core (serde + smol_str only) so the xtask and
//! CLI can reference them without pulling in the daemon's heavy deps.
//! The parse → merge → validate pipeline and the config manager live
//! here too, with the placeholder dry-run checker injected by the daemon
//! (it needs template + allocation types that would create a cycle).

#![deny(missing_docs)]
// `ConfigDiagnostic` is a flat, self-describing struct (code, service,
// fields, message, location) rather than a boxed enum payload, by design —
// see its doc comment. That makes it larger than the lint's threshold, but
// it is only ever returned on the config-validation error path, which runs
// at most once per reload; boxing it at every one of the pipeline's

pub mod defaults;
pub mod docs;
pub mod fields;
pub mod file;
pub mod flags;
pub mod manager;
pub mod merge;
pub mod parse;
pub mod placement;
pub mod runtime;
pub mod units;
pub mod validate;

pub use file::{PathSources, resolve_config_path, resolve_from_env};
pub use merge::{Migration, resolve_inheritance, resolve_migrations};
pub use parse::{RawConfig, RawService, parse_toml};
pub use validate::{
    AllocationMode, AutoRestartSettings, CommandConfig, ConfigDiagnostic, ConfigDiagnosticReport,
    ConfigPipelineError, DaemonSettings, DeviceReserves, DeviceSlot, DiagnosticLocation,
    EffectiveConfig, ErrorRateTrigger, ErrorStatusClass, Filters, GenerationStallTrigger,
    HealthSettings, IkSettings, Lifecycle, LlamaCppConfig, NumaStrategy, OffloadMode, PeriodicMode,
    PeriodicTrigger, PlacementPolicy, Runtime, RuntimeConfig, ServiceConfig, SpecCollapseTrigger,
    SplitMode, Template, TemplateConfig, TrackingSettings, TtftStallTrigger, ValidationErrorCode,
    validate, validate_with_checks,
};

/// Load, parse, merge, validate, and preflight a config file from disk.
pub fn load_config(
    path: &std::path::Path,
) -> Result<(EffectiveConfig, Vec<Migration>), ananke_errors::ExpectedError> {
    load_config_with_fs(
        path,
        &ananke_fs::LocalFs,
        &ananke_fs::Fs::read_to_string(&ananke_fs::LocalFs, path)
            .map_err(|_| ananke_errors::ExpectedError::config_file_missing(path.to_path_buf()))?,
    )
}

/// Variant of [`load_config`] that uses an explicit filesystem for the
/// GGUF preflight (but takes the TOML source directly rather than reading
/// it through the fs).
pub fn load_config_with_fs(
    origin: &std::path::Path,
    fs: &dyn ananke_fs::Fs,
    source: &str,
) -> Result<(EffectiveConfig, Vec<Migration>), ananke_errors::ExpectedError> {
    let (effective, migrations) =
        load_config_from_str_with_checks(source, &validate::NoopPlaceholderChecker)
            .map_err(|error| error.into_expected_error(origin.to_path_buf()))?;
    preflight_ggufs(origin, &effective, fs)?;
    Ok((effective, migrations))
}

/// Parse, merge, and validate a TOML config from a string, with no GGUF
/// preflight. The daemon's full load path is [`load_config_with_fs`].
pub fn load_config_from_str(
    source: &str,
) -> Result<(EffectiveConfig, Vec<Migration>), validate::ConfigDiagnosticReport> {
    load_config_from_str_with_checks(source, &validate::NoopPlaceholderChecker)
        .map_err(validate::ConfigPipelineError::into_report)
}

/// [`load_config_from_str`] with an injected placeholder dry-run checker.
pub fn load_config_from_str_with_checks(
    source: &str,
    checker: &dyn validate::PlaceholderChecker,
) -> Result<(EffectiveConfig, Vec<Migration>), validate::ConfigPipelineError> {
    let mut raw = parse_toml(source).map_err(|diagnostic| {
        validate::ConfigPipelineError::Parse(validate::ConfigDiagnosticReport::from(diagnostic))
    })?;
    let mut report = validate::ConfigDiagnosticReport::new();
    let mut merge_failed = false;
    if let Err(merge_report) = resolve_inheritance(&mut raw) {
        merge_failed = true;
        report.extend(merge_report);
    }
    let migrations = match resolve_migrations(&mut raw) {
        Ok(migrations) => migrations,
        Err(migration_report) => {
            merge_failed = true;
            report.extend(migration_report);
            Vec::new()
        }
    };
    match validate_with_checks(&raw, checker) {
        Ok(effective) if report.is_empty() => Ok((effective, migrations)),
        Ok(_) => Err(if merge_failed {
            validate::ConfigPipelineError::Merge(report)
        } else {
            validate::ConfigPipelineError::Validation(report)
        }),
        Err(validation_report) => {
            report.extend(validation_report);
            Err(if merge_failed {
                validate::ConfigPipelineError::Merge(report)
            } else {
                validate::ConfigPipelineError::Validation(report)
            })
        }
    }
}

/// Walk every llama-cpp service's GGUF through `fs` and ensure the reader
/// can enumerate each tensor table.
pub fn preflight_ggufs(
    origin: &std::path::Path,
    cfg: &EffectiveConfig,
    fs: &dyn ananke_fs::Fs,
) -> Result<(), ananke_errors::ExpectedError> {
    for svc in &cfg.services {
        let Some(lc) = svc.llama_cpp() else {
            continue;
        };
        ananke_gguf::read(fs, &lc.model).map_err(|e| {
            ananke_errors::ExpectedError::config_unparseable(
                origin.to_path_buf(),
                format!("service {}: {}", svc.name, e),
            )
        })?;
        if let Some(mmproj) = &lc.mmproj {
            ananke_gguf::read(fs, mmproj.as_path()).map_err(|e| {
                ananke_errors::ExpectedError::config_unparseable(
                    origin.to_path_buf(),
                    format!("service {} mmproj: {}", svc.name, e),
                )
            })?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loader_preserves_pipeline_phase_for_parser_failures() {
        let error = load_config_from_str_with_checks(
            "this is not valid toml [[[",
            &validate::NoopPlaceholderChecker,
        )
        .expect_err("malformed TOML must fail");
        assert!(matches!(error, ConfigPipelineError::Parse(_)));
    }

    #[test]
    fn loader_preserves_pipeline_phase_for_merge_failures() {
        let source = r#"
[[service]]
name = "broken"
template = "command"
extends = "missing"
port = 12000
command = ["/bin/true"]
allocation.mode = "static"
allocation.reserve_gb = 1
"#;
        let error = load_config_from_str_with_checks(source, &validate::NoopPlaceholderChecker)
            .expect_err("missing parent must fail");
        let ConfigPipelineError::Merge(report) = error else {
            panic!("expected a merge phase error");
        };
        assert!(
            report
                .as_slice()
                .iter()
                .any(|diagnostic| { diagnostic.code() == ValidationErrorCode::MergeConstraint })
        );
    }

    #[test]
    fn loader_preserves_pipeline_phase_for_semantic_failures() {
        let source = r#"
[daemon]
management_listen = "not-an-address"
"#;
        let error = load_config_from_str_with_checks(source, &validate::NoopPlaceholderChecker)
            .expect_err("invalid listen address must fail");
        let ConfigPipelineError::Validation(report) = error else {
            panic!("expected a validation phase error");
        };
        assert_eq!(
            report.as_slice()[0].code(),
            ValidationErrorCode::ValueInvalid
        );
    }

    #[test]
    fn loader_keeps_effective_services_in_name_order() {
        let source = r#"
[[service]]
name = "zeta"
template = "command"
port = 12000
command = ["/bin/true"]
allocation.mode = "static"
allocation.reserve_gb = 1

[[service]]
name = "alpha"
template = "command"
port = 12001
command = ["/bin/true"]
allocation.mode = "static"
allocation.reserve_gb = 1
"#;
        let (effective, _) =
            load_config_from_str_with_checks(source, &validate::NoopPlaceholderChecker)
                .expect("valid command services should load");
        assert_eq!(
            effective
                .services
                .iter()
                .map(|service| service.name.as_str())
                .collect::<Vec<_>>(),
            ["alpha", "zeta"]
        );
    }
}
