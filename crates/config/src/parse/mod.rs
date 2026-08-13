//! Parse a TOML string into a `RawConfig` typed tree (pre-merge, pre-validation).
//!
//! `RawService` is a `#[serde(tag = "template")]` enum with one variant per
//! template kind. Template-specific fields live on the corresponding variant's
//! struct; fields shared across templates live on `RawServiceCommon` and are
//! flattened into each variant. This makes wrong-template fields a parse error
//! rather than a runtime surprise, and lets downstream code pattern-match on a
//! typed variant instead of reaching through a bag of `Option`s.

mod auto_restart;
mod command;
mod common;
mod llama_cpp;
mod service;
mod top_level;

pub use auto_restart::{
    RawAutoRestart, RawErrorRateSettings, RawGenerationStallSettings, RawPeriodicSettings,
    RawSpecCollapseSettings, RawTtftStallSettings, Toggle,
};
pub use command::{RawCommandService, RawOpenAiProxy};
pub use common::{RawFilters, RawHealth, RawServiceCommon, RawServiceDevices, RawTracking};
pub use llama_cpp::{
    EstimationConfig, RawExpertOffload, RawIkSettings, RawLlamaCppService, RawRuntime,
    SamplingConfig,
};
pub use service::RawService;
pub use top_level::{
    CpuConfig, DaemonConfig, DefaultsConfig, DevicesConfig, OpenAiApiConfig, RawAllocation,
    RawConfig,
};

/// Default concurrency cap on pending start requests waiting for the same
/// supervisor to finish starting before they are rejected with `QueueFull`.
pub use crate::docs::DEFAULT_START_QUEUE_DEPTH;
use crate::validate::{ConfigDiagnostic, DiagnosticLocation};

/// Parse a TOML string into a raw config tree, rejecting unknown fields.
///
/// Every parse type carries `#[serde(deny_unknown_fields)]`, so a stale or
/// mistyped key is a hard error here rather than a silently-ignored one. That
/// is a deliberate choice over collecting unknowns via `serde_ignored`: the
/// report-and-continue model would both duplicate these errors and let bad
/// config through with only a warning.
pub fn parse_toml(source: &str) -> Result<RawConfig, ConfigDiagnostic> {
    toml_edit::de::from_str::<RawConfig>(source)
        .map(|mut config| {
            config.service_source_indices = (0..config.services.len()).collect();
            config
        })
        .map_err(|e| {
            let location = e
                .span()
                .map(|span| DiagnosticLocation::from_range(source, span));
            ConfigDiagnostic::parse(e.to_string(), location)
        })
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn parses_minimal_llama_cpp() {
        let toml = r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
"#;
        let cfg = parse_toml(toml).unwrap();
        assert_eq!(cfg.services.len(), 1);
        let svc = &cfg.services[0];
        assert_eq!(svc.common().name.as_deref(), Some("demo"));
        assert_eq!(svc.common().port, Some(11435));
        let RawService::LlamaCpp(lc) = svc else {
            panic!("expected LlamaCpp variant");
        };
        assert_eq!(lc.model.as_ref().unwrap().to_str(), Some("/m/x.gguf"));
    }

    #[test]
    fn parses_minimal_command() {
        let toml = r#"
[[service]]
name = "svc"
template = "command"
port = 11500
command = ["/bin/true"]
"#;
        let cfg = parse_toml(toml).unwrap();
        let svc = &cfg.services[0];
        let RawService::Command(cmd) = svc else {
            panic!("expected Command variant");
        };
        assert_eq!(cmd.command.as_ref().unwrap().as_slice(), ["/bin/true"]);
    }

    #[test]
    fn parses_dotted_keys() {
        let toml = r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
sampling.temperature = 0.7
devices.placement = "gpu-only"
devices.placement_override = { "gpu:0" = 18944 }
"#;
        let cfg = parse_toml(toml).unwrap();
        let RawService::LlamaCpp(lc) = &cfg.services[0] else {
            panic!("expected LlamaCpp");
        };
        assert_eq!(
            lc.common.devices.as_ref().unwrap().placement.as_deref(),
            Some("gpu-only")
        );
        assert_eq!(
            lc.common
                .devices
                .as_ref()
                .unwrap()
                .placement_override
                .as_ref()
                .unwrap()["gpu:0"],
            18944
        );
    }

    #[test]
    fn rejects_unparseable() {
        let toml = "this is not valid toml [[[";
        let err = parse_toml(toml).unwrap_err();
        assert!(format!("{err}").contains("parse"));
    }

    #[test]
    fn rejects_llama_cpp_field_on_command_template() {
        // `model` belongs to the llama-cpp variant; with a tagged enum it is
        // not a known field of the command variant, so serde rejects at parse.
        let toml = r#"
[[service]]
name = "svc"
template = "command"
port = 11500
command = ["/bin/true"]
model = "/m/x.gguf"
"#;
        let err = parse_toml(toml);
        assert!(
            err.is_err(),
            "expected parse error for llama-cpp field on command template, got {:?}",
            err.ok()
        );
    }

    #[test]
    fn rejects_unknown_template() {
        let toml = r#"
[[service]]
name = "svc"
template = "does-not-exist"
port = 11500
"#;
        let err = parse_toml(toml).unwrap_err();
        assert!(format!("{err}").contains("does-not-exist"));
    }

    #[test]
    fn rejects_missing_template() {
        // Tagged enum requires the discriminator; missing template is a parse error.
        let toml = r#"
[[service]]
name = "svc"
port = 11500
"#;
        let err = parse_toml(toml).unwrap_err();
        assert!(format!("{err}").contains("template"));
    }
}
