//! Validate the `llama-cpp`-template body of a service: model paths, runtime
//! flavour, KV cache settings, split mode, and the launcher template.

use std::path::PathBuf;

use smol_str::SmolStr;

use crate::{
    flags,
    parse::{RawExpertOffload, RawLlamaCppService, RawRuntime},
    validate::{
        ConfigDiagnostic, ConstraintReason, IkSettings, LlamaCppConfig, LlamaCppReason,
        NumaStrategy, OffloadMode, PlaceholderChecker, Runtime, RuntimeConfig, ValidationErrorCode,
    },
};

pub(crate) fn validate_llama_cpp(
    name: &SmolStr,
    lc: &RawLlamaCppService,
    daemon_llama_server: Option<&std::path::Path>,
    checker: &dyn PlaceholderChecker,
) -> Result<LlamaCppConfig, crate::validate::ConfigDiagnostic> {
    let model = lc.model.clone().ok_or_else(|| {
        ConfigDiagnostic::constraint(
            ValidationErrorCode::TemplateConstraint,
            Some(name.to_string()),
            vec!["model".into()],
            ConstraintReason::LlamaCpp(LlamaCppReason::ModelMissing),
        )
    })?;
    let runtime = match &lc.runtime {
        None | Some(RawRuntime::LlamaCpp) => {
            // ik's --spec-type takes "mtp:..." specs; mainline's takes
            // "draft-mtp". Catch the cross-runtime mixup at parse time.
            if let Some(st) = lc.spec_type.as_deref()
                && st.starts_with("mtp:")
            {
                return Err(ConfigDiagnostic::constraint(
                    ValidationErrorCode::TemplateConstraint,
                    Some(name.to_string()),
                    vec!["spec_type".into()],
                    ConstraintReason::LlamaCpp(LlamaCppReason::SpecTypeWrongDialect {
                        spec_type: st.to_string(),
                        expected: "\"draft-mtp\" (or set runtime = { kind = \"ik-llama\" })",
                    }),
                ));
            }
            RuntimeConfig::Mainline
        }
        Some(RawRuntime::IkLlama(ik)) => {
            if let Some(m) = ik.mla
                && m > 3
            {
                return Err(ConfigDiagnostic::constraint(
                    ValidationErrorCode::TemplateConstraint,
                    Some(name.to_string()),
                    vec!["runtime.mla".into()],
                    ConstraintReason::LlamaCpp(LlamaCppReason::MlaOutOfRange { value: m }),
                ));
            }
            if let Some(st) = lc.spec_type.as_deref()
                && st.starts_with("draft-")
            {
                return Err(ConfigDiagnostic::constraint(
                    ValidationErrorCode::TemplateConstraint,
                    Some(name.to_string()),
                    vec!["spec_type".into()],
                    ConstraintReason::LlamaCpp(LlamaCppReason::SpecTypeWrongDialect {
                        spec_type: st.to_string(),
                        expected: "\"mtp:n_max=4,p_min=0.5\"",
                    }),
                ));
            }
            if ik.dsa == Some(true) {
                for (key, val) in [
                    ("cache_type_k", lc.cache_type_k.as_deref()),
                    ("cache_type_v", lc.cache_type_v.as_deref()),
                ] {
                    if let Some(v) = val
                        && v != "f16"
                    {
                        return Err(ConfigDiagnostic::constraint(
                            ValidationErrorCode::TemplateConstraint,
                            Some(name.to_string()),
                            vec![key.into()],
                            ConstraintReason::LlamaCpp(LlamaCppReason::DsaRequiresF16Kv {
                                key,
                                value: v.to_string(),
                            }),
                        ));
                    }
                }
            }
            RuntimeConfig::Ik(IkSettings {
                mla: ik.mla,
                dsa: ik.dsa.unwrap_or(false),
                attn_max_batch: ik.attn_max_batch,
                runtime_repack: ik.runtime_repack.unwrap_or(false),
            })
        }
    };

    if let RuntimeConfig::Ik(ik) = &runtime
        && ik.attn_max_batch == Some(0)
    {
        return Err(ConfigDiagnostic::constraint(
            ValidationErrorCode::TemplateConstraint,
            Some(name.to_string()),
            vec!["runtime.attn_max_batch".into()],
            ConstraintReason::LlamaCpp(LlamaCppReason::AttnMaxBatchZero),
        ));
    }

    let flash = lc.flash_attn.unwrap_or(false);
    // ik_llama predates mainline's FA-required-for-quantised-KV rule and
    // handles quantised caches without the flag, so the check is
    // mainline-only.
    if runtime.fork() == Runtime::Mainline {
        for (key, val) in [
            ("cache_type_k", lc.cache_type_k.as_deref()),
            ("cache_type_v", lc.cache_type_v.as_deref()),
        ] {
            if let Some(v) = val
                && v != "f16"
                && !flash
            {
                return Err(ConfigDiagnostic::constraint(
                    ValidationErrorCode::TemplateConstraint,
                    Some(name.to_string()),
                    vec![key.into()],
                    ConstraintReason::LlamaCpp(LlamaCppReason::QuantizedKvRequiresFlashAttn {
                        key,
                        value: v.to_string(),
                    }),
                ));
            }
        }
    }

    if lc.draft_model.is_some() && lc.spec_type.is_none() {
        return Err(ConfigDiagnostic::constraint(
            ValidationErrorCode::TemplateConstraint,
            Some(name.to_string()),
            vec!["draft_model".into(), "spec_type".into()],
            ConstraintReason::LlamaCpp(LlamaCppReason::DraftModelRequiresSpecType),
        ));
    }

    let launcher = match &lc.launcher {
        None => None,
        Some(argv) => {
            if argv.is_empty() {
                return Err(ConfigDiagnostic::constraint(
                    ValidationErrorCode::TemplateConstraint,
                    Some(name.to_string()),
                    vec!["launcher".into()],
                    ConstraintReason::LlamaCpp(LlamaCppReason::LauncherEmpty),
                ));
            }
            checker.check(name, "launcher", argv)?;
            Some(argv.clone())
        }
    };
    let binary = lc
        .llama_server
        .clone()
        .or_else(|| daemon_llama_server.map(std::path::Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("llama-server"));

    let expert_offload = match &lc.expert_offload {
        None => OffloadMode::Off,
        Some(RawExpertOffload::Layers(n)) => OffloadMode::Layers(*n),
        Some(RawExpertOffload::Mode(s)) => match s.as_str() {
            flags::expert_offload::OFF => OffloadMode::Off,
            flags::expert_offload::AUTO => OffloadMode::Auto,
            other => {
                return Err(ConfigDiagnostic::constraint(
                    ValidationErrorCode::TemplateConstraint,
                    Some(name.to_string()),
                    vec!["expert_offload".into()],
                    ConstraintReason::LlamaCpp(LlamaCppReason::ExpertOffloadInvalid {
                        value: other.to_string(),
                        expected: flags::quoted_list(flags::expert_offload::ALL),
                    }),
                ));
            }
        },
    };

    let numa = match lc.numa.as_deref() {
        None => None,
        Some(s) => Some(NumaStrategy::from_flag(s).ok_or_else(|| {
            ConfigDiagnostic::constraint(
                ValidationErrorCode::TemplateConstraint,
                Some(name.to_string()),
                vec!["numa".into()],
                ConstraintReason::LlamaCpp(LlamaCppReason::NumaInvalid {
                    value: s.to_string(),
                    expected: NumaStrategy::valid_values(),
                }),
            )
        })?),
    };

    Ok(LlamaCppConfig {
        runtime,
        model,
        mmproj: lc.mmproj.clone(),
        context: lc.context,
        n_gpu_layers: lc.n_gpu_layers,
        expert_offload,
        flash_attn: lc.flash_attn,
        cache_type_k: lc.cache_type_k.clone(),
        cache_type_v: lc.cache_type_v.clone(),
        mmap: lc.mmap,
        mlock: lc.mlock,
        parallel: lc.parallel,
        spec_type: lc.spec_type.clone(),
        spec_draft_n_max: lc.spec_draft_n_max,
        draft_model: lc.draft_model.clone(),
        kv_unified: lc.kv_unified,
        cache_idle_slots: lc.cache_idle_slots,
        cache_ram_mb: lc.cache_ram_mb,
        metrics: lc.metrics,
        slots: lc.slots,
        batch_size: lc.batch_size,
        ubatch_size: lc.ubatch_size,
        threads: lc.threads,
        threads_batch: lc.threads_batch,
        numa,
        jinja: lc.jinja,
        chat_template_file: lc.chat_template_file.clone(),
        override_tensor: lc.override_tensor.clone().unwrap_or_default(),
        sampling: lc.sampling.clone().unwrap_or_default(),
        estimation: lc.estimation.clone().unwrap_or_default(),
        binary,
        launcher,
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::{
        parse::parse_toml,
        validate::{
            ConfigDiagnosticKind, ConstraintReason, LlamaCppReason, ServiceReason,
            ValidationErrorCode, test_fixtures::parse_and_merge, validate,
        },
    };

    const GOOD: &str = r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 8192
flash_attn = true
cache_type_k = "q8_0"
cache_type_v = "q8_0"
devices.placement = "gpu-only"
devices.placement_override = { "gpu:0" = 18944 }
lifecycle = "persistent"
"#;

    #[test]
    fn draft_model_requires_spec_type() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
draft_model = "/m/mtp-draft.gguf"
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert_eq!(diag.code(), ValidationErrorCode::TemplateConstraint);
        assert!(matches!(
            &*diag.kind,
            ConfigDiagnosticKind::Fields {
                reason: ConstraintReason::LlamaCpp(LlamaCppReason::DraftModelRequiresSpecType),
                ..
            }
        ));
    }

    #[test]
    fn invalid_enum_values_enumerate_the_accepted_ones() {
        // The accepted values are the whole point of these diagnostics — an
        // operator with a typo needs the list, not just a rejection.
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
numa = "sideways"
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert!(matches!(
            &*diag.kind,
            ConfigDiagnosticKind::Fields {
                reason: ConstraintReason::LlamaCpp(LlamaCppReason::NumaInvalid { .. }),
                ..
            }
        ));
        assert!(diag.to_string().contains(&NumaStrategy::valid_values()));

        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
expert_offload = "maybe"
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert!(matches!(
            &*diag.kind,
            ConfigDiagnosticKind::Fields {
                reason: ConstraintReason::LlamaCpp(LlamaCppReason::ExpertOffloadInvalid { .. }),
                ..
            }
        ));
        assert!(
            diag.to_string()
                .contains(&flags::quoted_list(flags::expert_offload::ALL))
        );
    }

    #[test]
    fn draft_model_with_spec_type_validates() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
flash_attn = true
spec_type = "draft-mtp"
draft_model = "/m/mtp-draft.gguf"
kv_unified = true
cache_idle_slots = false
metrics = true
slots = true
lifecycle = "persistent"
"#,
        );
        let ec = validate(&cfg).unwrap();
        let lc = ec.services[0].llama_cpp().unwrap();
        assert_eq!(
            lc.draft_model.as_deref(),
            Some(std::path::Path::new("/m/mtp-draft.gguf"))
        );
        assert_eq!(lc.kv_unified, Some(true));
        assert_eq!(lc.cache_idle_slots, Some(false));
        assert_eq!(lc.metrics, Some(true));
        assert_eq!(lc.slots, Some(true));
    }
    #[test]
    fn n_cpu_moe_is_rejected_as_unknown_field() {
        // `n_cpu_moe` is not a config key; deny_unknown_fields surfaces it
        // as an error rather than dropping it in silence.
        let err = parse_toml(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
n_cpu_moe = 8
"#,
            Path::new("/t"),
        )
        .unwrap_err();
        // The parser rejects unknown fields; verify the error names the field.
        assert!(err.to_string().contains("n_cpu_moe"));
    }
    #[test]
    fn phase3_accepts_missing_placement_override() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "gpu-only"
lifecycle = "persistent"
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert!(ec.services[0].placement_override.is_empty());
    }
    #[test]
    fn rejects_quantised_kv_without_flash_attn() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
flash_attn = false
cache_type_k = "q8_0"
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert_eq!(diag.code(), ValidationErrorCode::TemplateConstraint);
        assert!(matches!(
            &*diag.kind,
            ConfigDiagnosticKind::Fields {
                reason: ConstraintReason::LlamaCpp(
                    LlamaCppReason::QuantizedKvRequiresFlashAttn { .. }
                ),
                ..
            }
        ));
    }

    #[test]
    fn rejects_cpu_only_with_ngl_nonzero() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
n_gpu_layers = 10
devices.placement = "cpu-only"
devices.placement_override = { "cpu" = 1000 }
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert_eq!(diag.code(), ValidationErrorCode::TemplateConstraint);
        assert!(matches!(
            &*diag.kind,
            ConfigDiagnosticKind::Fields {
                reason: ConstraintReason::Service(ServiceReason::CpuOnlyWithGpuLayers { .. }),
                ..
            }
        ));
    }
    #[test]
    fn llama_server_defaults_to_path_lookup() {
        let cfg = parse_and_merge(GOOD);
        let ec = validate(&cfg).unwrap();
        let lc = ec.services[0].llama_cpp().unwrap();
        assert_eq!(lc.binary, PathBuf::from("llama-server"));
        assert!(lc.launcher.is_none());
    }

    #[test]
    fn daemon_llama_server_default_applies_when_service_unset() {
        let cfg = parse_and_merge(
            r#"
[daemon]
llama_server = "/opt/llama-build/llama-server"

[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let ec = validate(&cfg).unwrap();
        let lc = ec.services[0].llama_cpp().unwrap();
        assert_eq!(lc.binary, PathBuf::from("/opt/llama-build/llama-server"));
    }

    #[test]
    fn service_llama_server_overrides_daemon_default() {
        let cfg = parse_and_merge(
            r#"
[daemon]
llama_server = "/opt/global"

[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
llama_server = "/opt/per-service"
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let ec = validate(&cfg).unwrap();
        let lc = ec.services[0].llama_cpp().unwrap();
        assert_eq!(lc.binary, PathBuf::from("/opt/per-service"));
    }
}
