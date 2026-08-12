//! Runtime-flavour vocabulary for llama-cpp services: NUMA strategy, the
//! mainline-vs-ik_llama runtime split, and the expert-offload mode.

use crate::{
    flags,
    validate::{flag_variant, variant_flag},
};
pub use crate::{placement::OffloadMode, runtime::Runtime};

/// NUMA thread-and-memory placement strategy for a llama-cpp service,
/// emitted as llama.cpp's `--numa <strategy>`. Resolved from the `numa`
/// config value; unset emits no flag (llama.cpp's default).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumaStrategy {
    /// Spread worker threads across all nodes and interleave the model's
    /// memory allocation across them — balances memory-bandwidth load on
    /// multi-node / multi-CCD hosts (e.g. Threadripper).
    Distribute,
    /// Confine threads and allocation to a single NUMA node.
    Isolate,
    /// Defer placement to an external `numactl` mask.
    Numactl,
}

impl NumaStrategy {
    /// Variant ↔ flag binding, sourced from
    /// [`crate::flags::numa`] (see [`SplitMode::VARIANTS`]).
    const VARIANTS: &'static [(Self, &'static str)] = &[
        (Self::Distribute, flags::numa::DISTRIBUTE),
        (Self::Isolate, flags::numa::ISOLATE),
        (Self::Numactl, flags::numa::NUMACTL),
    ];

    /// The `--numa` flag value.
    pub fn as_flag(self) -> &'static str {
        variant_flag(Self::VARIANTS, self)
    }

    /// Parse an accepted `numa` string into a variant.
    pub fn from_flag(s: &str) -> Option<Self> {
        flag_variant(Self::VARIANTS, s)
    }

    /// Accepted values as a quoted list for operator-facing errors.
    pub fn valid_values() -> String {
        flags::quoted_list(flags::numa::ALL)
    }
}

/// Serving runtime for a llama-cpp-template service, mirroring
/// [`crate::parse::RawRuntime`]: which fork serves, plus the fork's
/// validated knobs where it has any.
///
/// Guardrail: the knobs hang off the variant rather than off a
/// `{ fork, ik: Option<_> }` pair, so "mainline carrying ik settings" is
/// unrepresentable instead of merely unexpected. [`Self::fork`] projects out
/// the bare [`Runtime`] marker that the calibration dataset also records.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum RuntimeConfig {
    /// Upstream llama.cpp, which has no runtime table of its own.
    #[default]
    Mainline,
    /// ikawrakow's fork, whose knobs switch spawn and estimation to its flag
    /// and memory conventions.
    Ik(IkSettings),
}

impl RuntimeConfig {
    /// Which fork this is, without its settings.
    pub fn fork(&self) -> Runtime {
        match self {
            RuntimeConfig::Mainline => Runtime::Mainline,
            RuntimeConfig::Ik(_) => Runtime::Ik,
        }
    }

    /// The fork settings, when this is the fork runtime.
    pub fn ik(&self) -> Option<&IkSettings> {
        match self {
            RuntimeConfig::Mainline => None,
            RuntimeConfig::Ik(ik) => Some(ik),
        }
    }
}

/// Validated ik_llama.cpp settings. See
/// [`crate::parse::RawIkSettings`] for per-field semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IkSettings {
    /// `-mla` kernel mode (0-3).
    pub mla: Option<u32>,
    /// DSA sparse attention (`-dsa -fidx`).
    pub dsa: bool,
    /// `-amb` attention scratch cap in MiB.
    pub attn_max_batch: Option<u32>,
    /// `-rtr` runtime repacking.
    pub runtime_repack: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate::{
        ConfigDiagnosticKind, ConstraintReason, ValidationErrorCode,
        test_fixtures::parse_and_merge, validate,
    };

    #[test]
    fn numa_vocab_is_single_sourced_and_complete() {
        for &(variant, flag) in NumaStrategy::VARIANTS {
            assert_eq!(variant.as_flag(), flag);
            assert_eq!(NumaStrategy::from_flag(flag), Some(variant));
        }
        for variant in [
            NumaStrategy::Distribute,
            NumaStrategy::Isolate,
            NumaStrategy::Numactl,
        ] {
            match variant {
                NumaStrategy::Distribute | NumaStrategy::Isolate | NumaStrategy::Numactl => {}
            }
            assert!(
                NumaStrategy::VARIANTS.iter().any(|&(v, _)| v == variant),
                "{variant:?} missing from NumaStrategy::VARIANTS"
            );
        }
        assert_eq!(NumaStrategy::from_flag("bogus"), None);
        assert_eq!(
            NumaStrategy::valid_values(),
            flags::quoted_list(flags::numa::ALL)
        );
    }
    #[test]
    fn expert_offload_parses_auto_and_count() {
        let auto = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
expert_offload = "auto"
devices.placement = "hybrid"
lifecycle = "persistent"
"#,
        );
        let ec = validate(&auto).unwrap();
        assert_eq!(
            ec.services[0].llama_cpp().unwrap().expert_offload,
            OffloadMode::Auto
        );

        let count = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
expert_offload = 16
devices.placement = "hybrid"
lifecycle = "persistent"
"#,
        );
        let ec = validate(&count).unwrap();
        assert_eq!(
            ec.services[0].llama_cpp().unwrap().expert_offload,
            OffloadMode::Layers(16)
        );
    }
    #[test]
    fn ik_runtime_parses_and_validates() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 131072
spec_type = "mtp:n_max=4,p_min=0.5"
runtime = { kind = "ik-llama", mla = 1, dsa = true, attn_max_batch = 512 }
lifecycle = "persistent"
"#,
        );
        let e = validate(&cfg).unwrap();
        let svc = &e.services[0];
        let lc = svc.llama_cpp().unwrap();
        assert_eq!(lc.runtime.fork(), Runtime::Ik);
        let ik = lc.runtime.ik().expect("ik runtime");
        assert_eq!(ik.mla, Some(1));
        assert!(ik.dsa);
        assert_eq!(ik.attn_max_batch, Some(512));
        assert!(!ik.runtime_repack);
    }

    #[test]
    fn ik_runtime_rejects_unknown_keys_in_table() {
        // deny_unknown_fields must hold through the internally-tagged
        // enum's newtype variant — a typo in the runtime table is a hard
        // error, not a silent no-op.
        let err = crate::parse::parse_toml(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
runtime = { kind = "ik-llama", mla = 1, n_cpu_moe = 4 }
"#,
            std::path::Path::new("/fake/ananke.toml"),
        )
        .unwrap_err();
        assert!(
            format!("{err}").contains("n_cpu_moe"),
            "unknown runtime key must be rejected, got: {err}"
        );
    }

    #[test]
    fn ik_runtime_gates_spec_type_dialects() {
        // Mainline service with ik-dialect spec_type.
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
spec_type = "mtp:n_max=4"
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert_eq!(diag.code(), ValidationErrorCode::TemplateConstraint);
        assert!(matches!(
            &*diag.kind,
            ConfigDiagnosticKind::Fields {
                reason: ConstraintReason::LlamaCppSpecTypeWrongDialect { .. },
                ..
            }
        ));

        // ik service with mainline-dialect spec_type.
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
spec_type = "draft-mtp"
runtime = { kind = "ik-llama" }
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert_eq!(diag.code(), ValidationErrorCode::TemplateConstraint);
        assert!(matches!(
            &*diag.kind,
            ConfigDiagnosticKind::Fields {
                reason: ConstraintReason::LlamaCppSpecTypeWrongDialect { .. },
                ..
            }
        ));
    }

    #[test]
    fn ik_dsa_requires_f16_kv() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
cache_type_k = "q8_0"
flash_attn = true
runtime = { kind = "ik-llama", dsa = true }
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert_eq!(diag.code(), ValidationErrorCode::TemplateConstraint);
        assert!(matches!(
            &*diag.kind,
            ConfigDiagnosticKind::Fields {
                reason: ConstraintReason::LlamaCppDsaRequiresF16Kv { .. },
                ..
            }
        ));
    }

    #[test]
    fn rejects_attn_max_batch_zero() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
runtime = { kind = "ik-llama", attn_max_batch = 0 }
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert_eq!(diag.code(), ValidationErrorCode::TemplateConstraint);
        assert!(matches!(
            &*diag.kind,
            ConfigDiagnosticKind::Fields {
                reason: ConstraintReason::LlamaCppAttnMaxBatchZero,
                ..
            }
        ));
    }
    #[test]
    fn expert_offload_requires_hybrid_placement() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
expert_offload = "auto"
devices.placement = "gpu-only"
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert_eq!(diag.code(), ValidationErrorCode::TemplateConstraint);
        assert!(matches!(
            &*diag.kind,
            ConfigDiagnosticKind::Fields {
                reason: ConstraintReason::ExpertOffloadRequiresHybridPlacement,
                ..
            }
        ));
    }

    #[test]
    fn expert_offload_rejects_sharded_split() {
        // A sharded (tensor/row) split has no CPU half, so it cannot honour an
        // expert offload to host RAM — reject the combination explicitly rather
        // than leaving the operator to infer it from the placement constraints.
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
expert_offload = "auto"
devices.placement = "gpu-only"
devices.split = "tensor"
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert_eq!(diag.code(), ValidationErrorCode::TemplateConstraint);
        assert!(matches!(
            &*diag.kind,
            ConfigDiagnosticKind::Fields {
                reason: ConstraintReason::ExpertOffloadConflictsShardedSplit { .. },
                ..
            }
        ));
    }
}
