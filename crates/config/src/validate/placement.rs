//! Placement and lifecycle vocabulary: which template a service uses, how it
//! claims memory, when it runs, and the global device reserves it sees.

use std::collections::BTreeMap;

pub use crate::placement::{DeviceReserves, DeviceSlot, PlacementPolicy};
use crate::validate::{AllocationReason, ConstraintReason, gib_to_mib};

/// Which template a service uses: `llama-cpp` or `command`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Template {
    /// A llama.cpp model served by llama-server.
    LlamaCpp,
    /// Arbitrary external argv managed by ananke.
    Command,
}

impl Template {
    /// The template name as it appears in config files.
    pub fn as_str(self) -> &'static str {
        match self {
            Template::LlamaCpp => "llamacpp",
            Template::Command => "command",
        }
    }
}

/// How the allocator reserves memory for a service: none (estimated by the
/// packer), a fixed reservation, or a dynamic balloon range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationMode {
    /// Llama-cpp services: placement decided by estimator/override; mode absent.
    None,
    /// A fixed reservation. Named device-neutrally because it lands on the
    /// CPU device for a cpu-only command service just as readily as on a GPU.
    Static {
        /// The fixed reservation, in MiB.
        reserve_mb: u64,
    },
    /// A balloon range: the reservation grows and shrinks between these
    /// bounds as the service's observed usage dictates.
    Dynamic {
        /// Lower bound of the balloon range, in MiB.
        min_mb: u64,
        /// Upper bound of the balloon range, in MiB.
        max_mb: u64,
        /// How long a borrower run must live before its memory may be reclaimed.
        min_borrower_runtime_ms: u64,
    },
}

impl AllocationMode {
    /// Resolve an allocation mode from a `(template, mode)` pair plus the
    /// associated reservation knobs. Shared by the TOML validator and the
    /// oneshot API so both paths agree on the semantics of `"static"`,
    /// `"dynamic"`, and the llama-cpp exclusions.
    pub fn from_parts(
        template: Template,
        mode: Option<&str>,
        reserve_gb: Option<f32>,
        min_reserve_gb: Option<f32>,
        max_reserve_gb: Option<f32>,
        min_borrower_runtime_ms: u64,
    ) -> Result<AllocationMode, ConstraintReason> {
        match mode {
            Some("static") => {
                let gb = reserve_gb.ok_or(ConstraintReason::Allocation(
                    AllocationReason::StaticRequiresReserveGb,
                ))?;
                Ok(AllocationMode::Static {
                    reserve_mb: gib_to_mib(gb),
                })
            }
            Some("dynamic") => {
                let min = min_reserve_gb.ok_or(ConstraintReason::Allocation(
                    AllocationReason::DynamicRequiresMinReserveGb,
                ))?;
                let max = max_reserve_gb.ok_or(ConstraintReason::Allocation(
                    AllocationReason::DynamicRequiresMaxReserveGb,
                ))?;
                if max <= min {
                    return Err(ConstraintReason::Allocation(
                        AllocationReason::MaxMustExceedMin,
                    ));
                }
                Ok(AllocationMode::Dynamic {
                    min_mb: gib_to_mib(min),
                    max_mb: gib_to_mib(max),
                    min_borrower_runtime_ms,
                })
            }
            Some(other) => Err(ConstraintReason::Allocation(
                AllocationReason::ModeUnknown {
                    value: other.into(),
                },
            )),
            // A llama-cpp service without one is estimated and packed, which
            // is the normal path. A command service cannot be: ananke does not
            // build its argv and so cannot know what it will allocate.
            None => match template {
                Template::LlamaCpp => Ok(AllocationMode::None),
                Template::Command => Err(ConstraintReason::Allocation(
                    AllocationReason::CommandRequiresMode,
                )),
            },
        }
    }
}

/// When a service starts and stops relative to the daemon's own lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lifecycle {
    /// Started at boot and restarted when it exits.
    Persistent,
    /// Started on first request and stopped after `idle_timeout`.
    OnDemand,
}

impl Lifecycle {
    /// The lifecycle name as it appears in config files.
    pub fn as_str(self) -> &'static str {
        match self {
            Lifecycle::Persistent => "persistent",
            Lifecycle::OnDemand => "ondemand",
        }
    }
}

/// Request-scrubbing rules applied before a request is forwarded to a
/// service.
#[derive(Debug, Clone, Default)]
pub struct Filters {
    /// Query parameters stripped from every proxied request.
    pub strip_params: Vec<String>,
    /// Operator-supplied values injected into the request body. Opaque
    /// to ananke — the shape is whatever the upstream engine accepts.
    pub set_params: BTreeMap<String, serde_json::Value>,
}

/// Readiness-probe settings for a service. `None` path means no probe.
#[derive(Debug, Clone)]
pub struct HealthSettings {
    /// HTTP path to probe for readiness. `None` means no health check —
    /// the service transitions to Running immediately after spawn.
    pub http_path: Option<String>,
    /// How long a probe may take before it counts as failed.
    pub timeout_ms: u64,
    /// How often the probe runs while the service is up.
    pub probe_interval_ms: u64,
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::{
        parse::parse_toml,
        validate::{
            ConfigDiagnosticKind, ConstraintReason, ServiceReason, test_fixtures::parse_and_merge,
            validate,
        },
    };

    #[test]
    fn rejects_oneshot_lifecycle_in_service_block() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "oneshot"
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert!(matches!(
            &*diag.kind,
            ConfigDiagnosticKind::Fields {
                reason: ConstraintReason::Service(ServiceReason::LifecycleOneshotInvalid),
                ..
            }
        ));
    }

    #[test]
    fn phase2_accepts_on_demand() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "on_demand"
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert_eq!(ec.services[0].lifecycle, Lifecycle::OnDemand);
    }

    #[test]
    fn default_lifecycle_is_on_demand() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert_eq!(ec.services[0].lifecycle, Lifecycle::OnDemand);
    }

    #[test]
    fn parses_filters() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }
filters.strip_params = ["temperature"]
filters.set_params = { max_tokens = 4096 }
"#,
        );
        let ec = validate(&cfg).unwrap();
        let s = &ec.services[0];
        assert_eq!(s.filters.strip_params, vec!["temperature"]);
        assert!(s.filters.set_params.contains_key("max_tokens"));
    }

    #[test]
    fn parses_idle_timeout() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "on_demand"
idle_timeout = "5m"
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert_eq!(ec.services[0].idle_timeout_ms, 300_000);
    }
    #[test]
    fn command_template_with_static_allocation() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "comfy"
template = "command"
command = ["python", "main.py"]
port = 8188
lifecycle = "on_demand"
allocation.mode = "static"
allocation.reserve_gb = 6
"#,
        );
        let ec = validate(&cfg).unwrap();
        let svc = &ec.services[0];
        assert_eq!(svc.template(), Template::Command);
        assert!(matches!(
            svc.allocation_mode,
            AllocationMode::Static { reserve_mb: 6144 }
        ));
    }

    /// `vram_gb` / `min_vram_gb` / `max_vram_gb` are device-specific aliases
    /// for the device-neutral reservation keys. Configs on disk use them, so
    /// they have to keep parsing to the same allocation.
    #[test]
    fn legacy_vram_gb_keys_still_parse() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "static-legacy"
template = "command"
command = ["python", "main.py"]
port = 8188
lifecycle = "on_demand"
allocation.mode = "static"
allocation.vram_gb = 6

[[service]]
name = "dynamic-legacy"
template = "command"
command = ["python", "main.py"]
port = 8189
lifecycle = "on_demand"
allocation.mode = "dynamic"
allocation.min_vram_gb = 4
allocation.max_vram_gb = 20
"#,
        );
        let ec = validate(&cfg).unwrap();
        let mode = |name: &str| {
            ec.services
                .iter()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("service {name} must be present"))
                .allocation_mode
        };
        assert!(matches!(
            mode("static-legacy"),
            AllocationMode::Static { reserve_mb: 6144 }
        ));
        assert!(matches!(
            mode("dynamic-legacy"),
            AllocationMode::Dynamic {
                min_mb: 4096,
                max_mb: 20480,
                ..
            }
        ));
    }

    #[test]
    fn command_template_with_dynamic_allocation() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "comfy"
template = "command"
command = ["python", "main.py"]
port = 8188
lifecycle = "on_demand"
allocation.mode = "dynamic"
allocation.min_reserve_gb = 4
allocation.max_reserve_gb = 20
"#,
        );
        let ec = validate(&cfg).unwrap();
        let svc = &ec.services[0];
        assert!(matches!(
            svc.allocation_mode,
            AllocationMode::Dynamic {
                min_mb: 4096,
                max_mb: 20480,
                ..
            }
        ));
    }
    #[test]
    fn llama_cpp_allocation_mode_rejected_at_parse() {
        // With a tagged enum, `allocation` isn't a field on the llama-cpp
        // variant; serde rejects it before the validator runs.
        let res = parse_toml(
            r#"
[[service]]
name = "llama"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
allocation.mode = "static"
allocation.reserve_gb = 4
"#,
            Path::new("/t"),
        );
        assert!(
            res.is_err(),
            "expected parse error for allocation on llama-cpp; got {:?}",
            res.ok()
        );
    }
    #[test]
    fn dynamic_rejects_max_le_min() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "comfy"
template = "command"
command = ["python"]
port = 8188
allocation.mode = "dynamic"
allocation.min_reserve_gb = 10
allocation.max_reserve_gb = 5
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert!(matches!(
            &*diag.kind,
            ConfigDiagnosticKind::Fields {
                reason: ConstraintReason::Allocation(AllocationReason::MaxMustExceedMin),
                ..
            }
        ));
    }
}
