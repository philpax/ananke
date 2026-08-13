//! The top-level validation pass: resolve daemon-global settings, then walk
//! every `[[service]]` block into an [`EffectiveConfig`].

use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    sync::Arc,
};

use smol_str::SmolStr;

use crate::{
    docs::DEFAULT_OPENAI_MAX_BODY_MB,
    fields,
    parse::RawConfig,
    validate::{
        ConfigDiagnostic, ConfigDiagnosticReport, DaemonSettings, DeviceReserves, EffectiveConfig,
        NoopPlaceholderChecker, PlaceholderChecker, PrivatePortAllocator, PrivatePortRange,
        ValidationErrorCode, parse_duration_ms, validate_service,
    },
};

/// Validate a `RawConfig` with the no-op placeholder checker. Library
/// callers with a template-based checker should use [`validate_with_checks`].
pub fn validate(cfg: &RawConfig) -> Result<EffectiveConfig, ConfigDiagnosticReport> {
    validate_with_checks(cfg, &NoopPlaceholderChecker)
}

/// [`validate`] with an injected placeholder dry-run checker. The daemon
/// passes its template-based checker here; lib-internal callers use the
/// no-op default.
pub fn validate_with_checks(
    cfg: &RawConfig,
    checker: &dyn PlaceholderChecker,
) -> Result<EffectiveConfig, ConfigDiagnosticReport> {
    let data_dir = cfg.daemon.data_dir.clone().unwrap_or_else(|| {
        std::env::var("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()))
                    .join(".local")
                    .join("share")
            })
            .join("ananke")
    });

    let mut report = ConfigDiagnosticReport::new();
    let shutdown_timeout_str = if cfg.daemon.shutdown_timeout.is_empty() {
        "120s"
    } else {
        &cfg.daemon.shutdown_timeout
    };
    let shutdown_timeout_ms = match parse_duration_ms(shutdown_timeout_str) {
        Ok(value) => value,
        Err(error) => {
            report.push(ConfigDiagnostic::value(
                crate::validate::ValidationErrorCode::DurationInvalid,
                "daemon.shutdown_timeout",
                shutdown_timeout_str,
                Some(error.to_string()),
            ));
            120_000
        }
    };

    let management_addr = if cfg.daemon.management_listen.is_empty() {
        crate::defaults::MANAGEMENT_LISTEN.into()
    } else {
        cfg.daemon.management_listen.clone()
    };
    let mgmt_socket_addr = match management_addr.parse::<std::net::SocketAddr>() {
        Ok(address) => Some(address),
        Err(error) => {
            report.push(ConfigDiagnostic::value(
                crate::validate::ValidationErrorCode::ValueInvalid,
                "daemon.management_listen",
                management_addr.clone(),
                Some(error.to_string()),
            ));
            None
        }
    };
    if let Some(address) = mgmt_socket_addr
        && !address.ip().is_loopback()
        && !cfg.daemon.allow_external_management
    {
        report.push(ConfigDiagnostic::constraint(
            ValidationErrorCode::ValueInvalid,
            None,
            &[fields::daemon::MANAGEMENT_LISTEN, fields::daemon::ALLOW_EXTERNAL_MANAGEMENT],
            "daemon.management_listen is non-loopback but daemon.allow_external_management is false; the management API has no authentication".to_string(),
        ));
    }
    let management_port = mgmt_socket_addr.map(|address| address.port());
    let openai_listen = cfg
        .openai_api
        .listen
        .clone()
        .unwrap_or_else(|| crate::defaults::OPENAI_LISTEN.into());

    let openai_max_body_bytes = cfg
        .openai_api
        .max_body_mb
        .unwrap_or(DEFAULT_OPENAI_MAX_BODY_MB)
        .saturating_mul(1024 * 1024)
        .min(usize::MAX as u64) as usize;

    let private_port_range = match PrivatePortRange::from_config(
        cfg.daemon.private_port_start,
        cfg.daemon.private_port_end,
    ) {
        Ok(range) => range,
        Err(error) => {
            report.push(error);
            PrivatePortRange {
                start: crate::validate::DEFAULT_PRIVATE_PORT_START,
                end: crate::validate::DEFAULT_PRIVATE_PORT_END,
            }
        }
    };
    let mut private_ports = PrivatePortAllocator::new(private_port_range);
    let daemon_llama_server = cfg.daemon.llama_server.clone();
    let (device_reserves, reserve_report) = resolve_device_reserves(&cfg.devices);
    report.extend(reserve_report);
    let device_reserves = Arc::new(device_reserves);

    let mut names: BTreeSet<SmolStr> = BTreeSet::new();
    let mut ports: BTreeSet<u16> = BTreeSet::new();
    let mut out = Vec::new();

    let daemon_ctx = DaemonValidationCtx {
        defaults: &cfg.defaults,
        management_port,
        daemon_llama_server: daemon_llama_server.as_deref(),
        reserves: &device_reserves,
        devices: &cfg.devices,
        placeholder_checker: checker,
    };
    for (resolved_index, raw) in cfg.services.iter().enumerate() {
        let source_index = cfg
            .service_source_indices
            .get(resolved_index)
            .copied()
            .unwrap_or(resolved_index);
        let mut candidate_names = names.clone();
        let mut candidate_ports = ports.clone();
        let mut candidate_private_ports = private_ports.clone();
        let mut candidate_state = ServiceValidationState {
            names: &mut candidate_names,
            ports: &mut candidate_ports,
            private_ports: &mut candidate_private_ports,
        };
        match validate_service(raw, &daemon_ctx, &mut candidate_state) {
            Ok(svc) => {
                names = candidate_names;
                ports = candidate_ports;
                private_ports = candidate_private_ports;
                out.push(svc);
            }
            Err(error) => {
                report.push(error.with_service_context(source_index, raw.common().name.as_deref()))
            }
        }
    }

    report.sort_by_source_index();
    if !report.is_empty() {
        return Err(report);
    }

    Ok(EffectiveConfig {
        daemon: DaemonSettings {
            management_listen: management_addr,
            openai_listen,
            data_dir,
            shutdown_timeout_ms,
            allow_external_management: cfg.daemon.allow_external_management,
            allow_external_services: cfg.daemon.allow_external_services,
            openai_allow_cors: cfg.openai_api.allow_cors,
            openai_max_body_bytes,
        },
        services: out,
    })
}

/// Resolve the global `[devices]` reserve knobs into a [`DeviceReserves`].
/// `gpu_reserved_mb` keys are GPU id strings (`"0"`); a non-numeric key is a
/// hard config error rather than a silently ignored reservation.
fn resolve_device_reserves(
    dev: &crate::parse::DevicesConfig,
) -> (DeviceReserves, ConfigDiagnosticReport) {
    let mut per_gpu_mb = BTreeMap::new();
    let mut report = ConfigDiagnosticReport::new();
    for (key, mb) in &dev.gpu_reserved_mb {
        match key.parse::<u32>() {
            Ok(id) => {
                per_gpu_mb.insert(id, *mb);
            }
            Err(_) => report.push(ConfigDiagnostic::value(
                crate::validate::ValidationErrorCode::ValueInvalid,
                "devices.gpu_reserved_mb",
                key,
                Some("a numeric GPU id such as `0`".into()),
            )),
        }
    }
    (
        DeviceReserves {
            default_gpu_mb: dev.default_gpu_reserved_mb.unwrap_or(0),
            per_gpu_mb,
            cpu_bytes: dev
                .cpu
                .reserved_gb
                .unwrap_or(0)
                .saturating_mul(1024 * 1024 * 1024),
        },
        report,
    )
}

/// Daemon-scoped inputs that don't change across services within a
/// single `validate` call. Grouped into a struct so per-service
/// validation doesn't need a long arg list (and so clippy stops
/// flagging it).
pub(crate) struct DaemonValidationCtx<'a> {
    pub(crate) defaults: &'a crate::parse::DefaultsConfig,
    pub(crate) management_port: Option<u16>,
    pub(crate) daemon_llama_server: Option<&'a std::path::Path>,
    /// Global device reserves resolved from `[devices]`, shared with every
    /// service so the packer can read them. The `Arc` is cloned per service.
    pub(crate) reserves: &'a Arc<DeviceReserves>,
    /// Raw `[devices]` config so per-service validation can check the
    /// configured GPU count when `gpu_allow` is unset.
    pub(crate) devices: &'a crate::parse::DevicesConfig,
    /// Placeholder dry-run checker injected by the daemon.
    pub(crate) placeholder_checker: &'a dyn PlaceholderChecker,
}

/// Mutable bookkeeping that accumulates across the per-service loop:
/// the set of names seen so duplicates can be rejected, the same for
/// ports, and the allocator that hands out private loopback ports.
pub(crate) struct ServiceValidationState<'a> {
    pub(crate) names: &'a mut BTreeSet<SmolStr>,
    pub(crate) ports: &'a mut BTreeSet<u16>,
    pub(crate) private_ports: &'a mut PrivatePortAllocator,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate::{
        ConfigDiagnosticKind, DeviceSlot, TemplateConfig, ValidationErrorCode,
        test_fixtures::parse_and_merge,
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
    fn validates_good() {
        let cfg = parse_and_merge(GOOD);
        let ec = validate(&cfg).unwrap();
        assert_eq!(ec.services.len(), 1);
        assert_eq!(ec.services[0].name, "demo");
        assert_eq!(ec.services[0].port, 11435);
        assert!(ec.services[0].private_port != 11435);
        assert_eq!(
            ec.services[0].placement_override[&DeviceSlot::Gpu(0)],
            18944
        );
        assert!(matches!(
            ec.services[0].template_config,
            TemplateConfig::LlamaCpp(_)
        ));
    }
    #[test]
    fn resolves_global_reserves_and_headroom() {
        let cfg = parse_and_merge(
            r#"
[devices]
default_gpu_reserved_mb = 512
gpu_reserved_mb = { "1" = 4096 }
[devices.cpu]
reserved_gb = 8

[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "hybrid"
devices.gpu_headroom_mb = 1024
lifecycle = "persistent"
"#,
        );
        let ec = validate(&cfg).unwrap();
        let svc = &ec.services[0];
        assert_eq!(svc.gpu_headroom_mb, 1024);
        assert_eq!(svc.reserves.default_gpu_mb, 512);
        assert_eq!(svc.reserves.per_gpu_mb.get(&1).copied(), Some(4096));
        assert_eq!(svc.reserves.cpu_bytes, 8 * 1024 * 1024 * 1024);
    }

    #[test]
    fn rejects_duplicate_port() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }

[[service]]
name = "b"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert!(matches!(
            &*diag.kind,
            ConfigDiagnosticKind::Value {
                code: ValidationErrorCode::ServicePortDuplicate,
                ..
            }
        ));
    }

    #[test]
    fn env_inherit_defaults_to_true() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "svc"
template = "command"
command = ["/bin/true"]
port = 11500
allocation.mode = "static"
allocation.reserve_gb = 1
"#,
        );
        let validated = validate(&cfg).unwrap();
        let svc = validated.services.iter().find(|s| s.name == "svc").unwrap();
        assert!(svc.env_inherit, "env_inherit should default to true");
    }

    #[test]
    fn env_inherit_false_parsed() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "svc"
template = "command"
command = ["/bin/true"]
port = 11500
allocation.mode = "static"
allocation.reserve_gb = 1
env_inherit = false
"#,
        );
        let validated = validate(&cfg).unwrap();
        let svc = validated.services.iter().find(|s| s.name == "svc").unwrap();
        assert!(!svc.env_inherit, "env_inherit should be false");
    }
}

