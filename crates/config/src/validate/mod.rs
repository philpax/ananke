//! Validate a post-merge `RawConfig`, producing an `EffectiveConfig` of
//! per-service validated configs plus daemon-global settings.

use ananke_errors::ExpectedError;
use smol_str::SmolStr;

mod auto_restart_types;
mod auto_restart_validation;
mod command_validation;
mod common;
mod container_validation;
mod llama_cpp_validation;
mod metadata;
mod orchestrate;
mod placement;
mod port_pool;
mod restart_triggers;
mod runtime;
mod service_validation;
mod split_mode;
mod tracking;
mod types;

#[cfg(any(test, feature = "test-fakes"))]
pub mod test_fixtures;

// Config-default constants live in the `ananke-config` leaf crate so the
// xtask and CLI can reference them without pulling in the daemon's heavy
// deps. Re-exported here so they are reachable as
// `crate::validate::DEFAULT_*`.
pub use auto_restart_types::{
    AutoRestartSettings, DEFAULT_AUTO_RESTART_PERIODIC_MODE, ErrorRateTrigger, ErrorStatusClass,
    GenerationStallTrigger, PeriodicMode, PeriodicTrigger, SpecCollapseTrigger, TtftStallTrigger,
};
pub(crate) use auto_restart_validation::validate_auto_restart;
pub(crate) use command_validation::{command_uses_port_placeholder, validate_command};
pub use common::{fail, flag_variant, gib_to_mib, parse_duration_ms, variant_flag};
pub(crate) use container_validation::{validate_bridge_command_endpoint, validate_container};
pub(crate) use llama_cpp_validation::validate_llama_cpp;
pub(crate) use metadata::{build_ananke_metadata, toml_value_to_json};
pub(crate) use orchestrate::{DaemonValidationCtx, ServiceValidationState};
pub use orchestrate::{validate, validate_with_checks};

pub use crate::docs::{
    DEFAULT_AUTO_RESTART_FLAP_WINDOW_MS, DEFAULT_AUTO_RESTART_GENERATION_STALL_MS,
    DEFAULT_AUTO_RESTART_GENERATION_STALL_POLL_MS, DEFAULT_AUTO_RESTART_MAX_ERROR_RATE,
    DEFAULT_AUTO_RESTART_MAX_RESTARTS, DEFAULT_AUTO_RESTART_MIN_REQUESTS,
    DEFAULT_AUTO_RESTART_MIN_UPTIME_MS, DEFAULT_AUTO_RESTART_POLL_INTERVAL_MS,
    DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_MIN_DRAFT_TOKENS,
    DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_POLL_MS, DEFAULT_AUTO_RESTART_SPEC_COLLAPSE_WINDOW_MS,
    DEFAULT_AUTO_RESTART_TTFT_STALL_MS, DEFAULT_AUTO_RESTART_WINDOW_MS, DEFAULT_DRAIN_TIMEOUT_MS,
    DEFAULT_EXTENDED_STREAM_DRAIN_MS, DEFAULT_HEALTH_PROBE_INTERVAL_MS, DEFAULT_HEALTH_TIMEOUT_MS,
    DEFAULT_IDLE_TIMEOUT_MS, DEFAULT_MAX_REQUEST_DURATION_MS, DEFAULT_MIN_BORROWER_RUNTIME_MS,
    DEFAULT_OPENAI_MAX_BODY_BYTES, DEFAULT_OPENAI_MAX_BODY_MB, DEFAULT_PRIVATE_PORT_END,
    DEFAULT_PRIVATE_PORT_START, DEFAULT_SERVICE_PRIORITY, DEFAULT_STOP_GRACE_MS,
    RESERVED_SERVICE_NAME_PREFIX,
};
// Placeholder dry-run checking cannot live in this crate: it needs the
// template substitution + allocation types, which would create a
// config → templates → devices → placement → config cycle. The daemon
// injects it through `validate_with_checks`.
/// Dry-run checker for placeholder-substituted argv, injected by the
/// daemon because the template engine lives outside this crate.
pub trait PlaceholderChecker {
    /// Dry-run substitute `argv` for `field`, failing on unresolved or
    /// malformed placeholders.
    fn check(&self, name: &SmolStr, field: &str, argv: &[String]) -> Result<(), ExpectedError>;

    /// Dry-run substitute every value in an environment map. Env values go
    /// through the same substitution as argv at spawn time, so a typo in
    /// one has to fail here rather than at the launch it breaks.
    fn check_env(
        &self,
        name: &SmolStr,
        field: &str,
        env: &std::collections::BTreeMap<String, String>,
    ) -> Result<(), ExpectedError>;
}

/// Checker that skips the dry-run. Used by `validate` when the caller
/// doesn't supply a checker (lib-internal tests); the daemon always uses
/// `validate_with_checks` with the real template-based checker.
pub struct NoopPlaceholderChecker;

impl PlaceholderChecker for NoopPlaceholderChecker {
    fn check(&self, _name: &SmolStr, _field: &str, _argv: &[String]) -> Result<(), ExpectedError> {
        Ok(())
    }

    fn check_env(
        &self,
        _name: &SmolStr,
        _field: &str,
        _env: &std::collections::BTreeMap<String, String>,
    ) -> Result<(), ExpectedError> {
        Ok(())
    }
}
pub use placement::{
    AllocationMode, DeviceReserves, DeviceSlot, Filters, HealthSettings, Lifecycle,
    PlacementPolicy, Template,
};
pub(crate) use port_pool::{PrivatePortAllocator, PrivatePortRange};
pub(crate) use restart_triggers::{
    validate_error_rate, validate_generation_stall, validate_spec_collapse, validate_ttft_stall,
};
pub use runtime::{IkSettings, NumaStrategy, OffloadMode, Runtime, RuntimeConfig};
pub(crate) use service_validation::validate_service;
pub use split_mode::SplitMode;
pub use tracking::TrackingSettings;
pub(crate) use tracking::validate_tracking;
pub use types::{
    CommandConfig, ContainerConfig, ContainerIpc, ContainerMount, ContainerNetwork,
    ContainerPortPublication, ContainerRuntime, DaemonSettings, EffectiveConfig, LlamaCppConfig,
    OpenAiProxyConfig, ServiceConfig, TemplateConfig,
};
