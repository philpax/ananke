//! Validation of a post-merge `RawConfig`.
//!
//! The validate pipeline lives in `ananke-config`; this module re-exports
//! it and keeps the daemon-side placeholder dry-run checker (which needs
//! the template substitution + allocation types that cannot live in the
//! config crate).

pub mod placeholders;

#[cfg(any(test, feature = "test-fakes"))]
pub use ananke_config::validate::test_fixtures;
pub use ananke_config::{
    docs::{
        DEFAULT_DRAIN_TIMEOUT_MS, DEFAULT_EXTENDED_STREAM_DRAIN_MS,
        DEFAULT_HEALTH_PROBE_INTERVAL_MS, DEFAULT_HEALTH_TIMEOUT_MS,
        DEFAULT_MAX_REQUEST_DURATION_MS, DEFAULT_MIN_BORROWER_RUNTIME_MS,
        DEFAULT_OPENAI_MAX_BODY_BYTES, DEFAULT_OPENAI_MAX_BODY_MB, DEFAULT_PRIVATE_PORT_END,
        DEFAULT_PRIVATE_PORT_START, DEFAULT_SERVICE_PRIORITY,
    },
    validate::{
        AllocationMode, AutoRestartSettings, CommandConfig, DaemonSettings, DeviceReserves,
        DeviceSlot, EffectiveConfig, ErrorRateTrigger, ErrorStatusClass, Filters,
        GenerationStallTrigger, HealthSettings, IkSettings, Lifecycle, LlamaCppConfig,
        NumaStrategy, OffloadMode, PeriodicMode, PeriodicTrigger, PlaceholderChecker,
        PlacementPolicy, Runtime, RuntimeConfig, ServiceConfig, SpecCollapseTrigger, SplitMode,
        Template, TemplateConfig, TrackingSettings, TtftStallTrigger, parse_duration_ms, validate,
        validate_with_checks,
    },
};
pub use placeholders::DaemonPlaceholderChecker;
