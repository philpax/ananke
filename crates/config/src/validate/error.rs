//! Structured diagnostics produced by the configuration parse, merge, and
//! semantic-validation pipeline.

use std::{fmt, ops::Range, path::PathBuf};

use ananke_api::config::validate::{
    ValidationContext, ValidationError, ValidationErrorCode, ValidationLocation,
};

/// A source location in the original configuration text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticLocation {
    /// Zero-based byte offset of the beginning of the span.
    pub start: usize,
    /// Exclusive zero-based byte offset of the end of the span.
    pub end: usize,
    /// One-based line containing the beginning of the span.
    pub line: u32,
    /// One-based column containing the beginning of the span.
    pub column: u32,
}

impl DiagnosticLocation {
    /// Construct a location from a byte range and source text.
    pub fn from_range(source: &str, range: Range<usize>) -> Self {
        let (line, column) = byte_offset_to_line_column(source, range.start);
        Self {
            start: range.start,
            end: range.end,
            line,
            column,
        }
    }
}

/// Convert a byte offset into a one-based line and column.
///
/// Offsets in the middle of a UTF-8 code point are clamped to the beginning of
/// that code point. End-of-file is a valid position.
pub fn byte_offset_to_line_column(source: &str, offset: usize) -> (u32, u32) {
    let offset = offset.min(source.len());
    let offset = (0..=offset)
        .rev()
        .find(|candidate| source.is_char_boundary(*candidate))
        .unwrap_or(0);
    let prefix = &source[..offset];
    let line = prefix.bytes().filter(|b| *b == b'\n').count() as u32 + 1;
    let column = prefix
        .rsplit_once('\n')
        .map_or(prefix.chars().count(), |(_, rest)| rest.chars().count()) as u32
        + 1;
    (line, column)
}

/// Parse error for duration strings.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum DurationParseError {
    /// The numeric portion failed to parse.
    InvalidNumber { input: String },
    /// The suffix is not recognised.
    UnrecognisedSuffix { input: String },
}

impl fmt::Display for DurationParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidNumber { input } => write!(f, "invalid number in `{input}`"),
            Self::UnrecognisedSuffix { input } => write!(f, "unrecognised duration: {input}"),
        }
    }
}

/// Structured reason for a fields-level constraint violation.
///
/// Replaces the free-form `reason: String` so consumers can match on the
/// specific rule that failed rather than substring-searching a rendered message.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum ConstraintReason {
    // --- Llama-cpp template ---
    LlamaCppModelMissing,
    LlamaCppSpecTypeWrongDialect {
        spec_type: String,
        expected: &'static str,
    },
    LlamaCppMlaOutOfRange {
        value: u32,
    },
    LlamaCppDsaRequiresF16Kv {
        key: &'static str,
        value: String,
    },
    LlamaCppAttnMaxBatchZero,
    LlamaCppQuantizedKvRequiresFlashAttn {
        key: &'static str,
        value: String,
    },
    LlamaCppDraftModelRequiresSpecType,
    LlamaCppLauncherEmpty,
    LlamaCppExpertOffloadInvalid {
        value: String,
        expected: String,
    },
    LlamaCppNumaInvalid {
        value: String,
        expected: String,
    },
    // --- Command template ---
    CommandMissingCommand,
    CommandEmptyCommand,
    CommandEmptyShutdownCommand,
    CommandUpstreamModelEmpty,
    // --- Service-level ---
    LifecycleOneshotInvalid,
    LifecycleUnknown {
        value: String,
    },
    ModalityUnknown {
        value: String,
    },
    CpuOnlyWithGpuLayers {
        n_gpu_layers: i32,
    },
    PlacementUnknown {
        value: String,
    },
    PlacementOverrideEmpty,
    PlacementOverrideKeyInvalid {
        key: String,
    },
    PlacementOverrideZero {
        key: String,
    },
    GpuOnlyWithCpuOverride,
    SplitUnknown {
        value: String,
        expected: String,
    },
    ExpertOffloadConflictsShardedSplit {
        split: String,
    },
    ExpertOffloadRequiresHybridPlacement,
    ShardedSplitRequiresGpuOnly {
        split: String,
    },
    ShardedSplitLlamaCppOnly {
        split: String,
    },
    ShardedSplitConflictsOverrideTensor {
        split: String,
    },
    TensorSplitWeightsRequiresSharded,
    // --- Duration parsing ---
    DurationParseError {
        error: DurationParseError,
    },
    // --- Auto-restart ---
    PeriodicNeedsInterval,
    SpecCollapseRequiresSpecType,
    PeriodicModeInvalid {
        value: String,
    },
    SpecCollapseWindowZero,
    SpecCollapseMinDraftTokensZero,
    SpecCollapsePollIntervalZero,
    GenerationStallTimeoutZero,
    GenerationStallPollIntervalZero,
    TtftStallTimeoutZero,
    ErrorRateOutOfRange {
        value: String,
    },
    ErrorStatusClassInvalid {
        value: String,
    },
    PeriodicMissingInterval,
    // --- Allocation ---
    AllocationStaticRequiresReserveGb,
    AllocationDynamicRequiresMinReserveGb,
    AllocationDynamicRequiresMaxReserveGb,
    AllocationMaxMustExceedMin,
    AllocationModeUnknown {
        value: String,
    },
    AllocationCommandRequiresMode,
    // --- Daemon ---
    DaemonNonLoopbackWithoutFlag,
    // --- Metadata ---
    MetadataInvalid {
        field: String,
        error: String,
    },
    // --- Filters ---
    FilterSetParamsInvalid {
        key: String,
        error: String,
    },
    // --- Private port ---
    PrivatePortExhausted {
        range_start: u16,
        range_end: u16,
        width: u32,
    },
}

impl fmt::Display for ConstraintReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LlamaCppModelMissing => write!(f, "template llama-cpp requires `model`"),
            Self::LlamaCppSpecTypeWrongDialect {
                spec_type,
                expected,
            } => write!(
                f,
                "spec_type `{spec_type}` uses the wrong dialect (expected {expected})"
            ),
            Self::LlamaCppMlaOutOfRange { value } => {
                write!(f, "runtime.mla={value} is invalid (ik_llama accepts 0-3)")
            }
            Self::LlamaCppDsaRequiresF16Kv { key, value } => {
                write!(f, "runtime.dsa=true requires f16 KV, but {key}={value}")
            }
            Self::LlamaCppAttnMaxBatchZero => {
                write!(f, "runtime.attn_max_batch must be > 0")
            }
            Self::LlamaCppQuantizedKvRequiresFlashAttn { key, value } => {
                write!(
                    f,
                    "{key}={value} requires flash_attn=true (llama.cpp requires FA for quantised KV)"
                )
            }
            Self::LlamaCppDraftModelRequiresSpecType => {
                write!(f, "draft_model requires spec_type to be set")
            }
            Self::LlamaCppLauncherEmpty => write!(f, "launcher is present but empty"),
            Self::LlamaCppExpertOffloadInvalid { value, expected } => {
                write!(
                    f,
                    "expert_offload `{value}` is invalid (expected {expected}, or an integer layer count)"
                )
            }
            Self::LlamaCppNumaInvalid { value, expected } => {
                write!(f, "numa `{value}` is invalid (expected {expected})")
            }
            Self::CommandMissingCommand => write!(f, "command template requires `command`"),
            Self::CommandEmptyCommand => write!(f, "command is empty"),
            Self::CommandEmptyShutdownCommand => {
                write!(f, "shutdown_command is present but empty")
            }
            Self::CommandUpstreamModelEmpty => {
                write!(f, "openai_proxy.upstream_model must be a non-empty string")
            }
            Self::LifecycleOneshotInvalid => {
                write!(
                    f,
                    "lifecycle `oneshot` is invalid in a [[service]] block (API-only)"
                )
            }
            Self::LifecycleUnknown { value } => write!(f, "unknown lifecycle `{value}`"),
            Self::ModalityUnknown { value } => {
                write!(f, "unknown modality `{value}` (valid: `chat`, `embedding`)")
            }
            Self::CpuOnlyWithGpuLayers { n_gpu_layers } => {
                write!(
                    f,
                    "devices.placement=cpu-only with n_gpu_layers={n_gpu_layers} is invalid"
                )
            }
            Self::PlacementUnknown { value } => write!(f, "unknown placement `{value}`"),
            Self::PlacementOverrideEmpty => {
                write!(f, "devices.placement_override is empty")
            }
            Self::PlacementOverrideKeyInvalid { key } => {
                write!(f, "invalid placement_override key `{key}`")
            }
            Self::PlacementOverrideZero { key } => {
                write!(f, "placement_override for {key} is zero")
            }
            Self::GpuOnlyWithCpuOverride => {
                write!(f, "placement=gpu-only but placement_override includes cpu")
            }
            Self::SplitUnknown { value, expected } => {
                write!(f, "unknown devices.split `{value}` (expected {expected})")
            }
            Self::ExpertOffloadConflictsShardedSplit { split } => {
                write!(
                    f,
                    "expert_offload cannot be combined with devices.split=`{split}` (sharded split is GPU-only; expert offload targets the CPU)"
                )
            }
            Self::ExpertOffloadRequiresHybridPlacement => {
                write!(
                    f,
                    "expert_offload requires placement=hybrid (expert tensors offload to CPU)"
                )
            }
            Self::ShardedSplitRequiresGpuOnly { split } => {
                write!(
                    f,
                    "devices.split=`{split}` requires placement=gpu-only (tensor/row split cannot spill to CPU)"
                )
            }
            Self::ShardedSplitLlamaCppOnly { split } => {
                write!(
                    f,
                    "devices.split=`{split}` is only valid for llama-cpp services"
                )
            }
            Self::ShardedSplitConflictsOverrideTensor { split } => {
                write!(
                    f,
                    "devices.split=`{split}` cannot be combined with override_tensor"
                )
            }
            Self::TensorSplitWeightsRequiresSharded => {
                write!(
                    f,
                    "devices.tensor_split_weights is only valid with a sharded split mode (`row` or `tensor`)"
                )
            }
            Self::DurationParseError { error } => write!(f, "{error}"),
            Self::PeriodicNeedsInterval => {
                write!(
                    f,
                    "auto_restart.periodic = true needs an interval; write `periodic = {{ interval = \"6h\" }}`"
                )
            }
            Self::SpecCollapseRequiresSpecType => {
                write!(
                    f,
                    "auto_restart.spec_collapse requires spec_type to be set (without speculative decoding, responses carry no draft counts and the watchdog can never fire)"
                )
            }
            Self::PeriodicModeInvalid { value } => {
                write!(
                    f,
                    "auto_restart.periodic.mode must be `immediate`, `on-idle`, or `on-request`, got `{value}`"
                )
            }
            Self::SpecCollapseWindowZero => {
                write!(
                    f,
                    "auto_restart.spec_collapse.window must be greater than zero"
                )
            }
            Self::SpecCollapseMinDraftTokensZero => {
                write!(
                    f,
                    "auto_restart.spec_collapse.min_draft_tokens must be greater than zero"
                )
            }
            Self::SpecCollapsePollIntervalZero => {
                write!(
                    f,
                    "auto_restart.spec_collapse.poll_interval must be greater than zero"
                )
            }
            Self::GenerationStallTimeoutZero => {
                write!(
                    f,
                    "auto_restart.generation_stall.timeout must be greater than zero"
                )
            }
            Self::GenerationStallPollIntervalZero => {
                write!(
                    f,
                    "auto_restart.generation_stall.poll_interval must be greater than zero"
                )
            }
            Self::TtftStallTimeoutZero => {
                write!(
                    f,
                    "auto_restart.ttft_stall.timeout must be greater than zero"
                )
            }
            Self::ErrorRateOutOfRange { value } => {
                write!(
                    f,
                    "auto_restart.error_rate.max_error_rate must be in (0.0, 1.0], got {value}"
                )
            }
            Self::ErrorStatusClassInvalid { value } => {
                write!(
                    f,
                    "auto_restart.error_rate.error_statuses must be `5xx` or `4xx+5xx`, got `{value}`"
                )
            }
            Self::PeriodicMissingInterval => {
                write!(f, "auto_restart.periodic requires an `interval`")
            }
            Self::AllocationStaticRequiresReserveGb => {
                write!(f, "allocation.mode=static requires reserve_gb")
            }
            Self::AllocationDynamicRequiresMinReserveGb => {
                write!(f, "allocation.mode=dynamic requires min_reserve_gb")
            }
            Self::AllocationDynamicRequiresMaxReserveGb => {
                write!(f, "allocation.mode=dynamic requires max_reserve_gb")
            }
            Self::AllocationMaxMustExceedMin => {
                write!(f, "max_reserve_gb must be > min_reserve_gb")
            }
            Self::AllocationModeUnknown { value } => {
                write!(f, "unknown allocation.mode `{value}`")
            }
            Self::AllocationCommandRequiresMode => {
                write!(
                    f,
                    "command template requires allocation.mode (static|dynamic)"
                )
            }
            Self::DaemonNonLoopbackWithoutFlag => {
                write!(
                    f,
                    "daemon.management_listen is non-loopback but daemon.allow_external_management is false; the management API has no authentication"
                )
            }
            Self::MetadataInvalid { field, error } => write!(f, "{field}: {error}"),
            Self::FilterSetParamsInvalid { key, error } => {
                write!(f, "filters.set_params[{key}]: {error}")
            }
            Self::PrivatePortExhausted {
                range_start,
                range_end,
                width,
            } => {
                write!(
                    f,
                    "private_port_range [{range_start}, {range_end}] exhausted ({width} slots) — widen the range or reduce service count"
                )
            }
        }
    }
}

/// Structured reason for merge and inheritance failures.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum MergeReason {
    Cycle,
    ParentNotFound,
    ParentResolvedToNothing,
    ServiceNotFound,
    TemplateMismatch { child: String, parent: String },
    PortMustOverride,
    MigrationCycle,
    MissingNameDuringMigration,
}

impl fmt::Display for MergeReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cycle => write!(f, "extends cycle"),
            Self::ParentNotFound => write!(f, "parent does not exist"),
            Self::ParentResolvedToNothing => write!(f, "parent resolved to nothing"),
            Self::ServiceNotFound => {
                write!(f, "service not found during extends resolution")
            }
            Self::TemplateMismatch { child, parent } => {
                write!(
                    f,
                    "template `{child}` does not match parent's template `{parent}`; cross-template extends is not allowed"
                )
            }
            Self::PortMustOverride => write!(f, "must override port from parent"),
            Self::MigrationCycle => write!(f, "migrate_from cycle"),
            Self::MissingNameDuringMigration => {
                write!(f, "service without a name during migrate_from resolution")
            }
        }
    }
}

/// Typed placeholder substitution failures owned by the config domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlaceholderError {
    /// `{reserve_mb}` was used with a dynamic allocation.
    ReserveMbOnDynamic,
    /// `{reserve_mb}` was used with multiple static devices.
    ReserveMbMultiDevice,
    /// A placeholder name is not recognized.
    UnknownPlaceholder(String),
    /// `{args}` was embedded in a launcher argument.
    SplatInsideArg,
}

impl PlaceholderError {
    /// Stable category used by the wire adapter.
    pub fn category(&self) -> &'static str {
        match self {
            Self::ReserveMbOnDynamic => "reserve_mb_on_dynamic",
            Self::ReserveMbMultiDevice => "reserve_mb_multi_device",
            Self::UnknownPlaceholder(_) => "unknown_placeholder",
            Self::SplatInsideArg => "splat_inside_arg",
        }
    }
}

impl fmt::Display for PlaceholderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReserveMbOnDynamic => {
                write!(f, "{{reserve_mb}} is invalid with a dynamic allocation")
            }
            Self::ReserveMbMultiDevice => write!(
                f,
                "{{reserve_mb}} is valid only with a single-device static allocation"
            ),
            Self::UnknownPlaceholder(name) => write!(f, "unknown placeholder {{{name}}}"),
            Self::SplatInsideArg => write!(
                f,
                "splat placeholder {{args}} must be the entire launcher entry, not embedded"
            ),
        }
    }
}

/// Typed detail for value diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueDiagnosticDetail {
    /// The value has no specialized rendering rule.
    Generic,
    /// The private-port range has inverted bounds.
    PrivatePortRangeInvalid,
    /// A tracking cgroup path is empty.
    TrackingEmpty,
    /// A tracking cgroup path is relative.
    TrackingNotAbsolute,
    /// A tracking cgroup path ends with a slash.
    TrackingTrailingSlash,
    /// A tracking cgroup path contains unsupported characters.
    TrackingInvalidCharacters,
}

/// The complete typed domain diagnostic. Its variants own the stable code and
/// every fact needed by renderers and boundary adapters.
// The associated-data variants are public for typed adapters, but their fields
// are fully described by the variant-level API and mirrored by `context()`.
#[expect(
    missing_docs,
    reason = "associated diagnostic payload fields are documented by context()"
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigDiagnosticKind {
    /// The parser rejected the TOML source.
    Parse { parser_message: String },
    /// Inheritance or migration resolution failed.
    Merge {
        service: Option<String>,
        index: Option<usize>,
        parent: Option<String>,
        reason: MergeReason,
    },
    /// A field-level constraint failed.
    Field {
        code: ValidationErrorCode,
        field: String,
        offending: Option<String>,
        expected: Option<String>,
    },
    /// A scalar value failed validation.
    Value {
        code: ValidationErrorCode,
        detail: ValueDiagnosticDetail,
        field: String,
        offending: String,
        expected: Option<String>,
    },
    /// A collection count failed validation.
    Count {
        code: ValidationErrorCode,
        field: String,
        got: usize,
        expected: usize,
    },
    /// An indexed collection value failed validation.
    Index {
        code: ValidationErrorCode,
        field: String,
        index: usize,
        value: String,
        expected: Option<String>,
    },
    /// Multiple fields participate in one constraint.
    Fields {
        code: ValidationErrorCode,
        fields: Vec<String>,
        service: Option<String>,
        reason: ConstraintReason,
    },
    /// Placeholder substitution failed.
    Placeholder {
        code: ValidationErrorCode,
        service: Option<String>,
        field: String,
        argv_index: Option<usize>,
        argument: Option<String>,
        error: PlaceholderError,
    },
    /// A service identity or location was invalid.
    Service {
        code: ValidationErrorCode,
        service: Option<String>,
        index: Option<usize>,
        field: Option<String>,
    },
}

/// One typed configuration diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigDiagnostic {
    /// The complete typed diagnostic payload.
    pub kind: Box<ConfigDiagnosticKind>,
    /// Optional source location from the parser.
    pub location: Option<DiagnosticLocation>,
    /// Original zero-based service index, when the diagnostic belongs to a service.
    pub source_index: Option<usize>,
    /// Owning service name, when the diagnostic belongs to a named service.
    ///
    /// Attached by the orchestrator rather than by the individual validators, so
    /// that a diagnostic raised before the name is read — a missing `port`, say —
    /// still reaches the operator attributed to the block it came from.
    pub service: Option<String>,
}

impl ConfigDiagnostic {
    /// Construct a diagnostic from a typed payload and optional source location.
    pub fn new(kind: ConfigDiagnosticKind, location: Option<DiagnosticLocation>) -> Self {
        Self {
            kind: Box::new(kind),
            location,
            source_index: None,
            service: None,
        }
    }

    /// Attach the owning `[[service]]` block's source index and name.
    pub fn with_service_context(mut self, source_index: usize, service: Option<&str>) -> Self {
        self.source_index = Some(source_index);
        if self.service.is_none() {
            self.service = service.map(str::to_owned);
        }
        self
    }

    /// Construct a parser diagnostic, retaining the parser's original message.
    pub fn parse(message: impl Into<String>, location: Option<DiagnosticLocation>) -> Self {
        Self::new(
            ConfigDiagnosticKind::Parse {
                parser_message: message.into(),
            },
            location,
        )
    }

    /// Construct a merge diagnostic.
    pub fn merge(
        service: Option<String>,
        index: Option<usize>,
        parent: Option<String>,
        reason: MergeReason,
    ) -> Self {
        Self::new(
            ConfigDiagnosticKind::Merge {
                service,
                index,
                parent,
                reason,
            },
            None,
        )
    }

    /// Construct a field-value diagnostic.
    pub fn value(
        code: ValidationErrorCode,
        field: impl Into<String>,
        offending: impl Into<String>,
        expected: Option<String>,
    ) -> Self {
        Self::value_with_detail(
            code,
            ValueDiagnosticDetail::Generic,
            field,
            offending,
            expected,
        )
    }

    /// Construct a value diagnostic with typed rendering detail.
    pub fn value_with_detail(
        code: ValidationErrorCode,
        detail: ValueDiagnosticDetail,
        field: impl Into<String>,
        offending: impl Into<String>,
        expected: Option<String>,
    ) -> Self {
        Self::new(
            ConfigDiagnosticKind::Value {
                code,
                detail,
                field: field.into(),
                offending: offending.into(),
                expected,
            },
            None,
        )
    }

    /// Construct a collection-count diagnostic.
    pub fn count(
        code: ValidationErrorCode,
        field: impl Into<String>,
        got: usize,
        expected: usize,
    ) -> Self {
        Self::new(
            ConfigDiagnosticKind::Count {
                code,
                field: field.into(),
                got,
                expected,
            },
            None,
        )
    }

    /// Construct an indexed collection diagnostic.
    pub fn index(
        code: ValidationErrorCode,
        field: impl Into<String>,
        index: usize,
        value: impl Into<String>,
        expected: Option<String>,
    ) -> Self {
        Self::new(
            ConfigDiagnosticKind::Index {
                code,
                field: field.into(),
                index,
                value: value.into(),
                expected,
            },
            None,
        )
    }

    /// Construct a constraint diagnostic with explicit stable code and fields.
    pub fn constraint(
        code: ValidationErrorCode,
        service: Option<String>,
        fields: Vec<String>,
        reason: ConstraintReason,
    ) -> Self {
        Self::new(
            ConfigDiagnosticKind::Fields {
                code,
                fields,
                service,
                reason,
            },
            None,
        )
    }

    /// Construct a placeholder diagnostic.
    pub fn placeholder(
        service: Option<String>,
        field: impl Into<String>,
        argv_index: Option<usize>,
        argument: Option<String>,
        error: PlaceholderError,
    ) -> Self {
        Self::new(
            ConfigDiagnosticKind::Placeholder {
                code: ValidationErrorCode::PlaceholderInvalid,
                service,
                field: field.into(),
                argv_index,
                argument,
                error,
            },
            None,
        )
    }

    /// Return the stable code derived from the typed payload.
    pub const fn code(&self) -> ValidationErrorCode {
        match &*self.kind {
            ConfigDiagnosticKind::Parse { .. } => ValidationErrorCode::Parse,
            ConfigDiagnosticKind::Merge { .. } => ValidationErrorCode::MergeConstraint,
            ConfigDiagnosticKind::Field { code, .. }
            | ConfigDiagnosticKind::Value { code, .. }
            | ConfigDiagnosticKind::Count { code, .. }
            | ConfigDiagnosticKind::Index { code, .. }
            | ConfigDiagnosticKind::Fields { code, .. }
            | ConfigDiagnosticKind::Placeholder { code, .. }
            | ConfigDiagnosticKind::Service { code, .. } => *code,
        }
    }

    /// Derive the wire-shaped context for a boundary adapter.
    pub fn context(&self) -> ValidationContext {
        match &*self.kind {
            ConfigDiagnosticKind::Parse { parser_message } => ValidationContext::Parse {
                parser_message: parser_message.clone(),
            },
            ConfigDiagnosticKind::Merge {
                service,
                index,
                parent,
                reason,
            } => ValidationContext::Merge {
                service: service.clone(),
                index: *index,
                parent: parent.clone(),
                reason: reason.to_string(),
            },
            ConfigDiagnosticKind::Field {
                field,
                offending,
                expected,
                ..
            } => ValidationContext::Field {
                field: field.clone(),
                offending: offending.clone(),
                expected: expected.clone(),
            },
            ConfigDiagnosticKind::Value {
                field,
                offending,
                expected,
                ..
            } => ValidationContext::Value {
                field: field.clone(),
                offending: offending.clone(),
                expected: expected.clone(),
            },
            ConfigDiagnosticKind::Count {
                field,
                got,
                expected,
                ..
            } => ValidationContext::Count {
                field: field.clone(),
                got: *got,
                expected: *expected,
            },
            ConfigDiagnosticKind::Index {
                field,
                index,
                value,
                expected,
                ..
            } => ValidationContext::Index {
                field: field.clone(),
                index: *index,
                value: value.clone(),
                expected: expected.clone(),
            },
            ConfigDiagnosticKind::Fields {
                fields,
                service,
                reason,
                ..
            } => ValidationContext::Fields {
                fields: fields.clone(),
                service: service.clone(),
                reason: reason.to_string(),
            },
            ConfigDiagnosticKind::Placeholder {
                service,
                field,
                argv_index,
                argument,
                error,
                ..
            } => ValidationContext::Placeholder {
                service: service.clone(),
                field: field.clone(),
                argv_index: *argv_index,
                argument: argument.clone(),
                category: error.category().to_string(),
            },
            ConfigDiagnosticKind::Service {
                service,
                index,
                field,
                ..
            } => ValidationContext::Service {
                service: service.clone(),
                index: *index,
                field: field.clone(),
            },
        }
    }

    /// Return the field path, when the diagnostic has one.
    pub fn path(&self) -> Option<&str> {
        match &*self.kind {
            ConfigDiagnosticKind::Field { field, .. }
            | ConfigDiagnosticKind::Value { field, .. }
            | ConfigDiagnosticKind::Count { field, .. }
            | ConfigDiagnosticKind::Index { field, .. }
            | ConfigDiagnosticKind::Placeholder { field, .. } => Some(field),
            ConfigDiagnosticKind::Service { field, .. } => field.as_deref(),
            ConfigDiagnosticKind::Parse { .. }
            | ConfigDiagnosticKind::Merge { .. }
            | ConfigDiagnosticKind::Fields { .. } => None,
        }
    }

    /// Return the owning service name, preferring the one the payload carries.
    pub fn service_name(&self) -> Option<&str> {
        self.embedded_service().or(self.service.as_deref())
    }

    /// The service name the payload renders itself, if any.
    ///
    /// Kinds listed here already write `service {name}` in their own `Display`,
    /// so the shared prefix must not repeat it.
    fn embedded_service(&self) -> Option<&str> {
        match &*self.kind {
            ConfigDiagnosticKind::Merge { service, .. }
            | ConfigDiagnosticKind::Fields { service, .. }
            | ConfigDiagnosticKind::Placeholder { service, .. }
            | ConfigDiagnosticKind::Service { service, .. } => service.as_deref(),
            ConfigDiagnosticKind::Parse { .. }
            | ConfigDiagnosticKind::Field { .. }
            | ConfigDiagnosticKind::Value { .. }
            | ConfigDiagnosticKind::Count { .. }
            | ConfigDiagnosticKind::Index { .. } => None,
        }
    }

    /// Whether the payload writes its own service identity in `Display`.
    const fn renders_own_service(&self) -> bool {
        matches!(
            &*self.kind,
            ConfigDiagnosticKind::Merge { .. }
                | ConfigDiagnosticKind::Fields { .. }
                | ConfigDiagnosticKind::Placeholder { .. }
                | ConfigDiagnosticKind::Service { .. }
        )
    }
}

impl From<ConfigDiagnostic> for ValidationError {
    fn from(diagnostic: ConfigDiagnostic) -> Self {
        Self {
            code: diagnostic.code(),
            message: diagnostic.to_string(),
            path: diagnostic.path().map(str::to_owned),
            service: diagnostic.service_name().map(str::to_owned),
            service_index: diagnostic.source_index,
            context: diagnostic.context(),
            location: diagnostic
                .location
                .as_ref()
                .map(|location| ValidationLocation {
                    start: location.start,
                    end: location.end,
                    line: location.line,
                    column: location.column,
                }),
        }
    }
}

impl fmt::Display for ConfigDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.renders_own_service() {
            match (self.service.as_deref(), self.source_index) {
                (Some(service), _) => write!(f, "service {service}: ")?,
                (None, Some(index)) => write!(f, "service[{index}]: ")?,
                (None, None) => {}
            }
        }
        match &*self.kind {
            ConfigDiagnosticKind::Parse { parser_message } => {
                write!(f, "parse error: {parser_message}")
            }
            ConfigDiagnosticKind::Merge {
                service,
                parent,
                reason,
                ..
            } => match (service, parent) {
                (Some(service), Some(parent)) => {
                    write!(f, "service {service} extends {parent}: {reason}")
                }
                (Some(service), None) => write!(f, "service {service}: {reason}"),
                _ => write!(f, "{reason}"),
            },
            ConfigDiagnosticKind::Field {
                field,
                offending,
                expected,
                ..
            } => {
                write!(f, "{field}: invalid field")?;
                if let Some(offending) = offending {
                    write!(f, " `{offending}`")?;
                }
                if let Some(expected) = expected {
                    write!(f, " (expected {expected})")?;
                }
                Ok(())
            }
            ConfigDiagnosticKind::Value {
                code,
                detail,
                field,
                offending,
                expected,
            } => match detail {
                ValueDiagnosticDetail::PrivatePortRangeInvalid => write!(
                    f,
                    "daemon.private_port_end must exceed daemon.private_port_start (got {offending})"
                ),
                ValueDiagnosticDetail::TrackingEmpty => write!(
                    f,
                    "tracking.cgroup_parent is empty — omit the field or supply a non-empty cgroup path"
                ),
                ValueDiagnosticDetail::TrackingNotAbsolute => write!(
                    f,
                    "tracking.cgroup_parent must be an absolute cgroup v2 path starting with `/` (got `{offending}`)"
                ),
                ValueDiagnosticDetail::TrackingTrailingSlash => write!(
                    f,
                    "tracking.cgroup_parent must not end with `/` (got `{offending}`)"
                ),
                ValueDiagnosticDetail::TrackingInvalidCharacters => write!(
                    f,
                    "tracking.cgroup_parent may only contain alphanumeric, `.`, `_`, `/`, and `-` characters (got `{offending}`)"
                ),
                ValueDiagnosticDetail::Generic => match code {
                    ValidationErrorCode::GpuAllowDuplicate => write!(
                        f,
                        "service: {field} must not contain duplicate GPU ids when tensor_split_weights is set"
                    ),
                    ValidationErrorCode::GpuAllowUnsorted => write!(
                        f,
                        "service: {field} must be in ascending GPU-id order when tensor_split_weights is set (got {offending})"
                    ),
                    ValidationErrorCode::ServiceNameDuplicate => {
                        write!(f, "duplicate service name `{offending}`")
                    }
                    ValidationErrorCode::ServicePortDuplicate => {
                        write!(f, "duplicate service port {offending}")
                    }
                    ValidationErrorCode::ServicePortManagementCollision => write!(
                        f,
                        "service port {offending} collides with daemon.management_listen"
                    ),
                    ValidationErrorCode::PrivatePortExhausted => write!(f, "{offending}"),
                    _ => {
                        write!(f, "{field}: invalid value `{offending}`")?;
                        if let Some(expected) = expected {
                            write!(f, " (expected {expected})")?;
                        }
                        Ok(())
                    }
                },
            },
            ConfigDiagnosticKind::Count {
                code,
                field,
                got,
                expected,
            } => {
                if *code == ValidationErrorCode::TensorSplitWeightsCount {
                    write!(
                        f,
                        "service: {field} has {got} entries but {expected} allowed GPU(s) (set via gpu_allow or [devices].gpu_ids)"
                    )
                } else {
                    write!(f, "{field}: got {got} entries, expected {expected}")
                }
            }
            ConfigDiagnosticKind::Index {
                code,
                field,
                index,
                value,
                expected,
            } => {
                if *code == ValidationErrorCode::TensorSplitWeightInvalid {
                    return write!(
                        f,
                        "service: {field}[{index}] must be a positive finite number, got {value}"
                    );
                }
                write!(f, "{field}[{index}]: invalid value `{value}`")?;
                if let Some(expected) = expected {
                    write!(f, " (expected {expected})")?;
                }
                Ok(())
            }
            ConfigDiagnosticKind::Fields {
                fields,
                service,
                reason,
                ..
            } => {
                if let Some(service) = service {
                    write!(f, "service {service} ")?;
                }
                write!(f, "{}: {reason}", fields.join(", "))
            }
            ConfigDiagnosticKind::Placeholder {
                service,
                field,
                argv_index,
                argument,
                error,
                ..
            } => {
                if let Some(service) = service {
                    write!(f, "service {service} ")?;
                }
                write!(f, "{field}")?;
                if let Some(index) = argv_index {
                    write!(f, "[{index}]")?;
                }
                if let Some(argument) = argument {
                    write!(f, " {argument:?}")?;
                }
                write!(f, ": {error}")
            }
            ConfigDiagnosticKind::Service {
                service,
                index,
                field,
                ..
            } => {
                if let Some(service) = service {
                    write!(f, "service {service}")?;
                } else if let Some(index) = index {
                    write!(f, "service[{index}]")?;
                } else {
                    write!(f, "service")?;
                }
                if let Some(field) = field {
                    write!(f, ": {field}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for ConfigDiagnostic {}

/// Ordered accumulation of configuration diagnostics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConfigDiagnosticReport {
    diagnostics: Vec<ConfigDiagnostic>,
}

impl ConfigDiagnosticReport {
    /// Construct an empty report.
    pub const fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
        }
    }

    /// Append one diagnostic.
    pub fn push(&mut self, diagnostic: ConfigDiagnostic) {
        self.diagnostics.push(diagnostic);
    }

    /// Append all diagnostics from another report.
    pub fn extend(&mut self, other: Self) {
        self.diagnostics.extend(other.diagnostics);
    }

    /// Return whether no diagnostics were collected.
    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }

    /// Return the number of diagnostics.
    pub fn len(&self) -> usize {
        self.diagnostics.len()
    }

    /// Borrow the ordered diagnostics.
    pub fn as_slice(&self) -> &[ConfigDiagnostic] {
        &self.diagnostics
    }

    /// Consume the report into its ordered diagnostics.
    pub fn into_vec(self) -> Vec<ConfigDiagnostic> {
        self.diagnostics
    }

    /// Place service diagnostics after global diagnostics in original source order.
    pub fn sort_by_source_index(&mut self) {
        self.diagnostics.sort_by_key(|diagnostic| {
            (
                diagnostic.source_index.is_some(),
                diagnostic.source_index.unwrap_or(usize::MAX),
            )
        });
    }

    /// Convert the report to the compatibility startup error.
    pub fn into_expected_error(self, origin: PathBuf) -> ananke_errors::ExpectedError {
        let cause = self
            .diagnostics
            .iter()
            .map(|diagnostic| {
                if let Some(location) = &diagnostic.location {
                    format!("{}:{}: {}", location.line, location.column, diagnostic)
                } else {
                    diagnostic.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("; ");
        ananke_errors::ExpectedError::config_unparseable(origin, cause)
    }
}

impl IntoIterator for ConfigDiagnosticReport {
    type Item = ConfigDiagnostic;
    type IntoIter = std::vec::IntoIter<ConfigDiagnostic>;
    fn into_iter(self) -> Self::IntoIter {
        self.diagnostics.into_iter()
    }
}

impl fmt::Display for ConfigDiagnosticReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, diagnostic) in self.diagnostics.iter().enumerate() {
            if index != 0 {
                write!(f, "; ")?;
            }
            write!(f, "{diagnostic}")?;
        }
        Ok(())
    }
}

/// Failure category for the full config pipeline.
#[derive(Debug)]
pub enum ConfigPipelineError {
    /// TOML parser failure(s).
    Parse(ConfigDiagnosticReport),
    /// Inheritance or migration failure(s).
    Merge(ConfigDiagnosticReport),
    /// Semantic validation failure(s).
    Validation(ConfigDiagnosticReport),
    /// Operational GGUF preflight failure.
    Preflight(ananke_errors::ExpectedError),
}

impl ConfigPipelineError {
    /// Consume a validation-phase error as its ordered report.
    ///
    /// This adapter is only valid for parse, merge, and semantic validation
    /// failures. Operational preflight remains an `ExpectedError` and must be
    /// handled by [`Self::into_expected_error`].
    pub fn into_report(self) -> ConfigDiagnosticReport {
        match self {
            Self::Parse(report) | Self::Merge(report) | Self::Validation(report) => report,
            Self::Preflight(error) => panic!("preflight is not a validation report: {error}"),
        }
    }

    /// Convert the pipeline result to the startup-facing operational error.
    pub fn into_expected_error(self, origin: PathBuf) -> ananke_errors::ExpectedError {
        match self {
            Self::Parse(report) | Self::Merge(report) | Self::Validation(report) => {
                report.into_expected_error(origin)
            }
            Self::Preflight(error) => error,
        }
    }
}

impl fmt::Display for ConfigPipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(report) => write!(f, "{report}"),
            Self::Merge(report) => write!(f, "{report}"),
            Self::Validation(report) => write!(f, "{report}"),
            Self::Preflight(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for ConfigPipelineError {}

impl From<ConfigDiagnostic> for ananke_errors::ExpectedError {
    fn from(diagnostic: ConfigDiagnostic) -> Self {
        ConfigDiagnosticReport {
            diagnostics: vec![diagnostic],
        }
        .into_expected_error(PathBuf::from("<config>"))
    }
}

impl From<ConfigDiagnostic> for ConfigDiagnosticReport {
    fn from(diagnostic: ConfigDiagnostic) -> Self {
        Self {
            diagnostics: vec![diagnostic],
        }
    }
}

impl From<ConfigDiagnosticReport> for ananke_errors::ExpectedError {
    fn from(report: ConfigDiagnosticReport) -> Self {
        report.into_expected_error(PathBuf::from("<config>"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_offsets_handle_multiline_utf8_and_eof() {
        let source = "α = 1\nname = \"é\"";
        assert_eq!(byte_offset_to_line_column(source, 0), (1, 1));
        assert_eq!(byte_offset_to_line_column(source, 7), (2, 1));
        assert_eq!(byte_offset_to_line_column(source, source.len()), (2, 11));
    }

    #[test]
    fn typed_kinds_retain_rule_data_and_render_without_string_dispatch() {
        let tracking = ConfigDiagnostic::value_with_detail(
            ValidationErrorCode::TrackingConstraint,
            ValueDiagnosticDetail::TrackingTrailingSlash,
            "tracking.cgroup_parent",
            "/system.slice/",
            Some("a path without a trailing slash".into()),
        );
        assert!(matches!(
            *tracking.kind,
            ConfigDiagnosticKind::Value {
                detail: ValueDiagnosticDetail::TrackingTrailingSlash,
                ..
            }
        ));
        assert!(tracking.to_string().contains("must not end with"));

        let placeholder = ConfigDiagnostic::placeholder(
            Some("demo".into()),
            "command",
            Some(1),
            Some("{bogus}".into()),
            PlaceholderError::UnknownPlaceholder("bogus".into()),
        );
        assert!(matches!(
            *placeholder.kind,
            ConfigDiagnosticKind::Placeholder {
                error: PlaceholderError::UnknownPlaceholder(ref name),
                ..
            } if name == "bogus"
        ));
        assert!(placeholder.to_string().contains("unknown placeholder"));
    }

    #[test]
    fn report_preserves_order_and_converts_all_messages() {
        let mut report = ConfigDiagnosticReport::new();
        report.push(ConfigDiagnostic::value(
            ValidationErrorCode::ValueInvalid,
            "service.port",
            "x",
            Some("an integer".into()),
        ));
        report.push(ConfigDiagnostic::parse("bad TOML", None));
        assert_eq!(report.len(), 2);
        assert!(report.to_string().contains("service.port"));
        assert!(
            report
                .into_expected_error(PathBuf::from("/tmp/config.toml"))
                .to_string()
                .contains("bad TOML")
        );
    }
}
