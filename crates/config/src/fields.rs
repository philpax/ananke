//! Canonical TOML field paths, named once so a diagnostic, a doc row, and
//! the parse struct cannot drift apart.
//!
//! A diagnostic names the field it rejected, and a test asserts on that name
//! rather than on the rendered sentence. The path is therefore part of the
//! contract, not decoration, and belongs in one place.
//!
//! Every path is written the way the operator writes it, relative to the
//! table that encloses it: a field of `[[service]]` is bare, a field of a
//! sub-table is prefixed by that sub-table, and a field of a top-level table
//! is prefixed by that table. `RawServiceCommon` and the per-template fields
//! are `#[serde(flatten)]`ed into `[[service]]`, so they are bare regardless
//! of which struct declares them — there is no `service.`, `llama_cpp.`, or
//! `command.` prefix to write in TOML, and so none to report.

/// Field paths under `[service.allocation]`.
pub mod allocation {
    /// `allocation.min_borrower_runtime`
    pub const MIN_BORROWER_RUNTIME: &str = "allocation.min_borrower_runtime";
    /// `allocation.mode`
    pub const MODE: &str = "allocation.mode";
}

/// Field paths under `[service.auto_restart]`.
pub mod auto_restart {
    /// `auto_restart.error_rate.error_statuses`
    pub const ERROR_RATE_ERROR_STATUSES: &str = "auto_restart.error_rate.error_statuses";
    /// `auto_restart.error_rate.max_error_rate`
    pub const ERROR_RATE_MAX_ERROR_RATE: &str = "auto_restart.error_rate.max_error_rate";
    /// `auto_restart.error_rate.poll_interval`
    pub const ERROR_RATE_POLL_INTERVAL: &str = "auto_restart.error_rate.poll_interval";
    /// `auto_restart.error_rate.window`
    pub const ERROR_RATE_WINDOW: &str = "auto_restart.error_rate.window";
    /// `auto_restart.generation_stall.poll_interval`
    pub const GENERATION_STALL_POLL_INTERVAL: &str = "auto_restart.generation_stall.poll_interval";
    /// `auto_restart.generation_stall.timeout`
    pub const GENERATION_STALL_TIMEOUT: &str = "auto_restart.generation_stall.timeout";
    /// `auto_restart.periodic`
    pub const PERIODIC: &str = "auto_restart.periodic";
    /// `auto_restart.periodic.interval`
    pub const PERIODIC_INTERVAL: &str = "auto_restart.periodic.interval";
    /// `auto_restart.periodic.mode`
    pub const PERIODIC_MODE: &str = "auto_restart.periodic.mode";
    /// `auto_restart.spec_collapse`
    pub const SPEC_COLLAPSE: &str = "auto_restart.spec_collapse";
    /// `auto_restart.spec_collapse.min_draft_tokens`
    pub const SPEC_COLLAPSE_MIN_DRAFT_TOKENS: &str = "auto_restart.spec_collapse.min_draft_tokens";
    /// `auto_restart.spec_collapse.poll_interval`
    pub const SPEC_COLLAPSE_POLL_INTERVAL: &str = "auto_restart.spec_collapse.poll_interval";
    /// `auto_restart.spec_collapse.window`
    pub const SPEC_COLLAPSE_WINDOW: &str = "auto_restart.spec_collapse.window";
    /// `auto_restart.ttft_stall.timeout`
    pub const TTFT_STALL_TIMEOUT: &str = "auto_restart.ttft_stall.timeout";
}

/// Field paths under `[daemon]`.
pub mod daemon {
    /// `daemon.allow_external_management`
    pub const ALLOW_EXTERNAL_MANAGEMENT: &str = "daemon.allow_external_management";
    /// `daemon.management_listen`
    pub const MANAGEMENT_LISTEN: &str = "daemon.management_listen";
    /// `daemon.private_port_end`
    pub const PRIVATE_PORT_END: &str = "daemon.private_port_end";
    /// `daemon.private_port_start`
    pub const PRIVATE_PORT_START: &str = "daemon.private_port_start";
    /// `daemon.shutdown_timeout`
    pub const SHUTDOWN_TIMEOUT: &str = "daemon.shutdown_timeout";
}

/// Field paths under `[service.devices]`.
pub mod devices {
    /// `devices.gpu_allow`
    pub const GPU_ALLOW: &str = "devices.gpu_allow";
    /// `devices.placement`
    pub const PLACEMENT: &str = "devices.placement";
    /// `devices.placement_override`
    pub const PLACEMENT_OVERRIDE: &str = "devices.placement_override";
    /// `devices.split`
    pub const SPLIT: &str = "devices.split";
    /// `devices.tensor_split_weights`
    pub const TENSOR_SPLIT_WEIGHTS: &str = "devices.tensor_split_weights";
}

/// Field paths under the global `[devices]` table (distinct from
/// `[service.devices]`, see the `devices` module above).
pub mod global_devices {
    /// `devices.gpu_reserved_mb`
    pub const GPU_RESERVED_MB: &str = "devices.gpu_reserved_mb";
}

/// Field paths under `[service.health]`.
pub mod health {
    /// `health.probe_interval`
    pub const PROBE_INTERVAL: &str = "health.probe_interval";
    /// `health.timeout`
    pub const TIMEOUT: &str = "health.timeout";
}

/// Field paths under `[service.openai_proxy]`.
pub mod openai_proxy {
    /// `openai_proxy.upstream_model`
    pub const UPSTREAM_MODEL: &str = "openai_proxy.upstream_model";
}

/// Field paths under `[service.runtime]`.
pub mod runtime {
    /// `runtime.attn_max_batch`
    pub const ATTN_MAX_BATCH: &str = "runtime.attn_max_batch";
    /// `runtime.mla`
    pub const MLA: &str = "runtime.mla";
}

/// Field paths written at the top level of a `[[service]]` block: the
/// flattened common fields and both templates' own fields.
pub mod service {
    /// `command`
    pub const COMMAND: &str = "command";
    /// `draft_model`
    pub const DRAFT_MODEL: &str = "draft_model";
    /// `drain_timeout`
    pub const DRAIN_TIMEOUT: &str = "drain_timeout";
    /// `expert_offload`
    pub const EXPERT_OFFLOAD: &str = "expert_offload";
    /// `extended_stream_drain`
    pub const EXTENDED_STREAM_DRAIN: &str = "extended_stream_drain";
    /// `idle_timeout`
    pub const IDLE_TIMEOUT: &str = "idle_timeout";
    /// `launcher`
    pub const LAUNCHER: &str = "launcher";
    /// `lifecycle`
    pub const LIFECYCLE: &str = "lifecycle";
    /// `max_request_duration`
    pub const MAX_REQUEST_DURATION: &str = "max_request_duration";
    /// `metadata`
    pub const METADATA: &str = "metadata";
    /// `modality`
    pub const MODALITY: &str = "modality";
    /// `model`
    pub const MODEL: &str = "model";
    /// `n_gpu_layers`
    pub const N_GPU_LAYERS: &str = "n_gpu_layers";
    /// `name`
    pub const NAME: &str = "name";
    /// `numa`
    pub const NUMA: &str = "numa";
    /// `port`
    pub const PORT: &str = "port";
    /// `shutdown_command`
    pub const SHUTDOWN_COMMAND: &str = "shutdown_command";
    /// `spec_type`
    pub const SPEC_TYPE: &str = "spec_type";
}

/// Field paths under `[service.tracking]`.
pub mod tracking {
    /// `tracking.cgroup_parent`
    pub const CGROUP_PARENT: &str = "tracking.cgroup_parent";
}
