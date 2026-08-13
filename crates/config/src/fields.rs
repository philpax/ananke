//! Canonical TOML field paths, named once so a diagnostic, a doc row, and
//! the parse struct cannot drift apart.
//!
//! A diagnostic names the field it rejected, and a test asserts on that name
//! rather than on the rendered sentence. The path is therefore part of the
//! contract, not decoration, and belongs in one place.

/// Field paths under `[allocation]`.
pub mod allocation {
    /// `allocation.min_borrower_runtime`
    pub const MIN_BORROWER_RUNTIME: &str = "allocation.min_borrower_runtime";
    /// `allocation.mode`
    pub const MODE: &str = "allocation.mode";
}

/// Field paths under `[auto_restart]`.
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

/// Field paths under `[command]`.
pub mod command {
    /// `command.command`
    pub const COMMAND: &str = "command.command";
    /// `command.openai_proxy.upstream_model`
    pub const OPENAI_PROXY_UPSTREAM_MODEL: &str = "command.openai_proxy.upstream_model";
    /// `command.shutdown_command`
    pub const SHUTDOWN_COMMAND: &str = "command.shutdown_command";
}

/// Field paths under `[daemon]`.
pub mod daemon {
    /// `daemon.allow_external_management`
    pub const ALLOW_EXTERNAL_MANAGEMENT: &str = "daemon.allow_external_management";
    /// `daemon.management_listen`
    pub const MANAGEMENT_LISTEN: &str = "daemon.management_listen";
}

/// Field paths under `[devices]`.
pub mod devices {
    /// `devices.placement`
    pub const PLACEMENT: &str = "devices.placement";
    /// `devices.placement_override`
    pub const PLACEMENT_OVERRIDE: &str = "devices.placement_override";
    /// `devices.split`
    pub const SPLIT: &str = "devices.split";
    /// `devices.tensor_split_weights`
    pub const TENSOR_SPLIT_WEIGHTS: &str = "devices.tensor_split_weights";
}

/// Field paths under `[health]`.
pub mod health {
    /// `health.probe_interval`
    pub const PROBE_INTERVAL: &str = "health.probe_interval";
    /// `health.timeout`
    pub const TIMEOUT: &str = "health.timeout";
}

/// Field paths under `[llama_cpp]`.
pub mod llama_cpp {
    /// `llama_cpp.expert_offload`
    pub const EXPERT_OFFLOAD: &str = "llama_cpp.expert_offload";
    /// `llama_cpp.n_gpu_layers`
    pub const N_GPU_LAYERS: &str = "llama_cpp.n_gpu_layers";
}

/// Field paths under `[runtime]`.
pub mod runtime {
    /// `runtime.attn_max_batch`
    pub const ATTN_MAX_BATCH: &str = "runtime.attn_max_batch";
    /// `runtime.mla`
    pub const MLA: &str = "runtime.mla";
}

/// Service-level field paths.
pub mod service {
    /// `draft_model`
    pub const DRAFT_MODEL: &str = "draft_model";
    /// `expert_offload`
    pub const EXPERT_OFFLOAD: &str = "expert_offload";
    /// `launcher`
    pub const LAUNCHER: &str = "launcher";
    /// `model`
    pub const MODEL: &str = "model";
    /// `numa`
    pub const NUMA: &str = "numa";
    /// `service.drain_timeout`
    pub const DRAIN_TIMEOUT: &str = "service.drain_timeout";
    /// `service.extended_stream_drain`
    pub const EXTENDED_STREAM_DRAIN: &str = "service.extended_stream_drain";
    /// `service.idle_timeout`
    pub const IDLE_TIMEOUT: &str = "service.idle_timeout";
    /// `service.lifecycle`
    pub const LIFECYCLE: &str = "service.lifecycle";
    /// `service.max_request_duration`
    pub const MAX_REQUEST_DURATION: &str = "service.max_request_duration";
    /// `service.metadata`
    pub const METADATA: &str = "service.metadata";
    /// `service.modality`
    pub const MODALITY: &str = "service.modality";
    /// `spec_type`
    pub const SPEC_TYPE: &str = "spec_type";
}
