//! Validate a single `[[service]]` block into a [`ServiceConfig`], resolving
//! defaults, placement, lifecycle, timeouts, and the template-specific body.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use ananke_api::shared::modality::Modality;
use ananke_errors::ExpectedError;
use smol_str::SmolStr;
use tracing::warn;

use crate::{
    docs::{
        DEFAULT_DRAIN_TIMEOUT_MS, DEFAULT_EXTENDED_STREAM_DRAIN_MS,
        DEFAULT_HEALTH_PROBE_INTERVAL_MS, DEFAULT_HEALTH_TIMEOUT_MS, DEFAULT_IDLE_TIMEOUT_MS,
        DEFAULT_MAX_REQUEST_DURATION_MS, DEFAULT_MIN_BORROWER_RUNTIME_MS, DEFAULT_SERVICE_PRIORITY,
        DEFAULT_STOP_GRACE_MS,
    },
    parse::RawService,
    validate::{
        AllocationMode, DaemonValidationCtx, DeviceSlot, Filters, HealthSettings, Lifecycle,
        PlacementPolicy, ServiceConfig, ServiceValidationState, SplitMode, Template,
        TemplateConfig, build_ananke_metadata, command_uses_port_placeholder, fail,
        parse_duration_ms, toml_value_to_json, validate_auto_restart,
        validate_bridge_command_endpoint, validate_command, validate_container, validate_llama_cpp,
        validate_tracking,
    },
};

pub(crate) fn validate_service(
    index: usize,
    raw: &RawService,
    daemon: &DaemonValidationCtx<'_>,
    state: &mut ServiceValidationState<'_>,
) -> Result<ServiceConfig, ExpectedError> {
    let common = raw.common();
    let name = common
        .name
        .clone()
        .ok_or_else(|| fail(format!("service[{index}] missing name")))?;
    let port = common
        .port
        .ok_or_else(|| fail(format!("service {name} missing port")))?;

    if name.starts_with(super::RESERVED_SERVICE_NAME_PREFIX) {
        return Err(fail(format!(
            "service name `{name}` starts with `{}`, which is reserved for manual reservations",
            super::RESERVED_SERVICE_NAME_PREFIX
        )));
    }
    if !state.names.insert(name.clone()) {
        return Err(fail(format!("duplicate service name `{name}`")));
    }
    if !state.ports.insert(port) {
        return Err(fail(format!("duplicate service port {port}")));
    }
    if Some(port) == daemon.management_port {
        return Err(fail(format!(
            "service {name} port {port} collides with daemon.management_listen"
        )));
    }

    let (allocation_mode, template_config, raw_container) = match raw {
        RawService::LlamaCpp(lc) => {
            // Reject launcher + container: both replace the host execution boundary.
            if lc.launcher.is_some() && lc.container.is_some() {
                return Err(fail(format!(
                    "service {name}: launcher and [service.container] are mutually exclusive (both replace the host execution boundary)"
                )));
            }
            let tc = validate_llama_cpp(
                &name,
                lc,
                daemon.daemon_llama_server,
                daemon.placeholder_checker,
            )?;
            // llama-cpp never takes an allocation.mode; none of the dynamic
            // knobs apply here.
            let alloc = AllocationMode::from_parts(
                Template::LlamaCpp,
                None,
                None,
                None,
                None,
                DEFAULT_MIN_BORROWER_RUNTIME_MS,
            )
            .map_err(|e| fail(format!("service {name}: {e}")))?;
            (
                alloc,
                TemplateConfig::LlamaCpp(Box::new(tc)),
                lc.container.clone(),
            )
        }
        RawService::Command(cmd) => {
            // Reject shutdown_command + container: native runtime cleanup replaces it.
            if cmd.shutdown_command.is_some() && cmd.container.is_some() {
                return Err(fail(format!(
                    "service {name}: shutdown_command is not allowed with [service.container] (native runtime cleanup replaces it)"
                )));
            }
            let raw_alloc = cmd.allocation.clone().unwrap_or_default();
            let runtime_ms = raw_alloc
                .min_borrower_runtime
                .as_deref()
                .map(parse_duration_ms)
                .transpose()
                .map_err(|e| fail(format!("service {name} min_borrower_runtime: {e}")))?
                .unwrap_or(DEFAULT_MIN_BORROWER_RUNTIME_MS);
            let alloc = AllocationMode::from_parts(
                Template::Command,
                raw_alloc.mode.as_deref(),
                raw_alloc.reserve_gb,
                raw_alloc.min_reserve_gb,
                raw_alloc.max_reserve_gb,
                runtime_ms,
            )
            .map_err(|e| fail(format!("service {name}: {e}")))?;
            let tc = validate_command(&name, cmd, daemon.placeholder_checker)?;
            (alloc, TemplateConfig::Command(tc), cmd.container.clone())
        }
    };

    let lifecycle_str = common
        .lifecycle
        .clone()
        .unwrap_or_else(|| SmolStr::new("on_demand"));
    let lifecycle = match lifecycle_str.as_str() {
        "persistent" => Lifecycle::Persistent,
        "on_demand" => Lifecycle::OnDemand,
        other => return Err(fail(format!("service {name}: unknown lifecycle `{other}`"))),
    };

    let metadata = build_ananke_metadata(common.metadata.as_ref())
        .map_err(|e| fail(format!("service {name} metadata: {e}")))?;
    let modality = match common.modality.as_deref() {
        None | Some("chat") => Modality::Chat,
        Some("embedding") => Modality::Embedding,
        Some(other) => {
            return Err(fail(format!(
                "service {name}: unknown modality `{other}` (valid: `chat`, `embedding`)"
            )));
        }
    };
    // llama-cpp always speaks OpenAI. Command services opt in by
    // setting `[service.openai_proxy] upstream_model = ...`; that's
    // also where the model-name rewrite to the upstream lives.
    let openai_compat = match &template_config {
        TemplateConfig::LlamaCpp(_) => true,
        TemplateConfig::Command(cmd) => cmd.openai_proxy.is_some(),
    };

    let dev = common.devices.clone().unwrap_or_default();
    let n_gpu_layers = match &template_config {
        TemplateConfig::LlamaCpp(lc) => lc.n_gpu_layers,
        TemplateConfig::Command(_) => None,
    };
    let placement_policy = match dev.placement.as_deref().unwrap_or("gpu-only") {
        "gpu-only" => PlacementPolicy::GpuOnly,
        "cpu-only" => {
            // Invariant: the guard below only enters when `n_gpu_layers` is
            // non-zero, which implies it is `Some`.
            if let Some(n) = n_gpu_layers
                && n != 0
            {
                return Err(fail(format!(
                    "service {name}: devices.placement=cpu-only with n_gpu_layers={} is invalid",
                    n
                )));
            }
            PlacementPolicy::CpuOnly
        }
        "hybrid" => PlacementPolicy::Hybrid,
        other => return Err(fail(format!("service {name}: unknown placement `{other}`"))),
    };

    if let TemplateConfig::LlamaCpp(lc) = &template_config
        && lc.context.is_none()
    {
        warn!(
            service = %name,
            "no context set; the estimator will default to 4096 tokens"
        );
    }

    let raw_override = dev.placement_override.clone().unwrap_or_default();
    if dev.placement_override.is_some() && raw_override.is_empty() {
        return Err(fail(format!(
            "service {name}: devices.placement_override is empty"
        )));
    }
    let mut placement_override = BTreeMap::new();
    for (k, v) in raw_override {
        let slot = match k.as_str() {
            "cpu" => DeviceSlot::Cpu,
            s if s.starts_with("gpu:") => {
                let n: u32 = s[4..].parse().map_err(|_| {
                    fail(format!(
                        "service {name}: invalid placement_override key `{s}`"
                    ))
                })?;
                DeviceSlot::Gpu(n)
            }
            other => {
                return Err(fail(format!(
                    "service {name}: invalid placement_override key `{other}`"
                )));
            }
        };
        if v == 0 {
            return Err(fail(format!(
                "service {name}: placement_override for {k} is zero"
            )));
        }
        placement_override.insert(slot, v);
    }

    if placement_policy == PlacementPolicy::GpuOnly
        && placement_override.contains_key(&DeviceSlot::Cpu)
    {
        return Err(fail(format!(
            "service {name}: placement=gpu-only but placement_override includes cpu"
        )));
    }

    let gpu_allow = dev.gpu_allow.clone().unwrap_or_default();
    let gpu_headroom_mb = dev.gpu_headroom_mb.unwrap_or(0);

    let split_mode = match dev.split.as_deref() {
        None => SplitMode::Layer,
        Some(s) => SplitMode::from_flag(s).ok_or_else(|| {
            fail(format!(
                "service {name}: unknown devices.split `{s}` (expected {})",
                SplitMode::valid_values()
            ))
        })?,
    };

    // Expert offload moves expert tensors to the CPU. That makes it
    // incompatible with a sharded (tensor/row) split — which divides every
    // layer across the GPUs in parallel with no CPU half — and it requires a
    // CPU-allowing placement. Reject both combinations at load time rather than
    // silently producing a placement that can't honour the request.
    if let TemplateConfig::LlamaCpp(lc) = &template_config
        && lc.expert_offload.is_enabled()
    {
        if split_mode.is_sharded() {
            return Err(fail(format!(
                "service {name}: expert_offload cannot be combined with devices.split=`{}` (sharded split is GPU-only; expert offload targets the CPU)",
                split_mode.as_flag()
            )));
        }
        if placement_policy != PlacementPolicy::Hybrid {
            return Err(fail(format!(
                "service {name}: expert_offload requires placement=hybrid (expert tensors offload to CPU)"
            )));
        }
    }
    if split_mode.is_sharded() {
        // Tensor/row split shards every layer across all spanned GPUs in
        // parallel; there is no CPU half and no per-tensor override to honour.
        if placement_policy != PlacementPolicy::GpuOnly {
            return Err(fail(format!(
                "service {name}: devices.split=`{}` requires placement=gpu-only (tensor/row split cannot spill to CPU)",
                split_mode.as_flag()
            )));
        }
        match &template_config {
            TemplateConfig::Command(_) => {
                return Err(fail(format!(
                    "service {name}: devices.split=`{}` is only valid for llama-cpp services",
                    split_mode.as_flag()
                )));
            }
            TemplateConfig::LlamaCpp(lc) if !lc.override_tensor.is_empty() => {
                return Err(fail(format!(
                    "service {name}: devices.split=`{}` cannot be combined with override_tensor",
                    split_mode.as_flag()
                )));
            }
            TemplateConfig::LlamaCpp(_) => {}
        }
    }

    let tensor_split_weights = dev.tensor_split_weights.clone();
    if let Some(ref weights) = tensor_split_weights {
        if !split_mode.is_sharded() {
            return Err(fail(format!(
                "service {name}: devices.tensor_split_weights is only valid with a sharded split mode (`row` or `tensor`)"
            )));
        }
        // When gpu_allow is set, validate it is sorted ascending and free of
        // duplicates. The packer sorts GPUs ascending before pairing weights
        // by index, so the pairing is correct regardless, but catching mistakes
        // here ensures the config matches the documented expectation and
        // prevents a duplicate like `[0, 0]` from causing a runtime count
        // mismatch (the runtime GPU snapshot deduplicates, so `gpu_allow =
        // [0, 0]` with 2 weights would otherwise slip through and fail at
        // pack time).
        if !gpu_allow.is_empty() {
            // Check duplicates first so `[0, 0]` reports the duplicate error,
            // not the ordering error. Use a set to catch non-adjacent
            // duplicates like `[1, 0, 1]` too. Then check non-decreasing order
            // so a strictly-descending `[1, 0]` reports the ordering error.
            let unique: BTreeSet<u32> = gpu_allow.iter().copied().collect();
            if unique.len() != gpu_allow.len() {
                return Err(fail(format!(
                    "service {name}: devices.gpu_allow must not contain duplicate GPU ids when tensor_split_weights is set"
                )));
            }
            if !gpu_allow.windows(2).all(|w| w[0] <= w[1]) {
                return Err(fail(format!(
                    "service {name}: devices.gpu_allow must be in ascending GPU-id order when tensor_split_weights is set (got {gpu_allow:?})"
                )));
            }
        }
        let n_allowed = if !gpu_allow.is_empty() {
            gpu_allow.len()
        } else if let Some(ref gpu_ids) = daemon.devices.gpu_ids {
            gpu_ids.len()
        } else {
            // The visible GPU count is not known until runtime; validate count
            // only when the operator has constrained it in config. Warn so the
            // operator knows the count check is deferred to placement time.
            warn!(
                service = %name,
                "tensor_split_weights has {} entries but the GPU count is not constrained (set gpu_allow or [devices].gpu_ids); count will be checked at placement time",
                weights.len()
            );
            weights.len()
        };
        if weights.len() != n_allowed {
            return Err(fail(format!(
                "service {name}: devices.tensor_split_weights has {} entries but {} allowed GPU(s) (set via gpu_allow or [devices].gpu_ids)",
                weights.len(),
                n_allowed
            )));
        }
        for (i, &w) in weights.iter().enumerate() {
            if !w.is_finite() || w <= 0.0 {
                return Err(fail(format!(
                    "service {name}: devices.tensor_split_weights[{i}] must be a positive finite number, got {w}"
                )));
            }
        }
    }

    let health_raw = common.health.clone().unwrap_or_default();
    let health = HealthSettings {
        http_path: match &health_raw.http {
            Some(s) if s.is_empty() => None,
            Some(s) => Some(s.clone()),
            None => Some("/v1/models".into()),
        },
        timeout_ms: health_raw
            .timeout
            .map(|s| {
                parse_duration_ms(&s)
                    .map_err(|e| fail(format!("service {name} health.timeout: {e}")))
            })
            .transpose()?
            .unwrap_or(DEFAULT_HEALTH_TIMEOUT_MS),
        probe_interval_ms: health_raw
            .probe_interval
            .map(|s| {
                parse_duration_ms(&s)
                    .map_err(|e| fail(format!("service {name} health.probe_interval: {e}")))
            })
            .transpose()?
            .unwrap_or(DEFAULT_HEALTH_PROBE_INTERVAL_MS),
    };

    let priority = common
        .priority
        .or(daemon.defaults.priority)
        .unwrap_or(DEFAULT_SERVICE_PRIORITY);
    let idle_timeout_ms = common
        .idle_timeout
        .as_deref()
        .or(daemon.defaults.idle_timeout.as_deref())
        .map(parse_duration_ms)
        .transpose()
        .map_err(|e| fail(format!("service {name} idle_timeout: {e}")))?
        .unwrap_or(DEFAULT_IDLE_TIMEOUT_MS);
    let drain_timeout_ms = common
        .drain_timeout
        .as_deref()
        .map(parse_duration_ms)
        .transpose()
        .map_err(|e| fail(format!("service {name} drain_timeout: {e}")))?
        .unwrap_or(DEFAULT_DRAIN_TIMEOUT_MS);
    let stop_grace_ms = common
        .stop_grace
        .as_deref()
        .map(parse_duration_ms)
        .transpose()
        .map_err(|e| fail(format!("service {name} stop_grace: {e}")))?
        .unwrap_or(DEFAULT_STOP_GRACE_MS);
    let extended_stream_drain_ms = common
        .extended_stream_drain
        .as_deref()
        .map(parse_duration_ms)
        .transpose()
        .map_err(|e| fail(format!("service {name} extended_stream_drain: {e}")))?
        .unwrap_or(DEFAULT_EXTENDED_STREAM_DRAIN_MS);
    let max_request_duration_ms = common
        .max_request_duration
        .as_deref()
        .map(parse_duration_ms)
        .transpose()
        .map_err(|e| fail(format!("service {name} max_request_duration: {e}")))?
        .unwrap_or(DEFAULT_MAX_REQUEST_DURATION_MS);

    let mut filters = Filters::default();
    if let Some(raw_filters) = &common.filters {
        if let Some(strip) = &raw_filters.strip_params {
            filters.strip_params = strip.clone();
        }
        if let Some(set) = &raw_filters.set_params {
            for (k, v) in set {
                let json_val = toml_value_to_json(v.clone())
                    .map_err(|e| fail(format!("service {name} filters.set_params[{k}]: {e}")))?;
                filters.set_params.insert(k.clone(), json_val);
            }
        }
    }

    let start_queue_depth = common
        .start_queue_depth
        .unwrap_or(crate::parse::DEFAULT_START_QUEUE_DEPTH);

    let extra_args = common.extra_args.clone().unwrap_or_default();
    // extra_args_append is consumed into extra_args during merge for extending
    // services, but for non-extending services it's still present here. Fold it
    // in so downstream sees a single list.
    let mut all_extra = extra_args;
    if let Some(append) = &common.extra_args_append {
        all_extra.extend(append.iter().cloned());
    }
    let env = common.env.clone().unwrap_or_default();
    daemon.placeholder_checker.check_env(&name, "env", &env)?;
    let env_inherit = common.env_inherit.unwrap_or(true);

    // Allocate a private loopback port. Default is auto-assignment from
    // the daemon's pool; a command service may override with a fixed
    // port (used when the external service binds a predictable host
    // port, e.g. a docker container). If the operator didn't override,
    // warn when their `command`/`env` never substitutes `${port}` — that
    // suggests the child binds a fixed port ananke doesn't know about.
    let private_port_override = match &template_config {
        TemplateConfig::Command(cmd) => cmd.private_port_override,
        TemplateConfig::LlamaCpp(_) => None,
    };
    let private_port = if let Some(fixed) = private_port_override {
        if state.private_ports.contains(fixed) {
            warn!(
                service = %name,
                port = fixed,
                range_start = state.private_ports.range.start,
                range_end = state.private_ports.range.end,
                "private_port override falls inside the auto-assignment pool; a later auto-assigned service may collide — move this port outside [private_port_start, private_port_end]"
            );
        }
        fixed
    } else {
        let p = state.private_ports.allocate(&name)?;
        if let TemplateConfig::Command(cmd) = &template_config
            && !command_uses_port_placeholder(cmd, common.env.as_ref())
        {
            warn!(
                service = %name,
                private_port = p,
                "auto-assigned private_port is never referenced via `${{port}}` in the command or env — the child likely binds a different port and ananke's proxy will fail to forward. Either substitute `${{port}}` or set `private_port` to match the child's actual port"
            );
        }
        p
    };

    let tracking = validate_tracking(&name, common.tracking.as_ref())?;

    // auto_restart resolves as a whole block: a service's own block replaces
    // `[defaults.auto_restart]` entirely rather than merging field-by-field.
    let has_spec_type = match &template_config {
        TemplateConfig::LlamaCpp(lc) => lc.spec_type.is_some(),
        TemplateConfig::Command(_) => false,
    };
    let auto_restart = validate_auto_restart(
        &name,
        common
            .auto_restart
            .as_ref()
            .or(daemon.defaults.auto_restart.as_ref()),
        template_config.template(),
        has_spec_type,
        common.auto_restart.is_some(),
    )?;

    // Validate container configuration (if present).
    let container = match raw_container {
        None => None,
        Some(raw_ct) => {
            let ct = validate_container(&name, &raw_ct)?;
            // `container.env` is substituted at spawn like any argv entry,
            // so a typo in one has to fail here rather than at launch.
            daemon
                .placeholder_checker
                .check_env(&name, "container.env", &ct.env)?;
            // A bridge-networked command service publishes a loopback host
            // port onto a container port; ananke can only guarantee the
            // published endpoint is reachable if the command actually binds
            // the resolved interface and port it was given.
            if let TemplateConfig::Command(cmd) = &template_config {
                validate_bridge_command_endpoint(&name, &ct, &cmd.command, &env)?;
            }
            Some(ct)
        }
    };

    Ok(ServiceConfig {
        name,
        port,
        private_port,
        lifecycle,
        priority,
        health,
        placement_override,
        placement_policy,
        gpu_allow,
        split_mode,
        tensor_split_weights,
        gpu_headroom_mb,
        reserves: Arc::clone(daemon.reserves),
        filters,
        idle_timeout_ms,
        drain_timeout_ms,
        stop_grace_ms,
        extended_stream_drain_ms,
        max_request_duration_ms,
        auto_restart,
        allocation_mode,
        openai_compat,
        description: common.description.clone(),
        modality,
        start_queue_depth,
        extra_args: all_extra,
        env,
        env_inherit,
        tracking,
        metadata,
        template_config,
        container,
    })
}
