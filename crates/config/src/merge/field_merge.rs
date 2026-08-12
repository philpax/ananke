//! Per-template, per-field deep-merge rules applied to a resolved
//! `extends` pair: scalars take the child's value, sub-tables merge
//! field-by-field, and arrays are replaced outright.

use std::collections::BTreeMap;

use smol_str::SmolStr;

use crate::{
    parse::{RawCommandService, RawLlamaCppService, RawServiceCommon},
    validate::{ConfigDiagnostic, MergeReason},
};

pub(crate) fn merge_llama_cpp(
    parent: &RawLlamaCppService,
    child: &RawLlamaCppService,
    child_name: &SmolStr,
) -> Result<RawLlamaCppService, ConfigDiagnostic> {
    let common = merge_common(&parent.common, &child.common, child_name)?;

    macro_rules! inherit {
        ($field:ident) => {
            child.$field.clone().or_else(|| parent.$field.clone())
        };
    }

    Ok(RawLlamaCppService {
        common,
        // The runtime table inherits whole, not per-field: a child
        // overriding the parent's runtime replaces it entirely.
        runtime: inherit!(runtime),
        model: inherit!(model),
        mmproj: inherit!(mmproj),
        context: inherit!(context),
        n_gpu_layers: inherit!(n_gpu_layers),
        expert_offload: inherit!(expert_offload),
        flash_attn: inherit!(flash_attn),
        cache_type_k: inherit!(cache_type_k),
        cache_type_v: inherit!(cache_type_v),
        mmap: inherit!(mmap),
        mlock: inherit!(mlock),
        parallel: inherit!(parallel),
        spec_type: inherit!(spec_type),
        spec_draft_n_max: inherit!(spec_draft_n_max),
        draft_model: inherit!(draft_model),
        kv_unified: inherit!(kv_unified),
        cache_idle_slots: inherit!(cache_idle_slots),
        cache_ram_mb: inherit!(cache_ram_mb),
        metrics: inherit!(metrics),
        slots: inherit!(slots),
        batch_size: inherit!(batch_size),
        ubatch_size: inherit!(ubatch_size),
        threads: inherit!(threads),
        threads_batch: inherit!(threads_batch),
        numa: inherit!(numa),
        jinja: inherit!(jinja),
        chat_template_file: inherit!(chat_template_file),
        override_tensor: inherit!(override_tensor),
        llama_server: inherit!(llama_server),
        launcher: inherit!(launcher),
        sampling: match (parent.sampling.clone(), child.sampling.clone()) {
            (None, x) => x,
            (x, None) => x,
            (Some(p), Some(c)) => Some(crate::parse::SamplingConfig {
                temperature: c.temperature.or(p.temperature),
                top_p: c.top_p.or(p.top_p),
                top_k: c.top_k.or(p.top_k),
                min_p: c.min_p.or(p.min_p),
                repeat_penalty: c.repeat_penalty.or(p.repeat_penalty),
            }),
        },
        estimation: match (parent.estimation.clone(), child.estimation.clone()) {
            (None, x) => x,
            (x, None) => x,
            (Some(p), Some(c)) => Some(crate::parse::EstimationConfig {
                compute_buffer_mb: c.compute_buffer_mb.or(p.compute_buffer_mb),
                safety_factor: c.safety_factor.or(p.safety_factor),
            }),
        },
    })
}

pub(crate) fn merge_command(
    parent: &RawCommandService,
    child: &RawCommandService,
    child_name: &SmolStr,
) -> Result<RawCommandService, ConfigDiagnostic> {
    let common = merge_common(&parent.common, &child.common, child_name)?;

    Ok(RawCommandService {
        common,
        command: child.command.clone().or_else(|| parent.command.clone()),
        workdir: child.workdir.clone().or_else(|| parent.workdir.clone()),
        allocation: match (parent.allocation.clone(), child.allocation.clone()) {
            (None, x) => x,
            (x, None) => x,
            (Some(p), Some(c)) => Some(crate::parse::RawAllocation {
                mode: c.mode.or(p.mode),
                reserve_gb: c.reserve_gb.or(p.reserve_gb),
                min_reserve_gb: c.min_reserve_gb.or(p.min_reserve_gb),
                max_reserve_gb: c.max_reserve_gb.or(p.max_reserve_gb),
                min_borrower_runtime: c.min_borrower_runtime.or(p.min_borrower_runtime),
            }),
        },
        private_port: child.private_port.or(parent.private_port),
        shutdown_command: child
            .shutdown_command
            .clone()
            .or_else(|| parent.shutdown_command.clone()),
        openai_proxy: child
            .openai_proxy
            .clone()
            .or_else(|| parent.openai_proxy.clone()),
    })
}

fn merge_common(
    parent: &RawServiceCommon,
    child: &RawServiceCommon,
    child_name: &SmolStr,
) -> Result<RawServiceCommon, ConfigDiagnostic> {
    // Child must supply its own port; inheriting silently from a parent leads to
    // port conflicts that are hard to diagnose, so we make it an explicit error.
    if child.port.is_none() {
        return Err(ConfigDiagnostic::merge(
            Some(child_name.to_string()),
            None,
            parent.name.as_ref().map(ToString::to_string),
            MergeReason::PortMustOverride,
        ));
    }

    let mut merged = parent.clone();

    macro_rules! take {
        ($field:ident) => {
            if child.$field.is_some() {
                merged.$field = child.$field.clone();
            }
        };
    }

    merged.name = child.name.clone();
    merged.port = child.port;
    // `extends` and `migrate_from` are not propagated to children of children.
    merged.extends = None;
    merged.migrate_from = None;

    take!(lifecycle);
    take!(priority);
    take!(idle_timeout);
    take!(description);
    take!(drain_timeout);
    take!(extended_stream_drain);
    take!(max_request_duration);
    take!(start_queue_depth);
    take!(env_inherit);
    // auto_restart is a cohesive policy block: a child that sets any
    // auto_restart field replaces the parent's whole block rather than
    // merging field-by-field, mirroring how it overrides `[defaults]`.
    take!(auto_restart);

    merged.metadata = deep_merge_map(parent.metadata.clone(), child.metadata.clone());
    merged.env = deep_merge_strs(parent.env.clone(), child.env.clone());

    merged.filters = match (parent.filters.clone(), child.filters.clone()) {
        (None, x) => x,
        (x, None) => x,
        (Some(p), Some(c)) => Some(crate::parse::RawFilters {
            strip_params: c.strip_params.or(p.strip_params),
            set_params: deep_merge_map(p.set_params, c.set_params),
        }),
    };

    merged.devices = match (parent.devices.clone(), child.devices.clone()) {
        (None, x) => x,
        (x, None) => x,
        (Some(p), Some(c)) => Some(crate::parse::RawServiceDevices {
            placement: c.placement.or(p.placement),
            gpu_allow: c.gpu_allow.or(p.gpu_allow),
            placement_override: c.placement_override.or(p.placement_override),
            gpu_headroom_mb: c.gpu_headroom_mb.or(p.gpu_headroom_mb),
            split: c.split.or(p.split),
            tensor_split_weights: c.tensor_split_weights.or(p.tensor_split_weights),
        }),
    };

    merged.health = match (parent.health.clone(), child.health.clone()) {
        (None, x) => x,
        (x, None) => x,
        (Some(p), Some(c)) => Some(crate::parse::RawHealth {
            http: c.http.or(p.http),
            timeout: c.timeout.or(p.timeout),
            probe_interval: c.probe_interval.or(p.probe_interval),
        }),
    };

    // extra_args: child replaces parent if present; otherwise inherit parent's value.
    // Then fold in the accumulated *_append chain so that downstream code sees the final
    // concatenated list in extra_args and does not need to re-apply *_append separately.
    let base_args = child
        .extra_args
        .clone()
        .or_else(|| parent.extra_args.clone())
        .unwrap_or_default();
    let mut accumulated: Vec<String> = base_args;
    if let Some(parent_append) = &parent.extra_args_append {
        accumulated.extend(parent_append.iter().cloned());
    }
    if let Some(child_append) = &child.extra_args_append {
        accumulated.extend(child_append.iter().cloned());
    }
    merged.extra_args = if accumulated.is_empty() {
        None
    } else {
        Some(accumulated)
    };
    merged.extra_args_append = None;

    Ok(merged)
}

fn deep_merge_map<V: Clone>(
    parent: Option<BTreeMap<String, V>>,
    child: Option<BTreeMap<String, V>>,
) -> Option<BTreeMap<String, V>> {
    match (parent, child) {
        (None, x) => x,
        (x, None) => x,
        (Some(mut p), Some(c)) => {
            for (k, v) in c {
                p.insert(k, v);
            }
            Some(p)
        }
    }
}

fn deep_merge_strs(
    parent: Option<BTreeMap<String, String>>,
    child: Option<BTreeMap<String, String>>,
) -> Option<BTreeMap<String, String>> {
    deep_merge_map(parent, child)
}

#[cfg(test)]
mod tests {
    use crate::{
        merge::{
            resolve_inheritance,
            test_support::{find_llama, parse},
        },
        validate::{ConfigDiagnosticKind, MergeReason},
    };

    #[test]
    fn child_scalar_overrides_parent() {
        let mut cfg = parse(
            r#"
[[service]]
name = "base"
template = "llama-cpp"
model = "/m/a.gguf"
port = 11000
context = 8192

[[service]]
name = "child"
template = "llama-cpp"
extends = "base"
port = 11001
context = 16384
"#,
        );
        resolve_inheritance(&mut cfg).unwrap();
        let c = find_llama(&cfg, "child");
        assert_eq!(c.context, Some(16384));
        assert_eq!(c.model.as_ref().unwrap().to_str(), Some("/m/a.gguf"));
    }
    #[test]
    fn extra_args_append_concatenates() {
        let mut cfg = parse(
            r#"
[[service]]
name = "base"
template = "llama-cpp"
model = "/m/a.gguf"
port = 11000
extra_args = ["--metrics"]
extra_args_append = ["--flash"]

[[service]]
name = "child"
template = "llama-cpp"
extends = "base"
port = 11001
extra_args_append = ["--verbose"]
"#,
        );
        resolve_inheritance(&mut cfg).unwrap();
        let c = find_llama(&cfg, "child");
        let args = c.common.extra_args.clone().unwrap_or_default();
        let idx_flash = args.iter().position(|a| a == "--flash");
        let idx_verbose = args.iter().position(|a| a == "--verbose");
        assert!(idx_flash.is_some(), "missing --flash in {args:?}");
        assert!(idx_verbose.is_some(), "missing --verbose in {args:?}");
        assert!(idx_flash.unwrap() < idx_verbose.unwrap());
    }
    #[test]
    fn inheriting_port_is_error() {
        let mut cfg = parse(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/a.gguf"
port = 11000

[[service]]
name = "b"
template = "llama-cpp"
extends = "a"
"#,
        );
        let err = resolve_inheritance(&mut cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert!(matches!(
            &*diag.kind,
            ConfigDiagnosticKind::Merge {
                reason: MergeReason::PortMustOverride,
                ..
            }
        ));
    }
    #[test]
    fn env_inherit_inherited_from_parent() {
        let mut cfg = parse(
            r#"
[[service]]
name = "base"
template = "command"
command = ["/bin/true"]
port = 11000
env_inherit = false

[[service]]
name = "child"
template = "command"
extends = "base"
port = 11001
"#,
        );
        resolve_inheritance(&mut cfg).unwrap();
        let child = cfg
            .services
            .iter()
            .find(|s| s.common().name.as_deref() == Some("child"))
            .unwrap();
        assert_eq!(
            child.common().env_inherit,
            Some(false),
            "child should inherit env_inherit=false from parent"
        );
    }
    #[test]
    fn env_inherit_child_overrides_parent() {
        let mut cfg = parse(
            r#"
[[service]]
name = "base"
template = "command"
command = ["/bin/true"]
port = 11000
env_inherit = false

[[service]]
name = "child"
template = "command"
extends = "base"
port = 11001
env_inherit = true
"#,
        );
        resolve_inheritance(&mut cfg).unwrap();
        let child = cfg
            .services
            .iter()
            .find(|s| s.common().name.as_deref() == Some("child"))
            .unwrap();
        assert_eq!(
            child.common().env_inherit,
            Some(true),
            "child env_inherit should override parent"
        );
    }
}
