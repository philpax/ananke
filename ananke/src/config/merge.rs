//! Resolve `extends` inheritance and `*_append` concatenation before validation.
//!
//! Rules (spec §6.3):
//! - Scalars: child overrides parent.
//! - Sub-tables: deep-merge field-by-field.
//! - Arrays: child replaces parent outright.
//! - `*_append` siblings: parent's resolved list ++ child's `*_append`;
//!   `child.extra_args` falls back to `parent.extra_args` if not specified.
//! - `extends` is transitive; cycles are errors.
//! - `name` and `port` must be overridden; inheriting either is an error.
//! - `extends` and `migrate_from` are not themselves inherited.
//! - Parent and child must have the same template (same `RawService` variant).
//!   Cross-template `extends` is an error — inheriting e.g. llama-cpp fields
//!   into a command service is always a misconfiguration.

use std::collections::{BTreeMap, BTreeSet};

use smol_str::SmolStr;

use crate::{
    config::parse::{
        RawCommandService, RawConfig, RawLlamaCppService, RawService, RawServiceCommon,
    },
    errors::ExpectedError,
};

pub fn resolve_inheritance(cfg: &mut RawConfig) -> Result<(), ExpectedError> {
    // Fold [[persistent_service]] into [[service]] with lifecycle=persistent default.
    for mut ps in std::mem::take(&mut cfg.persistent_services) {
        if ps.common().lifecycle.is_none() {
            ps.common_mut().lifecycle = Some(SmolStr::new("persistent"));
        }
        cfg.services.push(ps);
    }

    // Index services by name; require names and disallow duplicates.
    let mut by_name: BTreeMap<SmolStr, RawService> = BTreeMap::new();
    for s in std::mem::take(&mut cfg.services) {
        let name = s.common().name.clone().ok_or_else(|| {
            ExpectedError::config_unparseable(
                std::path::PathBuf::from("<config>"),
                "service block missing name".into(),
            )
        })?;
        if by_name.insert(name.clone(), s).is_some() {
            return Err(ExpectedError::config_unparseable(
                std::path::PathBuf::from("<config>"),
                format!("duplicate service name: {name}"),
            ));
        }
    }

    // Topologically resolve each service's extends chain.
    let mut resolved: BTreeMap<SmolStr, RawService> = BTreeMap::new();
    let names: Vec<SmolStr> = by_name.keys().cloned().collect();
    for name in &names {
        resolve_one(name, &by_name, &mut resolved, &mut BTreeSet::new())?;
    }

    cfg.services = resolved.into_values().collect();
    Ok(())
}

fn resolve_one(
    name: &SmolStr,
    source: &BTreeMap<SmolStr, RawService>,
    resolved: &mut BTreeMap<SmolStr, RawService>,
    stack: &mut BTreeSet<SmolStr>,
) -> Result<(), ExpectedError> {
    if resolved.contains_key(name) {
        return Ok(());
    }
    if !stack.insert(name.clone()) {
        return Err(ExpectedError::config_unparseable(
            std::path::PathBuf::from("<config>"),
            format!("extends cycle involving service {name}"),
        ));
    }

    let raw = source.get(name).cloned().ok_or_else(|| {
        ExpectedError::config_unparseable(
            std::path::PathBuf::from("<config>"),
            format!("service {name} not found during extends resolution"),
        )
    })?;

    let merged = match raw.common().extends.clone() {
        None => raw,
        Some(parent_name) => {
            if !source.contains_key(&parent_name) {
                return Err(ExpectedError::config_unparseable(
                    std::path::PathBuf::from("<config>"),
                    format!("service {name} extends {parent_name} which does not exist"),
                ));
            }
            resolve_one(&parent_name, source, resolved, stack)?;
            let parent = resolved.get(&parent_name).unwrap().clone();
            merge_service(&parent, &raw, name)?
        }
    };

    stack.remove(name);
    resolved.insert(name.clone(), merged);
    Ok(())
}

fn merge_service(
    parent: &RawService,
    child: &RawService,
    child_name: &SmolStr,
) -> Result<RawService, ExpectedError> {
    match (parent, child) {
        (RawService::LlamaCpp(p), RawService::LlamaCpp(c)) => {
            Ok(RawService::LlamaCpp(merge_llama_cpp(p, c, child_name)?))
        }
        (RawService::Command(p), RawService::Command(c)) => {
            Ok(RawService::Command(merge_command(p, c, child_name)?))
        }
        _ => Err(ExpectedError::config_unparseable(
            std::path::PathBuf::from("<config>"),
            format!(
                "service {child_name}: template `{}` does not match parent's template `{}`; \
                 cross-template extends is not allowed",
                child.template_label(),
                parent.template_label(),
            ),
        )),
    }
}

fn merge_llama_cpp(
    parent: &RawLlamaCppService,
    child: &RawLlamaCppService,
    child_name: &SmolStr,
) -> Result<RawLlamaCppService, ExpectedError> {
    let common = merge_common(&parent.common, &child.common, child_name)?;

    macro_rules! inherit {
        ($field:ident) => {
            child.$field.clone().or_else(|| parent.$field.clone())
        };
    }

    Ok(RawLlamaCppService {
        common,
        model: inherit!(model),
        mmproj: inherit!(mmproj),
        context: inherit!(context),
        n_gpu_layers: inherit!(n_gpu_layers),
        n_cpu_moe: inherit!(n_cpu_moe),
        flash_attn: inherit!(flash_attn),
        cache_type_k: inherit!(cache_type_k),
        cache_type_v: inherit!(cache_type_v),
        mmap: inherit!(mmap),
        mlock: inherit!(mlock),
        parallel: inherit!(parallel),
        batch_size: inherit!(batch_size),
        ubatch_size: inherit!(ubatch_size),
        threads: inherit!(threads),
        threads_batch: inherit!(threads_batch),
        jinja: inherit!(jinja),
        chat_template_file: inherit!(chat_template_file),
        override_tensor: inherit!(override_tensor),
        sampling: match (parent.sampling.clone(), child.sampling.clone()) {
            (None, x) => x,
            (x, None) => x,
            (Some(p), Some(c)) => Some(crate::config::parse::SamplingConfig {
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
            (Some(p), Some(c)) => Some(crate::config::parse::EstimationConfig {
                compute_buffer_mb: c.compute_buffer_mb.or(p.compute_buffer_mb),
                safety_factor: c.safety_factor.or(p.safety_factor),
                allow_fallback: c.allow_fallback.or(p.allow_fallback),
            }),
        },
    })
}

fn merge_command(
    parent: &RawCommandService,
    child: &RawCommandService,
    child_name: &SmolStr,
) -> Result<RawCommandService, ExpectedError> {
    let common = merge_common(&parent.common, &child.common, child_name)?;

    Ok(RawCommandService {
        common,
        command: child.command.clone().or_else(|| parent.command.clone()),
        workdir: child.workdir.clone().or_else(|| parent.workdir.clone()),
        allocation: match (parent.allocation.clone(), child.allocation.clone()) {
            (None, x) => x,
            (x, None) => x,
            (Some(p), Some(c)) => Some(crate::config::parse::RawAllocation {
                mode: c.mode.or(p.mode),
                vram_gb: c.vram_gb.or(p.vram_gb),
                min_vram_gb: c.min_vram_gb.or(p.min_vram_gb),
                max_vram_gb: c.max_vram_gb.or(p.max_vram_gb),
                min_borrower_runtime: c.min_borrower_runtime.or(p.min_borrower_runtime),
            }),
        },
    })
}

fn merge_common(
    parent: &RawServiceCommon,
    child: &RawServiceCommon,
    child_name: &SmolStr,
) -> Result<RawServiceCommon, ExpectedError> {
    // Child must supply its own port; inheriting silently from a parent leads to
    // port conflicts that are hard to diagnose, so we make it an explicit error.
    if child.port.is_none() {
        return Err(ExpectedError::config_unparseable(
            std::path::PathBuf::from("<config>"),
            format!("service {child_name} must override port from parent"),
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
    take!(warming_grace);
    take!(description);
    take!(drain_timeout);
    take!(extended_stream_drain);
    take!(max_request_duration);
    take!(start_queue_depth);

    merged.metadata = deep_merge_map(parent.metadata.clone(), child.metadata.clone());
    merged.env = deep_merge_strs(parent.env.clone(), child.env.clone());

    merged.filters = match (parent.filters.clone(), child.filters.clone()) {
        (None, x) => x,
        (x, None) => x,
        (Some(p), Some(c)) => Some(crate::config::parse::RawFilters {
            strip_params: c.strip_params.or(p.strip_params),
            set_params: deep_merge_map(p.set_params, c.set_params),
        }),
    };

    merged.devices = match (parent.devices.clone(), child.devices.clone()) {
        (None, x) => x,
        (x, None) => x,
        (Some(p), Some(c)) => Some(crate::config::parse::RawServiceDevices {
            placement: c.placement.or(p.placement),
            gpu_allow: c.gpu_allow.or(p.gpu_allow),
            placement_override: c.placement_override.or(p.placement_override),
        }),
    };

    merged.health = match (parent.health.clone(), child.health.clone()) {
        (None, x) => x,
        (x, None) => x,
        (Some(p), Some(c)) => Some(crate::config::parse::RawHealth {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Migration {
    pub old_name: SmolStr,
    pub new_name: SmolStr,
}

/// Resolve `migrate_from` chains into an ordered list of (old, new) pairs.
///
/// Returns pairs in topological order (sources before dependents) so the
/// database layer can reparent sequentially. Cycles are errors.
pub fn resolve_migrations(cfg: &mut RawConfig) -> Result<Vec<Migration>, ExpectedError> {
    let mut out: Vec<Migration> = Vec::new();
    let by_name: BTreeMap<SmolStr, &RawService> = cfg
        .services
        .iter()
        .map(|s| (s.common().name.clone().unwrap(), s))
        .collect();

    let mut visiting: BTreeSet<SmolStr> = BTreeSet::new();
    let mut visited: BTreeSet<SmolStr> = BTreeSet::new();

    fn visit(
        name: &SmolStr,
        by_name: &BTreeMap<SmolStr, &RawService>,
        visiting: &mut BTreeSet<SmolStr>,
        visited: &mut BTreeSet<SmolStr>,
        out: &mut Vec<Migration>,
    ) -> Result<(), ExpectedError> {
        if visited.contains(name) {
            return Ok(());
        }
        if !visiting.insert(name.clone()) {
            return Err(ExpectedError::config_unparseable(
                std::path::PathBuf::from("<config>"),
                format!("migrate_from cycle involving {name}"),
            ));
        }
        if let Some(svc) = by_name.get(name)
            && let Some(old) = &svc.common().migrate_from
        {
            if by_name.contains_key(old) {
                visit(old, by_name, visiting, visited, out)?;
            }
            out.push(Migration {
                old_name: old.clone(),
                new_name: name.clone(),
            });
        }
        visiting.remove(name);
        visited.insert(name.clone());
        Ok(())
    }

    let names: Vec<SmolStr> = by_name.keys().cloned().collect();
    for n in &names {
        visit(n, &by_name, &mut visiting, &mut visited, &mut out)?;
    }

    // Clear the migrate_from field on services so downstream code doesn't re-process.
    for svc in cfg.services.iter_mut() {
        svc.common_mut().migrate_from = None;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::config::parse::parse_toml;

    fn parse(src: &str) -> RawConfig {
        parse_toml(src, Path::new("/t")).unwrap()
    }

    fn find_llama<'a>(cfg: &'a RawConfig, name: &str) -> &'a RawLlamaCppService {
        let svc = cfg
            .services
            .iter()
            .find(|s| s.common().name.as_deref() == Some(name))
            .unwrap();
        match svc {
            RawService::LlamaCpp(lc) => lc,
            _ => panic!("expected llama-cpp service"),
        }
    }

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
    fn transitive_extends() {
        let mut cfg = parse(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/a.gguf"
port = 11000
context = 4096

[[service]]
name = "b"
template = "llama-cpp"
extends = "a"
port = 11001

[[service]]
name = "c"
template = "llama-cpp"
extends = "b"
port = 11002
context = 32768
"#,
        );
        resolve_inheritance(&mut cfg).unwrap();
        let c = find_llama(&cfg, "c");
        assert_eq!(c.context, Some(32768));
        assert_eq!(c.model.as_ref().unwrap().to_str(), Some("/m/a.gguf"));
    }

    #[test]
    fn cycle_is_error() {
        let mut cfg = parse(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/a.gguf"
port = 11000
extends = "b"

[[service]]
name = "b"
template = "llama-cpp"
model = "/m/a.gguf"
port = 11001
extends = "a"
"#,
        );
        let err = resolve_inheritance(&mut cfg).unwrap_err();
        assert!(format!("{err}").contains("cycle"));
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
        assert!(format!("{err}").contains("port"), "error: {err}");
    }

    #[test]
    fn missing_extends_target_is_error() {
        let mut cfg = parse(
            r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/a.gguf"
port = 11000
extends = "does-not-exist"
"#,
        );
        let err = resolve_inheritance(&mut cfg).unwrap_err();
        assert!(format!("{err}").contains("does-not-exist"));
    }

    #[test]
    fn cross_template_extends_is_error() {
        let mut cfg = parse(
            r#"
[[service]]
name = "base"
template = "command"
port = 11000
command = ["/bin/true"]

[[service]]
name = "child"
template = "llama-cpp"
extends = "base"
port = 11001
model = "/m/a.gguf"
"#,
        );
        let err = resolve_inheritance(&mut cfg).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("template") && msg.contains("cross-template"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn migrate_from_chain_resolved_in_order() {
        let mut cfg = parse(
            r#"
[[service]]
name = "c"
template = "llama-cpp"
model = "/m/x.gguf"
port = 12002
migrate_from = "b"

[[service]]
name = "b"
template = "llama-cpp"
model = "/m/x.gguf"
port = 12001
migrate_from = "a"
"#,
        );
        resolve_inheritance(&mut cfg).unwrap();
        let migrations = resolve_migrations(&mut cfg).unwrap();
        let b_idx = migrations.iter().position(|m| m.new_name == "b").unwrap();
        let c_idx = migrations.iter().position(|m| m.new_name == "c").unwrap();
        assert!(b_idx < c_idx);
        assert_eq!(migrations[b_idx].old_name, "a");
        assert_eq!(migrations[c_idx].old_name, "b");
    }

    #[test]
    fn migrate_from_missing_source_is_warning_not_error() {
        let mut cfg = parse(
            r#"
[[service]]
name = "b"
template = "llama-cpp"
model = "/m/x.gguf"
port = 12001
migrate_from = "does-not-exist"
"#,
        );
        resolve_inheritance(&mut cfg).unwrap();
        let migrations = resolve_migrations(&mut cfg).unwrap();
        assert_eq!(migrations.len(), 1);
        assert_eq!(migrations[0].old_name, "does-not-exist");
    }
}
