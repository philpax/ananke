//! Topological resolution of `extends` chains: cycle detection, dispatch to
//! the per-template field merge, and same-template enforcement.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::collections::{BTreeMap, BTreeSet};

use smol_str::SmolStr;

use crate::{
    merge::field_merge::{merge_command, merge_llama_cpp},
    parse::{RawConfig, RawService},
    validate::{ConfigDiagnostic, ConfigDiagnosticReport, MergeReason},
};

#[derive(Clone)]
struct IndexedService {
    service: RawService,
    source_index: usize,
}

/// Resolve every service's `extends` chain, merging inherited fields into
/// each service and enforcing that a service only extends the same template.
///
/// The effective services remain BTreeMap ordered, while their original source
/// indexes are retained on [`RawConfig`] for diagnostic ordering.
pub fn resolve_inheritance(cfg: &mut RawConfig) -> Result<(), ConfigDiagnosticReport> {
    let mut report = ConfigDiagnosticReport::new();
    let source_indices = if cfg.service_source_indices.len() == cfg.services.len() {
        cfg.service_source_indices.clone()
    } else {
        (0..cfg.services.len()).collect()
    };
    let mut by_name: BTreeMap<SmolStr, IndexedService> = BTreeMap::new();
    let mut skipped = Vec::new();
    for (source_index, service) in std::mem::take(&mut cfg.services).into_iter().enumerate() {
        let source_index = source_indices
            .get(source_index)
            .copied()
            .unwrap_or(source_index);
        let Some(name) = service.common().name.clone() else {
            report.push(
                ConfigDiagnostic::value(
                    crate::validate::ValidationErrorCode::FieldMissing,
                    format!("service[{source_index}].name"),
                    "<missing>",
                    Some("a service name".into()),
                )
                .with_source_index(source_index),
            );
            skipped.push(IndexedService {
                service,
                source_index,
            });
            continue;
        };
        if by_name.contains_key(&name) {
            report.push(
                ConfigDiagnostic::value(
                    crate::validate::ValidationErrorCode::ServiceNameDuplicate,
                    "service.name",
                    name.to_string(),
                    Some("a unique service name".into()),
                )
                .with_source_index(source_index),
            );
            skipped.push(IndexedService {
                service,
                source_index,
            });
            continue;
        }
        by_name.insert(
            name,
            IndexedService {
                service,
                source_index,
            },
        );
    }

    let mut resolved: BTreeMap<SmolStr, IndexedService> = BTreeMap::new();
    let names: Vec<SmolStr> = by_name.keys().cloned().collect();
    for name in &names {
        if let Err(child_report) = resolve_one(name, &by_name, &mut resolved, &mut BTreeSet::new())
        {
            report.extend(child_report);
            if let Some(entry) = by_name.get(name) {
                skipped.push(entry.clone());
            }
        }
    }

    cfg.services = resolved
        .values()
        .map(|entry| entry.service.clone())
        .chain(skipped.iter().map(|entry| entry.service.clone()))
        .collect();
    cfg.service_source_indices = resolved
        .values()
        .map(|entry| entry.source_index)
        .chain(skipped.iter().map(|entry| entry.source_index))
        .collect();
    if report.is_empty() {
        Ok(())
    } else {
        Err(report)
    }
}

fn resolve_one(
    name: &SmolStr,
    source: &BTreeMap<SmolStr, IndexedService>,
    resolved: &mut BTreeMap<SmolStr, IndexedService>,
    stack: &mut BTreeSet<SmolStr>,
) -> Result<(), ConfigDiagnosticReport> {
    if resolved.contains_key(name) {
        return Ok(());
    }
    if !stack.insert(name.clone()) {
        return Err(ConfigDiagnosticReport::from(ConfigDiagnostic::merge(
            Some(name.to_string()),
            source.get(name).map(|entry| entry.source_index),
            None,
            MergeReason::Cycle,
        )));
    }

    let Some(raw_entry) = source.get(name).cloned() else {
        return Err(ConfigDiagnosticReport::from(ConfigDiagnostic::merge(
            Some(name.to_string()),
            None,
            None,
            MergeReason::ServiceNotFound,
        )));
    };
    let raw = raw_entry.service.clone();
    let merged = match raw.common().extends.clone() {
        None => raw,
        Some(parent_name) => {
            if !source.contains_key(&parent_name) {
                return Err(ConfigDiagnosticReport::from(ConfigDiagnostic::merge(
                    Some(name.to_string()),
                    Some(raw_entry.source_index),
                    Some(parent_name.to_string()),
                    MergeReason::ParentNotFound,
                )));
            }
            resolve_one(&parent_name, source, resolved, stack)?;
            let Some(parent) = resolved
                .get(&parent_name)
                .map(|entry| entry.service.clone())
            else {
                return Err(ConfigDiagnosticReport::from(ConfigDiagnostic::merge(
                    Some(name.to_string()),
                    Some(raw_entry.source_index),
                    Some(parent_name.to_string()),
                    MergeReason::ParentResolvedToNothing,
                )));
            };
            merge_service(&parent, &raw, name).map_err(ConfigDiagnosticReport::from)?
        }
    };

    stack.remove(name);
    resolved.insert(
        name.clone(),
        IndexedService {
            service: merged,
            source_index: raw_entry.source_index,
        },
    );
    Ok(())
}

fn merge_service(
    parent: &RawService,
    child: &RawService,
    child_name: &SmolStr,
) -> Result<RawService, ConfigDiagnostic> {
    match (parent, child) {
        (RawService::LlamaCpp(p), RawService::LlamaCpp(c)) => Ok(RawService::LlamaCpp(Box::new(
            merge_llama_cpp(p, c, child_name)?,
        ))),
        (RawService::Command(p), RawService::Command(c)) => Ok(RawService::Command(Box::new(
            merge_command(p, c, child_name)?,
        ))),
        _ => Err(ConfigDiagnostic::merge(
            Some(child_name.to_string()),
            None,
            parent.common().name.as_ref().map(ToString::to_string),
            MergeReason::TemplateMismatch {
                child: child.template_label().to_string(),
                parent: parent.template_label().to_string(),
            },
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        merge::test_support::{find_llama, parse},
        validate::{ConfigDiagnosticKind, MergeReason},
    };

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
        let diag = &err.as_slice()[0];
        assert!(matches!(
            &*diag.kind,
            ConfigDiagnosticKind::Merge {
                reason: MergeReason::Cycle,
                ..
            }
        ));
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
        let diag = &err.as_slice()[0];
        assert!(matches!(
            &*diag.kind,
            ConfigDiagnosticKind::Merge {
                reason: MergeReason::ParentNotFound,
                ..
            }
        ));
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
        let diag = &err.as_slice()[0];
        assert!(matches!(
            &*diag.kind,
            ConfigDiagnosticKind::Merge {
                reason: MergeReason::TemplateMismatch { .. },
                ..
            }
        ));
    }
}
