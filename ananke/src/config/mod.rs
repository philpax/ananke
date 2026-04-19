//! Configuration loading, parsing, inheritance merging, and validation.

use std::path::Path;

use crate::errors::ExpectedError;

pub mod file;
pub mod manager;
pub mod merge;
pub mod parse;
pub mod validate;

pub use file::{PathSources, resolve_config_path, resolve_from_env};
pub use merge::{Migration, resolve_inheritance, resolve_migrations};
pub use parse::{RawConfig, RawService, parse_toml};
pub use validate::{
    AllocationMode, CommandConfig, DaemonSettings, DeviceSlot, EffectiveConfig, Filters,
    HealthSettings, Lifecycle, LlamaCppConfig, PlacementPolicy, ServiceConfig, Template,
    TemplateConfig, validate,
};

/// Load, parse, merge, validate, and preflight a config file from disk.
///
/// Unlike [`load_config_from_str`], this also walks each llama-cpp service's
/// GGUF model to surface unsupported dtypes (e.g. a new quant format that
/// ananke's `GgufType` table hasn't been taught about) at config-load time
/// rather than at first-request time. The alternative — silent dtype
/// fallback — produced 4× over-reservations for MXFP4 experts before being
/// caught.
pub fn load_config(path: &Path) -> Result<(EffectiveConfig, Vec<Migration>), ExpectedError> {
    let source = std::fs::read_to_string(path)
        .map_err(|_| ExpectedError::config_file_missing(path.to_path_buf()))?;
    let (effective, migrations) = load_config_from_str(&source, path)?;
    preflight_ggufs(&effective)?;
    Ok((effective, migrations))
}

pub fn load_config_from_str(
    source: &str,
    origin: &Path,
) -> Result<(EffectiveConfig, Vec<Migration>), ExpectedError> {
    let mut raw = parse_toml(source, origin)?;
    resolve_inheritance(&mut raw)?;
    let migrations = resolve_migrations(&mut raw)?;
    let effective = validate(&raw)?;
    Ok((effective, migrations))
}

/// Walk every llama-cpp service's GGUF and ensure the reader can enumerate
/// its tensor table. Errors here surface at daemon boot or config reload —
/// unknown dtypes, unreadable shards, bad magic — before traffic touches
/// the estimator or placement engine.
pub fn preflight_ggufs(cfg: &EffectiveConfig) -> Result<(), ExpectedError> {
    for svc in &cfg.services {
        let Some(lc) = svc.llama_cpp() else {
            continue;
        };
        crate::gguf::read(&lc.model).map_err(|e| {
            ExpectedError::config_unparseable(
                std::path::PathBuf::from("<preflight>"),
                format!("service {}: {}", svc.name, e),
            )
        })?;
        if let Some(mmproj) = &lc.mmproj {
            crate::gguf::read(mmproj.as_path()).map_err(|e| {
                ExpectedError::config_unparseable(
                    std::path::PathBuf::from("<preflight>"),
                    format!("service {} mmproj: {}", svc.name, e),
                )
            })?;
        }
    }
    Ok(())
}
