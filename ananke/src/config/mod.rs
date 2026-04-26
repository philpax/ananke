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
    TemplateConfig, TrackingSettings, validate,
};

/// Load, parse, merge, validate, and preflight a config file from disk.
///
/// Unlike [`load_config_from_str`], this also walks each llama-cpp service's
/// GGUF model to surface unsupported dtypes at config-load time rather
/// than at first-request time. The alternative — silent dtype fallback —
/// produced 4× over-reservations for MXFP4 experts before being caught.
pub fn load_config(path: &Path) -> Result<(EffectiveConfig, Vec<Migration>), ExpectedError> {
    let fs = crate::system::LocalFs;
    let source = crate::system::Fs::read_to_string(&fs, path)
        .map_err(|_| ExpectedError::config_file_missing(path.to_path_buf()))?;
    load_config_with_fs(path, &fs, &source)
}

/// Variant of [`load_config`] that uses an explicit filesystem for the
/// GGUF preflight (but takes the TOML source directly rather than reading
/// it through the fs). Used by `ConfigManager`'s validation path so tests
/// can preflight against a synthetic filesystem.
pub fn load_config_with_fs(
    origin: &Path,
    fs: &dyn crate::system::Fs,
    source: &str,
) -> Result<(EffectiveConfig, Vec<Migration>), ExpectedError> {
    let (effective, migrations) = load_config_from_str(source, origin)?;
    preflight_ggufs(&effective, fs)?;
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

/// Walk every llama-cpp service's GGUF through `fs` and ensure the reader
/// can enumerate each tensor table. Errors here surface at daemon boot or
/// config reload — unknown dtypes, unreadable shards, bad magic — before
/// traffic touches the estimator or placement engine.
///
/// Takes an explicit [`crate::system::Fs`] so tests can swap in
/// [`crate::system::InMemoryFs`] preloaded with synthetic bytes and avoid
/// touching a tempdir.
pub fn preflight_ggufs(
    cfg: &EffectiveConfig,
    fs: &dyn crate::system::Fs,
) -> Result<(), ExpectedError> {
    for svc in &cfg.services {
        let Some(lc) = svc.llama_cpp() else {
            continue;
        };
        crate::gguf::read(fs, &lc.model).map_err(|e| {
            ExpectedError::config_unparseable(
                std::path::PathBuf::from("<preflight>"),
                format!("service {}: {}", svc.name, e),
            )
        })?;
        if let Some(mmproj) = &lc.mmproj {
            crate::gguf::read(fs, mmproj.as_path()).map_err(|e| {
                ExpectedError::config_unparseable(
                    std::path::PathBuf::from("<preflight>"),
                    format!("service {} mmproj: {}", svc.name, e),
                )
            })?;
        }
    }
    Ok(())
}
