//! Configuration loading, parsing, inheritance merging, and validation.

use std::path::Path;

use crate::errors::ExpectedError;

pub mod file;
pub mod merge;
pub mod parse;
pub mod validate;

pub use file::{PathSources, resolve_config_path, resolve_from_env};
pub use merge::{Migration, resolve_inheritance, resolve_migrations};
pub use parse::{RawConfig, RawService, parse_toml};
pub use validate::{
    DaemonSettings, DeviceSlot, EffectiveConfig, HealthSettings, Lifecycle, PlacementPolicy,
    ServiceConfig, Template, validate,
};

pub fn load_config(path: &Path) -> Result<(EffectiveConfig, Vec<Migration>), ExpectedError> {
    let source = std::fs::read_to_string(path)
        .map_err(|_| ExpectedError::config_file_missing(path.to_path_buf()))?;
    load_config_from_str(&source, path)
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
