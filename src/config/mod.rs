//! Configuration loading, parsing, inheritance merging, and validation.

pub mod file;
pub mod merge;
pub mod parse;

pub use file::{PathSources, resolve_config_path, resolve_from_env};
pub use merge::resolve_inheritance;
pub use parse::{RawConfig, RawService, parse_toml};
