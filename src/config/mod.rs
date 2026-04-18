//! Configuration loading, parsing, inheritance merging, and validation.

pub mod file;

pub use file::{PathSources, resolve_config_path, resolve_from_env};
