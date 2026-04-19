//! Config path resolution.

use std::path::{Path, PathBuf};

use crate::errors::ExpectedError;

/// Sources checked for the config file, in priority order.
pub struct PathSources<'a> {
    pub env_ananke_config: Option<&'a str>,
    pub cli_config: Option<&'a Path>,
    pub xdg_config_home: Option<&'a Path>,
    pub home: Option<&'a Path>,
}

pub fn resolve_config_path(sources: PathSources<'_>) -> Result<PathBuf, ExpectedError> {
    if let Some(p) = sources.env_ananke_config {
        return Ok(PathBuf::from(p));
    }
    if let Some(p) = sources.cli_config {
        return Ok(p.to_path_buf());
    }
    if let Some(xdg) = sources.xdg_config_home {
        return Ok(xdg.join("ananke").join("config.toml"));
    }
    if let Some(home) = sources.home {
        return Ok(home.join(".config").join("ananke").join("config.toml"));
    }
    Ok(PathBuf::from("/etc/ananke/config.toml"))
}

/// Resolve config path from the live process environment.
/// Called from `main` via the library. Tests inject via [`resolve_config_path`].
pub fn resolve_from_env(cli_config: Option<&Path>) -> Result<PathBuf, ExpectedError> {
    let env = std::env::var("ANANKE_CONFIG").ok();
    let xdg = std::env::var("XDG_CONFIG_HOME").ok().map(PathBuf::from);
    let home = std::env::var("HOME").ok().map(PathBuf::from);
    resolve_config_path(PathSources {
        env_ananke_config: env.as_deref(),
        cli_config,
        xdg_config_home: xdg.as_deref(),
        home: home.as_deref(),
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn env_wins_over_cli() {
        let path = resolve_config_path(PathSources {
            env_ananke_config: Some("/a/env.toml"),
            cli_config: Some(Path::new("/b/cli.toml")),
            xdg_config_home: None,
            home: None,
        })
        .unwrap();
        assert_eq!(path, PathBuf::from("/a/env.toml"));
    }

    #[test]
    fn cli_wins_over_xdg() {
        let path = resolve_config_path(PathSources {
            env_ananke_config: None,
            cli_config: Some(Path::new("/b/cli.toml")),
            xdg_config_home: Some(Path::new("/home/u/.config")),
            home: None,
        })
        .unwrap();
        assert_eq!(path, PathBuf::from("/b/cli.toml"));
    }

    #[test]
    fn xdg_default() {
        let path = resolve_config_path(PathSources {
            env_ananke_config: None,
            cli_config: None,
            xdg_config_home: Some(Path::new("/home/u/.config")),
            home: None,
        })
        .unwrap();
        assert_eq!(path, PathBuf::from("/home/u/.config/ananke/config.toml"));
    }

    #[test]
    fn xdg_falls_back_to_home_dot_config() {
        let path = resolve_config_path(PathSources {
            env_ananke_config: None,
            cli_config: None,
            xdg_config_home: None,
            home: Some(Path::new("/home/u")),
        })
        .unwrap();
        assert_eq!(path, PathBuf::from("/home/u/.config/ananke/config.toml"));
    }

    #[test]
    fn etc_fallback() {
        let path = resolve_config_path(PathSources {
            env_ananke_config: None,
            cli_config: None,
            xdg_config_home: None,
            home: None,
        })
        .unwrap();
        assert_eq!(path, PathBuf::from("/etc/ananke/config.toml"));
    }
}
