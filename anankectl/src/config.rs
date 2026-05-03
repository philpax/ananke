//! Client-side configuration loaded from `~/.config/anankectl/config.toml`.
//!
//! Holds user preferences for the CLI itself (separate from the daemon's
//! TOML config, which is managed via `anankectl server-config`). Today
//! the only key is `endpoint`; new keys can be added without changing
//! the public surface.
//!
//! Resolution order for any value backed by config (e.g. `--endpoint`):
//! 1. CLI flag, 2. dedicated environment variable, 3. config file,
//! 4. built-in default.

use std::path::PathBuf;

use crate::client::ApiClientError;

/// Recognised keys, in display order. Add new entries here as the schema
/// grows so `set`/`unset`/`list`/`get` agree on what's valid.
pub const KNOWN_KEYS: &[&str] = &["endpoint"];

/// Parsed view of the client config file. Missing files load as all-`None`.
#[derive(Debug, Default, Clone)]
pub struct ClientConfig {
    pub endpoint: Option<String>,
}

/// Path to the client config file: `$XDG_CONFIG_HOME/anankectl/config.toml`,
/// falling back to `$HOME/.config/anankectl/config.toml`.
pub fn config_path() -> Result<PathBuf, ApiClientError> {
    let base = if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
        PathBuf::from(xdg)
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".config")
    } else {
        return Err(ApiClientError::Usage(
            "cannot determine config dir: neither XDG_CONFIG_HOME nor HOME is set".into(),
        ));
    };
    Ok(base.join("anankectl").join("config.toml"))
}

/// Load the client config from disk. A missing file returns the default
/// (all-`None`) config; a malformed file is surfaced as a usage error so
/// the user sees the problem rather than getting silent fallbacks.
pub fn load() -> Result<ClientConfig, ApiClientError> {
    let path = config_path()?;
    if !path.exists() {
        return Ok(ClientConfig::default());
    }
    let text = std::fs::read_to_string(&path)
        .map_err(|e| ApiClientError::Usage(format!("read {}: {e}", path.display())))?;
    let doc: toml_edit::DocumentMut = text
        .parse()
        .map_err(|e| ApiClientError::Usage(format!("parse {}: {e}", path.display())))?;
    let endpoint = doc
        .get("endpoint")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    Ok(ClientConfig { endpoint })
}

/// Load the underlying TOML document for in-place editing. Used by
/// `anankectl config set`/`unset` so existing comments and formatting
/// survive the round-trip.
pub fn load_doc() -> Result<(PathBuf, toml_edit::DocumentMut), ApiClientError> {
    let path = config_path()?;
    let doc = if path.exists() {
        let text = std::fs::read_to_string(&path)
            .map_err(|e| ApiClientError::Usage(format!("read {}: {e}", path.display())))?;
        text.parse::<toml_edit::DocumentMut>()
            .map_err(|e| ApiClientError::Usage(format!("parse {}: {e}", path.display())))?
    } else {
        toml_edit::DocumentMut::new()
    };
    Ok((path, doc))
}

pub fn save_doc(
    path: &std::path::Path,
    doc: &toml_edit::DocumentMut,
) -> Result<(), ApiClientError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| ApiClientError::Usage(format!("create {}: {e}", parent.display())))?;
    }
    std::fs::write(path, doc.to_string())
        .map_err(|e| ApiClientError::Usage(format!("write {}: {e}", path.display())))?;
    Ok(())
}

pub fn ensure_known_key(key: &str) -> Result<(), ApiClientError> {
    if KNOWN_KEYS.contains(&key) {
        Ok(())
    } else {
        Err(ApiClientError::Usage(format!(
            "unknown config key '{key}' (known: {})",
            KNOWN_KEYS.join(", ")
        )))
    }
}

/// Validate a value before writing it. Keeps obviously-wrong inputs out
/// of the file rather than blowing up on the next CLI invocation.
pub fn validate_value(key: &str, value: &str) -> Result<(), ApiClientError> {
    match key {
        "endpoint" => {
            reqwest::Url::parse(value)
                .map_err(|e| ApiClientError::Usage(format!("invalid endpoint URL: {e}")))?;
            Ok(())
        }
        _ => Err(ApiClientError::Usage(format!("unknown config key '{key}'"))),
    }
}

/// Resolve the management endpoint by walking the precedence chain.
/// Returns the chosen string and never fails for "no value found" — the
/// built-in default is always available.
pub fn resolve_endpoint(cli_value: Option<String>) -> Result<String, ApiClientError> {
    if let Some(e) = cli_value {
        return Ok(e);
    }
    if let Ok(e) = std::env::var("ANANKE_ENDPOINT")
        && !e.is_empty()
    {
        return Ok(e);
    }
    let cfg = load()?;
    if let Some(e) = cfg.endpoint {
        return Ok(e);
    }
    Ok(ananke_api::defaults::MANAGEMENT_ENDPOINT.to_string())
}
