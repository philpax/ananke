//! Client-side config CLI: `anankectl config {get,set,unset,list,path,edit}`.
//!
//! Reads/writes `~/.config/anankectl/config.toml` via the `crate::config`
//! module. Daemon-side configuration lives under `anankectl server-config`.

use crate::{
    client::ApiClientError,
    config::{KNOWN_KEYS, config_path, ensure_known_key, load, load_doc, save_doc, validate_value},
    output,
};

pub async fn get(json: bool, key: &str) -> Result<(), ApiClientError> {
    ensure_known_key(key)?;
    let cfg = load()?;
    let value = match key {
        "endpoint" => cfg.endpoint,
        _ => unreachable!("ensure_known_key gates this match"),
    };
    if json {
        output::print_json(&serde_json::json!({ "key": key, "value": value }));
    } else if let Some(v) = value {
        println!("{v}");
    }
    Ok(())
}

pub async fn set(json: bool, key: &str, value: &str) -> Result<(), ApiClientError> {
    ensure_known_key(key)?;
    validate_value(key, value)?;
    let (path, mut doc) = load_doc()?;
    doc[key] = toml_edit::value(value);
    save_doc(&path, &doc)?;
    if json {
        output::print_json(&serde_json::json!({ "key": key, "value": value, "path": path }));
    } else {
        println!("ok: set {key} = {value}");
    }
    Ok(())
}

pub async fn unset(json: bool, key: &str) -> Result<(), ApiClientError> {
    ensure_known_key(key)?;
    let (path, mut doc) = load_doc()?;
    let removed = doc.remove(key).is_some();
    save_doc(&path, &doc)?;
    if json {
        output::print_json(&serde_json::json!({ "key": key, "removed": removed }));
    } else if removed {
        println!("ok: unset {key}");
    } else {
        println!("ok: {key} was not set");
    }
    Ok(())
}

pub async fn list(json: bool) -> Result<(), ApiClientError> {
    let cfg = load()?;
    if json {
        output::print_json(&serde_json::json!({
            "endpoint": cfg.endpoint,
        }));
        return Ok(());
    }
    for key in KNOWN_KEYS {
        let value = match *key {
            "endpoint" => cfg.endpoint.clone(),
            _ => None,
        };
        match value {
            Some(v) => println!("{key} = {v}"),
            None => println!("{key} = (unset)"),
        }
    }
    Ok(())
}

pub async fn path(json: bool) -> Result<(), ApiClientError> {
    let p = config_path()?;
    if json {
        output::print_json(&serde_json::json!({ "path": p }));
    } else {
        println!("{}", p.display());
    }
    Ok(())
}

pub async fn edit() -> Result<(), ApiClientError> {
    let path = config_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| ApiClientError::Usage(format!("create {}: {e}", parent.display())))?;
    }
    let editor = std::env::var("EDITOR")
        .or_else(|_| std::env::var("VISUAL"))
        .unwrap_or_else(|_| "vi".to_string());
    let status = std::process::Command::new(&editor)
        .arg(&path)
        .status()
        .map_err(|e| ApiClientError::Usage(format!("spawn editor '{editor}': {e}")))?;
    if !status.success() {
        return Err(ApiClientError::Usage(format!(
            "editor '{editor}' exited with {status}"
        )));
    }
    Ok(())
}
