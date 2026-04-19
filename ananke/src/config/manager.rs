//! Owns the raw TOML + parsed `EffectiveConfig` plus all disk I/O for the
//! config file. Replaces the previous `Arc<EffectiveConfig>` held directly
//! by `AppState`.

use std::{io, path::PathBuf, sync::Arc, time::Duration};

use ananke_api::{Event, ValidationError};
use arc_swap::ArcSwap;
use base64::{Engine, engine::general_purpose::STANDARD as B64};
use parking_lot::{Mutex, RwLock};
use sha2::{Digest, Sha256};
use tracing::{info, warn};

use crate::{
    config::{EffectiveConfig, Migration, load_config},
    daemon::events::EventBus,
    errors::ExpectedError,
};

/// Base64-encoded SHA-256 of the raw TOML bytes. Callers treat it as opaque.
pub type ConfigHash = String;

/// Shared owner of config state. Cloned via `Arc<ConfigManager>`.
pub struct ConfigManager {
    raw: RwLock<String>,
    effective: ArcSwap<EffectiveConfig>,
    path: PathBuf,
    events: EventBus,
    _watcher: RwLock<Option<notify::RecommendedWatcher>>,
    boot_migrations: Mutex<Option<Vec<Migration>>>,
}

/// Failure modes from `ConfigManager::apply`.
#[derive(Debug)]
pub enum ApplyError {
    /// The caller's hash does not match the current server-side hash.
    HashMismatch { server_hash: ConfigHash },
    /// The new TOML failed validation.
    Invalid(Vec<ValidationError>),
    /// Writing the file to disk failed.
    PersistFailed(io::Error),
}

impl std::fmt::Display for ApplyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HashMismatch { server_hash } => {
                write!(f, "hash mismatch (server: {server_hash})")
            }
            Self::Invalid(errors) => write!(f, "invalid config ({} errors)", errors.len()),
            Self::PersistFailed(e) => write!(f, "persist failed: {e}"),
        }
    }
}

impl std::error::Error for ApplyError {}

impl ConfigManager {
    /// Load the config from disk, construct the manager, and spawn the
    /// `notify` watcher. The returned `Arc<ConfigManager>` is thread-safe and
    /// inexpensive to clone.
    pub async fn open(path: PathBuf, events: EventBus) -> Result<Arc<Self>, ExpectedError> {
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| ExpectedError::config_unparseable(path.clone(), e.to_string()))?;
        let (effective, migrations) = load_config(&path)?;
        let this = Arc::new(Self {
            raw: RwLock::new(raw),
            effective: ArcSwap::from_pointee(effective),
            path: path.clone(),
            events,
            _watcher: RwLock::new(None),
            boot_migrations: Mutex::new(Some(migrations)),
        });
        this.spawn_watcher();
        Ok(this)
    }

    /// Build a manager from a pre-parsed `EffectiveConfig` without touching
    /// disk or spawning a watcher. Intended for tests + any caller that has
    /// already loaded the config through another path.
    pub fn in_memory(effective: EffectiveConfig, events: EventBus) -> Arc<Self> {
        Arc::new(Self {
            raw: RwLock::new(String::new()),
            effective: ArcSwap::from_pointee(effective),
            path: std::path::PathBuf::from("<in-memory>"),
            events,
            _watcher: RwLock::new(None),
            boot_migrations: Mutex::new(Some(Vec::new())),
        })
    }

    /// Return the raw TOML content and its hash as a pair.
    pub fn raw(&self) -> (String, ConfigHash) {
        let raw = self.raw.read().clone();
        let hash = hash_of(&raw);
        (raw, hash)
    }

    /// Return a guard giving cheap access to the current `EffectiveConfig`.
    ///
    /// The guard derefs to `&EffectiveConfig`, so field access is transparent
    /// while the guard is held.
    pub fn effective(&self) -> arc_swap::Guard<Arc<EffectiveConfig>> {
        self.effective.load()
    }

    /// The path of the config file on disk.
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// Validate the given TOML without touching disk or the in-memory cache.
    pub fn validate(&self, toml: &str) -> Result<(), Vec<ValidationError>> {
        validate_toml(&self.path, toml)
    }

    /// Take the migrations that were produced at boot. Returns them exactly
    /// once; subsequent calls return an empty vec. This lets `daemon::run`
    /// apply the initial migrations without `ConfigManager::open` needing to
    /// return them directly.
    pub fn take_boot_migrations(&self) -> Vec<Migration> {
        self.boot_migrations.lock().take().unwrap_or_default()
    }

    /// Validate the new TOML, hash-check it against `if_match`, persist it
    /// to disk, update the in-memory snapshot, and publish `ConfigReloaded`.
    pub async fn apply(
        self: &Arc<Self>,
        new_toml: String,
        if_match: ConfigHash,
    ) -> Result<(), ApplyError> {
        {
            let (current_raw, current_hash) = self.raw();
            if current_hash != if_match {
                return Err(ApplyError::HashMismatch {
                    server_hash: current_hash,
                });
            }
            if current_raw == new_toml {
                return Ok(());
            }
        }

        validate_toml(&self.path, &new_toml).map_err(ApplyError::Invalid)?;
        persist_atomically(&self.path, &new_toml).map_err(ApplyError::PersistFailed)?;
        self.reload_from_disk();
        Ok(())
    }

    fn reload_from_disk(self: &Arc<Self>) {
        let raw = match std::fs::read_to_string(&self.path) {
            Ok(s) => s,
            Err(e) => {
                warn!(error = %e, "config reload: read failed");
                return;
            }
        };
        let (effective, _migs) = match load_config(&self.path) {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "config reload: validate failed; keeping live config");
                return;
            }
        };
        let changed = diff_services(&self.effective.load(), &effective);
        *self.raw.write() = raw;
        self.effective.store(Arc::new(effective));
        info!(?changed, "config reloaded");
        self.events.publish(Event::ConfigReloaded {
            at_ms: crate::tracking::now_unix_ms(),
            changed_services: changed,
        });
    }

    fn spawn_watcher(self: &Arc<Self>) {
        use notify::{RecursiveMode, Watcher};
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<()>();
        let dir = self.path.parent().map(|p| p.to_path_buf());
        let target = self.path.clone();
        let mut watcher = match notify::recommended_watcher(move |res: Result<notify::Event, _>| {
            if let Ok(ev) = res
                && ev.paths.iter().any(|p| p == &target)
            {
                let _ = tx.send(());
            }
        }) {
            Ok(w) => w,
            Err(e) => {
                warn!(error = %e, "notify watcher init failed");
                return;
            }
        };
        if let Some(d) = &dir
            && let Err(e) = watcher.watch(d, RecursiveMode::NonRecursive)
        {
            warn!(error = %e, path = %d.display(), "notify watch failed");
            return;
        }
        *self._watcher.write() = Some(watcher);

        let me = Arc::clone(self);
        tokio::spawn(async move {
            let mut debounce = tokio::time::interval(Duration::from_millis(500));
            debounce.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            debounce.tick().await;
            let mut pending = false;
            loop {
                tokio::select! {
                    msg = rx.recv() => match msg {
                        Some(()) => { pending = true; }
                        None => return,
                    },
                    _ = debounce.tick(), if pending => {
                        pending = false;
                        me.reload_from_disk();
                    }
                }
            }
        });
    }
}

fn hash_of(s: &str) -> ConfigHash {
    let digest = Sha256::digest(s.as_bytes());
    B64.encode(digest)
}

fn persist_atomically(path: &std::path::Path, content: &str) -> io::Result<()> {
    use std::io::Write;
    let dir = path.parent().unwrap_or(std::path::Path::new("."));
    let tmp = tempfile::Builder::new()
        .prefix(".ananke-config-")
        .suffix(".toml")
        .tempfile_in(dir)?;
    {
        let mut f = tmp.as_file();
        f.write_all(content.as_bytes())?;
        f.sync_all()?;
    }
    tmp.persist(path).map_err(|e| e.error)?;
    Ok(())
}

fn validate_toml(path: &std::path::Path, content: &str) -> Result<(), Vec<ValidationError>> {
    let parent = path.parent().unwrap_or(std::path::Path::new("."));
    let tmp_path = parent.join(".ananke-config-validate.toml");
    if let Err(e) = std::fs::write(&tmp_path, content) {
        return Err(vec![ValidationError {
            line: 0,
            column: 0,
            message: format!("write temp file: {e}"),
        }]);
    }
    let result = load_config(&tmp_path);
    let _ = std::fs::remove_file(&tmp_path);
    match result {
        Ok(_) => Ok(()),
        Err(e) => Err(vec![ValidationError {
            line: 0,
            column: 0,
            message: e.to_string(),
        }]),
    }
}

fn diff_services(old: &EffectiveConfig, new: &EffectiveConfig) -> Vec<smol_str::SmolStr> {
    use std::collections::BTreeSet;
    let old_names: BTreeSet<_> = old.services.iter().map(|s| s.name.clone()).collect();
    let new_names: BTreeSet<_> = new.services.iter().map(|s| s.name.clone()).collect();
    let mut changed = Vec::new();
    for name in &new_names {
        if !old_names.contains(name) {
            changed.push(name.clone());
        }
    }
    for name in &old_names {
        if !new_names.contains(name) {
            changed.push(name.clone());
        }
    }
    changed.sort();
    changed.dedup();
    changed
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    const VALID_TOML: &str = r#"
[daemon]
management_listen = "127.0.0.1:0"

[openai_api]
listen = "127.0.0.1:0"

[[service]]
name = "demo"
template = "llama-cpp"
model = "/tmp/fake.gguf"
port = 11435
devices.placement = "cpu-only"
devices.placement_override = { cpu = 100 }
lifecycle = "on_demand"
"#;

    #[tokio::test]
    async fn apply_rejects_stale_if_match() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("ananke.toml");
        std::fs::write(&path, VALID_TOML).unwrap();
        let manager = ConfigManager::open(path.clone(), EventBus::new())
            .await
            .unwrap();
        let result = manager
            .apply(VALID_TOML.to_string(), "wrong-hash".to_string())
            .await;
        assert!(matches!(result, Err(ApplyError::HashMismatch { .. })));
    }

    #[tokio::test]
    async fn apply_writes_and_reloads_on_valid_input() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("ananke.toml");
        std::fs::write(&path, VALID_TOML).unwrap();
        let manager = ConfigManager::open(path.clone(), EventBus::new())
            .await
            .unwrap();
        let (_current, hash) = manager.raw();
        let new_toml = VALID_TOML.replace("\"demo\"", "\"demo2\"");
        let result = manager.apply(new_toml.clone(), hash).await;
        assert!(matches!(result, Ok(())));
        let (raw_after, _) = manager.raw();
        assert_eq!(raw_after, new_toml);
        let eff = manager.effective();
        assert_eq!(eff.services[0].name.as_str(), "demo2");
    }

    #[tokio::test]
    async fn apply_rejects_invalid_toml() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("ananke.toml");
        std::fs::write(&path, VALID_TOML).unwrap();
        let manager = ConfigManager::open(path.clone(), EventBus::new())
            .await
            .unwrap();
        let (_, hash) = manager.raw();
        let bad = "this is not toml";
        let result = manager.apply(bad.to_string(), hash).await;
        assert!(matches!(result, Err(ApplyError::Invalid(_))));
    }
}
