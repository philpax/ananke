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
    config::{EffectiveConfig, Migration, load_config_with_fs},
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
    fs: Arc<dyn crate::system::Fs>,
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
    /// inexpensive to clone. Uses [`crate::system::LocalFs`] for all
    /// filesystem I/O — tests with synthetic configs should use [`Self::open_with_fs`].
    pub async fn open(path: PathBuf, events: EventBus) -> Result<Arc<Self>, ExpectedError> {
        Self::open_with_fs(path, events, Arc::new(crate::system::LocalFs)).await
    }

    /// Variant of [`Self::open`] that uses an explicit filesystem. Production
    /// passes `LocalFs`; tests can pass an `InMemoryFs`.
    pub async fn open_with_fs(
        path: PathBuf,
        events: EventBus,
        fs: Arc<dyn crate::system::Fs>,
    ) -> Result<Arc<Self>, ExpectedError> {
        let raw = fs
            .read_to_string(&path)
            .map_err(|e| ExpectedError::config_unparseable(path.clone(), e.to_string()))?;
        let (effective, migrations) = load_config_with_fs(&path, fs.as_ref(), &raw)?;
        let this = Arc::new(Self {
            raw: RwLock::new(raw),
            effective: ArcSwap::from_pointee(effective),
            path: path.clone(),
            events,
            _watcher: RwLock::new(None),
            boot_migrations: Mutex::new(Some(migrations)),
            fs,
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
            fs: Arc::new(crate::system::InMemoryFs::new()),
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
        validate_toml(self.fs.as_ref(), &self.path, toml)
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

        validate_toml(self.fs.as_ref(), &self.path, &new_toml).map_err(ApplyError::Invalid)?;
        persist_atomically(self.fs.as_ref(), &self.path, &new_toml)
            .map_err(ApplyError::PersistFailed)?;
        self.reload_from_disk();
        Ok(())
    }

    fn reload_from_disk(self: &Arc<Self>) {
        let raw = match self.fs.read_to_string(&self.path) {
            Ok(s) => s,
            Err(e) => {
                warn!(error = %e, "config reload: read failed");
                return;
            }
        };
        {
            let current = self.raw.read();
            if *current == raw {
                // File content matches the in-memory buffer (typical after a PUT-triggered notify fire).
                return;
            }
        }
        let (effective, _migs) = match load_config_with_fs(&self.path, self.fs.as_ref(), &raw) {
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

fn persist_atomically(
    fs: &dyn crate::system::Fs,
    path: &std::path::Path,
    content: &str,
) -> io::Result<()> {
    // Write sibling tempfile then atomic rename onto `path`. On POSIX,
    // `rename` within a single filesystem is atomic, so a partial write
    // can never be observed at `path`.
    let parent = path.parent().unwrap_or(std::path::Path::new("."));
    let basename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("config.toml");
    let tmp = parent.join(format!(".{basename}.{}.tmp", std::process::id()));
    fs.write(&tmp, content.as_bytes())?;
    match fs.rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = fs.remove_file(&tmp);
            Err(e)
        }
    }
}

fn validate_toml(
    fs: &dyn crate::system::Fs,
    path: &std::path::Path,
    content: &str,
) -> Result<(), Vec<ValidationError>> {
    // Parse + validate + preflight against the current filesystem without
    // touching `path` itself. `load_config_with_fs` takes the raw TOML
    // directly so we don't need a sibling tempfile.
    match load_config_with_fs(path, fs, content) {
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
    use std::path::PathBuf;

    use super::*;
    use crate::system::{Fs, InMemoryFs};

    /// Minimal but structurally valid GGUF v3 bytes so the config preflight
    /// (which calls `gguf::read`) accepts the referenced path.
    fn synth_gguf_bytes() -> Vec<u8> {
        let mut bytes = Vec::<u8>::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes()); // version
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&1u64.to_le_bytes()); // kv_count
        let arch_key = "general.architecture";
        bytes.extend_from_slice(&(arch_key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(arch_key.as_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes()); // string tag
        let arch_val = "qwen3";
        bytes.extend_from_slice(&(arch_val.len() as u64).to_le_bytes());
        bytes.extend_from_slice(arch_val.as_bytes());
        bytes
    }

    fn fixture() -> (Arc<dyn Fs>, String, PathBuf) {
        let fs = InMemoryFs::new();
        let gguf_path = PathBuf::from("/cfg/demo.gguf");
        fs.insert(&gguf_path, synth_gguf_bytes());
        let path = PathBuf::from("/cfg/ananke.toml");
        let toml = format!(
            r#"
[daemon]
management_listen = "127.0.0.1:0"

[openai_api]
listen = "127.0.0.1:0"

[[service]]
name = "demo"
template = "llama-cpp"
model = "{model}"
port = 11435
devices.placement = "cpu-only"
devices.placement_override = {{ cpu = 100 }}
lifecycle = "on_demand"
"#,
            model = gguf_path.display()
        );
        fs.write(&path, toml.as_bytes()).unwrap();
        (Arc::new(fs), toml, path)
    }

    #[tokio::test]
    async fn apply_rejects_stale_if_match() {
        let (fs, toml, path) = fixture();
        let manager = ConfigManager::open_with_fs(path, EventBus::new(), fs)
            .await
            .unwrap();
        let result = manager.apply(toml, "wrong-hash".to_string()).await;
        assert!(matches!(result, Err(ApplyError::HashMismatch { .. })));
    }

    #[tokio::test]
    async fn apply_writes_and_reloads_on_valid_input() {
        let (fs, toml, path) = fixture();
        let manager = ConfigManager::open_with_fs(path, EventBus::new(), fs)
            .await
            .unwrap();
        let (_current, hash) = manager.raw();
        let new_toml = toml.replace("\"demo\"", "\"demo2\"");
        let result = manager.apply(new_toml.clone(), hash).await;
        assert!(matches!(result, Ok(())));
        let (raw_after, _) = manager.raw();
        assert_eq!(raw_after, new_toml);
        let eff = manager.effective();
        assert_eq!(eff.services[0].name.as_str(), "demo2");
    }

    #[tokio::test]
    async fn apply_rejects_invalid_toml() {
        let (fs, _toml, path) = fixture();
        let manager = ConfigManager::open_with_fs(path, EventBus::new(), fs)
            .await
            .unwrap();
        let (_, hash) = manager.raw();
        let bad = "this is not toml";
        let result = manager.apply(bad.to_string(), hash).await;
        assert!(matches!(result, Err(ApplyError::Invalid(_))));
    }
}
