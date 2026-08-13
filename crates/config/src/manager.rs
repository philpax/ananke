//! Owns the raw TOML + parsed `EffectiveConfig` plus all disk I/O for the
//! config file. `AppState` holds this rather than an `EffectiveConfig`
//! directly, so a reload swaps one shared value.

use std::{io, path::PathBuf, sync::Arc, time::Duration};

use ananke_api::events::Event;
use ananke_errors::ExpectedError;
use ananke_events::EventBus;
use arc_swap::ArcSwap;
use base64::{Engine, engine::general_purpose::STANDARD as B64};
use parking_lot::{Mutex, RwLock};
use sha2::{Digest, Sha256};
use tracing::{info, warn};

use crate::{EffectiveConfig, Migration};

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
    fs: Arc<dyn ananke_fs::Fs>,
    placeholder_checker: Arc<dyn crate::validate::PlaceholderChecker>,
}

/// Failure modes from `ConfigManager::apply`.
#[derive(Debug)]
pub enum ApplyError {
    /// The caller's hash does not match the current server-side hash.
    HashMismatch {
        /// The hash the server currently holds, for the caller to diff against.
        server_hash: ConfigHash,
    },
    /// The new TOML failed validation.
    Invalid(crate::validate::ConfigDiagnosticReport),
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

impl std::error::Error for ApplyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::PersistFailed(e) => Some(e),
            Self::HashMismatch { .. } | Self::Invalid(_) => None,
        }
    }
}

impl ConfigManager {
    /// Load the config from disk, construct the manager, and spawn the
    /// `notify` watcher. The returned `Arc<ConfigManager>` is thread-safe and
    /// inexpensive to clone. Uses [`ananke_fs::LocalFs`] for all
    /// filesystem I/O — tests with synthetic configs should use [`Self::open_with_fs`].
    pub async fn open(path: PathBuf, events: EventBus) -> Result<Arc<Self>, ExpectedError> {
        Self::open_with_fs(path, events, Arc::new(ananke_fs::LocalFs)).await
    }

    /// Variant of [`Self::open`] with a daemon-owned placeholder checker.
    pub async fn open_with_checker(
        path: PathBuf,
        events: EventBus,
        placeholder_checker: Arc<dyn crate::validate::PlaceholderChecker>,
    ) -> Result<Arc<Self>, ExpectedError> {
        Self::open_with_fs_and_checker(
            path,
            events,
            Arc::new(ananke_fs::LocalFs),
            placeholder_checker,
        )
        .await
    }

    /// Variant of [`Self::open`] that uses an explicit filesystem. Production
    /// passes `LocalFs`; tests can pass an `InMemoryFs`.
    pub async fn open_with_fs(
        path: PathBuf,
        events: EventBus,
        fs: Arc<dyn ananke_fs::Fs>,
    ) -> Result<Arc<Self>, ExpectedError> {
        Self::open_with_fs_and_checker(
            path,
            events,
            fs,
            Arc::new(crate::validate::NoopPlaceholderChecker),
        )
        .await
    }

    /// Variant of [`Self::open_with_fs`] with a daemon-owned placeholder checker.
    pub async fn open_with_fs_and_checker(
        path: PathBuf,
        events: EventBus,
        fs: Arc<dyn ananke_fs::Fs>,
        placeholder_checker: Arc<dyn crate::validate::PlaceholderChecker>,
    ) -> Result<Arc<Self>, ExpectedError> {
        let raw = fs
            .read_to_string(&path)
            .map_err(|e| ExpectedError::config_unparseable(path.clone(), e.to_string()))?;
        let (effective, migrations) =
            crate::load_config_from_str_with_checks(&raw, placeholder_checker.as_ref())
                .map_err(|error| error.into_expected_error(path.clone()))?;
        crate::preflight_ggufs(&path, &effective, fs.as_ref())?;
        let this = Arc::new(Self {
            raw: RwLock::new(raw),
            effective: ArcSwap::from_pointee(effective),
            path: path.clone(),
            events,
            _watcher: RwLock::new(None),
            boot_migrations: Mutex::new(Some(migrations)),
            fs,
            placeholder_checker,
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
            fs: Arc::new(ananke_fs::InMemoryFs::new()),
            placeholder_checker: Arc::new(crate::validate::NoopPlaceholderChecker),
        })
    }

    /// Swap the in-memory effective config without persisting or publishing.
    /// Tests use this to stage the "post-reload" state before firing the
    /// `ConfigReloaded` event manually. Never used by production code.
    ///
    /// Lib-side tests use this directly; integration tests (compiled as
    /// their own binaries) reach it through the `test-fakes` feature gate.
    #[cfg(any(test, feature = "test-fakes"))]
    pub fn swap_effective_for_test(&self, new: EffectiveConfig) {
        self.effective.store(Arc::new(new));
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

    /// Whether the config file can be written to. Used by the API to
    /// tell the frontend whether to start the editor in read-only mode.
    pub fn writable(&self) -> bool {
        self.fs.writable(&self.path)
    }

    /// Validate the given TOML without touching disk or the in-memory cache.
    pub fn validate(&self, toml: &str) -> Result<(), crate::validate::ConfigDiagnosticReport> {
        crate::load_config_from_str_with_checks(toml, self.placeholder_checker.as_ref())
            .map(|_| ())
            .map_err(|error| error.into_report())
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

        let (effective, _migrations) =
            crate::load_config_from_str_with_checks(&new_toml, self.placeholder_checker.as_ref())
                .map_err(|error| ApplyError::Invalid(error.into_report()))?;
        persist_atomically(self.fs.as_ref(), &self.path, &new_toml)
            .map_err(ApplyError::PersistFailed)?;
        let changed = diff_services(&self.effective.load(), &effective);
        *self.raw.write() = new_toml;
        self.effective.store(Arc::new(effective));
        self.events.publish(Event::ConfigReloaded {
            at_ms: ananke_time::now_unix_ms(),
            changed_services: changed,
        });
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
        let (effective, _migs) = match crate::load_config_from_str_with_checks(
            &raw,
            self.placeholder_checker.as_ref(),
        ) {
            Ok((effective, migrations)) => {
                if let Err(e) = crate::preflight_ggufs(&self.path, &effective, self.fs.as_ref()) {
                    warn!(error = %e, "config reload: preflight failed; keeping live config");
                    return;
                }
                (effective, migrations)
            }
            Err(report) => {
                warn!(error = %report, "config reload: validate failed; keeping live config");
                return;
            }
        };
        let changed = diff_services(&self.effective.load(), &effective);
        *self.raw.write() = raw;
        self.effective.store(Arc::new(effective));
        info!(?changed, "config reloaded");
        self.events.publish(Event::ConfigReloaded {
            at_ms: ananke_time::now_unix_ms(),
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
    fs: &dyn ananke_fs::Fs,
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

    use ananke_fs::{Fs, InMemoryFs};
    use ananke_gguf::keys;

    use super::*;

    /// Minimal but structurally valid GGUF v3 bytes so the config preflight
    /// (which calls `gguf::read`) accepts the referenced path.
    fn synth_gguf_bytes() -> Vec<u8> {
        let mut bytes = Vec::<u8>::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes()); // version
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&1u64.to_le_bytes()); // kv_count
        let arch_key = keys::ARCHITECTURE;
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
