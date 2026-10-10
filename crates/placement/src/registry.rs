//! Shared lookup from service name to a service handle.
//!
//! Generic over the handle type so the daemon's `ServiceRegistry` can hold
//! `Arc<SupervisorHandle>` while the allocator holds an erased
//! `Arc<dyn KillHandle>` — the erased form is what lets the balloon
//! resolver fast-kill a peer without depending on the supervise crate.

use std::{collections::BTreeMap, sync::Arc};

use parking_lot::RwLock;
use smol_str::SmolStr;

/// The kill-capable surface of a service handle the allocator depends on.
///
/// The balloon resolver needs to fast-kill a peer on an over-committed
/// GPU without reaching into the supervise crate; `SupervisorHandle`
/// implements this trait in supervise.
// Clippy 1.99 double-counts `async_trait`'s boxed futures; older stable Clippy does not emit this lint.
#[allow(clippy::double_must_use)]
#[async_trait::async_trait]
pub trait KillHandle: Send + Sync {
    /// Fast-kill the service (short SIGTERM grace, then SIGKILL).
    async fn fast_kill(&self, reason: DrainReason);
}

/// Why a service is being drained or killed. Shared between the supervise
/// drain pipeline, the allocator's balloon fast-kill path, and the oneshot
/// TTL watchdog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainReason {
    Shutdown,
    IdleTimeout,
    Eviction,
    TtlExpired,
    UserKilled,
    ConfigChanged,
    /// Self-healing restart: the error-rate watchdog or periodic timer
    /// decided a `Running` service should be drained and respawned.
    AutoRestart,
}

/// Convert a `DeviceSlot` to the canonical string key used in
/// `AllocationChanged` reservations (`"cpu"` or `"gpu:N"`).
pub fn slot_to_key(slot: &ananke_config::placement::DeviceSlot) -> String {
    match slot {
        ananke_config::placement::DeviceSlot::Cpu => "cpu".to_string(),
        ananke_config::placement::DeviceSlot::Gpu(n) => format!("gpu:{n}"),
    }
}

pub struct ServiceRegistry<T> {
    inner: Arc<RwLock<BTreeMap<SmolStr, Arc<T>>>>,
}

// The map stores `Arc<T>`, so cloning the registry itself never clones a
// `T` — the derive would otherwise impose a spurious `T: Clone` bound.
impl<T> Clone for ServiceRegistry<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> Default for ServiceRegistry<T> {
    fn default() -> Self {
        Self {
            inner: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }
}

impl<T> ServiceRegistry<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&self, name: SmolStr, handle: Arc<T>) {
        self.inner.write().insert(name, handle);
    }

    pub fn get(&self, name: &str) -> Option<Arc<T>> {
        self.inner.read().get(name).cloned()
    }

    /// Evict `name` and return its handle if present. The caller typically
    /// awaits `shutdown()` on the returned handle to drain the underlying
    /// child before the `Arc` is dropped.
    pub fn remove(&self, name: &str) -> Option<Arc<T>> {
        self.inner.write().remove(name)
    }

    pub fn names(&self) -> Vec<SmolStr> {
        self.inner.read().keys().cloned().collect()
    }

    pub fn all(&self) -> Vec<(SmolStr, Arc<T>)> {
        self.inner
            .read()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_remove_roundtrip() {
        let registry: ServiceRegistry<u64> = ServiceRegistry::new();
        let handle: Arc<u64> = Arc::new(42);
        registry.insert(SmolStr::new("demo"), handle.clone());
        assert!(registry.get("demo").is_some());
        assert!(registry.get("missing").is_none());
        assert_eq!(registry.names(), vec![SmolStr::new("demo")]);
        let taken = registry.remove("demo").expect("registry had demo");
        assert_eq!(*taken, 42);
        assert!(registry.get("demo").is_none());
        assert_eq!(registry.names(), Vec::<SmolStr>::new());
    }

    #[test]
    fn all_returns_entries() {
        let registry: ServiceRegistry<u64> = ServiceRegistry::new();
        registry.insert(SmolStr::new("a"), Arc::new(1));
        registry.insert(SmolStr::new("b"), Arc::new(2));
        let entries = registry.all();
        assert_eq!(entries.len(), 2);
        assert!(entries.iter().any(|(n, h)| n.as_str() == "a" && **h == 1));
        assert!(entries.iter().any(|(n, h)| n.as_str() == "b" && **h == 2));
    }
}
