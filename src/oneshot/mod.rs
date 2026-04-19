//! Oneshot service registry and port pool.
//!
//! A oneshot is a short-lived, on-demand service spawned via `POST /api/oneshot`
//! and automatically torn down when its TTL expires or the caller deletes it.

pub mod handlers;
pub mod port_pool;
pub mod ttl;

use std::{collections::BTreeMap, sync::Arc};

use parking_lot::RwLock;
pub use port_pool::PortPool;
use smol_str::SmolStr;

/// Stable identifier for a oneshot. Auto-generated as a ULID when the
/// caller does not supply a name.
pub type OneshotId = SmolStr;

/// Snapshot of a live oneshot entry.
#[derive(Debug, Clone)]
pub struct OneshotRecord {
    pub id: OneshotId,
    /// The logical service name registered in the `ServiceRegistry`.
    pub service_name: SmolStr,
    /// The private port allocated from the `PortPool`.
    pub port: u16,
    /// TTL in milliseconds as supplied by the caller.
    pub ttl_ms: u64,
    /// Unix epoch milliseconds at which the oneshot was submitted.
    pub started_at_ms: u64,
}

/// Shared in-memory registry of all live oneshot records.
#[derive(Clone, Default)]
pub struct OneshotRegistry {
    inner: Arc<RwLock<BTreeMap<OneshotId, OneshotRecord>>>,
}

impl OneshotRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a new record. Overwrites any existing record with the same id.
    pub fn insert(&self, record: OneshotRecord) {
        self.inner.write().insert(record.id.clone(), record);
    }

    /// Look up a record by id.
    pub fn get(&self, id: &str) -> Option<OneshotRecord> {
        self.inner.read().get(id).cloned()
    }

    /// Remove and return a record. Returns `None` if not found.
    pub fn remove(&self, id: &str) -> Option<OneshotRecord> {
        self.inner.write().remove(id)
    }

    /// Snapshot all live records, sorted by id for deterministic output.
    pub fn list(&self) -> Vec<OneshotRecord> {
        self.inner.read().values().cloned().collect()
    }
}
