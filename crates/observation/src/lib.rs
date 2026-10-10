//! Per-service observed memory and the shared device snapshot.
//!
//! The snapshotter (`ananke-devices`) samples NVML + `/proc` each tick,
//! writes the current [`SharedSnapshot`] atomically, and feeds per-service
//! observations into [`ObservationTable`]. The allocator and tracking
//! read both back. The types live here (rather than in `devices` or
//! `tracking`) so both can depend on this crate without reaching into
//! each other.

mod observation;

use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

pub use ananke_placement::devices::DeviceSnapshot;
pub use observation::{ObservationTable, read_rss};
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use tokio::sync::watch;

/// Atomically replaced device measurements with producer-driven change notifications.
#[derive(Clone)]
pub struct SharedSnapshot {
    value: Arc<RwLock<DeviceSnapshot>>,
    changed: watch::Sender<()>,
}

impl SharedSnapshot {
    /// Borrow the current measurements.
    pub fn read(&self) -> RwLockReadGuard<'_, DeviceSnapshot> {
        self.value.read()
    }

    /// Replace measurements; dropping the guard notifies subscribers.
    pub fn write(&self) -> SnapshotWriteGuard<'_> {
        SnapshotWriteGuard {
            guard: self.value.write(),
            changed: &self.changed,
        }
    }

    /// Subscribe before reading to avoid losing changes during backfill.
    pub fn subscribe(&self) -> watch::Receiver<()> {
        self.changed.subscribe()
    }
}

/// A write guard that publishes once the producer finishes a measurement.
pub struct SnapshotWriteGuard<'a> {
    guard: RwLockWriteGuard<'a, DeviceSnapshot>,
    changed: &'a watch::Sender<()>,
}

impl Deref for SnapshotWriteGuard<'_> {
    type Target = DeviceSnapshot;
    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl DerefMut for SnapshotWriteGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}

impl Drop for SnapshotWriteGuard<'_> {
    fn drop(&mut self) {
        self.changed.send_replace(());
    }
}

/// Build an empty device snapshot with a change feed.
pub fn new_shared() -> SharedSnapshot {
    SharedSnapshot {
        value: Arc::new(RwLock::new(DeviceSnapshot::default())),
        changed: watch::channel(()).0,
    }
}
