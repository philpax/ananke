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

/// Atomically-replaced snapshot of the current device state, shared with
/// the allocator, the management API, and the device-sample db writer.
/// Readers never block the sampler, and every write notifies subscribers
/// once its guard drops.
#[derive(Clone)]
pub struct SharedSnapshot {
    value: Arc<RwLock<DeviceSnapshot>>,
    changed: watch::Sender<()>,
}

impl SharedSnapshot {
    pub fn read(&self) -> RwLockReadGuard<'_, DeviceSnapshot> {
        self.value.read()
    }

    /// Lock for replacement; dropping the guard notifies subscribers.
    pub fn write(&self) -> SnapshotWriteGuard<'_> {
        SnapshotWriteGuard {
            guard: self.value.write(),
            changed: &self.changed,
        }
    }

    /// Subscribe before reading, so a write landing between the two still
    /// wakes the receiver.
    pub fn subscribe(&self) -> watch::Receiver<()> {
        self.changed.subscribe()
    }
}

/// A write guard that notifies subscribers once the write finishes.
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
        // `send_replace` stores the value even with no receivers, so a
        // subscriber created later still starts from a seen version.
        self.changed.send_replace(());
    }
}

/// Build a fresh shared snapshot initialized to default (all-empty) state.
pub fn new_shared() -> SharedSnapshot {
    SharedSnapshot {
        value: Arc::new(RwLock::new(DeviceSnapshot::default())),
        changed: watch::channel(()).0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn write_notifies_subscribers_after_the_guard_drops() {
        let snapshot = new_shared();
        let mut rx = snapshot.subscribe();
        assert!(!rx.has_changed().unwrap());

        let mut guard = snapshot.write();
        guard.taken_at_ms = 7;
        assert!(!rx.has_changed().unwrap());
        drop(guard);

        rx.changed().await.unwrap();
        assert_eq!(snapshot.read().taken_at_ms, 7);
    }

    #[test]
    fn reads_do_not_notify() {
        let snapshot = new_shared();
        let rx = snapshot.subscribe();
        let _ = snapshot.read().taken_at_ms;
        assert!(!rx.has_changed().unwrap());
    }
}
