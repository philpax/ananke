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
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};

pub use ananke_placement::devices::DeviceSnapshot;
pub use observation::{ObservationTable, read_rss};
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use tokio::sync::{Notify, watch};

/// Atomically-replaced snapshot of the current device state, shared with
/// the allocator, the management API, and the device-sample db writer.
/// Readers never block the sampler, and every write notifies subscribers
/// once its guard drops.
#[derive(Clone)]
pub struct SharedSnapshot {
    value: Arc<RwLock<DeviceSnapshot>>,
    changed: watch::Sender<()>,
    refresh: Arc<Refresh>,
}

/// On-demand sampling. The sampler runs on a fixed cadence, so a reader
/// that has just freed memory (a child it waited for has exited) would
/// otherwise see the dead child's memory as still in use until the next
/// tick. Requests are numbered: the sampler notes the newest request before
/// it samples and publishes that number once the sample is written, so a
/// sample already under way when a request arrived never answers it.
struct Refresh {
    attached: AtomicBool,
    requested: AtomicU64,
    wake: Notify,
    served: watch::Sender<u64>,
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

    /// Ask the sampler for a sample started after this call, and wait up to
    /// `timeout` for it to be written. Returns at once, `false`, when no
    /// sampler is attached (tests that write the snapshot by hand), and
    /// `false` on timeout; the current snapshot then stands.
    pub async fn refresh(&self, timeout: Duration) -> bool {
        let r = &self.refresh;
        if !r.attached.load(Ordering::Acquire) {
            return false;
        }
        let mut served = r.served.subscribe();
        let ticket = r.requested.fetch_add(1, Ordering::AcqRel) + 1;
        r.wake.notify_one();
        tokio::time::timeout(timeout, served.wait_for(|s| *s >= ticket))
            .await
            .is_ok_and(|r| r.is_ok())
    }

    /// Register the one task that samples devices, so [`Self::refresh`]
    /// waits for it. Dropping the handle detaches it again.
    pub fn attach_sampler(&self) -> SamplerHandle {
        self.refresh.attached.store(true, Ordering::Release);
        SamplerHandle {
            refresh: self.refresh.clone(),
        }
    }
}

/// The sampler's side of [`SharedSnapshot::refresh`].
pub struct SamplerHandle {
    refresh: Arc<Refresh>,
}

/// The newest refresh request a sample covers, from [`SamplerHandle::begin`].
#[must_use]
pub struct SampleTicket(u64);

impl SamplerHandle {
    /// Resolves when a refresh has been requested since the last sample.
    pub async fn requested(&self) {
        self.refresh.wake.notified().await;
    }

    /// Call just before sampling.
    pub fn begin(&self) -> SampleTicket {
        SampleTicket(self.refresh.requested.load(Ordering::Acquire))
    }

    /// Call once the sample taken after [`Self::begin`] has been written.
    pub fn served(&self, ticket: SampleTicket) {
        self.refresh.served.send_if_modified(|s| {
            let newer = ticket.0 > *s;
            if newer {
                *s = ticket.0;
            }
            newer
        });
    }
}

impl Drop for SamplerHandle {
    fn drop(&mut self) {
        self.refresh.attached.store(false, Ordering::Release);
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
        refresh: Arc::new(Refresh {
            attached: AtomicBool::new(false),
            requested: AtomicU64::new(0),
            wake: Notify::new(),
            served: watch::channel(0).0,
        }),
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

    #[tokio::test]
    async fn refresh_without_a_sampler_returns_at_once() {
        let snapshot = new_shared();
        assert!(!snapshot.refresh(Duration::from_secs(60)).await);
    }

    /// A refresh is answered only by a sample that began after it was
    /// requested, never by one already under way.
    #[tokio::test]
    async fn refresh_waits_for_a_sample_begun_after_the_request() {
        let snapshot = new_shared();
        let sampler = snapshot.attach_sampler();
        // A sample under way before the request.
        let early = sampler.begin();

        let requester = snapshot.clone();
        let wait = tokio::spawn(async move { requester.refresh(Duration::from_secs(5)).await });
        sampler.requested().await;
        sampler.served(early);
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !wait.is_finished(),
            "an earlier sample answered the refresh"
        );

        let fresh = sampler.begin();
        snapshot.write().taken_at_ms = 9;
        sampler.served(fresh);
        assert!(wait.await.unwrap());
        assert_eq!(snapshot.read().taken_at_ms, 9);
    }

    #[tokio::test]
    async fn refresh_times_out_when_the_sampler_does_not_answer() {
        let snapshot = new_shared();
        let _sampler = snapshot.attach_sampler();
        assert!(!snapshot.refresh(Duration::from_millis(20)).await);
    }
}
