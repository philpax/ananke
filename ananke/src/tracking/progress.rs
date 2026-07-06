//! Per-service last-response-progress timestamps for the stall watchdog.
//!
//! The OpenAI proxy stamps the current time here every time it forwards a
//! response frame from a service's child. The time-to-first-token stall
//! watchdog reads it: a request whose per-request timer expires only warrants
//! a restart if the *whole service* has produced no frame within the window.
//! That distinguishes a genuinely wedged child from a request merely queued
//! behind a healthy long-running generation (llama.cpp serialises requests at
//! `--parallel 1`), which is otherwise indistinguishable from a stall.
//!
//! The timestamp is a [`tokio::time::Instant`], deliberately the same clock the
//! stall timer sleeps on, so the "no frame for `timeout`" check and the timer
//! never disagree — including under `tokio`'s paused-time test harness.

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use parking_lot::{Mutex, RwLock};
use smol_str::SmolStr;
use tokio::time::Instant;

#[derive(Clone, Default)]
pub struct ProgressTable {
    inner: Arc<RwLock<BTreeMap<SmolStr, ProgressCell>>>,
}

impl ProgressTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return (or lazily create) the progress cell for the given service. A
    /// freshly created cell is stamped "now", so a service that has never
    /// produced a frame is not treated as having been silent since the epoch.
    pub fn stamp(&self, service: &SmolStr) -> ProgressCell {
        {
            let guard = self.inner.read();
            if let Some(c) = guard.get(service) {
                return c.clone();
            }
        }
        let mut guard = self.inner.write();
        guard
            .entry(service.clone())
            .or_insert_with(ProgressCell::now)
            .clone()
    }
}

/// A shared, cheaply cloneable handle to one service's last-frame timestamp.
#[derive(Clone)]
pub struct ProgressCell(Arc<Mutex<Instant>>);

impl ProgressCell {
    fn now() -> Self {
        Self(Arc::new(Mutex::new(Instant::now())))
    }

    /// Record that a response frame was just forwarded for this service.
    pub fn record(&self) {
        *self.0.lock() = Instant::now();
    }

    /// How long since the service last forwarded a response frame.
    pub fn idle_for(&self) -> Duration {
        self.0.lock().elapsed()
    }
}
