//! The drain pipeline must wait for `inflight` to reach zero before it
//! issues SIGTERM. Uses [`FakeSpawner`] so the test is deterministic (no
//! real process, no real signals, virtual tokio time only).

use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use ananke::{
    supervise::{
        SpawnConfig,
        drain::{DrainConfig, DrainReason, drain_pipeline},
    },
    system::{FakeProcessState, FakeSpawner, ProcessSpawner},
};

fn spawn_cfg() -> SpawnConfig {
    SpawnConfig {
        binary: "fake".into(),
        args: Vec::new(),
        env: Default::default(),
    }
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn waits_for_inflight_zero() {
    let spawner = Arc::new(FakeSpawner::new());
    let mut child = spawner.spawn(&spawn_cfg()).await.unwrap();

    let counter = Arc::new(AtomicU64::new(2));
    let cfg = DrainConfig {
        max_request_duration: Duration::from_secs(1),
        drain_timeout: Duration::from_millis(0),
        extended_stream_drain: Duration::from_millis(0),
        sigterm_grace: Duration::from_millis(100),
    };

    let c = counter.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        c.store(0, Ordering::Relaxed);
    });

    drain_pipeline(&mut *child, &cfg, counter.clone(), DrainReason::Eviction).await;

    let snap = spawner.children();
    assert_eq!(snap.len(), 1);
    // The graceful path ran: SIGTERM marked the child terminated, not SIGKILL.
    assert_eq!(snap[0].state, FakeProcessState::SigTerm);
    assert_eq!(
        counter.load(Ordering::Relaxed),
        0,
        "drain must have observed inflight drop to zero"
    );
}
