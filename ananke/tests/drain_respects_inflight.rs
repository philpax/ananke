//! The drain pipeline must wait for `inflight` to reach zero before it
//! issues SIGTERM. Uses [`FakeSpawner`] so the test is deterministic (no
//! real process, no real signals, virtual tokio time only).
#![cfg(feature = "test-fakes")]

use std::{sync::Arc, time::Duration};

use ananke::supervise::{
    SpawnConfig,
    drain::{DrainConfig, DrainReason, drain_pipeline},
    workload::ManagedWorkload,
};
use ananke_proxy::{InflightCounter, InflightGuard};
use ananke_system::{FakeProcessState, FakeSpawner, ProcessSpawner};

fn spawn_cfg() -> SpawnConfig {
    SpawnConfig {
        binary: "fake".into(),
        args: Vec::new(),
        env: Default::default(),
        env_inherit: true,
    }
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn waits_for_inflight_zero() {
    let spawner = Arc::new(FakeSpawner::new());
    let child = spawner.spawn(&spawn_cfg()).await.unwrap();
    let mut workload = ManagedWorkload::Process(child);

    let counter = InflightCounter::detached();
    let guards = [
        InflightGuard::new(counter.clone()),
        InflightGuard::new(counter.clone()),
    ];
    let cfg = DrainConfig {
        max_request_duration: Duration::from_secs(1),
        drain_timeout: Duration::from_millis(0),
        extended_stream_drain: Duration::from_millis(0),
        sigterm_grace: Duration::from_millis(100),
    };

    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        drop(guards);
    });

    drain_pipeline(&mut workload, &cfg, counter.clone(), DrainReason::Eviction).await;

    let snap = spawner.children();
    assert_eq!(snap.len(), 1);
    // The graceful path ran: SIGTERM marked the child terminated, not SIGKILL.
    assert_eq!(snap[0].state, FakeProcessState::SigTerm);
    assert_eq!(
        counter.current(),
        0,
        "drain must have observed inflight drop to zero"
    );
}
