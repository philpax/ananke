//! If SIGTERM is ignored, the drain pipeline must escalate to SIGKILL
//! after the configured grace period. Uses [`FakeSpawner::ignoring_sigterm`]
//! so the escalation can be exercised without real signal plumbing.

use std::{
    sync::{Arc, atomic::AtomicU64},
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
async fn sigkill_after_sigterm_grace() {
    let spawner = Arc::new(FakeSpawner::ignoring_sigterm());
    let mut child = spawner.spawn(&spawn_cfg()).await.unwrap();

    let counter = Arc::new(AtomicU64::new(0));
    let cfg = DrainConfig {
        max_request_duration: Duration::from_millis(100),
        drain_timeout: Duration::from_millis(50),
        extended_stream_drain: Duration::from_millis(50),
        sigterm_grace: Duration::from_millis(500),
    };

    drain_pipeline(&mut *child, &cfg, counter, DrainReason::Shutdown).await;

    let snap = spawner.children();
    assert_eq!(snap.len(), 1);
    // SIGTERM was ignored, so the pipeline escalated.
    assert_eq!(snap[0].state, FakeProcessState::SigKill);
}
