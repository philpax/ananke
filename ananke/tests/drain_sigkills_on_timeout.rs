//! If SIGTERM is ignored, the drain pipeline must escalate to SIGKILL
//! after the configured grace period. Uses [`FakeSpawner::ignoring_sigterm`]
//! so the escalation can be exercised without real signal plumbing.
#![cfg(feature = "test-fakes")]

use std::{sync::Arc, time::Duration};

use ananke::supervise::{
    SpawnConfig,
    drain::{DrainConfig, DrainReason, drain_pipeline},
    workload::ManagedWorkload,
};
use ananke_proxy::InflightCounter;
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
async fn sigkill_after_sigterm_grace() {
    let spawner = Arc::new(FakeSpawner::ignoring_sigterm());
    let child = spawner.spawn(&spawn_cfg()).await.unwrap();
    let mut workload = ManagedWorkload::Process(child);

    let counter = InflightCounter::detached();
    let cfg = DrainConfig {
        max_request_duration: Duration::from_millis(100),
        drain_timeout: Duration::from_millis(50),
        extended_stream_drain: Duration::from_millis(50),
        sigterm_grace: Duration::from_millis(500),
    };

    drain_pipeline(&mut workload, &cfg, counter, DrainReason::Shutdown).await;

    let snap = spawner.children();
    assert_eq!(snap.len(), 1);
    // SIGTERM was ignored, so the pipeline escalated.
    assert_eq!(snap[0].state, FakeProcessState::SigKill);
}

/// A child needing twelve seconds to persist its state is killed by the
/// unchanged default, but exits normally with its service's twenty-second
/// grace. Paused Tokio time avoids a wall-clock wait in this regression.
#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn service_stop_grace_allows_slow_checkpoint_exit() {
    for (setting, exits_cleanly) in [("", false), ("stop_grace = \"20s\"", true)] {
        let source = format!(
            r#"
[[service]]
name = "checkpoint"
template = "command"
command = ["fake"]
port = 8001
allocation = {{ mode = "static", reserve_gb = 1 }}
{setting}
"#
        );
        let effective =
            ananke_config::load_config_from_str(&source, std::path::Path::new("/config.toml"))
                .unwrap()
                .0;
        let svc = &effective.services[0];
        let spawner = Arc::new(FakeSpawner::ignoring_sigterm());
        let child = spawner.spawn(&spawn_cfg()).await.unwrap();
        let pid = spawner.children()[0].pid;
        let mut workload = ManagedWorkload::Process(child);
        let cfg = DrainConfig {
            max_request_duration: Duration::from_millis(svc.max_request_duration_ms),
            drain_timeout: Duration::from_millis(svc.drain_timeout_ms),
            extended_stream_drain: Duration::from_millis(svc.extended_stream_drain_ms),
            sigterm_grace: Duration::from_millis(svc.stop_grace_ms),
        };
        let complete_checkpoint = async {
            tokio::time::sleep(Duration::from_secs(12)).await;
            spawner.exit(pid, 0)
        };
        let (_, did_exit) = tokio::join!(
            drain_pipeline(
                &mut workload,
                &cfg,
                InflightCounter::detached(),
                DrainReason::Eviction
            ),
            complete_checkpoint,
        );
        assert_eq!(did_exit, exits_cleanly);
        assert_eq!(
            spawner.children()[0].state,
            if exits_cleanly {
                FakeProcessState::SelfExited { code: 0 }
            } else {
                FakeProcessState::SigKill
            }
        );
    }
}
