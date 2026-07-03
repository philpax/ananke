//! Integration test: verify that the daemon's environment is inherited by
//! spawned child processes when `env_inherit = true` (the default), and
//! excluded when `env_inherit = false`.
//!
//! This exercises the full pipeline:
//! config → validate → render_argv → SpawnConfig → resolve_env() →
//! FakeSpawner → FakeChildSnapshot.
//!
//! Uses the `test-fakes` harness so no real process is spawned. The
//! `FakeSpawner` records the *resolved* env on each `FakeChildSnapshot`,
//! so we can assert that inherited vars like `$PATH` actually appear in
//! the child's env without launching a real process.
#![cfg(feature = "test-fakes")]

mod common;

use std::collections::BTreeMap;

use ananke::{
    config::{
        AllocationMode, CommandConfig, DeviceReserves, DeviceSlot, Filters, HealthSettings,
        Lifecycle, PlacementPolicy, ServiceConfig, SplitMode, TemplateConfig,
        parse::DEFAULT_START_QUEUE_DEPTH,
    },
    supervise::EnsureSource,
};
use common::build_harness;
use smol_str::SmolStr;

/// Build a CPU-only command-template service with the given `env_inherit`
/// setting and an `env` override for `MY_KEY`.
fn env_test_service(name: &str, port: u16, env_inherit: bool) -> ServiceConfig {
    let mut placement = BTreeMap::new();
    placement.insert(DeviceSlot::Cpu, 100);
    let mut env = BTreeMap::new();
    env.insert("MY_KEY".into(), "overridden".into());
    ServiceConfig {
        name: SmolStr::new(name),
        port,
        private_port: 0,
        lifecycle: Lifecycle::OnDemand,
        priority: 50,
        health: HealthSettings {
            http_path: Some("/health".into()),
            timeout_ms: 5_000,
            probe_interval_ms: 200,
        },
        placement_override: placement,
        placement_policy: PlacementPolicy::CpuOnly,
        gpu_allow: Vec::new(),
        split_mode: SplitMode::Layer,
        gpu_headroom_mb: 0,
        reserves: std::sync::Arc::new(DeviceReserves::default()),
        idle_timeout_ms: 60_000,
        drain_timeout_ms: 1_000,
        extended_stream_drain_ms: 1_000,
        max_request_duration_ms: 5_000,
        filters: Filters::default(),
        allocation_mode: AllocationMode::None,
        openai_compat: false,
        description: None,
        modality: ananke_api::shared::Modality::Chat,
        start_queue_depth: DEFAULT_START_QUEUE_DEPTH,
        extra_args: Vec::new(),
        env,
        env_inherit,
        tracking: ananke::config::TrackingSettings::default(),
        metadata: ananke_api::shared::AnankeMetadata::new(),
        template_config: TemplateConfig::Command(CommandConfig {
            command: vec!["echo".into(), "hello".into()],
            workdir: None,
            shutdown_command: None,
            private_port_override: None,
            openai_proxy: None,
        }),
    }
}

/// The inherited env we inject into the FakeSpawner after harness
/// construction.
fn fake_daemon_env() -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    env.insert("PATH".into(), "/usr/bin:/bin".into());
    env.insert("HOME".into(), "/home/test".into());
    env.insert("LANG".into(), "C.UTF-8".into());
    env
}

#[tokio::test(flavor = "multi_thread")]
async fn env_inherit_true_propagates_daemon_env() {
    let svc = env_test_service("inherit-true", common::free_port(), true);
    let h = build_harness(vec![svc]).await;

    // Inject a controlled inherited env *after* build_harness returns but
    // before ensure(). Safe because services are OnDemand — they don't
    // spawn until ensure() is called.
    h.process_spawner.set_inherited_env(fake_daemon_env());

    let handle = h
        .state
        .registry
        .get("inherit-true")
        .expect("service registered");
    let resp = handle
        .ensure(EnsureSource::UserRequest)
        .await
        .expect("ensure must respond");
    if let ananke::supervise::EnsureResponse::Waiting { mut rx } = resp {
        let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("ensure must complete within 5 s")
            .expect("StartOutcome must be sent");
        match outcome {
            ananke::supervise::StartOutcome::Ok => {}
            ananke::supervise::StartOutcome::Err(f) => panic!("start failed: {}", f.message),
        }
    }

    let children = h.process_spawner.children();
    let child = children
        .iter()
        .find(|c| c.binary == "echo")
        .expect("child must have been spawned");

    // Inherited vars are present.
    assert_eq!(
        child.env.get("PATH").map(String::as_str),
        Some("/usr/bin:/bin"),
        "PATH should be inherited; got env={:?}",
        child.env
    );
    assert_eq!(
        child.env.get("HOME").map(String::as_str),
        Some("/home/test"),
        "HOME should be inherited; got env={:?}",
        child.env
    );
    assert_eq!(
        child.env.get("LANG").map(String::as_str),
        Some("C.UTF-8"),
        "LANG should be inherited; got env={:?}",
        child.env
    );
    // Per-service env override is present.
    assert_eq!(
        child.env.get("MY_KEY").map(String::as_str),
        Some("overridden"),
        "MY_KEY should be present from per-service env; got env={:?}",
        child.env
    );
    // CUDA_VISIBLE_DEVICES is always set (empty for CPU-only).
    assert!(
        child.env.contains_key("CUDA_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES should always be present; got env={:?}",
        child.env
    );

    h.cleanup().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn env_inherit_false_excludes_daemon_env() {
    let svc = env_test_service("inherit-false", common::free_port(), false);
    let h = build_harness(vec![svc]).await;

    // Even with an inherited env present, env_inherit=false should
    // produce a child env containing only the per-service overrides.
    h.process_spawner.set_inherited_env(fake_daemon_env());

    let handle = h
        .state
        .registry
        .get("inherit-false")
        .expect("service registered");
    let resp = handle
        .ensure(EnsureSource::UserRequest)
        .await
        .expect("ensure must respond");
    if let ananke::supervise::EnsureResponse::Waiting { mut rx } = resp {
        let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("ensure must complete within 5 s")
            .expect("StartOutcome must be sent");
        match outcome {
            ananke::supervise::StartOutcome::Ok => {}
            ananke::supervise::StartOutcome::Err(f) => panic!("start failed: {}", f.message),
        }
    }

    let children = h.process_spawner.children();
    let child = children
        .iter()
        .find(|c| c.binary == "echo")
        .expect("child must have been spawned");

    // Inherited vars must NOT be present.
    assert!(
        !child.env.contains_key("PATH"),
        "PATH must not be present when env_inherit=false; got env={:?}",
        child.env
    );
    assert!(
        !child.env.contains_key("HOME"),
        "HOME must not be present when env_inherit=false; got env={:?}",
        child.env
    );
    assert!(
        !child.env.contains_key("LANG"),
        "LANG must not be present when env_inherit=false; got env={:?}",
        child.env
    );
    // Per-service env override is still present.
    assert_eq!(
        child.env.get("MY_KEY").map(String::as_str),
        Some("overridden"),
        "MY_KEY should be present from per-service env; got env={:?}",
        child.env
    );
    // CUDA_VISIBLE_DEVICES is always set (empty for CPU-only).
    assert!(
        child.env.contains_key("CUDA_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES should always be present; got env={:?}",
        child.env
    );

    h.cleanup().await;
}
