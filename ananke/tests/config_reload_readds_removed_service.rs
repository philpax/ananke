//! Integration test: a service removed by one reload and added back by a
//! later one serves on its per-service port again.
//!
//! The reconciler used to drain a removed service's supervisor but leave its
//! proxy listener running. Re-adding the service then failed to bind its port
//! ("Address already in use"), and the old listener answered every request
//! for the dead supervisor with "supervisor unreachable" until the daemon
//! restarted. Seen live on 2026-10-29, when a non-atomic write of the config
//! file made the watcher load an empty file and then the full one.
#![cfg(feature = "test-fakes")]

mod common;

use std::time::Duration;

use ananke::config::{EffectiveConfig, ServiceConfig};
use ananke_api::events::Event;
use common::{build_harness, minimal_llama_service};
use smol_str::SmolStr;

fn reload(h: &common::TestHarness, services: Vec<ServiceConfig>, changed: &str) {
    let daemon = h.state.config.effective().daemon.clone();
    h.state
        .config
        .swap_effective_for_test(EffectiveConfig { daemon, services });
    h.state.events.publish(Event::ConfigReloaded {
        at_ms: 0,
        changed_services: vec![SmolStr::new(changed)],
    });
}

async fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !cond() {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting for {what}"));
}

async fn health_status(client: &reqwest::Client, port: u16) -> Result<u16, reqwest::Error> {
    let resp = client
        .get(format!("http://127.0.0.1:{port}/health"))
        .send()
        .await?;
    Ok(resp.status().as_u16())
}

#[tokio::test(flavor = "multi_thread")]
async fn removed_then_readded_service_serves_on_its_port() {
    let alpha = minimal_llama_service("alpha", 0);
    let h = build_harness(vec![alpha.clone()]).await;
    let alpha = h.state.config.effective().services[0].clone();

    let mut beta = minimal_llama_service("beta", common::free_port());
    beta.private_port = alpha.private_port;
    let port = beta.port;
    // A fresh connection per request, so a kept-alive connection to an old
    // listener cannot answer for a new one.
    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(0)
        .build()
        .unwrap();

    reload(&h, vec![alpha.clone(), beta.clone()], "beta");
    wait_until("beta provisioned", || {
        h.state.registry.get("beta").is_some()
    })
    .await;
    // The listener binds inside the proxy task; give it a moment.
    let mut first = None;
    for _ in 0..40 {
        if let Ok(status) = health_status(&client, port).await {
            first = Some(status);
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert_eq!(first, Some(200), "beta never answered on its port");

    reload(&h, vec![alpha.clone()], "beta");
    wait_until("beta removed", || h.state.registry.get("beta").is_none()).await;
    // The reconciler closes the listener right after unregistering beta.
    let mut refused = false;
    for _ in 0..40 {
        if health_status(&client, port).await.is_err() {
            refused = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert!(
        refused,
        "the removed service's port should refuse connections"
    );

    reload(&h, vec![alpha.clone(), beta.clone()], "beta");
    wait_until("beta re-provisioned", || {
        h.state.registry.get("beta").is_some()
    })
    .await;
    let mut again = None;
    for _ in 0..40 {
        match health_status(&client, port).await {
            Ok(status) => {
                again = Some(status);
                break;
            }
            Err(_) => tokio::time::sleep(Duration::from_millis(25)).await,
        }
    }
    assert_eq!(
        again,
        Some(200),
        "the re-added beta should serve on its port again"
    );

    // Its own supervisor answered, not the removed one's.
    let beta_handle = h.state.registry.get("beta").unwrap();
    assert_eq!(
        beta_handle.peek_state(),
        ananke::supervise::state::ServiceState::Running
    );

    h.cleanup().await;
}
