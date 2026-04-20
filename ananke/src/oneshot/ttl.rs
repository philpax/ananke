//! Per-oneshot TTL watcher: when the wall clock exceeds the oneshot's TTL,
//! issue a `BeginDrain` on its supervisor and record the final state in the
//! `oneshots` SQLite table.

use std::{sync::Arc, time::Duration};

use parking_lot::Mutex;
use smol_str::SmolStr;
use tokio::sync::watch;
use tracing::info;

use crate::{
    db::Database,
    oneshot::{OneshotRegistry, PortPool},
    supervise::{drain::DrainReason, registry::ServiceRegistry},
};

/// Inputs to [`spawn_watcher`].
pub struct WatcherConfig {
    pub id: SmolStr,
    pub service_name: SmolStr,
    pub ttl: Duration,
    pub port: u16,
    pub registry: ServiceRegistry,
    pub oneshots: OneshotRegistry,
    pub db: Database,
    pub port_pool: Arc<Mutex<PortPool>>,
    pub shutdown: watch::Receiver<bool>,
}

/// Spawn a task that sleeps until `cfg.ttl` elapses, then drains the oneshot's
/// supervisor and cleans up the registry and port pool.
pub fn spawn_watcher(cfg: WatcherConfig) -> tokio::task::JoinHandle<()> {
    let WatcherConfig {
        id,
        service_name,
        ttl,
        port,
        registry,
        oneshots,
        db,
        port_pool,
        mut shutdown,
    } = cfg;
    tokio::spawn(async move {
        tokio::select! {
            _ = tokio::time::sleep(ttl) => {}
            res = shutdown.changed() => {
                // If the channel errors (sender dropped) or shutdown fires, exit.
                if res.is_err() || *shutdown.borrow() {
                    return;
                }
            }
        }
        info!(%id, "oneshot TTL expired; draining");
        if let Some(handle) = registry.get(&service_name) {
            handle.begin_drain(DrainReason::TtlExpired).await;
        }
        let now_ms = crate::tracking::now_unix_ms();
        let _ = db.mark_oneshot_ended(&id, now_ms).await;
        // Leave the record in place with `ended_at_ms` set so callers can
        // still observe the terminal state via `GET /api/oneshot/:id`.
        // Removing it here caused polling clients (stress scenario 05) to
        // see 404s immediately after TTL expiry.
        oneshots.mark_ended(&id, now_ms as u64, None);
        port_pool.lock().release(port);
    })
}
