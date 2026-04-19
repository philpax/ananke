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
    drain::DrainReason,
    oneshot::{OneshotRegistry, PortPool},
    service_registry::ServiceRegistry,
};

/// Spawn a task that sleeps until `ttl` elapses, then drains the oneshot's
/// supervisor and cleans up the registry and port pool.
#[allow(clippy::too_many_arguments)]
pub fn spawn_watcher(
    id: SmolStr,
    service_name: SmolStr,
    ttl: Duration,
    registry: ServiceRegistry,
    oneshots: OneshotRegistry,
    db: Database,
    port_pool: Arc<Mutex<PortPool>>,
    port: u16,
    mut shutdown: watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
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
        let now_ms = now_ms();
        {
            use crate::db::models::Oneshot;
            let mut handle = db.handle();
            if let Ok(Some(mut row)) = Oneshot::filter_by_id(id.to_string())
                .first()
                .exec(&mut handle)
                .await
            {
                let _ = row.update().ended_at(Some(now_ms)).exec(&mut handle).await;
            }
        }
        oneshots.remove(&id);
        port_pool.lock().release(port);
    })
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}
