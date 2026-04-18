//! 2-second-cadence device snapshotter.
//!
//! Samples NVML (if available) and /proc/meminfo once per tick and writes
//! into an `Arc<RwLock<DeviceSnapshot>>` shared with readers (allocator,
//! management API). Readers never block the sampler; the sampler replaces
//! the whole snapshot atomically.
//!
//! Also samples per-service observed memory peaks: for each running service,
//! sums NVML VRAM + /proc/<pid>/status VmRSS and calls `observation.update_peak`.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use tokio::sync::watch;
use tracing::{debug, warn};

use crate::devices::{CpuSnapshot, DeviceSnapshot, GpuProbe, GpuSnapshot, cpu};
use crate::observation::{ObservationTable, read_vm_rss};
use crate::service_registry::ServiceRegistry;

pub type SharedSnapshot = Arc<RwLock<DeviceSnapshot>>;

pub fn new_shared() -> SharedSnapshot {
    Arc::new(RwLock::new(DeviceSnapshot::default()))
}

pub fn spawn(
    snapshot: SharedSnapshot,
    probe: Option<Arc<dyn GpuProbe>>,
    observation: ObservationTable,
    registry: ServiceRegistry,
    mut shutdown: watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                _ = shutdown.changed() => { if *shutdown.borrow() { return; } }
                _ = interval.tick() => {
                    let next = sample(&probe);
                    *snapshot.write() = next;
                    sample_observation(&probe, &observation, &registry);
                }
            }
        }
    })
}

/// Sample per-service observed memory peaks.
///
/// For each service with a known PID, aggregates NVML VRAM usage (summed
/// across all GPUs for that PID) and `/proc/<pid>/status` VmRSS, then
/// calls `observation.update_peak` with the total.
fn sample_observation(
    probe: &Option<Arc<dyn GpuProbe>>,
    observation: &ObservationTable,
    registry: &ServiceRegistry,
) {
    for (name, _handle) in registry.all() {
        // snapshot() is async, so we use the synchronous pid state we have
        // via a best-effort approach: query the observation table's pids directly
        // by iterating known PIDs registered at spawn time.
        let pids = observation.pids(&name);
        if pids.is_empty() {
            continue;
        }
        let mut total: u64 = 0;

        // GPU VRAM from NVML — sum across all GPUs.
        if let Some(p) = probe {
            for gpu in p.list() {
                for proc in p.processes(gpu.id) {
                    if pids.contains(&proc.pid) {
                        total = total.saturating_add(proc.used_bytes);
                    }
                }
            }
        }

        // CPU RSS from /proc.
        for pid in &pids {
            if let Some(rss) = read_vm_rss(*pid) {
                total = total.saturating_add(rss);
            }
        }

        if total > 0 {
            observation.update_peak(&name, total);
        }
    }
}

fn sample(probe: &Option<Arc<dyn GpuProbe>>) -> DeviceSnapshot {
    let gpus: Vec<GpuSnapshot> = probe
        .as_ref()
        .map(|p| {
            p.list()
                .into_iter()
                .map(|info| {
                    let mem = p.query(info.id);
                    GpuSnapshot {
                        id: info.id,
                        name: info.name,
                        total_bytes: mem.as_ref().map(|m| m.total_bytes).unwrap_or(0),
                        free_bytes: mem.as_ref().map(|m| m.free_bytes).unwrap_or(0),
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    let cpu = match cpu::read() {
        Ok(c) => Some(CpuSnapshot {
            total_bytes: c.total_bytes,
            available_bytes: c.available_bytes,
        }),
        Err(e) => {
            debug!(error = %e, "cpu read failed");
            None
        }
    };

    if gpus.is_empty() && cpu.is_none() {
        warn!("device snapshot is empty — NVML and /proc/meminfo both failed");
    }

    DeviceSnapshot {
        gpus,
        cpu,
        taken_at_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::fake::{FakeGpu, FakeProbe};
    use crate::devices::probe::GpuInfo;
    use crate::observation::ObservationTable;
    use crate::service_registry::ServiceRegistry;

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn sampler_populates_snapshot() {
        let fake = FakeProbe::new(vec![FakeGpu {
            info: GpuInfo {
                id: 0,
                name: "Test".into(),
                total_bytes: 24 * 1024 * 1024 * 1024,
            },
            free_bytes: 20 * 1024 * 1024 * 1024,
            processes: Vec::new(),
        }]);
        let snapshot = new_shared();
        let (tx, rx) = watch::channel(false);
        let join = spawn(
            snapshot.clone(),
            Some(Arc::new(fake)),
            ObservationTable::new(),
            ServiceRegistry::new(),
            rx,
        );

        tokio::time::sleep(Duration::from_secs(3)).await;
        let s = snapshot.read().clone();
        assert_eq!(s.gpus.len(), 1);
        assert_eq!(s.gpus[0].free_bytes, 20 * 1024 * 1024 * 1024);

        tx.send(true).unwrap();
        let _ = join.await;
    }
}
