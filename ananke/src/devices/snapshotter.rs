//! Linux-only: 2-second-cadence device snapshotter.
//!
//! Samples NVML (if available) and /proc/meminfo once per tick and writes
//! into an `Arc<RwLock<DeviceSnapshot>>` shared with readers (allocator,
//! management API). Readers never block the sampler; the sampler replaces
//! the whole snapshot atomically.
//!
//! Also samples per-service observed memory peaks: for each running service,
//! sums NVML VRAM + /proc/<pid>/status VmRSS and calls `observation.update_peak`.

use std::{sync::Arc, time::Duration};

use parking_lot::RwLock;
use tokio::sync::watch;
use tracing::{debug, warn};

use crate::{
    devices::{CpuSnapshot, DeviceSnapshot, GpuProbe, GpuSnapshot, cpu},
    supervise::registry::ServiceRegistry,
    system::{
        ProcFs,
        proc::{descendants_from_map, parent_map, pids_in_cgroup_subtree},
    },
    tracking::observation::{ObservationTable, read_vm_rss},
};

pub type SharedSnapshot = Arc<RwLock<DeviceSnapshot>>;

/// Cadence at which NVML / `/proc/meminfo` / per-service RSS are re-sampled.
const SAMPLE_INTERVAL: Duration = Duration::from_secs(2);

pub fn new_shared() -> SharedSnapshot {
    Arc::new(RwLock::new(DeviceSnapshot::default()))
}

pub fn spawn(
    snapshot: SharedSnapshot,
    probe: Option<Arc<dyn GpuProbe>>,
    observation: ObservationTable,
    registry: ServiceRegistry,
    proc: Arc<dyn ProcFs>,
    mut shutdown: watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(SAMPLE_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                _ = shutdown.changed() => { if *shutdown.borrow() { return; } }
                _ = interval.tick() => {
                    let next = sample(&probe, proc.as_ref());
                    *snapshot.write() = next;
                    sample_observation(&probe, &observation, &registry, proc.as_ref());
                }
            }
        }
    })
}

/// Sample per-service observed memory peaks.
///
/// For each service with a known root PID, builds the full attribution
/// pid set from three sources and sums NVML VRAM + `/proc/<pid>/status`
/// VmRSS across the union:
///
/// 1. **Registered pids** — the immediate child the supervisor spawned.
/// 2. **Transitive descendants** — every pid whose parent chain leads
///    back to a registered pid. Catches wrapper scripts that fork
///    workers without `exec`.
/// 3. **Cgroup-resident pids** — every pid in the v2 subtree declared by
///    `[service.tracking].cgroup_parent`. Catches containerised
///    workloads (Docker, etc.) where the actual workload is reparented
///    out of the daemon's process tree, so pid lineage breaks.
///
/// The parent map is built once per tick and reused across services so
/// the per-tick cost stays at one `/proc` walk regardless of service
/// count.
fn sample_observation(
    probe: &Option<Arc<dyn GpuProbe>>,
    observation: &ObservationTable,
    registry: &ServiceRegistry,
    proc: &dyn ProcFs,
) {
    let parents = parent_map(proc);
    // Cache GPU compute-app lists per device id so a service iterating
    // multiple roots doesn't pay for a fresh NVML probe per pid.
    let gpu_processes: Vec<(u32, Vec<crate::devices::GpuProcess>)> = match probe {
        Some(p) => p
            .list()
            .into_iter()
            .map(|info| (info.id, p.processes(info.id)))
            .collect(),
        None => Vec::new(),
    };

    for (name, _handle) in registry.all() {
        let registered = observation.pids(&name);
        let cgroup_parent = observation.cgroup_parent(&name);
        if registered.is_empty() && cgroup_parent.is_none() {
            continue;
        }
        let pid_set = attributed_pid_set(&registered, cgroup_parent.as_deref(), &parents, proc);
        let (vram, rss) = attributed_bytes_split(&pid_set, &gpu_processes, proc);
        if vram + rss > 0 {
            observation.update_peak(&name, vram, rss);
        }
    }
}

/// Build the pid set the snapshotter should attribute to a single service:
/// registered pids ∪ transitive descendants ∪ cgroup-resident pids.
///
/// Public-to-the-crate so the snapshotter tests can assert on it directly
/// without standing up a full supervisor + registry.
fn attributed_pid_set(
    registered: &[u32],
    cgroup_parent: Option<&str>,
    parents: &std::collections::BTreeMap<u32, u32>,
    proc: &dyn ProcFs,
) -> std::collections::BTreeSet<u32> {
    let mut pid_set: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    for pid in registered {
        for descendant in descendants_from_map(parents, *pid) {
            pid_set.insert(descendant);
        }
    }
    if let Some(parent) = cgroup_parent {
        for pid in pids_in_cgroup_subtree(proc, parent) {
            pid_set.insert(pid);
        }
    }
    pid_set
}

/// Sum NVML-reported VRAM and `/proc/<pid>/status` RSS for every pid in
/// `pid_set`, returning the two components separately. `gpu_processes`
/// is the per-tick cache populated once in `sample_observation`.
///
/// Splitting matters because the dynamic-allocation pledge only models
/// VRAM — combining VRAM and RSS into a single peak and pledging that
/// would inflate the GPU pledge with python's interpreter RSS and
/// produce false over-commit signals.
fn attributed_bytes_split(
    pid_set: &std::collections::BTreeSet<u32>,
    gpu_processes: &[(u32, Vec<crate::devices::GpuProcess>)],
    proc: &dyn ProcFs,
) -> (u64, u64) {
    let mut vram: u64 = 0;
    for (_id, processes) in gpu_processes {
        for gp in processes {
            if pid_set.contains(&gp.pid) {
                vram = vram.saturating_add(gp.used_bytes);
            }
        }
    }
    let mut rss: u64 = 0;
    for pid in pid_set {
        if let Some(r) = read_vm_rss(proc, *pid) {
            rss = rss.saturating_add(r);
        }
    }
    (vram, rss)
}

fn sample(probe: &Option<Arc<dyn GpuProbe>>, proc: &dyn ProcFs) -> DeviceSnapshot {
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

    let cpu = match cpu::read(proc) {
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
        taken_at_ms: crate::tracking::now_unix_ms_u64(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        devices::{
            fake::{FakeGpu, FakeProbe},
            probe::GpuInfo,
        },
        supervise::registry::ServiceRegistry,
        system::InMemoryProcFs,
        tracking::observation::ObservationTable,
    };

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
        // Empty InMemoryProcFs: cpu::read returns an error, which is fine —
        // the test only asserts on the GPU side of the snapshot.
        let proc: Arc<dyn ProcFs> = Arc::new(InMemoryProcFs::new());
        let join = spawn(
            snapshot.clone(),
            Some(Arc::new(fake)),
            ObservationTable::new(),
            ServiceRegistry::new(),
            proc,
            rx,
        );

        tokio::time::sleep(Duration::from_secs(3)).await;
        let s = snapshot.read().clone();
        assert_eq!(s.gpus.len(), 1);
        assert_eq!(s.gpus[0].free_bytes, 20 * 1024 * 1024 * 1024);

        tx.send(true).unwrap();
        let _ = join.await;
    }

    /// Pid attribution must include transitive descendants of every
    /// registered pid, so a wrapper script that fork+execs a worker is
    /// covered without configuring `tracking.cgroup_parent`.
    #[test]
    fn attribution_includes_descendants() {
        let proc = InMemoryProcFs::new();
        // Tree: 100 (registered) → 200 (worker) → 300 (sub-worker).
        proc.set_parent(200, 100);
        proc.set_parent(300, 200);
        // Unrelated pid 999 must not be picked up.
        proc.set_parent(999, 1);
        let parents = parent_map(&proc);
        let set = attributed_pid_set(&[100], None, &parents, &proc);
        let mut pids: Vec<u32> = set.into_iter().collect();
        pids.sort();
        assert_eq!(pids, vec![100, 200, 300]);
    }

    /// Cgroup attribution catches pids that are NOT in the registered
    /// parent chain (e.g. a Docker container reparented to
    /// `containerd-shim`). Both sources are unioned.
    #[test]
    fn attribution_unions_cgroup_pids() {
        let proc = InMemoryProcFs::new();
        // Registered pid + one descendant.
        proc.set_parent(50, 10);
        // Containerised python lives in a cgroup under the declared parent.
        // It has *no* parent link to pid 10 (containerd-shim is its host
        // parent in reality; modelling it as parented to init is sufficient).
        proc.set_parent(700, 1);
        proc.set_cgroup(700, "/system.slice/ananke-comfyui.slice/docker-abc.scope");
        // Sibling cgroup that must NOT match.
        proc.set_parent(701, 1);
        proc.set_cgroup(701, "/system.slice/other.scope");

        let parents = parent_map(&proc);
        let set = attributed_pid_set(
            &[10],
            Some("/system.slice/ananke-comfyui.slice"),
            &parents,
            &proc,
        );
        let mut pids: Vec<u32> = set.into_iter().collect();
        pids.sort();
        assert_eq!(pids, vec![10, 50, 700]);
    }

    /// VRAM and RSS sums correctly across the union and stay separated
    /// in the return value — combining them in the snapshotter would
    /// inflate the dynamic pledge by python's interpreter RSS.
    #[test]
    fn attributed_bytes_split_keeps_vram_and_rss_apart() {
        use crate::devices::GpuProcess;
        let proc = InMemoryProcFs::new();
        proc.set_vm_rss(50, 1_000_000_000); // 1 GB RSS on a descendant.
        let pid_set: std::collections::BTreeSet<u32> = [10, 50, 700].into_iter().collect();
        let gpu_processes = vec![(
            0u32,
            vec![GpuProcess {
                pid: 700,
                used_bytes: 10_000_000_000, // 10 GB VRAM on the container pid.
                name: "python".into(),
            }],
        )];
        let (vram, rss) = attributed_bytes_split(&pid_set, &gpu_processes, &proc);
        assert_eq!(vram, 10_000_000_000);
        assert_eq!(rss, 1_000_000_000);
    }

    /// A service with neither a registered pid nor a cgroup_parent is a
    /// no-op (idle service that hasn't been started yet); the helper
    /// returns an empty set without panicking.
    #[test]
    fn attribution_empty_when_no_inputs() {
        let proc = InMemoryProcFs::new();
        let parents = parent_map(&proc);
        let set = attributed_pid_set(&[], None, &parents, &proc);
        assert!(set.is_empty());
    }
}
