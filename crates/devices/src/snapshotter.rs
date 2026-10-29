//! Linux-only: 2-second-cadence device snapshotter.
//!
//! Samples NVML (if available) and /proc/meminfo once per tick and writes
//! into an `Arc<RwLock<DeviceSnapshot>>` shared with readers (allocator,
//! management API). Readers never block the sampler; the sampler replaces
//! the whole snapshot atomically.
//!
//! Also samples per-service observed memory: for each running service, sums
//! NVML VRAM + /proc/<pid>/status VmRSS and calls `observation.record_sample`.

use std::{sync::Arc, time::Duration};

use ananke_observation::{ObservationTable, SharedSnapshot, read_rss};
use ananke_placement::{
    ServiceRegistry,
    devices::{CpuSnapshot, DeviceSnapshot, GpuSnapshot},
};
use ananke_system::{
    ProcFs, Rss,
    proc::{descendants_from_map, parent_map, pids_in_cgroup_subtree},
};
use tokio::sync::watch;
use tracing::{debug, warn};

use crate::{GpuProbe, cpu};

/// Cadence at which NVML / `/proc/meminfo` / per-service RSS are re-sampled.
const SAMPLE_INTERVAL: Duration = Duration::from_secs(2);

pub fn spawn<T: Send + Sync + 'static>(
    snapshot: SharedSnapshot,
    probe: Option<Arc<dyn GpuProbe>>,
    observation: ObservationTable,
    registry: ananke_placement::ServiceRegistry<T>,
    proc: Arc<dyn ProcFs>,
    mut shutdown: watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(SAMPLE_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // Besides the cadence, sample whenever a supervisor asks: one that
        // has just reaped its child needs the freed memory visible before
        // the next placement, not up to a tick later.
        let sampler = snapshot.attach_sampler();
        loop {
            tokio::select! {
                _ = shutdown.changed() => {
                    if *shutdown.borrow() { return; }
                    continue;
                }
                _ = interval.tick() => {}
                _ = sampler.requested() => {}
            }
            let ticket = sampler.begin();
            let next = sample(&probe, proc.as_ref());
            *snapshot.write() = next;
            sampler.served(ticket);
            sample_observation(&probe, &observation, &registry, proc.as_ref());
        }
    })
}

/// Sample per-service observed memory.
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
fn sample_observation<T: Send + Sync + 'static>(
    probe: &Option<Arc<dyn GpuProbe>>,
    observation: &ObservationTable,
    registry: &ServiceRegistry<T>,
    proc: &dyn ProcFs,
) {
    let parents = parent_map(proc);
    // Cache GPU compute-app lists per device id so a service iterating
    // multiple roots doesn't pay for a fresh NVML probe per pid.
    let gpu_processes: Vec<(u32, Vec<crate::GpuProcess>)> = match probe {
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
        let AttributedBytes { vram, rss } = attributed_bytes_split(&pid_set, &gpu_processes, proc);
        debug!(
            service = %name,
            registered_pids = ?registered,
            cgroup_parent = ?cgroup_parent.as_deref(),
            pid_set_len = pid_set.len(),
            vram_mb = vram / (1024 * 1024),
            rss_mb = rss.total / (1024 * 1024),
            rss_owned_mb = rss.owned / (1024 * 1024),
            rss_file_mb = rss.file / (1024 * 1024),
            "observation attribution sample"
        );
        // A tick that attributes nothing at all is a gap in the signal, not
        // an observation of zero: the pid set can go momentarily unreadable
        // (pid exiting between the `/proc` walk and the status read). Skip it
        // so the retained current reading stays the last thing we actually
        // saw, rather than dropping to zero and telling the balloon resolver
        // the service released everything.
        if vram + rss.total > 0 {
            observation.record_sample(&name, vram, rss);
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
/// One tick's per-service memory attribution, split by what each figure can
/// legitimately be compared against.
struct AttributedBytes {
    /// NVML-attributed VRAM.
    vram: u64,
    /// The host footprint, split by what each part can be compared against.
    rss: Rss,
}

fn attributed_bytes_split(
    pid_set: &std::collections::BTreeSet<u32>,
    gpu_processes: &[(u32, Vec<crate::GpuProcess>)],
    proc: &dyn ProcFs,
) -> AttributedBytes {
    let mut vram: u64 = 0;
    for (_id, processes) in gpu_processes {
        for gp in processes {
            if pid_set.contains(&gp.pid) {
                vram = vram.saturating_add(gp.used_bytes);
            }
        }
    }
    let mut rss = Rss::default();
    for pid in pid_set {
        if let Some(r) = read_rss(proc, *pid) {
            rss.total = rss.total.saturating_add(r.total);
            rss.owned = rss.owned.saturating_add(r.owned);
            rss.file = rss.file.saturating_add(r.file);
        }
    }
    AttributedBytes { vram, rss }
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
        taken_at_ms: ananke_time::now_unix_ms_u64(),
    }
}

#[cfg(test)]
mod tests {
    use ananke_observation::{ObservationTable, new_shared};
    use ananke_placement::ServiceRegistry;
    use ananke_system::InMemoryProcFs;

    use super::*;
    use crate::{
        fake::{FakeGpu, FakeProbe},
        probe::GpuInfo,
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
        let join = spawn::<()>(
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

    /// A container's workers are not descendants of the daemon at all — the
    /// runtime reparents them — so the cgroup is the only thing that reaches
    /// them. The inspected host PID must not be needed for attribution.
    #[test]
    fn container_cgroup_worker_is_attributed() {
        let proc = InMemoryProcFs::new();
        // Nothing is registered: a container launch records no usable host
        // pid of its own, only the cgroup its workers live in.
        proc.set_parent(900, 1);
        proc.set_cgroup(
            900,
            "/ananke.slice/ananke-muse-glimmer.slice/docker-deadbeef.scope",
        );
        // A second worker in the same container.
        proc.set_parent(901, 900);
        proc.set_cgroup(
            901,
            "/ananke.slice/ananke-muse-glimmer.slice/docker-deadbeef.scope",
        );
        // Another service's container under a sibling slice.
        proc.set_parent(950, 1);
        proc.set_cgroup(950, "/ananke.slice/ananke-ninfer.slice/docker-cafe.scope");

        let parents = parent_map(&proc);
        let set = attributed_pid_set(
            &[],
            Some("/ananke.slice/ananke-muse-glimmer.slice"),
            &parents,
            &proc,
        );
        let mut pids: Vec<u32> = set.into_iter().collect();
        pids.sort();
        assert_eq!(pids, vec![900, 901]);
    }

    /// pid 0 parents init, so walking its descendants reaches the entire
    /// process table. Nothing may register it — but if something does, the
    /// walk itself is what turns a missing host pid into "this service is
    /// using all the memory on the machine", so the shape is worth pinning.
    #[test]
    fn descendants_of_pid_zero_reach_everything() {
        let proc = InMemoryProcFs::new();
        // The real /proc shape: init and kthreadd both report ppid 0.
        proc.set_parent(1, 0);
        proc.set_parent(2, 0);
        proc.set_parent(500, 1);
        proc.set_parent(501, 500);
        let parents = parent_map(&proc);

        let everything = attributed_pid_set(&[0], None, &parents, &proc);
        assert!(
            everything.len() > 4,
            "registering pid 0 sweeps up unrelated processes: {everything:?}"
        );
        // A real pid stays confined to its own subtree.
        let confined = attributed_pid_set(&[500], None, &parents, &proc);
        assert_eq!(
            confined.into_iter().collect::<Vec<_>>(),
            vec![500, 501],
            "a real root attributes only its own descendants"
        );
    }

    /// VRAM and RSS sums correctly across the union and stay separated
    /// in the return value — combining them in the snapshotter would
    /// inflate the dynamic pledge by python's interpreter RSS.
    #[test]
    fn attributed_bytes_split_keeps_vram_and_rss_apart() {
        use crate::GpuProcess;
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
        let AttributedBytes { vram, rss, .. } =
            attributed_bytes_split(&pid_set, &gpu_processes, &proc);
        assert_eq!(vram, 10_000_000_000);
        assert_eq!(rss.total, 1_000_000_000);
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
