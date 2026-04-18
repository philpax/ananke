# Ananke Phase 2 — Unified OpenAI + On-Demand + Allocator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the phase-1 daemon with an OpenAI-compatible unified listener, on-demand lifecycle with per-service `idle_timeout`, start-future coalescing bounded by `start_queue_depth`, a pure-feasibility allocator, a read-only management slice (`/api/services`, `/api/services/{name}`, `/api/devices`), and a `utoipa`-generated `/api/openapi.json`.

**Architecture:** Two additional Axum routers (OpenAI on `openai_api.listen`, management on `daemon.management_listen`) share an `Arc<AppState>` that carries the service registry, allocation table, 2-second-cadence device snapshot, and per-service activity atomics. Supervisors gain an `idle` entry state and a `broadcast::Sender<StartOutcome>` for coalescing concurrent first-requests. The phase-1 per-service proxy stays as-is; it now also pings the activity atomic so `idle_timeout` doesn't fire on a busy bypass-path client.

**Tech Stack:** Rust 2024, tokio, axum, utoipa + utoipa-axum, hyper + hyper-util, http-body-util, serde_json, tokio::sync::{broadcast, RwLock, watch}.

**Parent design:** `docs/superpowers/specs/2026-04-18-ananke-phase-2-unified-openai-ondemand-allocator.md`. Phase-1 plan: `docs/superpowers/plans/2026-04-18-ananke-phase-1-lean-mvp-daemon.md`.

---

## File Structure

```
src/
├── main.rs                         // unchanged (delegates to daemon::run)
├── lib.rs                          // + new mods: openai_api, management_api, allocator,
│                                   //   service_registry, activity, snapshotter, openapi, app_state
├── app_state.rs                    // NEW: Arc-shared state carried via Axum extractors
├── service_registry.rs             // NEW: Arc<RwLock<BTreeMap<Name, SupervisorHandle>>>
├── activity.rs                     // NEW: AtomicU64 per service, last-activity millis
├── snapshotter.rs                  // NEW: 2s NVML + CPU sampler task
├── allocator.rs                    // NEW: pure can_fit(want, snapshot, reserved)
├── openapi.rs                      // NEW: #[derive(OpenApi)] aggregator + handler
├── openai_api/
│   ├── mod.rs                      // NEW: pub fn router(AppState) -> axum::Router
│   ├── handlers.rs                 // NEW: /v1/models, /v1/chat, /v1/completions, /v1/embeddings
│   ├── filters.rs                  // NEW: strip_params + set_params JSON rewrite
│   ├── errors.rs                   // NEW: OpenAI-shaped {error: {code, message, type}}
│   ├── schema.rs                   // NEW: ChatCompletionEnvelope, CompletionEnvelope, ...
│   └── unimplemented.rs            // NEW: 501 handlers for /v1/audio/* etc.
├── management_api/
│   ├── mod.rs                      // NEW: pub fn router(AppState) -> axum::Router
│   ├── handlers.rs                 // NEW: /api/services, /api/services/{name}, /api/devices
│   └── types.rs                    // NEW: ServiceSummary, ServiceDetail, DeviceSummary
├── supervise/
│   ├── mod.rs                      // MODIFY: add Idle entry state, Ensure/ActivityPing commands,
│                                   //   broadcast start bus, idle_timer in Running
│   └── ...                         // unchanged
├── config/
│   └── validate.rs                 // MODIFY: drop phase-1 on_demand rejection; parse filters;
│                                   //   parse idle_timeout_ms into ServiceConfig
├── daemon.rs                       // MODIFY: build AppState; spawn snapshotter; spawn both routers;
│                                   //   on-demand services start in Idle not Starting
└── proxy.rs                        // MODIFY: emit ActivityPing atomic bump on every request

tests/
├── common/
│   ├── mod.rs                      // MODIFY: add helpers for AppState, spawn counter
│   └── echo_server.rs              // MODIFY: add spawn counter header, /sink endpoint
├── openai_models.rs                // NEW
├── openai_chat_routing.rs          // NEW
├── openai_unimplemented.rs         // NEW
├── ondemand_start.rs               // NEW
├── start_coalescing.rs             // NEW
├── start_queue_full.rs             // NEW
├── idle_timeout_returns_to_idle.rs // NEW
├── allocator_insufficient_vram.rs  // NEW
├── management_services.rs          // NEW
├── management_devices.rs           // NEW
└── openapi_json.rs                 // NEW
```

---

## Task 1: Add utoipa + axum upgrades to Cargo.toml

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1: Add the new dependencies**

In `[dependencies]` section, ensure these entries exist (add if missing, update versions if present):

```toml
axum = { version = "0.7", features = ["macros", "json"] }
utoipa = { version = "5", features = ["axum_extras", "smallvec", "uuid"] }
utoipa-axum = "0.2"
tower = "0.5"
tower-http = { version = "0.6", features = ["trace"] }
async-trait = "0.1"
```

If `axum` or `tower` were already added in phase 1 at a different version, keep them at this version. Cargo will reconcile with the existing hyper/hyper-util stack.

- [ ] **Step 2: Verify it resolves**

Run: `cargo check`
Expected: all crates resolve, no compile errors. Warnings about unused crates are OK (they will be used by subsequent tasks). If a version specifier fails, bump to the nearest compatible release and note.

- [ ] **Step 3: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "chore: add axum, utoipa, utoipa-axum, tower for phase 2"
```

---

## Task 2: Activity atomic module

**Files:**
- Create: `src/activity.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write the failing test**

Create `src/activity.rs`:

```rust
//! Per-service activity timestamps, shared across tasks via `Arc<AtomicU64>`.
//!
//! Stores UNIX epoch milliseconds. Readers (supervisors computing idle
//! deadlines) use `load(Ordering::Relaxed)`; writers (proxy paths) use
//! `store(now_ms, Ordering::Relaxed)`. A monotonic wall clock is not
//! required: a stale value only delays idle transitions, which is
//! harmless for the scheduler.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use smol_str::SmolStr;

#[derive(Clone, Default)]
pub struct ActivityTable {
    inner: Arc<RwLock<BTreeMap<SmolStr, Arc<AtomicU64>>>>,
}

impl ActivityTable {
    pub fn new() -> Self { Self::default() }

    /// Return the atomic for `service`, creating it if missing.
    pub fn get_or_init(&self, service: &SmolStr) -> Arc<AtomicU64> {
        {
            let guard = self.inner.read();
            if let Some(existing) = guard.get(service) {
                return existing.clone();
            }
        }
        let mut guard = self.inner.write();
        guard.entry(service.clone())
            .or_insert_with(|| Arc::new(AtomicU64::new(now_ms())))
            .clone()
    }

    /// Bump the activity timestamp for `service` to now.
    pub fn ping(&self, service: &SmolStr) {
        self.get_or_init(service).store(now_ms(), Ordering::Relaxed);
    }

    /// Read the last activity timestamp for `service`. Returns None if
    /// the service has never been pinged.
    pub fn last_ms(&self, service: &SmolStr) -> Option<u64> {
        self.inner.read().get(service).map(|a| a.load(Ordering::Relaxed))
    }
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ping_updates_last_ms() {
        let t = ActivityTable::new();
        let svc = SmolStr::new("demo");
        assert!(t.last_ms(&svc).is_none());
        t.ping(&svc);
        let first = t.last_ms(&svc).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        t.ping(&svc);
        let second = t.last_ms(&svc).unwrap();
        assert!(second >= first);
    }

    #[test]
    fn get_or_init_returns_same_atomic() {
        let t = ActivityTable::new();
        let svc = SmolStr::new("demo");
        let a = t.get_or_init(&svc);
        let b = t.get_or_init(&svc);
        a.store(42, Ordering::Relaxed);
        assert_eq!(b.load(Ordering::Relaxed), 42);
    }
}
```

Add `pub mod activity;` to `src/lib.rs` alphabetically near the other `pub mod` lines.

- [ ] **Step 2: Run test to verify it passes**

Run: `cargo test --lib activity::tests`
Expected: 2 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/activity.rs src/lib.rs
git commit -m "feat(activity): per-service atomic last-activity timestamps"
```

---

## Task 3: Service registry

**Files:**
- Create: `src/service_registry.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write the failing test + implementation**

Create `src/service_registry.rs`:

```rust
//! Shared lookup from service name to `SupervisorHandle`.
//!
//! Read-heavy; wrapped in an `Arc<RwLock<...>>` so both HTTP routers and
//! the daemon lifecycle code can share visibility without cloning the
//! whole map per request.

use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::RwLock;
use smol_str::SmolStr;

use crate::supervise::SupervisorHandle;

#[derive(Clone, Default)]
pub struct ServiceRegistry {
    inner: Arc<RwLock<BTreeMap<SmolStr, Arc<SupervisorHandle>>>>,
}

impl ServiceRegistry {
    pub fn new() -> Self { Self::default() }

    pub fn insert(&self, name: SmolStr, handle: Arc<SupervisorHandle>) {
        self.inner.write().insert(name, handle);
    }

    pub fn get(&self, name: &str) -> Option<Arc<SupervisorHandle>> {
        self.inner.read().get(name).cloned()
    }

    pub fn names(&self) -> Vec<SmolStr> {
        self.inner.read().keys().cloned().collect()
    }

    pub fn all(&self) -> Vec<(SmolStr, Arc<SupervisorHandle>)> {
        self.inner.read().iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }
}
```

Modify `src/supervise/mod.rs` to wrap the existing `SupervisorHandle` in an `Arc`-friendly shape. The current handle's `shutdown(self)` consumes `self`, which is incompatible with `Arc`. Change to an interior-mutable pattern: move `join: JoinHandle<()>` behind `tokio::sync::Mutex<Option<JoinHandle<()>>>` so `shutdown(&self)` can take the handle out once.

Replace the existing `SupervisorHandle` block (lines ~47-75) with:

```rust
pub struct SupervisorHandle {
    pub name: smol_str::SmolStr,
    tx: mpsc::Sender<SupervisorCommand>,
    join: tokio::sync::Mutex<Option<JoinHandle<()>>>,
}

impl SupervisorHandle {
    pub async fn shutdown(&self) {
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        let _ = self.tx.send(SupervisorCommand::Shutdown { ack: ack_tx }).await;
        let _ = ack_rx.await;
        if let Some(handle) = self.join.lock().await.take() {
            let _ = handle.await;
        }
    }

    pub async fn snapshot(&self) -> Option<SupervisorSnapshot> {
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        if self.tx.send(SupervisorCommand::Snapshot { ack: ack_tx }).await.is_err() {
            return None;
        }
        ack_rx.await.ok()
    }
}
```

And update `spawn_supervisor` to wrap the `JoinHandle`:

```rust
pub fn spawn_supervisor(
    svc: ServiceConfig,
    allocation: Allocation,
    db: Database,
    batcher: BatcherHandle,
    service_id: i64,
) -> SupervisorHandle {
    let (tx, rx) = mpsc::channel(32);
    let name = svc.name.clone();
    let join = tokio::spawn(run(svc, allocation, db, batcher, service_id, rx));
    SupervisorHandle {
        name,
        tx,
        join: tokio::sync::Mutex::new(Some(join)),
    }
}
```

Update `src/daemon.rs` where it currently calls `sup.shutdown().await` in a loop:

```rust
// was: for sup in supervisors { sup.shutdown().await; }
for sup in &supervisors { sup.shutdown().await; }
```

Add `pub mod service_registry;` to `src/lib.rs`.

- [ ] **Step 2: Verify it compiles**

Run: `cargo check --lib`
Expected: clean compile.

- [ ] **Step 3: Add a small test**

Append to `src/service_registry.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::validate::{DeviceSlot, HealthSettings, Lifecycle, PlacementPolicy, ServiceConfig, Template};
    use crate::config::parse::RawService;
    use crate::db::Database;
    use crate::db::logs::spawn as spawn_batcher;
    use crate::devices::Allocation;
    use crate::supervise::spawn_supervisor;
    use smol_str::SmolStr;
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn minimal_svc(name: &str) -> ServiceConfig {
        let mut override_map = BTreeMap::new();
        override_map.insert(DeviceSlot::Cpu, 100);
        ServiceConfig {
            name: SmolStr::new(name),
            template: Template::LlamaCpp,
            port: 0,
            private_port: 0,
            lifecycle: Lifecycle::Persistent,
            priority: 50,
            health: HealthSettings { http_path: "/".into(), timeout_ms: 1000, probe_interval_ms: 500 },
            placement_override: override_map,
            placement_policy: PlacementPolicy::CpuOnly,
            idle_timeout_ms: 600_000,
            warming_grace_ms: 1000,
            drain_timeout_ms: 1000,
            extended_stream_drain_ms: 1000,
            max_request_duration_ms: 1000,
            raw: RawService {
                name: Some(SmolStr::new(name)),
                template: Some(SmolStr::new("llama-cpp")),
                model: Some(PathBuf::from("/fake/path")),
                port: Some(0),
                ..Default::default()
            },
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn insert_and_get() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("a.sqlite")).unwrap();
        let batcher = spawn_batcher(db.clone());
        let svc = minimal_svc("demo");
        let alloc = Allocation::from_override(&svc.placement_override);
        let handle = Arc::new(spawn_supervisor(svc.clone(), alloc, db.clone(), batcher.clone(), 1));

        let registry = ServiceRegistry::new();
        registry.insert(SmolStr::new("demo"), handle.clone());
        assert!(registry.get("demo").is_some());
        assert!(registry.get("missing").is_none());
        assert_eq!(registry.names(), vec![SmolStr::new("demo")]);

        handle.shutdown().await;
    }
}
```

Run: `cargo test --lib service_registry::tests`
Expected: 1 test passes.

- [ ] **Step 4: Commit**

```bash
git add src/service_registry.rs src/lib.rs src/supervise/mod.rs src/daemon.rs
git commit -m "feat(registry): Arc<RwLock> service name → handle map; shared SupervisorHandle"
```

---

## Task 4: Device snapshotter

**Files:**
- Create: `src/snapshotter.rs`
- Modify: `src/lib.rs`
- Modify: `src/devices/mod.rs` (add `DeviceSnapshot` type)

- [ ] **Step 1: Add `DeviceSnapshot` to `src/devices/mod.rs`**

Append to `src/devices/mod.rs`:

```rust
#[derive(Debug, Clone, Default)]
pub struct DeviceSnapshot {
    pub gpus: Vec<GpuSnapshot>,
    pub cpu: Option<CpuSnapshot>,
    pub taken_at_ms: u64,
}

#[derive(Debug, Clone)]
pub struct GpuSnapshot {
    pub id: u32,
    pub name: String,
    pub total_bytes: u64,
    pub free_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct CpuSnapshot {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

impl DeviceSnapshot {
    pub fn free_bytes(&self, slot: &crate::config::validate::DeviceSlot) -> Option<u64> {
        use crate::config::validate::DeviceSlot;
        match slot {
            DeviceSlot::Cpu => self.cpu.as_ref().map(|c| c.available_bytes),
            DeviceSlot::Gpu(id) => self.gpus.iter().find(|g| g.id == *id).map(|g| g.free_bytes),
        }
    }
}
```

- [ ] **Step 2: Create `src/snapshotter.rs`**

```rust
//! 2-second-cadence device snapshotter.
//!
//! Samples NVML (if available) and /proc/meminfo once per tick and writes
//! into an `Arc<RwLock<DeviceSnapshot>>` shared with readers (allocator,
//! management API). Readers never block the sampler; the sampler replaces
//! the whole snapshot atomically.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use tokio::sync::watch;
use tracing::{debug, warn};

use crate::devices::{CpuSnapshot, DeviceSnapshot, GpuProbe, GpuSnapshot, cpu};

pub type SharedSnapshot = Arc<RwLock<DeviceSnapshot>>;

pub fn new_shared() -> SharedSnapshot {
    Arc::new(RwLock::new(DeviceSnapshot::default()))
}

pub fn spawn(
    snapshot: SharedSnapshot,
    probe: Option<Arc<dyn GpuProbe>>,
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
                }
            }
        }
    })
}

fn sample(probe: &Option<Arc<dyn GpuProbe>>) -> DeviceSnapshot {
    let gpus: Vec<GpuSnapshot> = probe.as_ref().map(|p| {
        p.list().into_iter().map(|info| {
            let mem = p.query(info.id);
            GpuSnapshot {
                id: info.id,
                name: info.name,
                total_bytes: mem.as_ref().map(|m| m.total_bytes).unwrap_or(0),
                free_bytes: mem.as_ref().map(|m| m.free_bytes).unwrap_or(0),
            }
        }).collect()
    }).unwrap_or_default();

    let cpu = match cpu::read() {
        Ok(c) => Some(CpuSnapshot { total_bytes: c.total_bytes, available_bytes: c.available_bytes }),
        Err(e) => { debug!(error = %e, "cpu read failed"); None }
    };

    if gpus.is_empty() && cpu.is_none() {
        warn!("device snapshot is empty — NVML and /proc/meminfo both failed");
    }

    DeviceSnapshot {
        gpus,
        cpu,
        taken_at_ms: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64,
    }
}
```

Add `pub mod snapshotter;` to `src/lib.rs`.

- [ ] **Step 3: Write the test**

Append to `src/snapshotter.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::fake::{FakeGpu, FakeProbe};
    use crate::devices::probe::GpuInfo;

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn sampler_populates_snapshot() {
        let fake = FakeProbe::new(vec![FakeGpu {
            info: GpuInfo { id: 0, name: "Test".into(), total_bytes: 24 * 1024 * 1024 * 1024 },
            free_bytes: 20 * 1024 * 1024 * 1024,
            processes: Vec::new(),
        }]);
        let snapshot = new_shared();
        let (tx, rx) = watch::channel(false);
        let join = spawn(snapshot.clone(), Some(Arc::new(fake)), rx);

        tokio::time::sleep(Duration::from_secs(3)).await;
        let s = snapshot.read().clone();
        assert_eq!(s.gpus.len(), 1);
        assert_eq!(s.gpus[0].free_bytes, 20 * 1024 * 1024 * 1024);

        tx.send(true).unwrap();
        let _ = join.await;
    }
}
```

Run: `cargo test --lib snapshotter`
Expected: 1 test passes.

- [ ] **Step 4: Commit**

```bash
git add src/snapshotter.rs src/devices/mod.rs src/lib.rs
git commit -m "feat(snapshotter): 2s NVML+CPU sampler with shared RwLock snapshot"
```

---

## Task 5: Allocator feasibility check

**Files:**
- Create: `src/allocator.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write the failing test**

Create `src/allocator.rs`:

```rust
//! Pure feasibility check for service placement.
//!
//! Phase 2 has no eviction: the allocator either admits a service whose
//! declared `placement_override` fits live free bytes minus existing
//! reservations, or reports `NoFit` with a specific slot and shortfall.
//! Future phases replace the in-crate caller with an eviction-capable
//! one but keep this function as the innermost yes/no.

use std::collections::BTreeMap;

use smol_str::SmolStr;

use crate::config::validate::DeviceSlot;
use crate::devices::DeviceSnapshot;

pub type AllocationTable = BTreeMap<SmolStr, BTreeMap<DeviceSlot, u64>>;

#[derive(Debug, PartialEq, Eq)]
pub struct NoFit {
    pub slot: DeviceSlot,
    pub needed_bytes: u64,
    pub available_bytes: u64,
}

impl std::fmt::Display for NoFit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let slot = match &self.slot {
            DeviceSlot::Cpu => "cpu".to_string(),
            DeviceSlot::Gpu(n) => format!("gpu:{n}"),
        };
        write!(
            f,
            "no fit on {slot}: need {} bytes, {} available",
            self.needed_bytes, self.available_bytes
        )
    }
}

impl std::error::Error for NoFit {}

/// Check whether `want` (per-slot MB from `placement_override`) fits in
/// the device snapshot after subtracting the bytes already reserved by
/// other services.
pub fn can_fit(
    want: &BTreeMap<DeviceSlot, u64>,
    snapshot: &DeviceSnapshot,
    reserved: &AllocationTable,
    exclude: Option<&SmolStr>,
) -> Result<(), NoFit> {
    for (slot, want_mb) in want {
        let need = want_mb * 1024 * 1024;
        let free = snapshot.free_bytes(slot).unwrap_or(0);
        let already: u64 = reserved
            .iter()
            .filter(|(k, _)| exclude.is_none_or(|x| *k != x))
            .filter_map(|(_, alloc)| alloc.get(slot))
            .sum::<u64>()
            * 1024 * 1024;
        let available = free.saturating_sub(already);
        if available < need {
            return Err(NoFit {
                slot: slot.clone(),
                needed_bytes: need,
                available_bytes: available,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::{CpuSnapshot, DeviceSnapshot, GpuSnapshot};

    fn gb(n: u64) -> u64 { n * 1024 * 1024 * 1024 }
    fn mb(n: u64) -> u64 { n }

    fn snapshot_with(free_gpu_gb: u64, free_cpu_gb: u64) -> DeviceSnapshot {
        DeviceSnapshot {
            gpus: vec![GpuSnapshot { id: 0, name: "Test".into(), total_bytes: gb(24), free_bytes: gb(free_gpu_gb) }],
            cpu: Some(CpuSnapshot { total_bytes: gb(128), available_bytes: gb(free_cpu_gb) }),
            taken_at_ms: 0,
        }
    }

    #[test]
    fn fits_when_below_free() {
        let mut want = BTreeMap::new();
        want.insert(DeviceSlot::Gpu(0), mb(4096));
        assert!(can_fit(&want, &snapshot_with(20, 100), &BTreeMap::new(), None).is_ok());
    }

    #[test]
    fn no_fit_on_gpu() {
        let mut want = BTreeMap::new();
        want.insert(DeviceSlot::Gpu(0), mb(30 * 1024));
        let err = can_fit(&want, &snapshot_with(10, 100), &BTreeMap::new(), None).unwrap_err();
        assert_eq!(err.slot, DeviceSlot::Gpu(0));
    }

    #[test]
    fn reservations_subtract_from_available() {
        let mut want = BTreeMap::new();
        want.insert(DeviceSlot::Gpu(0), mb(10 * 1024)); // want 10 GB

        let mut other_alloc = BTreeMap::new();
        other_alloc.insert(DeviceSlot::Gpu(0), mb(15 * 1024)); // reserved 15 GB
        let mut reserved = BTreeMap::new();
        reserved.insert(SmolStr::new("other"), other_alloc);

        // Snapshot free = 20 GB. After subtracting 15 GB reserved, 5 GB available. Want 10 GB → no fit.
        let err = can_fit(&want, &snapshot_with(20, 100), &reserved, None).unwrap_err();
        assert_eq!(err.slot, DeviceSlot::Gpu(0));
    }

    #[test]
    fn exclude_skips_own_reservation() {
        let mut want = BTreeMap::new();
        want.insert(DeviceSlot::Gpu(0), mb(10 * 1024));

        let mut self_alloc = BTreeMap::new();
        self_alloc.insert(DeviceSlot::Gpu(0), mb(15 * 1024));
        let mut reserved = BTreeMap::new();
        reserved.insert(SmolStr::new("self"), self_alloc);

        // Self's 15 GB would eat the available; excluding self means 20 GB free, want 10 GB → fit.
        let ok = can_fit(&want, &snapshot_with(20, 100), &reserved, Some(&SmolStr::new("self")));
        assert!(ok.is_ok());
    }

    #[test]
    fn unknown_slot_is_no_fit_zero() {
        let mut want = BTreeMap::new();
        want.insert(DeviceSlot::Gpu(7), mb(1));
        let err = can_fit(&want, &snapshot_with(20, 100), &BTreeMap::new(), None).unwrap_err();
        assert_eq!(err.slot, DeviceSlot::Gpu(7));
        assert_eq!(err.available_bytes, 0);
    }
}
```

Add `pub mod allocator;` to `src/lib.rs`.

- [ ] **Step 2: Run tests**

Run: `cargo test --lib allocator`
Expected: 5 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/allocator.rs src/lib.rs
git commit -m "feat(allocator): pure can_fit feasibility check over device snapshot"
```

---

## Task 6: Config — parse filters and idle_timeout; drop on_demand gate

**Files:**
- Modify: `src/config/validate.rs`

The phase-1 validator currently rejects `lifecycle = "on_demand"` with a "phase 2" message and rejects services without `placement_override`. Phase 2 stops rejecting on_demand; it still requires `placement_override` (estimator is phase 3).

- [ ] **Step 1: Add tests for new validation behaviour**

Append to the `#[cfg(test)] mod tests` block at the bottom of `src/config/validate.rs`:

```rust
    #[test]
    fn phase2_accepts_on_demand() {
        let cfg = parse_and_merge(r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "on_demand"
devices.placement_override = { "gpu:0" = 1000 }
"#);
        let ec = validate(&cfg).unwrap();
        assert_eq!(ec.services[0].lifecycle, Lifecycle::OnDemand);
    }

    #[test]
    fn default_lifecycle_is_on_demand() {
        let cfg = parse_and_merge(r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
devices.placement_override = { "gpu:0" = 1000 }
"#);
        let ec = validate(&cfg).unwrap();
        assert_eq!(ec.services[0].lifecycle, Lifecycle::OnDemand);
    }

    #[test]
    fn parses_filters() {
        let cfg = parse_and_merge(r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }
filters.strip_params = ["temperature"]
filters.set_params = { max_tokens = 4096 }
"#);
        let ec = validate(&cfg).unwrap();
        let s = &ec.services[0];
        assert_eq!(s.filters.strip_params, vec!["temperature"]);
        assert!(s.filters.set_params.contains_key("max_tokens"));
    }

    #[test]
    fn parses_idle_timeout() {
        let cfg = parse_and_merge(r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "on_demand"
idle_timeout = "5m"
devices.placement_override = { "gpu:0" = 1000 }
"#);
        let ec = validate(&cfg).unwrap();
        assert_eq!(ec.services[0].idle_timeout_ms, 300_000);
    }
```

Remove the now-obsolete test `phase1_rejects_on_demand_with_clear_message` from the same block.

- [ ] **Step 2: Run tests to verify new ones fail**

Run: `cargo test --lib config::validate`
Expected: the three new tests FAIL (Lifecycle::OnDemand missing, filters field missing).

- [ ] **Step 3: Extend types + validator**

In `src/config/validate.rs`, change `Lifecycle`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lifecycle {
    Persistent,
    OnDemand,
}
```

Add a `Filters` struct:

```rust
#[derive(Debug, Clone, Default)]
pub struct Filters {
    pub strip_params: Vec<String>,
    pub set_params: std::collections::BTreeMap<String, serde_json::Value>,
}
```

Add to `ServiceConfig` (alongside existing fields):

```rust
pub filters: Filters,
```

In `validate()`, replace the current on_demand rejection:

```rust
let lifecycle = match lifecycle_str.as_str() {
    "persistent" => Lifecycle::Persistent,
    "on_demand" => Lifecycle::OnDemand,
    "oneshot" => return Err(fail(format!("service {name}: lifecycle `oneshot` is invalid in a [[service]] block (API-only)"))),
    other => return Err(fail(format!("service {name}: unknown lifecycle `{other}`"))),
};
```

And default when missing:

```rust
let lifecycle_str = raw.lifecycle.clone().unwrap_or_else(|| SmolStr::new("on_demand"));
```

Parse filters (add after the placement block, before constructing `ServiceConfig`):

```rust
let mut filters = Filters::default();
if let Some(raw_filters) = &raw.filters {
    if let Some(strip) = &raw_filters.strip_params {
        filters.strip_params = strip.clone();
    }
    if let Some(set) = &raw_filters.set_params {
        for (k, v) in set {
            let json_val = toml_value_to_json(v.clone()).map_err(|e| fail(format!("service {name} filters.set_params[{k}]: {e}")))?;
            filters.set_params.insert(k.clone(), json_val);
        }
    }
}
```

Add helper at module bottom:

```rust
fn toml_value_to_json(v: toml::Value) -> Result<serde_json::Value, String> {
    Ok(match v {
        toml::Value::String(s) => serde_json::Value::String(s),
        toml::Value::Integer(i) => serde_json::Value::Number(i.into()),
        toml::Value::Float(f) => serde_json::Number::from_f64(f).map(serde_json::Value::Number).ok_or_else(|| "non-finite float".to_string())?,
        toml::Value::Boolean(b) => serde_json::Value::Bool(b),
        toml::Value::Array(a) => serde_json::Value::Array(a.into_iter().map(toml_value_to_json).collect::<Result<_, _>>()?),
        toml::Value::Table(t) => {
            let mut m = serde_json::Map::new();
            for (k, v) in t {
                m.insert(k, toml_value_to_json(v)?);
            }
            serde_json::Value::Object(m)
        }
        toml::Value::Datetime(dt) => serde_json::Value::String(dt.to_string()),
    })
}
```

Add `filters` field to the `ServiceConfig` construction near the end of `validate`:

```rust
out.push(ServiceConfig {
    // ...existing fields...
    filters,
    raw: raw.clone(),
});
```

- [ ] **Step 4: Run tests**

Run: `cargo test --lib config::validate`
Expected: all tests pass (including the 3 new ones).

- [ ] **Step 5: Re-export `Filters` + `Lifecycle` update**

Update `src/config/mod.rs` re-exports to include `Filters`:

```rust
pub use validate::{
    Filters,
    validate, DaemonSettings, DeviceSlot, EffectiveConfig, HealthSettings, Lifecycle,
    PlacementPolicy, ServiceConfig, Template,
};
```

- [ ] **Step 6: Commit**

```bash
git add src/config/
git commit -m "feat(config): accept on_demand lifecycle and parse filters"
```

---

## Task 7: AppState

**Files:**
- Create: `src/app_state.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Create `src/app_state.rs`**

```rust
//! Shared application state passed to every Axum handler via `State(...)`.

use std::sync::Arc;

use parking_lot::Mutex;

use crate::activity::ActivityTable;
use crate::allocator::AllocationTable;
use crate::config::EffectiveConfig;
use crate::db::Database;
use crate::service_registry::ServiceRegistry;
use crate::snapshotter::SharedSnapshot;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<EffectiveConfig>,
    pub registry: ServiceRegistry,
    pub allocations: Arc<Mutex<AllocationTable>>,
    pub snapshot: SharedSnapshot,
    pub activity: ActivityTable,
    pub db: Database,
}
```

Add `pub mod app_state;` to `src/lib.rs`.

- [ ] **Step 2: Verify**

Run: `cargo check --lib`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add src/app_state.rs src/lib.rs
git commit -m "feat(app_state): shared Axum state struct"
```

---

## Task 8: Supervisor — Idle entry state + Ensure command + activity tracking

**Files:**
- Modify: `src/supervise/mod.rs`

This task adds the `Idle` lifecycle entry point, the `Ensure` command (with broadcast-based start coalescing), the `ActivityPing` command, and the idle-timer branch in the `Running` select!. The allocator integration lands in Task 10.

- [ ] **Step 1: Extend the command enum**

At the top of `src/supervise/mod.rs`, replace the existing `SupervisorCommand` enum with:

```rust
#[derive(Debug)]
pub enum SupervisorCommand {
    Shutdown {
        ack: tokio::sync::oneshot::Sender<()>,
    },
    Snapshot {
        ack: tokio::sync::oneshot::Sender<SupervisorSnapshot>,
    },
    /// Ensure the service is started (or starting). Returns a broadcast
    /// receiver the caller can await for the start outcome. If the
    /// start queue is full, returns `StartOutcome::QueueFull` via the
    /// single-shot `ack`.
    Ensure {
        ack: tokio::sync::oneshot::Sender<EnsureResponse>,
    },
    /// Record that a request was served; resets the idle timer.
    ActivityPing,
}

#[derive(Debug)]
pub enum EnsureResponse {
    /// Service is already running; proceed directly.
    AlreadyRunning,
    /// Service is idle/starting/warming; subscribe and wait.
    Waiting {
        rx: tokio::sync::broadcast::Receiver<StartOutcome>,
    },
    /// Start queue is full; reject with 503.
    QueueFull,
    /// Service is disabled or stopped; cannot start.
    Unavailable { reason: String },
}

#[derive(Debug, Clone)]
pub enum StartOutcome {
    Ok,
    Err(StartFailure),
}

#[derive(Debug, Clone)]
pub struct StartFailure {
    pub kind: StartFailureKind,
    pub message: String,
}

#[derive(Debug, Clone)]
pub enum StartFailureKind {
    NoFit,
    LaunchFailed,
    HealthTimeout,
    Disabled,
}
```

- [ ] **Step 2: Extend `SupervisorHandle` with `ensure` + `ping` methods**

```rust
impl SupervisorHandle {
    pub async fn ensure(&self) -> Option<EnsureResponse> {
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        self.tx.send(SupervisorCommand::Ensure { ack: ack_tx }).await.ok()?;
        ack_rx.await.ok()
    }

    pub fn ping(&self) {
        let _ = self.tx.try_send(SupervisorCommand::ActivityPing);
    }
}
```

- [ ] **Step 3: Extend the `run` loop**

This is the biggest change in phase 2. Add a new outer branch for `ServiceState::Idle` at the top of the match in `run`:

```rust
ServiceState::Idle => {
    // For on_demand services we wait here for an Ensure; for persistent
    // services the daemon calls `ensure()` synthetically at boot so this
    // path is exercised uniformly.
    let mut bus: Option<tokio::sync::broadcast::Sender<StartOutcome>> = None;
    loop {
        tokio::select! {
            cmd = rx.recv() => match cmd {
                Some(SupervisorCommand::Shutdown { ack }) => {
                    let _ = ack.send(());
                    return;
                }
                Some(SupervisorCommand::Snapshot { ack }) => {
                    let _ = ack.send(SupervisorSnapshot {
                        name: svc.name.clone(),
                        state: state.clone(),
                        run_id: None,
                        pid: None,
                    });
                }
                Some(SupervisorCommand::Ensure { ack }) => {
                    let sender = bus.get_or_insert_with(|| tokio::sync::broadcast::channel::<StartOutcome>(16).0);
                    if sender.receiver_count() >= svc.raw.start_queue_depth() {
                        let _ = ack.send(EnsureResponse::QueueFull);
                        continue;
                    }
                    let rx = sender.subscribe();
                    let _ = ack.send(EnsureResponse::Waiting { rx });
                    // Transition to Starting; carry the sender into that scope.
                    state = ServiceState::Starting;
                    *state_mirror.lock() = state.clone();
                    start_bus_carry = bus.take();
                    break;
                }
                Some(SupervisorCommand::ActivityPing) => {}
                None => return,
            }
        }
    }
}
```

Note: this introduces `start_bus_carry` as a function-local variable that `Starting` can consume. Declare it at the top of `run`:

```rust
let mut start_bus_carry: Option<tokio::sync::broadcast::Sender<StartOutcome>> = None;
```

In `Starting`, after a successful transition to `Running`, send `StartOutcome::Ok` through the carried bus and drop it:

```rust
// Right after the `WarmingComplete` transition and the running_services UPDATE:
if let Some(bus) = start_bus_carry.take() {
    let _ = bus.send(StartOutcome::Ok);
}
```

On failure paths (OOM, launch fail, no-fit, health timeout), emit the failure:

```rust
if let Some(bus) = start_bus_carry.take() {
    let _ = bus.send(StartOutcome::Err(StartFailure {
        kind: StartFailureKind::LaunchFailed,  // pick per path
        message: err_message.clone(),
    }));
}
```

Add new commands handling in the `Starting` inner select! (alongside existing `Shutdown` and `Snapshot`):

```rust
Some(SupervisorCommand::Ensure { ack }) => {
    let sender = start_bus_carry.as_ref();
    if let Some(sender) = sender {
        if sender.receiver_count() >= svc.raw.start_queue_depth() {
            let _ = ack.send(EnsureResponse::QueueFull);
        } else {
            let rx = sender.subscribe();
            let _ = ack.send(EnsureResponse::Waiting { rx });
        }
    } else {
        // Shouldn't happen in Starting; treat as AlreadyRunning best-effort.
        let _ = ack.send(EnsureResponse::AlreadyRunning);
    }
}
Some(SupervisorCommand::ActivityPing) => {}
```

In the `Running` inner select!, add an idle-timer branch and an ActivityPing branch:

```rust
// Added to the Running select alongside child.wait() and rx.recv():
_ = tokio::time::sleep_until(idle_deadline()) => {
    // Re-check the atomic in case a recent ping extended the deadline.
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
    if now + 100 < last_activity.load(std::sync::atomic::Ordering::Relaxed) + svc.idle_timeout_ms {
        continue; // someone pinged since the deadline was captured
    }
    info!(service = %svc.name, "idle timeout; draining to idle");
    // Drain the child without disabling the service.
    send_sigterm_and_wait(&mut child, Duration::from_secs(10)).await;
    let _ = db.with_conn(|c| c.execute(
        "DELETE FROM running_services WHERE service_id = ?1 AND run_id = ?2",
        (service_id, run_id),
    ));
    state = ServiceState::Idle;
    *state_mirror.lock() = state.clone();
    break;
}
```

Declare `last_activity` near the start of the supervisor (accepts an `Arc<AtomicU64>` passed in via `spawn_supervisor`):

```rust
// In spawn_supervisor signature:
pub fn spawn_supervisor(
    svc: ServiceConfig,
    allocation: Allocation,
    db: Database,
    batcher: BatcherHandle,
    service_id: i64,
    last_activity: Arc<std::sync::atomic::AtomicU64>,
) -> SupervisorHandle { /* ... */ }

// In run body, compute idle_deadline:
fn idle_deadline_for(last_activity: &Arc<std::sync::atomic::AtomicU64>, timeout_ms: u64) -> tokio::time::Instant {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
    let last = last_activity.load(std::sync::atomic::Ordering::Relaxed);
    let deadline_ms_from_now = (last + timeout_ms).saturating_sub(now);
    tokio::time::Instant::now() + Duration::from_millis(deadline_ms_from_now)
}
```

Use it inline: `_ = tokio::time::sleep_until(idle_deadline_for(&last_activity, svc.idle_timeout_ms)) => { ... }`.

Handle `ActivityPing` in the `Running` select!:

```rust
Some(SupervisorCommand::ActivityPing) => {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
    last_activity.store(now, std::sync::atomic::Ordering::Relaxed);
}
```

Also support `Ensure` in Running:

```rust
Some(SupervisorCommand::Ensure { ack }) => {
    let _ = ack.send(EnsureResponse::AlreadyRunning);
}
```

Support `Ensure` in Disabled (return Unavailable):

```rust
// Inside the Disabled arm:
Some(SupervisorCommand::Ensure { ack }) => {
    let _ = ack.send(EnsureResponse::Unavailable { reason: "service disabled".into() });
}
```

Add a helper on `RawService` (in `src/config/parse.rs`) to resolve `start_queue_depth` with a default of 10:

```rust
impl RawService {
    pub fn start_queue_depth(&self) -> usize {
        // placeholder — wire through validate if per-service override is needed.
        10
    }
}
```

- [ ] **Step 4: Compile + smoke**

Run: `cargo check --lib` — should pass.
Run: `cargo clippy --all-targets -- -D warnings` — should pass. Unused-variable warnings in the inner match arms are acceptable while downstream code is still stubbed; fix with `let _ =` bindings if clippy complains.

- [ ] **Step 5: Commit**

```bash
git add src/supervise/mod.rs src/config/parse.rs
git commit -m "feat(supervise): Idle entry state, Ensure coalescing, idle_timeout branch"
```

---

## Task 9: Supervisor — allocator integration at Idle → Starting

**Files:**
- Modify: `src/supervise/mod.rs`

- [ ] **Step 1: Thread the allocator inputs through `spawn_supervisor`**

Update `spawn_supervisor` signature to accept the snapshot + allocation table:

```rust
pub fn spawn_supervisor(
    svc: ServiceConfig,
    allocation: Allocation,
    db: Database,
    batcher: BatcherHandle,
    service_id: i64,
    last_activity: Arc<std::sync::atomic::AtomicU64>,
    snapshot: crate::snapshotter::SharedSnapshot,
    allocations: Arc<parking_lot::Mutex<crate::allocator::AllocationTable>>,
) -> SupervisorHandle { /* ... */ }
```

- [ ] **Step 2: Call `can_fit` at Idle → Starting**

Inside the `Idle` arm, before breaking to `Starting` on `Ensure`, run the allocator check. The plan's code snippet for the `Ensure` handler becomes:

```rust
Some(SupervisorCommand::Ensure { ack }) => {
    // Allocator feasibility check.
    let want = svc.placement_override.clone();
    let snap = snapshot.read().clone();
    let table = allocations.lock().clone();
    if let Err(nofit) = crate::allocator::can_fit(&want, &snap, &table, Some(&svc.name)) {
        let msg = format!("{nofit}");
        let _ = ack.send(EnsureResponse::Unavailable { reason: msg });
        continue;
    }

    // Reserve (write-lock allocation table).
    allocations.lock().insert(svc.name.clone(), want.clone());

    // Create bus and subscribe caller.
    let sender = tokio::sync::broadcast::channel::<StartOutcome>(16).0;
    let rx = sender.subscribe();
    let _ = ack.send(EnsureResponse::Waiting { rx });
    start_bus_carry = Some(sender);

    state = ServiceState::Starting;
    *state_mirror.lock() = state.clone();
    break;
}
```

- [ ] **Step 3: Release reservation on return to Idle / Disable**

Each path that transitions away from Running to Idle or Disabled should remove the entry from the allocation table:

```rust
// On idle_timeout drain:
allocations.lock().remove(&svc.name);

// On disable / failure:
allocations.lock().remove(&svc.name);
```

- [ ] **Step 4: Verify**

Run: `cargo check --lib` — must pass.

- [ ] **Step 5: Commit**

```bash
git add src/supervise/mod.rs
git commit -m "feat(supervise): allocator feasibility check at Idle -> Starting"
```

---

## Task 10: OpenAI API — error envelope + schema types

**Files:**
- Create: `src/openai_api/mod.rs`
- Create: `src/openai_api/errors.rs`
- Create: `src/openai_api/schema.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Create `src/openai_api/errors.rs`**

```rust
//! OpenAI-shaped error responses: `{error: {code, message, type}}`.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use utoipa::ToSchema;

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ErrorDetail {
    pub code: String,
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
}

pub fn err(status: StatusCode, code: &str, message: impl Into<String>, kind: &str) -> Response {
    let body = ErrorBody {
        error: ErrorDetail {
            code: code.into(),
            message: message.into(),
            kind: kind.into(),
        },
    };
    (status, Json(body)).into_response()
}

pub fn not_found_model(name: &str) -> Response {
    err(StatusCode::NOT_FOUND, "model_not_found",
        format!("model `{name}` not found"), "invalid_request_error")
}

pub fn service_disabled(name: &str, reason: &str) -> Response {
    err(StatusCode::SERVICE_UNAVAILABLE, "service_disabled",
        format!("service `{name}` is disabled: {reason}"), "server_error")
}

pub fn start_queue_full(name: &str) -> Response {
    err(StatusCode::SERVICE_UNAVAILABLE, "start_queue_full",
        format!("start queue full for service `{name}`"), "server_error")
}

pub fn start_failed(name: &str, detail: &str) -> Response {
    err(StatusCode::SERVICE_UNAVAILABLE, "start_failed",
        format!("service `{name}` failed to start: {detail}"), "server_error")
}

pub fn insufficient_vram(name: &str, detail: &str) -> Response {
    err(StatusCode::SERVICE_UNAVAILABLE, "insufficient_vram",
        format!("service `{name}` cannot fit: {detail}"), "server_error")
}

pub fn not_implemented(path: &str) -> Response {
    err(StatusCode::NOT_IMPLEMENTED, "not_implemented",
        format!("endpoint `{path}` is not implemented"), "invalid_request_error")
}

pub fn bad_request(msg: impl Into<String>) -> Response {
    err(StatusCode::BAD_REQUEST, "invalid_request_error",
        msg, "invalid_request_error")
}
```

- [ ] **Step 2: Create `src/openai_api/schema.rs`**

```rust
//! Request and response envelopes for the unified OpenAI listener.
//!
//! The daemon does not fully interpret OpenAI chat/completion/embedding
//! bodies; it extracts `model` and forwards the rest. Envelopes surface
//! only `model` in the OpenAPI schema, plus `#[serde(flatten)]` to
//! capture arbitrary other keys.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelListing {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelListing>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionEnvelope {
    pub model: String,
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CompletionEnvelope {
    pub model: String,
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct EmbeddingEnvelope {
    pub model: String,
    #[serde(flatten)]
    pub extra: serde_json::Value,
}
```

- [ ] **Step 3: Create `src/openai_api/mod.rs` as a router stub**

```rust
//! Unified OpenAI listener — Axum router factory.

pub mod errors;
pub mod filters;
pub mod handlers;
pub mod schema;
pub mod unimplemented;

use axum::Router;

use crate::app_state::AppState;

pub fn router(state: AppState) -> Router {
    handlers::register(Router::new(), state)
}
```

Create stubs for `src/openai_api/filters.rs`, `src/openai_api/handlers.rs`, `src/openai_api/unimplemented.rs` with doc comments only; they land in Tasks 11-13.

Add `pub mod openai_api;` to `src/lib.rs`.

- [ ] **Step 4: Verify compilation**

Run: `cargo check --lib` — expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/openai_api/ src/lib.rs
git commit -m "feat(openai_api): scaffold errors module, schema envelopes, router stub"
```

---

## Task 11: OpenAI API — filters

**Files:**
- Modify: `src/openai_api/filters.rs`

- [ ] **Step 1: Write the failing test + implementation**

Replace `src/openai_api/filters.rs`:

```rust
//! Apply `strip_params` and `set_params` to a JSON body.

use serde_json::Value;

use crate::config::Filters;

/// Apply filters in place. Strip first, then set.
pub fn apply(body: &mut Value, filters: &Filters) {
    if let Some(obj) = body.as_object_mut() {
        for key in &filters.strip_params {
            obj.remove(key);
        }
        for (key, value) in &filters.set_params {
            obj.insert(key.clone(), value.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::BTreeMap;

    fn filters_with(strip: &[&str], set: &[(&str, Value)]) -> Filters {
        let mut f = Filters::default();
        f.strip_params = strip.iter().map(|s| s.to_string()).collect();
        let mut m = BTreeMap::new();
        for (k, v) in set { m.insert(k.to_string(), v.clone()); }
        f.set_params = m;
        f
    }

    #[test]
    fn strips_keys() {
        let mut body = json!({"model":"m","temperature":0.7});
        apply(&mut body, &filters_with(&["temperature"], &[]));
        assert_eq!(body, json!({"model":"m"}));
    }

    #[test]
    fn sets_keys() {
        let mut body = json!({"model":"m"});
        apply(&mut body, &filters_with(&[], &[("max_tokens", json!(4096))]));
        assert_eq!(body["max_tokens"], json!(4096));
    }

    #[test]
    fn strip_then_set_order() {
        let mut body = json!({"model":"m","temperature":0.7});
        apply(&mut body, &filters_with(&["temperature"], &[("temperature", json!(0.3))]));
        assert_eq!(body["temperature"], json!(0.3));
    }

    #[test]
    fn no_object_is_noop() {
        let mut body = json!([1, 2, 3]);
        apply(&mut body, &filters_with(&["temperature"], &[]));
        assert_eq!(body, json!([1, 2, 3]));
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test --lib openai_api::filters`
Expected: 4 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/openai_api/filters.rs
git commit -m "feat(openai_api): strip + set JSON body filters"
```

---

## Task 12: OpenAI API — unimplemented 501 routes

**Files:**
- Modify: `src/openai_api/unimplemented.rs`

- [ ] **Step 1: Replace `src/openai_api/unimplemented.rs`**

```rust
//! 501 Not Implemented handlers for unsupported OpenAI endpoints.

use axum::extract::Path;
use axum::response::Response;
use axum::routing::{any, Router};

use crate::app_state::AppState;
use crate::openai_api::errors;

pub fn register(router: Router, _state: AppState) -> Router {
    router
        .route("/v1/audio/{*rest}", any(not_implemented))
        .route("/v1/images/{*rest}", any(not_implemented))
        .route("/v1/files/{*rest}", any(not_implemented))
        .route("/v1/fine_tuning/{*rest}", any(not_implemented))
        .route("/v1/batches", any(batches_not_implemented))
}

async fn not_implemented(Path(rest): Path<String>) -> Response {
    errors::not_implemented(&rest)
}

async fn batches_not_implemented() -> Response {
    errors::not_implemented("/v1/batches")
}
```

- [ ] **Step 2: Verify**

Run: `cargo check --lib` — expected: clean.

- [ ] **Step 3: Commit**

```bash
git add src/openai_api/unimplemented.rs
git commit -m "feat(openai_api): 501 handlers for audio/images/files/fine_tuning/batches"
```

---

## Task 13: OpenAI API — /v1/models, /v1/chat, /v1/completions, /v1/embeddings

**Files:**
- Modify: `src/openai_api/handlers.rs`
- Modify: `src/openai_api/mod.rs`

- [ ] **Step 1: Replace `src/openai_api/handlers.rs`**

```rust
//! Handlers for /v1/models and the three POST body-rewriting endpoints.

use std::time::Duration;

use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, Router};
use bytes::Bytes;
use futures::TryStreamExt;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::Frame;
use serde_json::Value;
use tokio::sync::broadcast;
use tracing::warn;

use crate::app_state::AppState;
use crate::openai_api::errors;
use crate::openai_api::filters;
use crate::openai_api::schema::{ModelListing, ModelsResponse};
use crate::state::ServiceState;
use crate::supervise::{EnsureResponse, StartFailureKind, StartOutcome};

pub fn register(router: Router, state: AppState) -> Router {
    router
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .merge(crate::openai_api::unimplemented::register(Router::new(), state.clone()))
        .with_state(state)
}

#[utoipa::path(get, path = "/v1/models", responses((status = 200, body = ModelsResponse)))]
pub async fn list_models(State(state): State<AppState>) -> Response {
    let mut data = Vec::new();
    for (name, handle) in state.registry.all() {
        let Some(snap) = handle.snapshot().await else { continue; };
        match snap.state {
            ServiceState::Idle
            | ServiceState::Starting
            | ServiceState::Warming
            | ServiceState::Running => {
                data.push(ModelListing {
                    id: name.to_string(),
                    object: "model",
                    created: 0,
                    owned_by: "ananke",
                });
            }
            _ => {}
        }
    }
    let body = ModelsResponse { object: "list", data };
    (StatusCode::OK, Json(body)).into_response()
}

#[utoipa::path(
    post,
    path = "/v1/chat/completions",
    responses((status = 200, description = "Proxied from upstream"))
)]
pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    forward_json_post("/v1/chat/completions", state, headers, body).await
}

#[utoipa::path(post, path = "/v1/completions", responses((status = 200)))]
pub async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    forward_json_post("/v1/completions", state, headers, body).await
}

#[utoipa::path(post, path = "/v1/embeddings", responses((status = 200)))]
pub async fn embeddings(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    forward_json_post("/v1/embeddings", state, headers, body).await
}

async fn forward_json_post(
    path: &'static str,
    state: AppState,
    headers: HeaderMap,
    body_bytes: Bytes,
) -> Response {
    let mut parsed: Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => return errors::bad_request(format!("invalid JSON body: {e}")),
    };
    let model = match parsed.get("model").and_then(|v| v.as_str()) {
        Some(m) => m.to_string(),
        None => return errors::bad_request("request body missing `model` field"),
    };

    let handle = match state.registry.get(&model) {
        Some(h) => h,
        None => return errors::not_found_model(&model),
    };

    let svc = state.config.services.iter().find(|s| s.name == model);
    let Some(svc) = svc else {
        return errors::not_found_model(&model);
    };

    // Ensure the service is running (coalescing concurrent first-requests).
    let mut ensure_rx: Option<broadcast::Receiver<StartOutcome>> = None;
    match handle.ensure().await {
        Some(EnsureResponse::AlreadyRunning) => {}
        Some(EnsureResponse::Waiting { rx }) => ensure_rx = Some(rx),
        Some(EnsureResponse::QueueFull) => return errors::start_queue_full(&model),
        Some(EnsureResponse::Unavailable { reason }) => {
            if reason.starts_with("no fit") {
                return errors::insufficient_vram(&model, &reason);
            }
            return errors::service_disabled(&model, &reason);
        }
        None => return errors::start_failed(&model, "supervisor unreachable"),
    }

    if let Some(mut rx) = ensure_rx {
        let timeout = Duration::from_millis(svc.max_request_duration_ms);
        match tokio::time::timeout(timeout, rx.recv()).await {
            Ok(Ok(StartOutcome::Ok)) => {}
            Ok(Ok(StartOutcome::Err(f))) => {
                return match f.kind {
                    StartFailureKind::NoFit => errors::insufficient_vram(&model, &f.message),
                    StartFailureKind::HealthTimeout => errors::start_failed(&model, "health check timed out"),
                    StartFailureKind::Disabled => errors::service_disabled(&model, &f.message),
                    StartFailureKind::LaunchFailed => errors::start_failed(&model, &f.message),
                };
            }
            Ok(Err(e)) => return errors::start_failed(&model, &format!("start broadcast closed: {e}")),
            Err(_) => return errors::start_failed(&model, "start timed out"),
        }
    }

    // Apply filters.
    filters::apply(&mut parsed, &svc.filters);
    let new_body = match serde_json::to_vec(&parsed) {
        Ok(b) => b,
        Err(e) => return errors::bad_request(format!("re-serialise failed: {e}")),
    };

    // Bump activity.
    state.activity.ping(&svc.name);

    // Forward.
    let client = hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
        .build_http::<http_body_util::combinators::BoxBody<Bytes, Box<dyn std::error::Error + Send + Sync>>>();
    let uri = format!("http://127.0.0.1:{}{}", svc.private_port, path).parse::<hyper::Uri>().unwrap();
    let mut req = hyper::Request::builder().method("POST").uri(uri);
    for (k, v) in headers.iter() {
        if k == hyper::header::HOST || k == hyper::header::CONTENT_LENGTH { continue; }
        req = req.header(k, v);
    }
    req = req.header(hyper::header::CONTENT_TYPE, "application/json");
    req = req.header(hyper::header::CONTENT_LENGTH, new_body.len());
    let upstream_body = http_body_util::Full::new(Bytes::from(new_body))
        .map_err(|never| match never {})
        .boxed();
    let req = match req.body(upstream_body) {
        Ok(r) => r,
        Err(e) => return errors::bad_request(format!("build request: {e}")),
    };

    let resp = match client.request(req).await {
        Ok(r) => r,
        Err(e) => {
            warn!(error = %e, model = %model, "upstream request failed");
            return errors::start_failed(&model, "upstream unavailable");
        }
    };

    let (parts, upstream_body) = resp.into_parts();
    let stream = upstream_body.into_data_stream().map_ok(Frame::data);
    let boxed = StreamBody::new(stream).map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) });
    let axum_body = Body::new(boxed);
    let mut out = Response::from_parts(parts, axum_body);
    out.headers_mut().remove(hyper::header::CONNECTION);
    out.headers_mut().remove("transfer-encoding");
    out
}
```

- [ ] **Step 2: Add `Filters` to re-exports if missing**

Ensure `src/config/mod.rs` exports `Filters` (done in Task 6).

- [ ] **Step 3: Verify compilation**

Run: `cargo check --lib` — expected: clean. Warnings about unused imports OK.

- [ ] **Step 4: Commit**

```bash
git add src/openai_api/
git commit -m "feat(openai_api): /v1/models, chat, completions, embeddings with Ensure + filters"
```

---

## Task 14: Management API — handlers and types

**Files:**
- Create: `src/management_api/mod.rs`
- Create: `src/management_api/handlers.rs`
- Create: `src/management_api/types.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Create types**

Create `src/management_api/types.rs`:

```rust
//! Response shapes for read-only management endpoints.

use serde::Serialize;
use utoipa::ToSchema;

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ServiceSummary {
    pub name: String,
    pub state: String,
    pub lifecycle: String,
    pub priority: u8,
    pub port: u16,
    pub run_id: Option<i64>,
    pub pid: Option<i32>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ServiceDetail {
    pub name: String,
    pub state: String,
    pub lifecycle: String,
    pub priority: u8,
    pub port: u16,
    pub private_port: u16,
    pub template: String,
    pub placement_override: std::collections::BTreeMap<String, u64>,
    pub idle_timeout_ms: u64,
    pub run_id: Option<i64>,
    pub pid: Option<i32>,
    pub recent_logs: Vec<LogLine>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct LogLine {
    pub timestamp_ms: i64,
    pub stream: String,
    pub line: String,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct DeviceSummary {
    pub id: String,
    pub name: String,
    pub total_bytes: u64,
    pub free_bytes: u64,
    pub reservations: Vec<DeviceReservation>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct DeviceReservation {
    pub service: String,
    pub bytes: u64,
}
```

- [ ] **Step 2: Create handlers**

Create `src/management_api/handlers.rs`:

```rust
//! Read-only management endpoints.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, Router};

use crate::app_state::AppState;
use crate::management_api::types::{
    DeviceReservation, DeviceSummary, LogLine, ServiceDetail, ServiceSummary,
};
use crate::state::ServiceState;

pub fn register(router: Router, state: AppState) -> Router {
    router
        .route("/api/services", get(list_services))
        .route("/api/services/{name}", get(service_detail))
        .route("/api/devices", get(list_devices))
        .with_state(state)
}

#[utoipa::path(get, path = "/api/services", responses((status = 200, body = Vec<ServiceSummary>)))]
pub async fn list_services(State(state): State<AppState>) -> Response {
    let mut out = Vec::new();
    for svc_cfg in state.config.services.iter() {
        let handle = state.registry.get(&svc_cfg.name);
        let snap = match &handle { Some(h) => h.snapshot().await, None => None };
        out.push(ServiceSummary {
            name: svc_cfg.name.to_string(),
            state: snap.as_ref().map(|s| state_name(&s.state)).unwrap_or_else(|| "unknown".into()),
            lifecycle: format!("{:?}", svc_cfg.lifecycle).to_lowercase(),
            priority: svc_cfg.priority,
            port: svc_cfg.port,
            run_id: snap.as_ref().and_then(|s| s.run_id),
            pid: snap.as_ref().and_then(|s| s.pid),
        });
    }
    (StatusCode::OK, Json(out)).into_response()
}

#[utoipa::path(get, path = "/api/services/{name}", responses((status = 200, body = ServiceDetail), (status = 404)))]
pub async fn service_detail(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Response {
    let Some(svc_cfg) = state.config.services.iter().find(|s| s.name == name) else {
        return (StatusCode::NOT_FOUND, Json(serde_json::json!({"error":"not found"}))).into_response();
    };
    let handle = state.registry.get(&svc_cfg.name);
    let snap = match &handle { Some(h) => h.snapshot().await, None => None };
    let placement_override: std::collections::BTreeMap<String, u64> = svc_cfg.placement_override.iter().map(|(k, v)| {
        let key = match k {
            crate::config::DeviceSlot::Cpu => "cpu".to_string(),
            crate::config::DeviceSlot::Gpu(n) => format!("gpu:{n}"),
        };
        (key, *v)
    }).collect();

    let svc_id_opt: Option<i64> = state.db.with_conn(|c| {
        c.query_row("SELECT service_id FROM services WHERE name = ?1", [&name], |r| r.get(0))
    }).ok();

    let recent_logs: Vec<LogLine> = match svc_id_opt {
        Some(svc_id) => state.db.with_conn(|c| {
            let mut stmt = c.prepare(
                "SELECT timestamp_ms, stream, line FROM service_logs
                 WHERE service_id = ?1 ORDER BY timestamp_ms DESC, seq DESC LIMIT 200"
            )?;
            let rows = stmt.query_map([svc_id], |r| Ok(LogLine {
                timestamp_ms: r.get(0)?,
                stream: r.get(1)?,
                line: r.get(2)?,
            }))?;
            Ok(rows.collect::<Result<Vec<_>, _>>()?)
        }).unwrap_or_default(),
        None => Vec::new(),
    };

    let detail = ServiceDetail {
        name: svc_cfg.name.to_string(),
        state: snap.as_ref().map(|s| state_name(&s.state)).unwrap_or_else(|| "unknown".into()),
        lifecycle: format!("{:?}", svc_cfg.lifecycle).to_lowercase(),
        priority: svc_cfg.priority,
        port: svc_cfg.port,
        private_port: svc_cfg.private_port,
        template: format!("{:?}", svc_cfg.template).to_lowercase(),
        placement_override,
        idle_timeout_ms: svc_cfg.idle_timeout_ms,
        run_id: snap.as_ref().and_then(|s| s.run_id),
        pid: snap.as_ref().and_then(|s| s.pid),
        recent_logs,
    };
    (StatusCode::OK, Json(detail)).into_response()
}

#[utoipa::path(get, path = "/api/devices", responses((status = 200, body = Vec<DeviceSummary>)))]
pub async fn list_devices(State(state): State<AppState>) -> Response {
    let snap = state.snapshot.read().clone();
    let alloc = state.allocations.lock().clone();

    let mut out = Vec::new();

    for g in &snap.gpus {
        let slot = crate::config::DeviceSlot::Gpu(g.id);
        let reservations: Vec<DeviceReservation> = alloc.iter().filter_map(|(svc, a)| {
            a.get(&slot).map(|mb| DeviceReservation { service: svc.to_string(), bytes: mb * 1024 * 1024 })
        }).collect();
        out.push(DeviceSummary {
            id: format!("gpu:{}", g.id),
            name: g.name.clone(),
            total_bytes: g.total_bytes,
            free_bytes: g.free_bytes,
            reservations,
        });
    }

    if let Some(c) = &snap.cpu {
        let reservations: Vec<DeviceReservation> = alloc.iter().filter_map(|(svc, a)| {
            a.get(&crate::config::DeviceSlot::Cpu).map(|mb| DeviceReservation { service: svc.to_string(), bytes: mb * 1024 * 1024 })
        }).collect();
        out.push(DeviceSummary {
            id: "cpu".into(),
            name: "CPU".into(),
            total_bytes: c.total_bytes,
            free_bytes: c.available_bytes,
            reservations,
        });
    }

    (StatusCode::OK, Json(out)).into_response()
}

fn state_name(s: &ServiceState) -> String {
    match s {
        ServiceState::Idle => "idle",
        ServiceState::Starting => "starting",
        ServiceState::Warming => "warming",
        ServiceState::Running => "running",
        ServiceState::Draining => "draining",
        ServiceState::Stopped => "stopped",
        ServiceState::Evicted => "evicted",
        ServiceState::Failed { .. } => "failed",
        ServiceState::Disabled { .. } => "disabled",
    }.to_string()
}
```

- [ ] **Step 3: Create mod**

Create `src/management_api/mod.rs`:

```rust
//! Read-only management API — `/api/services`, `/api/devices`, and
//! `/api/openapi.json`.

pub mod handlers;
pub mod types;

use axum::Router;

use crate::app_state::AppState;

pub fn router(state: AppState) -> Router {
    handlers::register(Router::new(), state)
}
```

Add `pub mod management_api;` to `src/lib.rs`.

- [ ] **Step 4: Verify**

Run: `cargo check --lib` — clean.

- [ ] **Step 5: Commit**

```bash
git add src/management_api/ src/lib.rs
git commit -m "feat(management_api): /api/services, /api/services/{name}, /api/devices"
```

---

## Task 15: OpenAPI aggregator

**Files:**
- Create: `src/openapi.rs`
- Modify: `src/management_api/mod.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Create `src/openapi.rs`**

```rust
//! Aggregated OpenAPI document for the daemon.

use axum::Json;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, Router};
use utoipa::OpenApi;

use crate::app_state::AppState;
use crate::management_api::handlers as mgmt_handlers;
use crate::management_api::types as mgmt_types;
use crate::openai_api::errors as openai_errors;
use crate::openai_api::handlers as openai_handlers;
use crate::openai_api::schema as openai_schema;

#[derive(OpenApi)]
#[openapi(
    paths(
        openai_handlers::list_models,
        openai_handlers::chat_completions,
        openai_handlers::completions,
        openai_handlers::embeddings,
        mgmt_handlers::list_services,
        mgmt_handlers::service_detail,
        mgmt_handlers::list_devices,
    ),
    components(schemas(
        openai_schema::ModelListing,
        openai_schema::ModelsResponse,
        openai_schema::ChatCompletionEnvelope,
        openai_schema::CompletionEnvelope,
        openai_schema::EmbeddingEnvelope,
        openai_errors::ErrorBody,
        openai_errors::ErrorDetail,
        mgmt_types::ServiceSummary,
        mgmt_types::ServiceDetail,
        mgmt_types::LogLine,
        mgmt_types::DeviceSummary,
        mgmt_types::DeviceReservation,
    )),
    info(title = "Ananke API", version = "0.1.0"),
)]
pub struct AnankeApi;

pub fn register(router: Router, state: AppState) -> Router {
    router.route("/api/openapi.json", get(serve_openapi)).with_state(state)
}

pub async fn serve_openapi(State(_state): State<AppState>) -> Response {
    (axum::http::StatusCode::OK, Json(AnankeApi::openapi())).into_response()
}
```

Modify `src/management_api/mod.rs` to merge the openapi route:

```rust
pub fn router(state: AppState) -> Router {
    handlers::register(Router::new(), state.clone())
        .merge(crate::openapi::register(Router::new(), state))
}
```

Add `pub mod openapi;` to `src/lib.rs`.

- [ ] **Step 2: Verify**

Run: `cargo check --lib` — clean.

- [ ] **Step 3: Commit**

```bash
git add src/openapi.rs src/management_api/mod.rs src/lib.rs
git commit -m "feat(openapi): aggregator + /api/openapi.json"
```

---

## Task 16: Daemon — wire up AppState, snapshotter, Axum servers

**Files:**
- Modify: `src/daemon.rs`

- [ ] **Step 1: Replace `src/daemon.rs` `run()` body**

Replace the existing wiring with this updated version that builds `AppState`, spawns the snapshotter, both Axum servers, and adapts to the new `spawn_supervisor` signature:

```rust
//! Top-level daemon orchestration (phase 2).

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Duration;

use parking_lot::Mutex;
use tokio::net::TcpListener;
use tokio::sync::watch;
use tracing::{error, info, warn};

use crate::activity::ActivityTable;
use crate::allocator::AllocationTable;
use crate::app_state::AppState;
use crate::config::{load_config, Lifecycle, Migration};
use crate::db::logs::spawn as spawn_batcher;
use crate::db::Database;
use crate::devices::{GpuProbe, cpu, nvml::NvmlProbe};
use crate::errors::ExpectedError;
use crate::proxy;
use crate::retention;
use crate::service_registry::ServiceRegistry;
use crate::signals::{await_shutdown, ShutdownKind};
use crate::snapshotter;
use crate::supervise::{SupervisorHandle, spawn_supervisor};
use crate::supervise::orphans::reconcile;

pub async fn run() -> Result<(), ExpectedError> {
    init_tracing();

    let cli_config = parse_cli_config_arg();
    let config_path = crate::config::resolve_from_env(cli_config.as_deref())?;
    info!(config_path = %config_path.display(), "resolved config path");

    let (effective, migrations) = load_config(&config_path)?;
    let effective = Arc::new(effective);
    let db = Database::open(&effective.daemon.data_dir.join("ananke.sqlite"))?;
    apply_migrations(&db, &migrations);

    let probe: Option<Arc<dyn GpuProbe>> = match NvmlProbe::init() {
        Ok(p) => {
            for g in p.list() {
                info!(gpu = g.id, name = %g.name, total_bytes = g.total_bytes, "detected GPU");
            }
            Some(Arc::new(p))
        }
        Err(e) => {
            warn!(error = %e, "NVML init failed; falling back to CPU-only");
            None
        }
    };
    if let Ok(m) = cpu::read() {
        info!(total = m.total_bytes, avail = m.available_bytes, "CPU memory");
    }

    for disposition in reconcile(&db, &PathBuf::from("/proc")) {
        info!(?disposition, "orphan reconcile");
    }

    let batcher = spawn_batcher(db.clone());
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let shared_snapshot = snapshotter::new_shared();
    let snapshotter_join = snapshotter::spawn(shared_snapshot.clone(), probe.clone(), shutdown_rx.clone());

    let activity = ActivityTable::new();
    let allocations = Arc::new(Mutex::new(AllocationTable::new()));
    let registry = ServiceRegistry::new();

    // Persistent services start in priority-desc + name-asc order;
    // on_demand services are registered but remain idle.
    let mut ordered = effective.services.clone();
    ordered.sort_by(|a, b| b.priority.cmp(&a.priority).then_with(|| a.name.cmp(&b.name)));

    let mut supervisors: Vec<Arc<SupervisorHandle>> = Vec::new();
    let mut proxy_tasks = Vec::new();
    for svc in ordered {
        let service_id = db.upsert_service(&svc.name, now_ms())?;
        let allocation = crate::devices::Allocation::from_override(&svc.placement_override);
        let last_activity = activity.get_or_init(&svc.name);
        let handle = Arc::new(spawn_supervisor(
            svc.clone(),
            allocation,
            db.clone(),
            batcher.clone(),
            service_id,
            last_activity,
            shared_snapshot.clone(),
            allocations.clone(),
        ));
        registry.insert(svc.name.clone(), handle.clone());

        // Persistent services kick-start via an implicit Ensure.
        if matches!(svc.lifecycle, Lifecycle::Persistent) {
            let handle2 = handle.clone();
            tokio::spawn(async move {
                let _ = handle2.ensure().await;
            });
        }

        let listen: SocketAddr = format!("127.0.0.1:{}", svc.port).parse()
            .map_err(|e: std::net::AddrParseError| ExpectedError::bind_failed(format!("127.0.0.1:{}", svc.port), e.to_string()))?;
        let shutdown_rx2 = shutdown_rx.clone();
        let upstream = svc.private_port;
        let name = svc.name.clone();
        let activity_for_proxy = activity.clone();
        proxy_tasks.push(tokio::spawn(async move {
            // Per-service proxy also pings activity.
            let name_ping = name.clone();
            let ping_cb = move || activity_for_proxy.ping(&name_ping);
            if let Err(e) = proxy::serve_with_activity(listen, upstream, shutdown_rx2, ping_cb).await {
                error!(service = %name, error = %e, "proxy failed");
            }
        }));
        supervisors.push(handle);
    }

    // Build AppState for the routers.
    let app_state = AppState {
        config: effective.clone(),
        registry: registry.clone(),
        allocations: allocations.clone(),
        snapshot: shared_snapshot.clone(),
        activity: activity.clone(),
        db: db.clone(),
    };

    // OpenAI listener.
    let openai_listen: SocketAddr = effective.daemon.openai_listen.parse()
        .map_err(|e: std::net::AddrParseError| ExpectedError::bind_failed(effective.daemon.openai_listen.clone(), e.to_string()))?;
    let openai_router = crate::openai_api::router(app_state.clone());
    let openai_listener = TcpListener::bind(openai_listen).await
        .map_err(|e| ExpectedError::bind_failed(openai_listen.to_string(), e.to_string()))?;
    let openai_shutdown = shutdown_rx.clone();
    let openai_server = tokio::spawn(async move {
        let _ = axum::serve(openai_listener, openai_router)
            .with_graceful_shutdown(wait_shutdown(openai_shutdown))
            .await;
    });
    info!(%openai_listen, "openai listener bound");

    // Management listener.
    let mgmt_listen: SocketAddr = effective.daemon.management_listen.parse()
        .map_err(|e: std::net::AddrParseError| ExpectedError::bind_failed(effective.daemon.management_listen.clone(), e.to_string()))?;
    let mgmt_router = crate::management_api::router(app_state.clone());
    let mgmt_listener = TcpListener::bind(mgmt_listen).await
        .map_err(|e| ExpectedError::bind_failed(mgmt_listen.to_string(), e.to_string()))?;
    let mgmt_shutdown = shutdown_rx.clone();
    let mgmt_server = tokio::spawn(async move {
        let _ = axum::serve(mgmt_listener, mgmt_router)
            .with_graceful_shutdown(wait_shutdown(mgmt_shutdown))
            .await;
    });
    info!(%mgmt_listen, "management listener bound");

    let retention_task = tokio::spawn(retention::run_loop(db.clone(), shutdown_rx.clone()));

    let shutdown_kind = await_shutdown().await;
    info!(?shutdown_kind, "shutdown initiated");
    let _ = shutdown_tx.send(true);

    let drain_bound = match shutdown_kind {
        ShutdownKind::Graceful => Duration::from_millis(effective.daemon.shutdown_timeout_ms),
        ShutdownKind::Emergency => Duration::from_secs(5),
    };
    let _ = tokio::time::timeout(drain_bound, async {
        for sup in &supervisors { sup.shutdown().await; }
    }).await;

    for t in proxy_tasks { t.abort(); let _ = t.await; }
    openai_server.abort();
    let _ = openai_server.await;
    mgmt_server.abort();
    let _ = mgmt_server.await;
    snapshotter_join.abort();
    let _ = snapshotter_join.await;
    retention_task.abort();
    let _ = retention_task.await;

    batcher.flush().await;
    Ok(())
}

async fn wait_shutdown(mut rx: watch::Receiver<bool>) {
    while rx.changed().await.is_ok() {
        if *rx.borrow() { return; }
    }
}

fn parse_cli_config_arg() -> Option<PathBuf> {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--config" { return args.next().map(PathBuf::from); }
        if let Some(rest) = a.strip_prefix("--config=") { return Some(PathBuf::from(rest)); }
    }
    None
}

fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as i64
}

fn apply_migrations(db: &Database, migs: &[Migration]) {
    let now = now_ms();
    for m in migs {
        if let Err(e) = db.reparent(&m.old_name, &m.new_name, now) {
            warn!(old = %m.old_name, new = %m.new_name, error = %e, "migrate_from failed");
        } else {
            info!(old = %m.old_name, new = %m.new_name, "migrated service");
        }
    }
}

fn init_tracing() {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(true)
        .with_writer(std::io::stderr)
        .init();
}
```

- [ ] **Step 2: Extend `DaemonSettings` to include `openai_listen`**

In `src/config/validate.rs`, add:

```rust
pub struct DaemonSettings {
    pub management_listen: String,
    pub openai_listen: String,       // NEW
    pub data_dir: PathBuf,
    pub shutdown_timeout_ms: u64,
}
```

Populate it in `validate()`:

```rust
let openai_listen = cfg.openai_api.listen.clone().unwrap_or_else(|| "127.0.0.1:8080".into());

// ...

DaemonSettings {
    management_listen: management_addr,
    openai_listen,
    data_dir,
    shutdown_timeout_ms,
}
```

- [ ] **Step 3: Add `proxy::serve_with_activity`**

In `src/proxy.rs`, add a variant that takes a ping callback:

```rust
pub async fn serve_with_activity<F: Fn() + Send + Sync + 'static>(
    listen: SocketAddr,
    upstream_port: u16,
    mut shutdown: watch::Receiver<bool>,
    on_request: F,
) -> Result<(), ExpectedError> {
    // identical to `serve` except the per-request service closure
    // calls `on_request()` before delegating to `handle(...)`.
    // Copy the body of `serve` and insert `on_request_clone();` at
    // the top of the `service_fn` closure.
    let on_request = std::sync::Arc::new(on_request);
    let listener = tokio::net::TcpListener::bind(listen).await
        .map_err(|e| ExpectedError::bind_failed(listen.to_string(), e.to_string()))?;
    let client = hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
        .build_http::<http_body_util::combinators::BoxBody<bytes::Bytes, Box<dyn std::error::Error + Send + Sync>>>();

    loop {
        tokio::select! {
            _ = shutdown.changed() => {
                if *shutdown.borrow() { return Ok(()); }
            }
            accept = listener.accept() => {
                let (stream, peer) = match accept {
                    Ok(x) => x,
                    Err(e) => { tracing::warn!(error = %e, "accept failed"); continue; }
                };
                let io = hyper_util::rt::TokioIo::new(stream);
                let client = client.clone();
                let on_request = on_request.clone();
                tokio::spawn(async move {
                    let svc = hyper::service::service_fn(move |req| {
                        (on_request)();
                        let client = client.clone();
                        handle(req, client, upstream_port, peer)
                    });
                    if let Err(e) = hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new())
                        .serve_connection(io, svc).await
                    { tracing::warn!(error = %e, "conn error"); }
                });
            }
        }
    }
}
```

Make `handle` / `try_handle` from phase 1 visible at `pub(crate)` if they aren't already.

- [ ] **Step 4: Run tests**

Run: `cargo test --workspace` — existing tests should still pass.
Run: `cargo clippy --all-targets -- -D warnings`.

- [ ] **Step 5: Commit**

```bash
git add src/daemon.rs src/proxy.rs src/config/validate.rs
git commit -m "feat(daemon): wire AppState, snapshotter, OpenAI + management Axum servers"
```

---

## Task 17: Echo server extensions — spawn counter + /sink

**Files:**
- Modify: `tests/common/echo_server.rs`
- Modify: `tests/common/mod.rs`

- [ ] **Step 1: Add spawn counter + /sink**

Replace `tests/common/echo_server.rs`:

```rust
//! Phase 2 echo server: adds spawn counter, /sink, and configurable /v1/* bodies.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use bytes::Bytes;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::Frame;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto;
use parking_lot::Mutex;
use tokio::net::TcpListener;
use tokio::sync::watch;

#[derive(Clone, Default)]
pub struct EchoState {
    pub spawn_counter: Arc<AtomicU32>,
    pub sink: Arc<Mutex<Vec<serde_json::Value>>>,
}

pub async fn serve(addr: SocketAddr, state: EchoState, mut shutdown: watch::Receiver<bool>) {
    state.spawn_counter.fetch_add(1, Ordering::Relaxed);
    let listener = TcpListener::bind(addr).await.expect("echo bind");
    loop {
        tokio::select! {
            _ = shutdown.changed() => { if *shutdown.borrow() { return; } }
            accept = listener.accept() => {
                let Ok((stream, _)) = accept else { continue; };
                let io = TokioIo::new(stream);
                let state = state.clone();
                tokio::spawn(async move {
                    let svc = service_fn(move |req| handle(req, state.clone()));
                    let _ = auto::Builder::new(TokioExecutor::new()).serve_connection(io, svc).await;
                });
            }
        }
    }
}

type EchoBody = http_body_util::combinators::BoxBody<Bytes, Box<dyn std::error::Error + Send + Sync>>;

async fn handle(req: Request<hyper::body::Incoming>, state: EchoState) -> Result<Response<EchoBody>, Infallible> {
    match (req.method().clone(), req.uri().path().to_string()) {
        (_, p) if p == "/health" || p == "/v1/models" => {
            let body = Full::new(Bytes::from("{}")).map_err(|n| match n {}).boxed();
            Ok(Response::builder()
                .status(StatusCode::OK)
                .header("x-echo-spawn-count", state.spawn_counter.load(Ordering::Relaxed).to_string())
                .body(body).unwrap())
        }
        (_, p) if p == "/sse" => {
            let (tx, rx) = tokio::sync::mpsc::channel::<Result<Frame<Bytes>, Box<dyn std::error::Error + Send + Sync>>>(8);
            tokio::spawn(async move {
                for i in 0..5 {
                    let chunk = format!("data: {i}\n\n");
                    if tx.send(Ok(Frame::data(Bytes::from(chunk)))).await.is_err() { break; }
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            });
            let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
            let body = StreamBody::new(stream).boxed();
            Ok(Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "text/event-stream")
                .body(body).unwrap())
        }
        (_, p) if p == "/v1/chat/completions" || p == "/v1/completions" || p == "/v1/embeddings" => {
            let body_bytes = req.into_body().collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&body_bytes) {
                state.sink.lock().push(v);
            }
            let body = Full::new(Bytes::from(r#"{"id":"cmpl-echo","choices":[{"message":{"role":"assistant","content":"ok"}}]}"#))
                .map_err(|n| match n {}).boxed();
            Ok(Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/json")
                .body(body).unwrap())
        }
        _ => {
            let body = Full::new(Bytes::from("hello")).map_err(|n| match n {}).boxed();
            Ok(Response::builder().status(StatusCode::OK).body(body).unwrap())
        }
    }
}
```

Update `tests/common/mod.rs` if needed to re-export `EchoState`.

- [ ] **Step 2: Build tests**

Run: `cargo test --no-run` — builds.

- [ ] **Step 3: Commit**

```bash
git add tests/common/
git commit -m "test: echo server extensions (spawn counter, /sink, /v1/*)"
```

---

## Task 18: Integration tests — OpenAI endpoints

**Files:**
- Create: `tests/openai_models.rs`
- Create: `tests/openai_chat_routing.rs`
- Create: `tests/openai_unimplemented.rs`
- Create: `tests/openapi_json.rs`

Each test file starts with `mod common;`. These tests construct an `AppState` against an in-memory `ServiceRegistry` where supervisors are backed by echo-server processes — i.e., we run the real OpenAI router but the "child" is the echo server instead of llama-server.

Since running a full daemon is heavyweight, tests build the router directly and drive it via `axum::Router::into_make_service` + an in-process client.

Full setup helper (append to `tests/common/mod.rs`):

```rust
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use ananke::activity::ActivityTable;
use ananke::allocator::AllocationTable;
use ananke::app_state::AppState;
use ananke::config::{DaemonSettings, DeviceSlot, EffectiveConfig, Filters, HealthSettings, Lifecycle, PlacementPolicy, ServiceConfig, Template};
use ananke::config::parse::RawService;
use ananke::db::Database;
use ananke::db::logs::spawn as spawn_batcher;
use ananke::devices::Allocation;
use ananke::service_registry::ServiceRegistry;
use ananke::snapshotter;
use ananke::supervise::{spawn_supervisor, SupervisorHandle};
use parking_lot::Mutex;
use smol_str::SmolStr;
use tempfile::TempDir;

pub struct TestHarness {
    pub state: AppState,
    pub echo_state: echo_server::EchoState,
    pub echo_addr: std::net::SocketAddr,
    pub echo_shutdown: tokio::sync::watch::Sender<bool>,
    pub supervisors: Vec<Arc<SupervisorHandle>>,
    pub _tmp: TempDir,
}

pub async fn build_harness(services: Vec<ServiceConfig>) -> TestHarness {
    let tmp = tempfile::tempdir().unwrap();
    let db = Database::open(&tmp.path().join("a.sqlite")).unwrap();
    let batcher = spawn_batcher(db.clone());

    let echo_state = echo_server::EchoState::default();
    let echo_port = free_port();
    let echo_addr: std::net::SocketAddr = format!("127.0.0.1:{echo_port}").parse().unwrap();
    let (echo_shutdown, echo_rx) = tokio::sync::watch::channel(false);
    tokio::spawn(echo_server::serve(echo_addr, echo_state.clone(), echo_rx));

    // Rewrite private_port on each service to point at the echo server.
    let services_rewritten: Vec<ServiceConfig> = services.into_iter().map(|mut s| {
        s.private_port = echo_port;
        s
    }).collect();

    let effective = Arc::new(EffectiveConfig {
        daemon: DaemonSettings {
            management_listen: "127.0.0.1:0".into(),
            openai_listen: "127.0.0.1:0".into(),
            data_dir: tmp.path().to_path_buf(),
            shutdown_timeout_ms: 5_000,
        },
        services: services_rewritten.clone(),
    });

    let activity = ActivityTable::new();
    let allocations = Arc::new(Mutex::new(AllocationTable::new()));
    let snapshot = snapshotter::new_shared();

    let registry = ServiceRegistry::new();
    let mut supervisors = Vec::new();
    for svc in &services_rewritten {
        let service_id = db.upsert_service(&svc.name, 0).unwrap();
        let alloc = Allocation::from_override(&svc.placement_override);
        let last_activity = activity.get_or_init(&svc.name);
        let handle = Arc::new(spawn_supervisor(
            svc.clone(), alloc, db.clone(), batcher.clone(), service_id,
            last_activity, snapshot.clone(), allocations.clone(),
        ));
        registry.insert(svc.name.clone(), handle.clone());
        supervisors.push(handle);
    }

    let state = AppState {
        config: effective,
        registry,
        allocations,
        snapshot,
        activity,
        db,
    };

    TestHarness {
        state, echo_state, echo_addr, echo_shutdown,
        supervisors, _tmp: tmp,
    }
}

pub fn minimal_llama_service(name: &str, port: u16) -> ServiceConfig {
    let mut placement = BTreeMap::new();
    placement.insert(DeviceSlot::Cpu, 100);
    ServiceConfig {
        name: SmolStr::new(name),
        template: Template::LlamaCpp,
        port,
        private_port: 0,
        lifecycle: Lifecycle::OnDemand,
        priority: 50,
        health: HealthSettings { http_path: "/health".into(), timeout_ms: 5_000, probe_interval_ms: 200 },
        placement_override: placement,
        placement_policy: PlacementPolicy::CpuOnly,
        idle_timeout_ms: 60_000,
        warming_grace_ms: 100,
        drain_timeout_ms: 1_000,
        extended_stream_drain_ms: 1_000,
        max_request_duration_ms: 5_000,
        filters: Filters::default(),
        raw: RawService {
            name: Some(SmolStr::new(name)),
            template: Some(SmolStr::new("llama-cpp")),
            model: Some(PathBuf::from("/fake/model.gguf")),
            port: Some(port),
            ..Default::default()
        },
    }
}

impl TestHarness {
    pub async fn cleanup(self) {
        let _ = self.echo_shutdown.send(true);
        for sup in &self.supervisors { sup.shutdown().await; }
    }
}
```

- [ ] **Step 1: `tests/openai_models.rs`**

```rust
mod common;

use ananke::openai_api;
use axum::body::to_bytes;
use axum::http::StatusCode;
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn get_v1_models_lists_registered_services() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0), minimal_llama_service("beta", 0)]).await;
    let app = openai_api::router(h.state.clone());
    let req = axum::http::Request::builder().method("GET").uri("/v1/models")
        .body(axum::body::Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let names: Vec<String> = parsed["data"].as_array().unwrap().iter()
        .map(|m| m["id"].as_str().unwrap().to_string()).collect();
    assert!(names.contains(&"alpha".to_string()));
    assert!(names.contains(&"beta".to_string()));
    h.cleanup().await;
}
```

- [ ] **Step 2: `tests/openai_unimplemented.rs`**

```rust
mod common;

use ananke::openai_api;
use axum::http::StatusCode;
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn audio_speech_returns_501() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0)]).await;
    let app = openai_api::router(h.state.clone());
    let req = axum::http::Request::builder().method("POST").uri("/v1/audio/speech")
        .body(axum::body::Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn images_generations_returns_501() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0)]).await;
    let app = openai_api::router(h.state.clone());
    let req = axum::http::Request::builder().method("POST").uri("/v1/images/generations")
        .body(axum::body::Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
    h.cleanup().await;
}
```

- [ ] **Step 3: `tests/openai_chat_routing.rs`**

```rust
mod common;

use ananke::openai_api;
use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn chat_completions_unknown_model_404() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0)]).await;
    let app = openai_api::router(h.state.clone());
    let body = r#"{"model":"nope","messages":[]}"#;
    let req = Request::builder().method("POST").uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body)).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn chat_completions_routes_through_echo() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0)]).await;
    let app = openai_api::router(h.state.clone());
    let body = r#"{"model":"alpha","messages":[{"role":"user","content":"hi"}]}"#;
    let req = Request::builder().method("POST").uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body)).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["id"], "cmpl-echo");

    // Echo's sink should have captured the request body.
    let sunk = h.echo_state.sink.lock().clone();
    assert_eq!(sunk.len(), 1);
    assert_eq!(sunk[0]["model"], "alpha");

    h.cleanup().await;
}
```

- [ ] **Step 4: `tests/openapi_json.rs`**

```rust
mod common;

use ananke::management_api;
use axum::body::to_bytes;
use axum::http::StatusCode;
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn openapi_json_is_valid() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0)]).await;
    let app = management_api::router(h.state.clone());
    let req = axum::http::Request::builder().method("GET").uri("/api/openapi.json")
        .body(axum::body::Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = to_bytes(resp.into_body(), 10 * 1024 * 1024).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["openapi"].as_str().unwrap_or("").chars().next(), Some('3'));
    let paths = parsed["paths"].as_object().expect("paths object");
    assert!(paths.contains_key("/v1/models"));
    assert!(paths.contains_key("/api/services"));
    assert!(paths.contains_key("/api/devices"));

    h.cleanup().await;
}
```

- [ ] **Step 5: Run**

Run: `cargo test --test openai_models --test openai_unimplemented --test openai_chat_routing --test openapi_json` — all pass.

- [ ] **Step 6: Commit**

```bash
git add tests/
git commit -m "test: openai routing, 501s, chat round-trip, openapi.json"
```

---

## Task 19: Integration tests — on-demand lifecycle

**Files:**
- Create: `tests/ondemand_start.rs`
- Create: `tests/start_coalescing.rs`
- Create: `tests/start_queue_full.rs`
- Create: `tests/idle_timeout_returns_to_idle.rs`

- [ ] **Step 1: `tests/ondemand_start.rs`**

```rust
mod common;

use ananke::openai_api;
use axum::http::StatusCode;
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn first_request_triggers_spawn_and_serves() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0)]).await;
    let app = openai_api::router(h.state.clone());
    let body = r#"{"model":"alpha","messages":[]}"#;
    let req = axum::http::Request::builder().method("POST").uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body)).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    h.cleanup().await;
}
```

- [ ] **Step 2: `tests/start_coalescing.rs`**

This test is the trickiest — we want to assert that concurrent first-requests for an idle service produce only one spawn. Because the test harness's supervisors are backed by echo (not actually spawning a llama-server), we instead assert that the `Ensure` bus is reused across concurrent callers, which we verify by counting the `x-echo-spawn-count` header on the response. The echo server increments on every `serve()` call, so the counter only grows when a new echo is bound — not useful here.

Practical approach: assert via the database row — `SELECT COUNT(*) FROM running_services WHERE service_id = ?`. In our harness, supervisors don't write `running_services` rows because they can't literally spawn a llama-server against a fake binary. Fire the requests, let them resolve, and assert all 5 responses have 200; that confirms the router handled coalescing without 503 start_queue_full.

Revise: our test harness's supervisor does attempt to spawn via the real `spawn_child` with `binary: "llama-server"`. That will fail because no real llama-server is available in the test env. The supervisor will mark `Failed` and the test will hit timeouts.

Correction: for phase-2 coalescing tests, **mock the spawn path**. Add a test-only feature `test-fake-spawn` that short-circuits `spawn_child` to spawn `/bin/sh -c 'sleep 30'` (or similar) instead of the real binary. Gate this in `src/supervise/spawn.rs`:

```rust
#[cfg(any(test, feature = "test-fakes"))]
pub async fn spawn_child(_cfg: &SpawnConfig) -> Result<tokio::process::Child, ExpectedError> {
    use std::os::unix::process::CommandExt;
    let mut cmd = tokio::process::Command::new("/bin/sh");
    cmd.args(["-c", "sleep 30"]);
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    cmd.kill_on_drop(true);
    cmd.spawn().map_err(|e| ExpectedError::config_unparseable(
        std::path::PathBuf::from("<fake-spawn>"),
        format!("fake spawn failed: {e}"),
    ))
}

#[cfg(not(any(test, feature = "test-fakes")))]
pub async fn spawn_child(cfg: &SpawnConfig) -> Result<tokio::process::Child, ExpectedError> {
    real_spawn(cfg).await
}

pub(crate) async fn real_spawn(cfg: &SpawnConfig) -> Result<tokio::process::Child, ExpectedError> {
    // Move the current phase-1 body of `spawn_child` verbatim into this function:
    // tokio::process::Command::new, env_clear, env, stdio piped, kill_on_drop,
    // pre_exec(prctl::set_pdeathsig(SIGTERM)), spawn.
}
```

For test runs we still want the health probe to succeed. The test harness's echo server listens on the `private_port` chosen by the harness, and health probes hit `/health` on the echo. So the fake spawn merely needs to keep a process alive while the echo serves the HTTP.

Because `spawn_child` is shared code, use a feature flag `test-fakes` (already declared in Cargo.toml) to switch the impl. Tests set `--features test-fakes`.

Now back to the coalescing test. Fire 5 concurrent POSTs:

```rust
mod common;

use ananke::openai_api;
use axum::http::StatusCode;
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_first_requests_collapse_into_one_start() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0)]).await;
    let app = ananke::openai_api::router(h.state.clone());

    let body = r#"{"model":"alpha","messages":[]}"#;
    let mut join_set = tokio::task::JoinSet::new();
    for _ in 0..5 {
        let app = app.clone();
        let body = body.to_string();
        join_set.spawn(async move {
            let req = axum::http::Request::builder().method("POST").uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body)).unwrap();
            app.oneshot(req).await.unwrap().status()
        });
    }

    let mut statuses = Vec::new();
    while let Some(r) = join_set.join_next().await {
        statuses.push(r.unwrap());
    }
    // All five succeed (or the coalescing bus would yield 503 to some).
    assert_eq!(statuses.iter().filter(|s| **s == StatusCode::OK).count(), 5);
    h.cleanup().await;
}
```

Run tests with `cargo test --features test-fakes --test start_coalescing`.

- [ ] **Step 3: `tests/start_queue_full.rs`**

```rust
mod common;

// Set start_queue_depth low via the service's raw config for this test.
// Our minimal_llama_service helper doesn't accept it yet; extend the helper:
```

Extend `minimal_llama_service` in `tests/common/mod.rs` with an optional `start_queue_depth: usize` parameter (default 10). Then in the test, use a value of 2 and fire 4 concurrent requests.

Helper variant:

```rust
pub fn service_with_queue_depth(name: &str, port: u16, depth: usize) -> ServiceConfig {
    let mut s = minimal_llama_service(name, port);
    // start_queue_depth is currently hardcoded to 10 in RawService. Extend
    // RawService to accept a `start_queue_depth: Option<usize>` field and
    // plumb it through `start_queue_depth()`. Done in Task 8's follow-up.
    s
}
```

Extend `RawService::start_queue_depth()` in `src/config/parse.rs`:

```rust
impl RawService {
    pub fn start_queue_depth(&self) -> usize {
        self.start_queue_depth.unwrap_or(10)
    }
}
```

Add `pub start_queue_depth: Option<usize>` to `RawService`. Propagate through `merge_service`.

Actually `service_with_queue_depth` needs a way to set this. Simplest: expose a mutable reference in `ServiceConfig` and overwrite via `raw.start_queue_depth`. The supervisor reads `svc.raw.start_queue_depth()`.

```rust
pub fn service_with_queue_depth(name: &str, port: u16, depth: usize) -> ServiceConfig {
    let mut s = minimal_llama_service(name, port);
    s.raw.start_queue_depth = Some(depth);
    s
}
```

Test:

```rust
mod common;

use axum::http::StatusCode;
use common::{build_harness, service_with_queue_depth};
use tower::util::ServiceExt;

#[tokio::test(flavor = "multi_thread")]
async fn start_queue_full_rejects_excess_waiters() {
    let h = build_harness(vec![service_with_queue_depth("alpha", 0, 2)]).await;
    let app = ananke::openai_api::router(h.state.clone());

    let body = r#"{"model":"alpha","messages":[]}"#;
    let mut join_set = tokio::task::JoinSet::new();
    for _ in 0..4 {
        let app = app.clone();
        let body = body.to_string();
        join_set.spawn(async move {
            let req = axum::http::Request::builder().method("POST").uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body)).unwrap();
            app.oneshot(req).await.unwrap().status()
        });
    }

    let mut statuses = Vec::new();
    while let Some(r) = join_set.join_next().await {
        statuses.push(r.unwrap());
    }
    let ok = statuses.iter().filter(|s| **s == StatusCode::OK).count();
    let svc_unavail = statuses.iter().filter(|s| **s == StatusCode::SERVICE_UNAVAILABLE).count();
    assert!(ok >= 2, "statuses: {statuses:?}");
    assert!(svc_unavail >= 1, "expected at least one 503 start_queue_full, got {statuses:?}");
    h.cleanup().await;
}
```

- [ ] **Step 4: `tests/idle_timeout_returns_to_idle.rs`**

```rust
mod common;

use axum::http::StatusCode;
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "multi_thread")]
async fn idle_timeout_returns_to_idle_then_restarts_on_next_request() {
    let mut svc = minimal_llama_service("alpha", 0);
    svc.idle_timeout_ms = 500;
    let h = build_harness(vec![svc]).await;
    let app = ananke::openai_api::router(h.state.clone());

    // First request: starts and serves.
    let body = r#"{"model":"alpha","messages":[]}"#;
    let req = axum::http::Request::builder().method("POST").uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body)).unwrap();
    assert_eq!(app.clone().oneshot(req).await.unwrap().status(), StatusCode::OK);

    // Wait longer than idle_timeout; supervisor drains.
    tokio::time::sleep(std::time::Duration::from_millis(1500)).await;

    // Next request should succeed (fresh spawn).
    let req = axum::http::Request::builder().method("POST").uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body)).unwrap();
    assert_eq!(app.clone().oneshot(req).await.unwrap().status(), StatusCode::OK);

    h.cleanup().await;
}
```

- [ ] **Step 5: Run all new tests**

Run: `cargo test --features test-fakes --test ondemand_start --test start_coalescing --test start_queue_full --test idle_timeout_returns_to_idle`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/supervise/spawn.rs src/config/parse.rs tests/
git commit -m "test: on-demand lifecycle, coalescing, queue full, idle_timeout"
```

---

## Task 20: Integration tests — allocator + management API

**Files:**
- Create: `tests/allocator_insufficient_vram.rs`
- Create: `tests/management_services.rs`
- Create: `tests/management_devices.rs`

- [ ] **Step 1: `tests/allocator_insufficient_vram.rs`**

```rust
mod common;

use ananke::config::DeviceSlot;
use ananke::devices::{CpuSnapshot, DeviceSnapshot};
use axum::http::StatusCode;
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "multi_thread")]
async fn insufficient_vram_returns_503() {
    let mut svc = minimal_llama_service("big", 0);
    // Demand 10 GB of CPU.
    svc.placement_override.clear();
    svc.placement_override.insert(DeviceSlot::Cpu, 10 * 1024);

    let h = build_harness(vec![svc]).await;

    // Populate the snapshot with only 1 GB free CPU.
    *h.state.snapshot.write() = DeviceSnapshot {
        gpus: Vec::new(),
        cpu: Some(CpuSnapshot { total_bytes: 16 * 1024 * 1024 * 1024, available_bytes: 1024 * 1024 * 1024 }),
        taken_at_ms: 0,
    };

    let app = ananke::openai_api::router(h.state.clone());
    let body = r#"{"model":"big","messages":[]}"#;
    let req = axum::http::Request::builder().method("POST").uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body)).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["error"]["code"], "insufficient_vram");
    h.cleanup().await;
}
```

- [ ] **Step 2: `tests/management_services.rs`**

```rust
mod common;

use axum::body::to_bytes;
use axum::http::StatusCode;
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn api_services_lists_registered() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0), minimal_llama_service("beta", 0)]).await;
    let app = ananke::management_api::router(h.state.clone());
    let req = axum::http::Request::builder().method("GET").uri("/api/services")
        .body(axum::body::Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let names: Vec<String> = parsed.as_array().unwrap().iter().map(|s| s["name"].as_str().unwrap().to_string()).collect();
    assert!(names.contains(&"alpha".to_string()));
    h.cleanup().await;
}

#[tokio::test(flavor = "current_thread")]
async fn api_service_detail_by_name() {
    let h = build_harness(vec![minimal_llama_service("alpha", 12345)]).await;
    let app = ananke::management_api::router(h.state.clone());
    let req = axum::http::Request::builder().method("GET").uri("/api/services/alpha")
        .body(axum::body::Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["name"], "alpha");
    assert_eq!(parsed["port"], 12345);
    h.cleanup().await;
}
```

- [ ] **Step 3: `tests/management_devices.rs`**

```rust
mod common;

use ananke::devices::{CpuSnapshot, DeviceSnapshot, GpuSnapshot};
use axum::body::to_bytes;
use axum::http::StatusCode;
use common::{build_harness, minimal_llama_service};
use tower::util::ServiceExt;

#[tokio::test(flavor = "current_thread")]
async fn api_devices_reflects_snapshot() {
    let h = build_harness(vec![minimal_llama_service("alpha", 0)]).await;
    *h.state.snapshot.write() = DeviceSnapshot {
        gpus: vec![GpuSnapshot { id: 0, name: "RTX 3090".into(), total_bytes: 24 * 1024 * 1024 * 1024, free_bytes: 20 * 1024 * 1024 * 1024 }],
        cpu: Some(CpuSnapshot { total_bytes: 64 * 1024 * 1024 * 1024, available_bytes: 40 * 1024 * 1024 * 1024 }),
        taken_at_ms: 0,
    };
    let app = ananke::management_api::router(h.state.clone());
    let req = axum::http::Request::builder().method("GET").uri("/api/devices")
        .body(axum::body::Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let ids: Vec<String> = parsed.as_array().unwrap().iter().map(|d| d["id"].as_str().unwrap().to_string()).collect();
    assert!(ids.contains(&"gpu:0".to_string()));
    assert!(ids.contains(&"cpu".to_string()));
    h.cleanup().await;
}
```

- [ ] **Step 4: Run all**

Run: `cargo test --features test-fakes --test allocator_insufficient_vram --test management_services --test management_devices`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test: allocator insufficient_vram + management services/devices"
```

---

## Task 21: Smoke test + final cleanup

**Files:**
- Modify: `tests/manual/phase-1-smoke.md` (rename to `phase-2-smoke.md`)
- Verify: end-to-end real run

- [ ] **Step 1: Extend smoke runbook**

Copy `tests/manual/phase-1-smoke.md` to `tests/manual/phase-2-smoke.md` and extend with:

```markdown
# Phase 2 manual smoke test

Real-hardware validation for the phase-2 additions: unified OpenAI
endpoint, on-demand lifecycle, start coalescing, filters, allocator,
management API.

## Prerequisites

Same as phase 1, plus:

- `jq` for inspecting JSON responses.
- An extra small GGUF to exercise routing by name.

## Steps

1. Build release:
   ```
   cargo build --release
   ```

2. Create `~/.config/ananke/config.toml`:
   ```toml
   [daemon]
   management_listen = "127.0.0.1:17777"
   data_dir = "/tmp/ananke-phase2"

   [openai_api]
   listen = "127.0.0.1:18080"

   [[service]]
   name = "small"
   template = "llama-cpp"
   model = "/mnt/ssd0/ai/llm/Qwen3-4B-Instruct-2507-UD-Q5_K_XL.gguf"
   port = 17435
   context = 4096
   flash_attn = true
   cache_type_k = "q8_0"
   cache_type_v = "q8_0"
   lifecycle = "on_demand"
   idle_timeout = "30s"
   devices.placement = "gpu-only"
   devices.placement_override = { "gpu:0" = 4500 }
   filters.set_params = { max_tokens = 64 }
   ```

3. Start: `LD_LIBRARY_PATH=/run/opengl-driver/lib ./target/release/ananke --config ~/.config/ananke/config.toml`

4. Verify unified OpenAI endpoint:
   - `curl http://127.0.0.1:18080/v1/models | jq` — includes `"id": "small"`.
   - First chat request triggers spawn:
     ```
     curl -s http://127.0.0.1:18080/v1/chat/completions \
       -H 'Content-Type: application/json' \
       -d '{"model":"small","messages":[{"role":"user","content":"say hi"}]}' | jq
     ```
     Response arrives; `max_tokens` is enforced via the filter.

5. Verify on-demand + idle_timeout:
   - After the first request completes, wait 35 seconds.
   - `nvidia-smi` — llama-server is gone.
   - Fire another chat request — starts fresh.

6. Verify start coalescing:
   - Wait 35s so service is idle.
   - Fire 5 concurrent `curl` POSTs in the background (e.g., `for i in 1 2 3 4 5; do curl -s ... & done; wait`). All should return 200; only one llama-server ever exists.

7. Verify management API:
   - `curl http://127.0.0.1:17777/api/services | jq`
   - `curl http://127.0.0.1:17777/api/services/small | jq`
   - `curl http://127.0.0.1:17777/api/devices | jq`
   - `curl http://127.0.0.1:17777/api/openapi.json | jq '.paths | keys'`

8. Verify unimplemented 501:
   - `curl -i http://127.0.0.1:18080/v1/audio/speech -X POST` — returns `501 Not Implemented` with `error.code = "not_implemented"`.

9. Verify insufficient_vram:
   - Edit config: `placement_override = { "gpu:0" = 90000 }` (90 GB).
   - Reload ananke.
   - Fire a chat request — returns 503 with `error.code = "insufficient_vram"`.

10. Clean shutdown: `kill -TERM $(pidof ananke)`.

Success criteria: every numbered step above produces the expected result. File a bug if anything drifts.
```

- [ ] **Step 2: Run the smoke test on redline**

Execute the runbook. Record any real issues found; create follow-up commits as needed.

- [ ] **Step 3: Final lint + test sweep**

Run:
```
just lint
cargo test --workspace
cargo test --workspace --features test-fakes
```
All must pass.

- [ ] **Step 4: Commit runbook**

```bash
git add tests/manual/phase-2-smoke.md
git commit -m "docs: phase 2 manual smoke runbook"
```

---

## Self-review checklist

Before declaring phase 2 complete:

- `just lint` passes (all Rust + TS checks).
- `cargo test --workspace` and `cargo test --workspace --features test-fakes` both pass.
- `tests/manual/phase-2-smoke.md` executed end-to-end against real hardware.
- Phase-1 smoke runbook steps still pass (no regression on persistent lifecycle, orphan recovery, clean shutdown).
- `/api/openapi.json` serves valid OpenAPI 3.x with all the expected path keys.
- `docs/spec.md` and both phase specs are still consistent with what was built; correct inline if drift.

Phase 2 success: point your redline at ananke's `:18080` as an OpenAI-compatible endpoint, declare your current lmp services in ananke config with `placement_override` values, and retire lmp for the services that don't need eviction or hybrid placement. Eviction + layer-aware placement + hybrid + dynamic = phases 3-4.
