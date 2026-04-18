# Phase 4: `command` template + balloon resolver + eviction + oneshots — design

Status: accepted 2026-04-18. Parent design: `docs/spec.md` (authoritative). Prior phases: phase-1 (`docs/superpowers/specs/2026-04-18-ananke-phase-1-lean-mvp-daemon.md`), phase-2 (`docs/superpowers/specs/2026-04-18-ananke-phase-2-unified-openai-ondemand-allocator.md`), phase-3 (`docs/superpowers/specs/2026-04-18-ananke-phase-3-estimator-layer-aware.md`).

## 1. Goal

Deliver the scheduler-complete daemon: add the `command` template (so ComfyUI-style services fit), dynamic-VRAM services with a balloon resolver, priority-based eviction with the full drain pipeline, and oneshots via the management API. After this phase the redline lmp config — including ComfyUI — transfers to ananke in full.

Non-goals: management-API *config-mutation* POST/PUT endpoints beyond oneshots, WebSocket event stream, `anankectl`, frontend, Nix module.

## 2. Scope

### 2.1 In scope

**`command` template** (spec §7.2):

- New `Template::Command` variant.
- Fields (on the service block): `command: Vec<String>`, `workdir: Option<PathBuf>`, `env: Option<BTreeMap<String, String>>`.
- Placeholder substitution at spawn time on both the command vector and env values:
  - `{port}` — the service's external port.
  - `{gpu_ids}` — comma-separated NVML ids from the allocation (useful for commands that don't respect `CUDA_VISIBLE_DEVICES`).
  - `{vram_mb}` — per-device reservation in MB; valid only for `static` single-GPU allocations (validation rejects use on dynamic or multi-device).
  - `{model}` — `model` field if present (empty if not).
  - `{name}` — service name.
- `CUDA_VISIBLE_DEVICES` auto-populated from the allocation (same as `llama-cpp`).
- `metadata.openai_compat: bool` (default false for `command`). When false, service is hidden from `/v1/models`.
- Health check remains `health.http` with default `/system_stats` for `command` (spec §7.2 example).

**Allocation modes** (spec §7.2, §8.4):

- `allocation.mode = "static"` + `allocation.vram_gb = N`: fixed reservation on a single GPU (or CPU if `placement = "cpu-only"`).
- `allocation.mode = "dynamic"` + `allocation.min_vram_gb = A` + `allocation.max_vram_gb = B`: starts with `min_vram_gb` reserved; can grow up to `max_vram_gb` observed.
- Validation: `dynamic` mode is incompatible with `llama-cpp` template (llama-cpp sizing comes from the estimator). Dynamic is `command`-template-only.
- `static` allocation on `llama-cpp` is also rejected — llama-cpp uses the estimator or `placement_override`, not `allocation`.

**Balloon resolver** (spec §8.4) — per-dynamic-service tokio task:

- Samples `observed_usage` every 2 s via the phase-3 observation infrastructure.
- 6-sample rolling window (12 s total).
- `elastic_region = max_vram_gb - current_observed - margin` (default margin 512 MB).
- Admits **elastic borrowers** into the elastic region; borrowers are tagged in the allocation table with `elastic = true`.
- **Growth detection**: slope across the 6-sample window. If slope positive AND projected next-sample > floor currently claimed by borrower → contention.
- **Contention resolution**:
  - Dynamic priority > borrower priority → evict borrower.
  - Borrower priority ≥ dynamic → evict dynamic.
- **Fast-path SIGKILL**: 5 s grace, not the normal drain. Used by the balloon resolver only.
- **`min_borrower_runtime`** (default 60 s): do not evict a borrower that started that recently. Dynamic service's queued request times out instead.
- **Jitter tolerance**: slope detection requires positive trend across majority of window, not a single sample.
- **Exceeded `max_vram_gb`**: observed > max × 1.1 for > 30 s → SIGKILL dynamic. Logged loudly.

**Priority-based eviction** (spec §8.1 step 5, §5.2):

- Invoked by the allocator when `can_fit` fails and the requesting service has higher priority than the lowest-priority running service.
- Collect candidates: `priority < want.priority` AND `lifecycle` is `on_demand` or `oneshot` (NOT `persistent` unless also lower priority — persistents are evictable too per §5.1, just strongly discouraged).
- Sort by: `idle`-first → lowest priority → smallest allocation.
- Evict the minimum number of candidates so that `can_fit` succeeds.
- Each eviction goes through the full drain pipeline (§10.3), not the balloon fast-path.
- If no evictable set makes the placement fit → return the existing `NoFit` / OOM retry / disable path.

**Full drain pipeline** (spec §10.3):

- Upgrade the phase-1 drain to the full six-step sequence:
  1. Mark `draining`; new requests return 503.
  2. Count in-flight requests via a per-supervisor `Arc<AtomicU64>` bumped by the unified + per-service proxies; wait for count to reach 0, bounded by `max_request_duration`.
  3. `drain_timeout` grace (default 30 s llama-cpp, 5 s command).
  4. If streaming requests are still active at end of drain, extend by `extended_stream_drain` (default 30 s).
  5. SIGTERM.
  6. 10 s grace, then SIGKILL.
- Worst-case eviction latency for `llama-cpp`: ~10.7 minutes (spec §10.3).
- In-flight tracking lives in the proxy and unified-listener handlers; they obtain the counter via the service registry → supervisor handle.

**Oneshots** (spec §5.1, §11.3):

- `POST /api/oneshot` launches a oneshot. Body mirrors `[[service]]` fields but `lifecycle = "oneshot"`, plus `ttl: <duration>` (max lifetime) and optional `id` (auto-generated ULID if missing).
- `GET /api/oneshot/{id}` returns oneshot status.
- `DELETE /api/oneshot/{id}` terminates a running oneshot (goes through the full drain).
- Port pool: configurable range (default 18000–18999). Exhaustion returns 503.
- Scheduled through the same allocator + estimator pipeline as regular services.
- Evictable by strictly-higher-priority placements; `ttl` is a ceiling, not a floor.
- Final statuses surfaced via the existing `oneshots` SQLite table (already created in phase 1): `Running`, `Exited { code }`, `Evicted`, `Failed { reason }`, `Killed { reason }`.
- Not restarted after any terminal state.
- Clients that need guaranteed runtime submit at high priority.

**Management API extensions**:

- `POST /api/oneshot`, `GET /api/oneshot`, `GET /api/oneshot/{id}`, `DELETE /api/oneshot/{id}` — the oneshot surface.
- `GET /api/services` — services list continues to cover oneshots while they're running (with `lifecycle = "oneshot"`).
- `ServiceSummary` + `ServiceDetail` gain an `elastic_borrower: Option<String>` field (set to the dynamic service name when this service is currently borrowing elastic VRAM).
- `DeviceSummary` reservation entries gain `elastic: bool` so the UI can render two-tone bars later (phase 6).
- OpenAPI aggregator includes the new paths and types.

### 2.2 Out of scope (still deferred)

- POST/PUT/DELETE mutations on regular services (`/api/services/{name}/{start,stop,restart,enable,disable}`) → phase 5.
- `GET /api/config`, `PUT /api/config`, `POST /api/config/validate` → phase 5.
- WebSocket streams (`/api/services/{name}/logs/stream`, `/api/events`) → phase 5.
- `anankectl` CLI → phase 5.
- Frontend → phase 6.
- Nix module → phase 7.

## 3. Architecture

```
    +-------------+     +-------------+     +----------------+
    |  OpenAI     |     |  Management |     |  Oneshot POST  |
    |  router     |     |  router     |     |  handler       |
    +------+------+     +------+------+     +--------+-------+
           |                   |                     |
           v                   v                     v
    +-------------------------------------------------------+
    |                  Daemon core (Arc-shared)             |
    |   • registry / allocator / snapshotter (phase 2-3)    |
    |   • NEW: EvictionPlanner                              |
    |   • NEW: OneshotController                            |
    |   • NEW: port allocator (for oneshots)                |
    +-------------------+-----------------+-----------------+
                        |                 |
        per-service supervisors              per-oneshot supervisors
        (existing; adds drain counter)       (new; spawned from API)
                        |                 |
                per-dynamic-service balloon resolver task
                    (samples observation; triggers contention)
```

### 3.1 Shared state (additions)

- `OneshotTable`: `Arc<RwLock<BTreeMap<OneshotId, Arc<SupervisorHandle>>>>`. Like `ServiceRegistry` but scoped to oneshot IDs.
- `PortPool`: `Arc<Mutex<BTreeSet<u16>>>` free ports in the oneshot range; `allocate() → Option<u16>`, `release(port)`.
- `InflightCounters`: `Arc<RwLock<BTreeMap<ServiceName, Arc<AtomicU64>>>>`. One counter per service; bumped by the proxy path.

### 3.2 Supervisor additions

- `SupervisorCommand::BeginDrain { reason, ack }` — full-drain entry point. Replaces the existing `Shutdown` path for graceful eviction (which still exists for daemon shutdown).
- `SupervisorCommand::FastKill { reason, ack }` — balloon fast-path (5 s grace then SIGKILL).
- Drain state machine transitions: `Running → Draining → Stopped | Evicted | Idle`.

### 3.3 New module layout

```
src/
├── templates/
│   ├── mod.rs              // NEW: Template enum + per-template argv/env rendering
│   └── placeholders.rs     // NEW: {port}, {gpu_ids}, {vram_mb}, {model}, {name}
├── balloon.rs              // NEW: per-dynamic-service resolver task
├── eviction.rs             // NEW: priority-based candidate selection
├── drain.rs                // NEW: upgraded drain pipeline (spec §10.3)
├── oneshot/
│   ├── mod.rs              // NEW: OneshotController, OneshotId, statuses
│   ├── handlers.rs         // NEW: POST/GET/DELETE /api/oneshot handlers
│   └── port_pool.rs        // NEW: port allocator
├── config/
│   └── validate.rs         // MODIFY: accept command, allocation modes, oneshot POST body
├── supervise/
│   ├── mod.rs              // MODIFY: BeginDrain, FastKill, inflight counter, elastic tag
│   └── spawn.rs            // MODIFY: command template argv + placeholder substitution
├── allocator.rs            // MODIFY: can_fit extended to ask EvictionPlanner for candidates
├── management_api/
│   ├── handlers.rs         // MODIFY: elastic_borrower in detail; /api/oneshot routes
│   └── types.rs            // MODIFY: extend ServiceSummary + DeviceSummary
├── openapi.rs              // MODIFY: add oneshot paths/types
├── proxy.rs                // MODIFY: bump inflight counter on each request
└── daemon.rs               // MODIFY: wire OneshotController + port pool + inflight table
```

## 4. Data flow

### 4.1 Dynamic service with balloon resolver

1. `command`-template service declared with `allocation.mode = "dynamic"`, `min = 4 GB`, `max = 20 GB`.
2. Daemon start: supervisor enters `Idle`. On first `Ensure` (or explicit start for persistent), allocator reserves `min_vram_gb` on the configured GPU.
3. Supervisor spawns child (e.g. ComfyUI via `python main.py --port {port}`), transitions through `Starting → Warming → Running`.
4. A balloon-resolver tokio task is spawned alongside the supervisor (one per dynamic service). Every 2 s it pulls the latest `observation.read_peak()` and appends to its 6-sample window.
5. When another service (borrower) requests placement that fits in the elastic region (`max - current_observed - margin`), allocator admits it with `elastic = true` on its reservation entry.
6. Balloon resolver detects growth via slope > 0 with projected next-sample > floor claimed by borrower → contention.
7. Contention resolution:
   - `dynamic.priority > borrower.priority` → issue `FastKill` to borrower; reclaim the space.
   - `borrower.priority ≥ dynamic.priority` → issue `FastKill` to dynamic (rare; spec note).
8. Steady-state: dynamic service grows/shrinks freely inside its `[min, max]` window; borrowers come and go.

### 4.2 Priority-based eviction at placement

1. New service (or oneshot) needs `can_fit` to succeed; it fails because existing services occupy the device.
2. Allocator calls `EvictionPlanner::select(want, existing_reservations, snapshot, requesting_priority)`.
3. Planner returns `Vec<ServiceName>` — minimum set of evictable services (priority < requesting, sorted idle → low-priority → small allocation).
4. For each candidate, supervisor receives `BeginDrain { reason: Eviction, ack }`.
5. Full drain pipeline runs (§10.3). Dropped requests from the evicted service surface to clients as 503 (unified listener) or connection close (per-service port).
6. After all acks return, allocator re-checks `can_fit`. If succeeds, reserve and proceed with the new placement.

### 4.3 Oneshot lifecycle

1. Client POSTs `/api/oneshot` with template + placement + TTL.
2. Handler allocates a port from the oneshot pool, assigns a ULID if none provided, inserts into `oneshots` SQLite row with `submitted_at = now`, status `Pending`.
3. Handler runs the full `Ensure` pipeline (estimate → pack → can_fit → evict if necessary → spawn).
4. Oneshot supervisor running; `GET /api/oneshot/{id}` returns status; log stream accessed via same `/api/services/{name}/logs/stream` path (phase 5 WebSocket).
5. TTL enforcement: a task per oneshot watches the clock. On TTL expiry → `BeginDrain { reason: TtlExpired }` → full drain → status `Exited` or `Killed`.
6. `DELETE /api/oneshot/{id}` → `BeginDrain { reason: UserKilled }`.
7. Evictable like any other service; if evicted, final status = `Evicted` and the `oneshots` row is updated with `ended_at + exit_code = None`.
8. Not restarted after any terminal state.
9. Port pool releases the port on terminal status.

### 4.4 Drain pipeline (§10.3)

State machine inside the supervisor on `BeginDrain`:

```
   Running
      | (BeginDrain received)
      v
   Draining  [refuses new requests; inflight counter observed]
      | wait until counter == 0 OR max_request_duration elapsed
      v
   DrainTimeoutGrace  [drain_timeout sleep]
      | if SSE clients still present → extend by extended_stream_drain
      v
   Signalling  [SIGTERM child]
      | 10 s grace
      v
   Killing  [SIGKILL if needed]
      v
   Stopped | Evicted | Idle  (depends on reason)
```

Refusing new requests is done by the proxy checking the supervisor's snapshot `state` on each request; if `Draining` → 503.

## 5. Key design calls

- **Balloon resolver is per-dynamic, not one global task.** Keeps reasoning local: each resolver only knows about its own service's window + which borrowers are currently using its elastic region.
- **Elastic tagging in the allocation table** (not a separate structure) — one place to look for placement truth.
- **Eviction is allocator-driven, not router-driven.** The unified OpenAI router calls `ensure()` → supervisor → allocator → planner. The router doesn't know about eviction; it just waits for the broadcast outcome.
- **Fast-path SIGKILL vs full drain** are two distinct code paths. Both eventually SIGKILL; fast-path skips the `max_request_duration` wait and `extended_stream_drain`.
- **`command` template placeholders are pre-substitution**. Once the process is running, the args are fixed. `{vram_mb}` on dynamic is a spec error because the number would be meaningless (balloon grows past it).
- **Oneshot port pool is in-memory** (lost on restart). A restart won't revive oneshots; spec §11.3 says "TTL is maximum lifetime, not minimum guarantee".
- **Inflight counter is per service, shared Arc**. Proxy increments on request start, decrements on request end (or connection close). SSE streams hold the counter until the stream closes. Single atomic, eventually consistent; small over-counting during bursts is acceptable.
- **Persistent services are evictable** by strictly higher priority (spec §5.1, §5.2). Operators who want pinning use `priority = 100`. Don't special-case persistence beyond this.
- **`max_vram_gb` enforcement is intentionally aggressive** — SIGKILL, not graceful drain — because a leaking dynamic service is pathological and deserves harsh handling.

## 6. Testing

### 6.1 Unit

- `templates::placeholders::substitute` — all 5 placeholders in command + env; invalid `{vram_mb}` on dynamic rejected.
- `balloon::detect_growth` — synthetic window arrays; positive slope vs flat; jitter noise ±10%.
- `eviction::select` — given (services, reservations, want, priority) returns correct minimum set; sort stability; ties broken by idle-first then smallest.
- `oneshot::port_pool` — allocate, release, exhaustion.
- `drain::pipeline` — state transitions under synthetic timing.

### 6.2 Integration

- `command_template_echo.rs` — service running `/bin/sh -c 'echo "$@"; sleep 300'` with placeholders in argv; assert substituted args visible in `/proc/<pid>/cmdline`.
- `dynamic_allocation_min_max.rs` — `command` template, dynamic mode, FakeProbe; allocator reserves `min_vram_gb` at spawn.
- `balloon_grows_evicts_borrower.rs` — dynamic service, fake observation driven via test hook, simulated borrower; balloon resolver fast-kills borrower when slope crosses threshold.
- `priority_eviction_low_prio_loses.rs` — 2 on_demand services at `priority=30` fill GPU; new request at `priority=70` triggers eviction of both.
- `persistent_is_evictable_by_higher_priority.rs` — persistent at `priority=50` evicted by `priority=80` request.
- `drain_respects_inflight.rs` — service running with an in-flight streaming request; `BeginDrain` waits for the stream to complete up to `max_request_duration`.
- `drain_sigkills_on_timeout.rs` — child ignores SIGTERM; after 10 s grace ananke SIGKILLs.
- `oneshot_post_spawns_and_ttl.rs` — POST /api/oneshot with `ttl = 2s`; oneshot starts, exits via TTL, `GET /api/oneshot/{id}` reports `Exited`.
- `oneshot_delete_cancels.rs` — POST oneshot, DELETE it; status `Killed { reason: UserKilled }`.
- `oneshot_port_exhaustion.rs` — configure pool of size 1; second POST returns 503.
- `oneshot_evicted_by_higher_priority.rs` — oneshot at priority 30 evicted by service request at priority 70; final status `Evicted`.
- `management_elastic_borrower_tag.rs` — dynamic service + borrower running; `/api/services/{borrower}` reports `elastic_borrower = Some("dynamic-service-name")`.

### 6.3 Smoke runbook (`tests/manual/phase-4-smoke.md`)

- ComfyUI command service with `dynamic` allocation. Load a workflow, observe GPU usage climbing; verify `/api/devices` reservation updates accordingly.
- Two llama-cpp services at different priorities sharing one GPU, one VRAM-hungry; higher-priority request successfully evicts the other.
- oneshot: POST a short script via `/api/oneshot` with `ttl = 10s`; watch it run and auto-expire.
- Drain timing: initiate eviction of a service with a live SSE stream; confirm the stream completes (bounded by `max_request_duration`) before SIGTERM.

## 7. Risks

1. **In-flight counting race**: a request that crashes mid-flight without decrementing would stall the drain until `max_request_duration`. Accept the worst-case; it's bounded.
2. **Balloon slope detection false positives**: noisy NVML samples could trigger a spurious eviction. Jitter tolerance + "positive trend across majority of window" mitigates. Tunable if problematic.
3. **Port pool exhaustion with long-running oneshots**: 1000 ports is generous but not infinite. If it ever trips, the 503 response is correct.
4. **Eviction cascades**: if evicting candidate A makes room for the requesting service but also frees capacity for candidate B (already reserved, now redundant), we don't rebalance. Acceptable for phase 4; future phases may want consolidation.
5. **TTL clocks vs daemon restart**: oneshots running at restart are adopted (phase-1 orphan recovery); their TTL clocks restart from the adoption point. Documented in the runbook.
6. **`persistent` eviction surprise**: users may not expect persistent services to be evictable. Documented in the spec (§5.1, §5.2); mitigated by the `priority = 100` pinning convention.

## 8. Success criteria

- `command` template service for ComfyUI spawns, serves requests, observes VRAM climb through the balloon resolver window.
- A lower-priority llama-cpp service is evicted to make room for a higher-priority one; the unified OpenAI listener serves the higher-priority request successfully after the eviction drain completes.
- Oneshots: POST/GET/DELETE all work; TTL expiry kills cleanly; evicted oneshots show correct final status.
- All prior-phase behaviours (unified OpenAI, on-demand, filters, estimator + placement, rolling correction, management API, openapi.json) continue working.
- `just lint` and `cargo test --workspace --features test-fakes` both pass.
- Redline lmp config — including ComfyUI — fully replaceable by ananke after this phase.
