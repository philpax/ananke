# Phase 2: Unified OpenAI + on-demand + allocator — design

Status: accepted 2026-04-18. Parent design: `docs/spec.md` (authoritative). Prior phase: `docs/superpowers/specs/2026-04-18-ananke-phase-1-lean-mvp-daemon.md`.

## 1. Goal

Extend the phase-1 daemon with:

- An OpenAI-compatible unified listener on `openai_api.listen`, routing requests by the `model` field in the request body, applying per-service request-body filters, and streaming SSE responses back.
- On-demand service lifecycle (`idle ⇄ starting → warming → running → draining → idle`) with per-service `idle_timeout`.
- Start-future coalescing so concurrent first-requests for the same idle service share a single spawn, bounded by `start_queue_depth`.
- A pure feasibility-check allocator that refuses to spawn when a service's `placement_override` no longer fits live device state.
- A minimal, read-only slice of the management API (`/api/services`, `/api/services/{name}`, `/api/devices`) plus an OpenAPI document at `/api/openapi.json`, annotated end-to-end with `utoipa`.

Non-goals for this phase: eviction, dynamic/command template, oneshots, full management-API mutations, GGUF estimator, layer-aware placement, frontend, Nix module.

## 2. Scope

### 2.1 In scope

**Unified OpenAI listener.**

- Axum router bound to `openai_api.listen` (default `127.0.0.1:8080`).
- Endpoints: `/v1/models` (GET), `/v1/chat/completions` (POST), `/v1/completions` (POST), `/v1/embeddings` (POST).
- `/v1/audio/*`, `/v1/images/*`, `/v1/files/*`, `/v1/fine_tuning/*`, `/v1/batches` return `501 Not Implemented` with `{error: {code, message, type}}`.
- Error shape matches OpenAI convention across the board: `{error: {code: <snake_case>, message: <human>, type: <class>}}`.
- Unknown `model` → `404 model_not_found`.
- Service `disabled` → `503 service_disabled` with reason embedded in message.
- Start queue full → `503 start_queue_full`.
- Start failed → `503 start_failed` with last stderr line in message.

**Lifecycle.**

- Accept `lifecycle = "on_demand"` in config (phase 1 gate removed). This becomes the implicit default when `lifecycle` is omitted.
- `persistent` services still start at boot per phase 1.
- `on_demand` services start in `idle`; first routed request triggers spawn; service returns to `idle` after `idle_timeout` (default 10m) of no traffic.
- `idle_timeout` is per-service; parsed as a duration string (same grammar as phase 1).

**Start-future coalescing.**

- Per supervisor: `Option<broadcast::Sender<Result<(), StartFailure>>>` stored behind the supervisor's state.
- First `Ensure` command in `idle`: create the channel, store, subscribe, transition to `starting`.
- Subsequent `Ensure`s in `starting`/`warming`: check `subscriber_count()` against `start_queue_depth`; if room, subscribe; otherwise return `StartQueueFull` to the caller.
- On `running` transition: broadcast `Ok(())`, drop the sender.
- On disable/fail: broadcast `Err(StartFailure { kind, message })`, drop.

**Allocator.**

- Pure function `can_fit(want: &BTreeMap<DeviceSlot, u64>, snapshot: &DeviceSnapshot, reserved: &AllocationTable) -> Result<(), NoFit>`.
- `DeviceSnapshot` is a snapshot-in-time of live NVML free bytes + CPU `MemAvailable` + external reservations inferred per spec §4.4.
- `AllocationTable` is `Arc<Mutex<BTreeMap<ServiceName, Allocation>>>` from phase 1; read-locked here.
- Called from supervisor at `idle → starting` transition; on `NoFit`, supervisor transitions to `Failed { retry_count: 0 }` and uses the phase-1 backoff (2s/5s/15s then `Disabled { NoFit }`).
- No eviction; that's phase 4.

**Request filters.**

- Top-level JSON key rewrite applied to POST bodies before forwarding upstream.
- `filters.strip_params` (`Vec<String>`): remove these top-level keys.
- `filters.set_params` (`BTreeMap<String, serde_json::Value>`): insert/override these top-level keys.
- Non-JSON or missing-body requests bypass filters.
- Order: strip first, then set (spec §10.2).

**Activity signaling.**

- Per-service `Arc<AtomicU64>` tracks `last_activity_ms` (UNIX epoch millis).
- Both the unified listener and the phase-1 per-service proxy bump the atomic on every incoming request.
- Supervisor's `Running` select! re-reads the atomic on each wakeup to compute `idle_deadline = last_activity_ms + idle_timeout_ms`.

**Minimal management API (read-only).**

- Axum router bound to `daemon.management_listen` (default `127.0.0.1:7777`).
- `GET /api/services` → `Vec<ServiceSummary>`.
- `GET /api/services/{name}` → `ServiceDetail` (full effective config + runtime state + last 200 log lines from SQLite).
- `GET /api/devices` → `Vec<DeviceSummary>`.
- `GET /api/openapi.json` — served directly via `utoipa::OpenApi` derive + handler.

**utoipa wiring.**

- All handlers annotated with `#[utoipa::path(...)]`.
- Request and response structs derive `ToSchema`.
- Shared aggregator `OpenApiDoc` struct derives `OpenApi` referencing all paths/schemas.
- Axum serves the generated JSON at `/api/openapi.json`.

### 2.2 Out of scope (deferred)

- Eviction / priority-based displacement → phase 4.
- Dynamic (`allocation.mode = "dynamic"`) services, balloon resolver → phase 4.
- `command` template → phase 4.
- Oneshots and the oneshot port pool → phase 4.
- POST/PUT/DELETE management mutations (`/api/services/{name}/{start,stop,restart,enable,disable}`, `/api/config`, `/api/oneshot`) → phase 5.
- WebSocket streams (`/api/services/{name}/logs/stream`, `/api/events`) → phase 5.
- `anankectl` CLI → phase 5.
- GGUF reader, architecture-aware estimator, rolling correction → phase 3.
- Layer-aware multi-GPU placement and automatic `-ngl`/`--tensor-split` derivation → phase 3.
- Frontend → phase 6.
- Nix module → phase 7.

## 3. Architecture

```
      +-----------------+        +------------------+
      |   OpenAI API    |        |  Management API  |
      |  Axum router    |        |  Axum router     |
      | 127.0.0.1:8080  |        | 127.0.0.1:7777   |
      +--------+--------+        +---------+--------+
               |                           |
        +------v---------------------------v------+
        |       Daemon core (Arc-shared)          |
        |   • service_registry (name -> handle)   |
        |   • allocation_table (Mutex)            |
        |   • device snapshotter (2s cadence)     |
        |   • activity table (AtomicU64 per svc)  |
        +-----------------+-----------------------+
                          |
         per-service supervisors with expanded state:
            idle <-> starting -> warming -> running -> draining -> idle
                        with per-service proxy still on its external port
```

### 3.1 Shared state

- **`ServiceRegistry`** (`Arc<RwLock<BTreeMap<SmolStr, SupervisorHandle>>>`): looked up by both routers.
- **`AllocationTable`** (`Arc<Mutex<BTreeMap<SmolStr, Allocation>>>`): already introduced in phase 1 as `Mutex<BTreeMap>`; now shared with the allocator.
- **`DeviceSnapshotter`**: a tokio task that samples NVML + CPU every 2 s, writing into `Arc<RwLock<DeviceSnapshot>>`. Supervisors and `/api/devices` read-lock it; nobody writes except the sampler.
- **`ActivityTable`** (`Arc<RwLock<BTreeMap<SmolStr, Arc<AtomicU64>>>>`): one atomic per service; created on supervisor spawn.

### 3.2 Listener wiring (in `daemon.rs`)

Two new Axum routers, each on its own `axum::serve` task with shutdown via the existing `watch::Receiver<bool>`:

- `openai_api::router()` → bound to `openai_api.listen`.
- `management_api::router()` → bound to `daemon.management_listen`.

Both receive `Arc<AppState>` via Axum's `State` extractor, where `AppState` bundles the shared state above.

## 4. Module layout

### 4.1 New modules

```
src/
├── openai_api/
│   ├── mod.rs               // pub fn router(state) -> axum::Router
│   ├── handlers.rs          // /v1/models, /v1/chat/completions, etc.
│   ├── filters.rs           // strip + set params on JSON body
│   ├── errors.rs            // OpenAI-shaped error responses
│   ├── schema.rs            // ToSchema structs for request/response envelopes
│   └── types.rs             // ModelListing, ErrorBody, etc.
├── management_api/
│   ├── mod.rs               // pub fn router(state) -> axum::Router
│   ├── handlers.rs          // /api/services, /api/services/{name}, /api/devices
│   └── types.rs             // ServiceSummary, ServiceDetail, DeviceSummary
├── allocator.rs             // pure feasibility check
├── service_registry.rs      // Arc<RwLock<BTreeMap<Name, Handle>>>
├── activity.rs              // Arc<AtomicU64>-based activity table
├── snapshotter.rs           // 2s NVML + CPU sampler task
└── openapi.rs               // utoipa::OpenApi aggregator
```

### 4.2 Modified modules

- `src/supervise/mod.rs`:
  - New `idle` entry state for on_demand services.
  - New commands: `Ensure { ack: oneshot::Sender<broadcast::Receiver<StartOutcome>> }`, `ActivityPing`.
  - New `StartOutcome` + `StartFailure` types.
  - `Running` select! gains `idle_timer` branch (via `tokio::time::sleep_until`) and `ActivityPing` branch.
  - Integrate allocator call at `idle → starting`.
- `src/config/validate.rs`:
  - Remove the "phase-1 rejects on_demand" gate.
  - Parse `filters` into `ServiceConfig` (strip_params, set_params).
  - Parse `idle_timeout` into `ServiceConfig.idle_timeout_ms`.
  - Keep `placement_override` requirement (estimator is still phase 3).
- `src/daemon.rs`:
  - Build `AppState`.
  - Spawn snapshotter task.
  - Spawn OpenAI + management axum servers; drive shutdown from the existing `watch` channel.
- `src/lib.rs`:
  - Declare new modules.
- `src/proxy.rs` (phase-1 per-service proxy):
  - Send `ActivityPing` to the supervisor via the activity atomic on each request.

## 5. Data flow

### 5.1 Unified OpenAI request

1. Client POSTs `/v1/chat/completions` to `openai_api.listen`.
2. Axum handler deserialises body as `ChatCompletionEnvelope { model: String, #[serde(flatten)] extra: serde_json::Value }`.
3. Look up `model` in `ServiceRegistry`. If absent → `404 model_not_found`.
4. Inspect supervisor snapshot (via `SupervisorHandle::snapshot()` extended this phase to report start-coalescing state):
   - `Disabled` / `Stopped` → `503 service_disabled`.
   - `Idle` / `Starting` / `Warming` → send `Ensure`; wait on returned `broadcast::Receiver<StartOutcome>` up to `max_request_duration`.
     - `StartOutcome::Ok` → proceed.
     - `StartOutcome::QueueFull` → `503 start_queue_full`.
     - `StartOutcome::Err(StartFailure)` → `503 start_failed` with the failure message (and any captured last stderr line).
   - `Running` → proceed directly.
5. Apply filters to `extra` before reserialisation: strip keys, then set keys.
6. Forward to `http://127.0.0.1:<private_port>` using the phase-1 proxy primitives (`hyper_util::client::legacy::Client` with streaming body).
7. On response: increment activity atomic; stream SSE back.

### 5.2 On-demand idle timer

- `last_activity_ms` updated on every proxy hit (both unified and per-service paths).
- Supervisor in `Running` tracks `idle_deadline_ms = last_activity_ms + idle_timeout_ms`.
- `Running` select! includes `tokio::time::sleep_until(deadline)`; on fire, check atomic once more — if still expired, drain → SIGTERM → transition to `idle`; if not, re-enter select! with fresh deadline.

### 5.3 Start-future coalescing

- Supervisor state adds `start_bus: Option<broadcast::Sender<StartOutcome>>`.
- On `Ensure` in `Idle`: create `broadcast::channel::<StartOutcome>(16)` (inner capacity irrelevant since we only send once), subscribe, store sender, transition to `Starting`.
- On `Ensure` in `Starting` / `Warming`: if `sender.receiver_count() >= start_queue_depth` return `StartOutcome::QueueFull` synchronously; else `subscribe` and return receiver.
- On `Running` transition: `sender.send(StartOutcome::Ok)`, then drop the sender.
- On disable/failure: `sender.send(StartOutcome::Err(...))`, drop.

**Router path for `Running`-state services.** The router inspects the supervisor snapshot; if state is `Running`, it skips `Ensure` entirely and proxies directly. Only non-running snapshots go through the `Ensure` → broadcast path. This keeps the running-state hot path free of extra hops and avoids needing to synthesise a pre-resolved broadcast receiver.

### 5.4 Management API read

- `GET /api/services`: lock `ServiceRegistry`, for each handle call `snapshot()`, map to `ServiceSummary`. O(N) services.
- `GET /api/services/{name}`: one `snapshot()` + one SQLite read for last 200 log lines.
- `GET /api/devices`: read-lock the `DeviceSnapshot`, fold in `AllocationTable` reservations, return `Vec<DeviceSummary>`.

### 5.5 OpenAPI

`openapi.rs` declares a single `#[derive(OpenApi)]` struct listing every `#[utoipa::path]` from `openai_api::handlers` and `management_api::handlers`, plus every `#[derive(ToSchema)]` type. A handler `GET /api/openapi.json` returns the serialised JSON.

## 6. Key design calls

- **Atomic for activity, not Instant**: cross-task timestamp visibility needs `AtomicU64` of epoch millis. `Instant::now()` is process-local but not `Copy` between arbitrary tasks cheaply.
- **Broadcast for start coalescing, not Vec<oneshot>**: one broadcast supports arbitrary concurrent waiters with one `send`. `subscriber_count()` also gives us the phase-2 queue-depth check for free.
- **Routers are stateless; state flows via `State(AppState)`**: Axum idiom; lets us add more routes in phase 5 without rewiring.
- **Per-service proxy (phase 1) still works**: unified listener is additive. Both paths update activity through the same atomic.
- **utoipa is phase-2 work, not phase-5**: zero marginal cost now, saves ~30 lines of hand-serialisation plus gives `/api/openapi.json` for free.
- **OpenAI body envelope via `serde_json::Value` flatten**: honest in the OpenAPI schema about what we don't interpret; keeps pass-through semantics.
- **No eviction; `NoFit` → Failed → retry**: lean on phase-1 retry backoff. The user who over-allocates gets a `Disabled { NoFit }` service after 3 tries, with a clear message. Eviction lands in phase 4.
- **No `anankectl` yet**: management API is the integration point; CLI is phase 5.

## 7. Testing

Per user request, explicit end-to-end integration tests that hit the HTTP surfaces, reusing the typed structs from the utoipa-annotated modules so redeclaration is not needed.

### 7.1 Unit

- `openai_api::filters`: `strip_params` removes listed keys; `set_params` overrides; non-JSON bypass; order (strip → set).
- `allocator::can_fit`: feasibility across GPU / CPU / mixed; empty allocation always fits; reserves accumulate.
- `activity`: atomic updates visible across tasks; monotonic.
- `snapshotter`: samples on tick; serves stale value while sampling; shutdown cleanly.
- `supervise`: state transitions for `Ensure` in each state; `ActivityPing` resets deadline; broadcast resolves on `running`.

### 7.2 Integration (under `tests/`, reusing the echo-server harness from phase 1)

- `openai_models.rs` — `GET /v1/models`: lists running + idle services; hides disabled/stopped.
- `openai_chat_routing.rs` — POST `/v1/chat/completions`: routes by `model`, filters applied (assert echo-server sees rewritten body), unknown model → 404, disabled → 503.
- `openai_unimplemented.rs` — POST `/v1/audio/speech`, `/v1/images/generations`: 501 with correct error shape.
- `ondemand_start.rs` — service in `idle`, request arrives, supervisor spawns, response served, supervisor returns to `running`.
- `start_coalescing.rs` — 5 concurrent first-requests to an `idle` service; echo server's spawn-counter hits exactly 1; all 5 responses succeed.
- `start_queue_full.rs` — `start_queue_depth = 2`; during a held start, fire 4 concurrent requests; 2 succeed, 2 get 503 `start_queue_full`.
- `idle_timeout_returns_to_idle.rs` — request served; `idle_timeout` elapses (configure to 2s for the test); supervisor drains child; subsequent request triggers fresh spawn.
- `allocator_insufficient_vram.rs` — FakeProbe reports low free bytes; placement_override exceeds; response is 503 `insufficient_vram` after the retry backoff concludes (or we expose a config flag to shorten backoff for this test).
- `management_services.rs` — `GET /api/services` and `GET /api/services/{name}` return typed shapes matching the utoipa-declared schemas.
- `management_devices.rs` — `GET /api/devices` reflects injected FakeProbe state + current reservations.
- `openapi_json.rs` — `GET /api/openapi.json` parses as valid OpenAPI 3.x with the expected paths and component schemas.

### 7.3 Test harness extensions

- Echo server gains:
  - A spawn counter (each invocation increments an `AtomicU32`; a header on the response reports the current value).
  - A `/sink` endpoint that records the raw body it received into an `Arc<Mutex<Vec<Value>>>`, for filter-application assertions.
- A helper to build a minimal `AppState` against in-memory `FakeProbe` + `Database` + a real `ServiceRegistry` populated with echo-server-backed "services", so tests drive the real OpenAI router without launching a real daemon.

## 8. Risks and open items

1. **Activity-vs-idle race.** Solution: compute the idle deadline inside the select! from the atomic, not from a captured timestamp. On spurious wake-up (atomic updated since deadline scheduled), recompute and re-enter select!.
2. **Broadcast channel cleanup.** If we leak a `Sender`, stale `StartOutcome` can reach new subscribers in the next cycle. Drop the sender explicitly when transitioning out of `Starting`/`Warming` to `Running` or `Idle` (after disable).
3. **`GET /v1/models` consistency.** Spec §10.2 lists `starting`/`warming`/`running`/`idle`. Hide `disabled` / `stopped` / `failed`. Include `idle` so users can see services that *could* run.
4. **Filter validation at config time.** Duplicate key in `set_params` and `strip_params` is a config error (strip + set is redundant; warn).
5. **Backpressure in start queue.** `broadcast::Sender::subscribe` doesn't bound subscribers by queue size; `start_queue_depth` is enforced by comparing `receiver_count()`. A race between two subscribers both seeing `count < N` then subscribing is possible but harmless — over-subscription by one or two is not a correctness issue.
6. **utoipa 5.x API.** Confirm during implementation that `axum` integration works via `utoipa-axum` (or direct `.route` wiring with `utoipa::OpenApi::openapi()`). Version-pin in Cargo.toml with a note.

## 9. Success criteria

- Unified listener accepts `POST /v1/chat/completions` with `stream: true` and forwards SSE chunks transparently.
- An on_demand service configured with `idle_timeout = 5s` starts on first request and returns to `idle` cleanly after 5s of inactivity; next request spawns fresh.
- Concurrent first-requests collapse into a single spawn.
- `/api/services`, `/api/services/{name}`, `/api/devices`, `/api/openapi.json` all return correct typed JSON.
- All phase-1 behaviours (clean shutdown, orphan recovery, per-service proxy) continue to work.
- `just lint` and `cargo test --workspace` (default + `--no-default-features`) pass.
- End-to-end smoke against a real llama-server on redline exercises the unified endpoint; `idle_timeout` visibly unloads and reloads; a concurrent-request burst shows a single spawn.
