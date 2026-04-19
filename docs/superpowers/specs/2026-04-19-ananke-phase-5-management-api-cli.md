# Phase 5 — management API expansion + WebSocket streams + `anankectl` CLI

Status: accepted 2026-04-19. Scope: single phase covering the full §11 (management API) and §14 (CLI) surface of the parent spec (`docs/spec.md`). Lands as one development branch with a workspace restructure first, then the feature work.

## 1. Goal

Bring the daemon to feature-complete for remote + scripted operation:

- Expand the management REST surface from 4 endpoints to 13, covering service lifecycle (start/stop/restart/enable/disable), paginated historical logs, and config read/write/validate.
- Add two WebSocket streams: a system-wide event bus (`/api/events`) and a per-service live log tail (`/api/services/{name}/logs/stream`).
- Ship the `anankectl` CLI binary as a separate crate that speaks HTTP-only against the management API.
- Support opt-in remote access to the management API for Tailscale-style trusted networks — without adding auth or TLS in this phase.

## 2. Scope

### 2.1 In scope

- Workspace restructure into three crates (`ananke`, `anankectl`, `ananke-api`).
- Nine new REST endpoints; four existing endpoints unchanged.
- Two WebSocket streams.
- Full `anankectl` CLI per parent spec §14, including `oneshot run` shorthand.
- `ConfigManager` abstraction — owns raw-TOML + parsed-config state, disk I/O, and the `notify` watcher. Replaces the current ad-hoc `Arc<EffectiveConfig>` held by `AppState`.
- Opt-in non-loopback `management_listen` gated by `daemon.allow_external_management`.

### 2.2 Out of scope

- Authentication / authorisation on the management API. If you bind to the network, you own the security boundary.
- TLS termination. Users are expected to use SSH tunnels, Tailscale, or a reverse proxy with TLS for encryption.
- Log pagination for `allocation_events` or `oneshots` history — only `service_logs` gets the paginated endpoint in this phase.
- Frontend work. This phase ships the API surface the frontend will consume; the React app itself is a later phase.

## 3. Architecture

```
                                     +------------------+
                  HTTP / WS          |   anankectl      |
  (browser UI,  ──────────────────→  |   (CLI crate)    |
   anankectl,                        +--------+---------+
   curl)                                      │  reqwest + clap
                                              │
                                       HTTP   │
                                              ▼
+---------------------------------------------------------------+
| ananke daemon (crate `ananke`)                                |
|                                                               |
|   +--------------------+   +-----------------+                |
|   |  axum routers      |←──| AppState        |                |
|   |   /api/openai/*    |   |  + ConfigManager|                |
|   |   /api/management  |   |  + EventBus     |                |
|   |   /api/oneshot     |   |  + Database     |                |
|   |   /api/events (ws) |   |  + Supervisors  |                |
|   |   /api/.../logs/   |   +--------+--------+                |
|   |       stream (ws)  |            │                         |
|   +--------------------+            │                         |
|             │                       │                         |
|             ▼                       ▼                         |
|   +--------------------+   +-----------------+                |
|   | supervise::*       |   | ConfigManager   |                |
|   |   SupervisorHandle |   |  raw + parsed   |                |
|   |   RunLoop          |   |  notify watcher |                |
|   +--------------------+   +-----------------+                |
|             │                                                 |
|             ▼                                                 |
|   +--------------------+                                      |
|   | EventBus           |←── StateChanged, AllocationChanged,  |
|   | (broadcast<Event>) |    ConfigReloaded, EstimatorDrift    |
|   +--------------------+                                      |
+---------------------------------------------------------------+
                                                           │
            ┌──────────────────────────────────────────────┘
            ▼
    +---------------+
    | ananke-api    |   request/response DTOs shared between
    | (lib crate)   |   the daemon's handlers and the CLI
    +---------------+
```

### 3.1 Workspace layout

```
ananke/
├── Cargo.toml              # [workspace] + [workspace.dependencies]
├── ananke/                 # daemon; everything currently in src/ moves here
│   ├── Cargo.toml          # bin "ananke" + lib (for harness reuse)
│   └── src/
├── anankectl/              # CLI
│   ├── Cargo.toml          # bin "anankectl"
│   └── src/
└── ananke-api/             # shared HTTP DTOs, no logic
    ├── Cargo.toml
    └── src/lib.rs
```

Dependencies live in `[workspace.dependencies]` at the root `Cargo.toml`; member crates pull them via `<name>.workspace = true`. Test deps follow the same pattern.

### 3.2 `ananke-api` contents

Pure serde DTOs — request bodies, response bodies, and the `Event` envelope. No business logic, no `tokio`, no `axum`. `utoipa` annotations live on the daemon's handlers, not on these types, so the shared crate stays dependency-light.

Types:

```rust
// Service views
pub struct ServiceSummary { name, state, lifecycle, priority, port, run_id, pid, elastic_borrower }
pub struct ServiceDetail  { name, state, lifecycle, priority, port, private_port, template,
                            placement_override, idle_timeout_ms, run_id, pid, recent_logs,
                            rolling_mean, rolling_samples, observed_peak_bytes, elastic_borrower }
pub struct LogLine        { timestamp_ms, stream, line, run_id, seq }

// Device views
pub struct DeviceSummary    { id, name, total_bytes, free_bytes, reservations }
pub struct DeviceReservation { service, bytes, elastic }

// Logs pagination
pub struct LogsResponse   { logs: Vec<LogLine>, next_cursor: Option<String> }

// Config
pub struct ConfigResponse { content: String, hash: String }
pub struct ConfigValidateRequest { content: String }
pub struct ConfigValidateResponse { valid: bool, errors: Vec<ValidationError> }
pub struct ValidationError { line: u32, column: u32, message: String }

// Oneshot
pub struct OneshotRequest  { name?, template, command?, workdir?, allocation, devices?, priority?, ttl?, port?, metadata? }
pub struct OneshotResponse { id, name, port, logs_url }
pub struct OneshotStatus   { id, name, state, port, submitted_at, started_at?, ended_at?, exit_code?, logs_url }

// Lifecycle acks
pub struct StartResponse  { status: "already_running" | "started" | "queue_full" | "unavailable", reason? }
pub struct StopResponse   { status: "not_running" | "drained" }
pub struct EnableResponse { status: "enabled" | "not_disabled" | "already_enabled" }
pub struct DisableResponse { status: "disabled" | "already_disabled" }

// Events bus
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum Event {
    StateChanged       { service: SmolStr, from: String, to: String, at_ms: i64 },
    AllocationChanged  { service: SmolStr, reservations: BTreeMap<String, u64>, at_ms: i64 },
    ConfigReloaded     { at_ms: i64, changed_services: Vec<SmolStr> },
    EstimatorDrift     { service: SmolStr, rolling_mean: f32, at_ms: i64 },
    Overflow           { dropped: u64 },
}
```

### 3.3 `ConfigManager`

Lives in `ananke/src/config/manager.rs`. Replaces the current `Arc<EffectiveConfig>` stored in `AppState`.

```rust
pub struct ConfigManager {
    raw:       Arc<RwLock<String>>,            // source of truth for GET /api/config
    effective: Arc<ArcSwap<EffectiveConfig>>,  // read-mostly; cheap snapshot for handlers
    path:      PathBuf,
    watcher:   Option<notify::RecommendedWatcher>,  // kept alive for its lifetime
    events:    EventBus,                       // to publish ConfigReloaded
}

impl ConfigManager {
    pub async fn open(path: &Path, events: EventBus) -> Result<Arc<Self>, ExpectedError>;
    pub fn raw(&self) -> (String, ConfigHash);
    pub fn effective(&self) -> arc_swap::Guard<Arc<EffectiveConfig>>;
    pub async fn apply(&self, new_toml: String, if_match: ConfigHash) -> Result<(), ApplyError>;
    pub fn validate(&self, toml: &str) -> Result<(), Vec<ValidationError>>;
    // Private: spawn_watcher task drives the same apply path on file change.
}

pub enum ApplyError {
    HashMismatch { server_hash: ConfigHash },
    Invalid(Vec<ValidationError>),
    PersistFailed(std::io::Error),
}
```

`ConfigHash` is a `sha2::Sha256`-derived base64 string. Writes go via `toml_edit::DocumentMut` → temp file → `fsync` → `rename`, then the buffer + parsed cache are replaced atomically via `ArcSwap`. The `notify` watcher exists inside the manager, not in `daemon::run`.

### 3.4 `EventBus`

Thin wrapper over `tokio::sync::broadcast`:

```rust
#[derive(Clone)]
pub struct EventBus { tx: broadcast::Sender<Event> }

impl EventBus {
    pub fn new() -> Self { Self { tx: broadcast::channel(EVENT_BUS_CAPACITY).0 } }
    pub fn publish(&self, e: Event) { let _ = self.tx.send(e); }
    pub fn subscribe(&self) -> broadcast::Receiver<Event> { self.tx.subscribe() }
}
```

`EVENT_BUS_CAPACITY = 1024`. Publishers are infallible from the caller's perspective. Subscribers handle `RecvError::Lagged(n)` by emitting `Event::Overflow { dropped: n }` on their WebSocket and resuming.

Publish sites added in this phase:
- `supervise::RunLoop::set_state` → `StateChanged` (old → new)
- Allocation table mutations (on spawn / drain / evict) → `AllocationChanged`
- `ConfigManager::apply` + notify-driven reload → `ConfigReloaded`
- `RollingTable::update` when the mean exceeds a threshold → `EstimatorDrift` (threshold v1 = log only; event emitted for every update that bumps the mean by > 5%)

### 3.5 Log-stream broadcast

`db::logs::BatcherHandle` gains:

```rust
pub fn subscribe(&self) -> broadcast::Receiver<LogLine>
```

Internally: the batcher holds a `broadcast::Sender<LogLine>` with capacity 256. On every successful transaction commit, it sends each line to the broadcast channel in the same order they were inserted. The WebSocket endpoint filters by `service_id` client-side.

## 4. REST surface

Unchanged endpoints (kept as-is):
- `GET /api/openapi.json`
- `GET /api/devices`
- `GET /api/services`
- `GET /api/services/{name}`
- `POST /api/oneshot`, `GET /api/oneshot`, `GET /api/oneshot/{id}`, `DELETE /api/oneshot/{id}`

New endpoints:

| Path | Method | Success | Error cases |
|---|---|---|---|
| `/api/services/{name}/start` | POST | 202 + `StartResponse` | 404 not_found, 409 service_disabled |
| `/api/services/{name}/stop` | POST | 202 + `StopResponse` | 404 not_found |
| `/api/services/{name}/restart` | POST | 202 + `StartResponse` | 404 not_found, 409 service_disabled |
| `/api/services/{name}/enable` | POST | 200 + `EnableResponse` | 404 not_found |
| `/api/services/{name}/disable` | POST | 200 + `DisableResponse` | 404 not_found |
| `/api/services/{name}/logs` | GET | 200 + `LogsResponse` | 404 not_found, 400 bad_query |
| `/api/config` | GET | 200 + `ConfigResponse` | — |
| `/api/config` | PUT | 202 no body | 412 hash_mismatch, 422 validation_failed, 500 persist_failed |
| `/api/config/validate` | POST | 200 + `ConfigValidateResponse` | — |

### 4.1 Logs endpoint

Query params (all optional):
- `since`, `until`: millisecond UNIX timestamps.
- `run`: filter to a specific `run_id`.
- `stream`: `stdout` | `stderr`.
- `limit`: 1..=1000, default 200.
- `before`: opaque cursor from a previous response's `next_cursor`.

Ordering: newest-first. `next_cursor` is present iff more rows exist before the last returned row.

Cursor is base64-encoded JSON `{run_id, seq}`. Opaque to clients; the daemon decodes and adds `(run_id, seq) < (cursor.run_id, cursor.seq)` to the toasty filter.

### 4.2 Service-lifecycle POSTs

All map to existing or new `SupervisorCommand` messages. Handlers never call `transition()` directly; external input cannot trigger a panic from the state machine's invariant check.

- `start` → `SupervisorCommand::Ensure { ack }`. Handler awaits the ack with timeout = service's `max_request_duration_ms`. `EnsureResponse` maps to `StartResponse`.
- `stop` → existing `SupervisorCommand::BeginDrain { reason: UserKilled, ack }`.
- `restart` → `stop` (wait ack) → `start` (wait ack). Serialised.
- `enable` → new `SupervisorCommand::Enable { ack }`. Supervisor handles: if state is `Disabled` → transition to Idle via `transition(state, Event::UserEnable)`, ack `Enabled`; else ack `NotDisabled`.
- `disable` → new `SupervisorCommand::Disable { ack }`. Supervisor handles: if state is Running → begin drain, then transition to Disabled via `transition(state, Event::UserDisable)`; else transition directly. Ack `Disabled`.

CLI-only: `retry <name>` is not its own endpoint. Implemented client-side as "if the service is Disabled, POST /enable; then POST /start". The supervisor's `Ensure` handler in the Failed phase is extended to cancel the pending backoff and transition immediately to Idle+Starting, so a `POST /start` on a Failed service re-attempts straight away rather than waiting out the backoff.

### 4.3 Config endpoints

`GET` / `POST /api/config/validate` are read-only and fast. `PUT` is slow (disk write + reload); the handler returns 202 the moment `ConfigManager::apply` returns `Ok`. Clients that want to confirm the reload completed wait on `Event::ConfigReloaded` via the WS.

`If-Match` is required on `PUT`. Handler parses the header and passes the hash to `ConfigManager::apply`. `ApplyError::HashMismatch { server_hash }` → 412 with the server's current hash in the body.

## 5. WebSocket surface

Both endpoints use axum's `WebSocketUpgrade`. Shared tokio-task layout: receiver task drives `broadcast::Receiver::recv()`; writer task writes JSON frames to the WS; both `select!` on an `app_state.shutdown` signal so clean daemon shutdown closes connections gracefully.

### 5.1 `/api/events`

Optional `?service=<name>` query param. Filtering is client-side in the subscriber task — only events whose `service` field matches are forwarded; service-less events (`ConfigReloaded`) always forward.

Lag handling: `RecvError::Lagged(n)` → emit `Event::Overflow { dropped: n }` and continue.

No backfill on connect.

### 5.2 `/api/services/{name}/logs/stream`

On connect:
1. Resolve `name` → `service_id` via `Database::upsert_service` (or a read-only lookup helper). 404 if the service isn't in the config.
2. Subscribe to the batcher's broadcast channel.
3. Stream `LogLine` frames that match `service_id`.

Lag handling identical to `/api/events`.

## 6. CLI (`anankectl`)

Subcommands (full surface per parent spec §14):

```
anankectl [--endpoint URL] [--json] <subcommand> [args]

devices
services [--all]
show <name>
start <name>
stop <name>
restart <name>
enable <name>
disable <name>
retry <name>                            # short-circuit Failed backoff; Disabled → Idle first
logs <name> [--follow] [--run N] [--since D] [--until D] [--limit N] [--stream stdout|stderr]
oneshot submit <file.toml>
oneshot run [--name N] [--priority N] [--ttl D] [--workdir PATH]
            [--placement gpu-only|cpu-only|hybrid]
            [--vram-gb N | --min-vram-gb N --max-vram-gb N]
            -- <command...>
oneshot list
oneshot kill <id>
config show
config validate [<file>]
config reload
reload                                  # alias for `config reload`
```

- `--endpoint URL` overrides `ANANKE_ENDPOINT` env, which overrides the default `http://127.0.0.1:17777`.
- `--json` on read subcommands emits the raw API JSON; default is a per-subcommand text layout.
- `--follow` on `logs` drains history via `GET`, then upgrades to the WS stream with a seq-based overlap dedup.
- `start` waits for `StateChanged{to: Running}` on the event WS (time-boxed by `max_request_duration_ms` from the service's config, fetched via `GET /api/services/{name}`) before exiting. Failure states exit non-zero with the message from the event.

Binary layout: `anankectl/src/` with `main.rs` dispatching into `commands/<name>.rs` modules. Integration tests: `anankectl/tests/<case>.rs`, launching the daemon in a tempdir via a shared test harness that reuses the existing fixtures from `ananke/tests/common`.

Exit codes:
- 0: success.
- 1: HTTP-level error (4xx/5xx) — body's `error.message` printed to stderr.
- 2: usage error (clap validation failure, etc.).
- 3: connection error (daemon not reachable).

## 7. External-access gate

New config field:

```toml
[daemon]
management_listen = "0.0.0.0:17777"
allow_external_management = true        # required when management_listen is non-loopback
```

Validation in `config::validate` rejects non-loopback bind addresses unless the flag is true. The error message references §11 of the spec and mentions Tailscale/SSH as the trusted-network expectation.

On daemon startup, if the bind is non-loopback, a single `tracing::warn!` line fires:

```
WARN management API reachable from <bind_addr> — no authentication enabled;
     trust your network perimeter (e.g. Tailscale) or terminate TLS + auth at a reverse proxy.
```

## 8. Error handling

- New endpoints return the existing OpenAI-shaped error body: `{"error": {"code": "<slug>", "message": "...", "type": "server_error"}}`.
- WebSocket error cases (bad service name, subscribe failure) close the connection with a structured close frame carrying a text reason.
- `ApplyError::PersistFailed` surfaces the underlying `io::Error` in the response body and logs it at `error!` level on the daemon side.
- `transition()` stays panicking-on-illegal for internal callers. External HTTP input cannot reach those call sites (handlers route through `SupervisorCommand`).

## 9. Testing

Unit:
- `ananke-api` DTOs round-trip through serde (serialize → deserialize → equality).
- `ConfigManager::apply` — hash mismatch, invalid TOML with span errors, successful apply publishes a `ConfigReloaded` event.
- `EventBus` basic subscribe + publish + lag → overflow.
- Log cursor encode / decode round-trip.

Integration (daemon side, in `ananke/tests/`):
- Each new endpoint: happy path + 404 + 400/409/412/422 as applicable.
- WS: connect, observe an event after triggering a state change, disconnect cleanly.
- Service-lifecycle endpoints: start/stop/restart/enable/disable produce the expected sequence of events.
- External-listen gate: non-loopback without the flag refuses to start.

Integration (CLI side, in `anankectl/tests/`):
- Each subcommand hits its endpoint, exit code correct, stdout format correct.
- `logs --follow` receives live lines after spawning a service that prints.
- `config show | config validate - | config apply` roundtrip.

Extended smoke: unchanged from prior phases — the phase-5 work doesn't touch the scheduling / estimator / spawn paths.

## 10. Implementation sequencing

One development branch, structured as a series of focused commits:

1. Workspace restructure (no feature work). Green tests at end.
2. `ananke-api` crate with all DTOs. Daemon migrates its handlers to return these types. No new endpoints yet; existing four unchanged behaviourally.
3. `EventBus` + publisher sites (state change, allocation change). Unit tests.
4. `ConfigManager`. Move the `notify` watcher + parsed-config storage into it. Daemon replaces its direct `Arc<EffectiveConfig>` with `manager.effective()` calls.
5. Service-lifecycle endpoints (start/stop/restart/enable/disable) + new `SupervisorCommand` variants.
6. Paginated logs endpoint.
7. Config endpoints (GET/PUT/validate).
8. `/api/events` WebSocket.
9. `/api/services/{name}/logs/stream` WebSocket + batcher broadcast.
10. External-access gate + validation + startup warn.
11. `anankectl` crate scaffolding + common commands (devices, services, show).
12. `anankectl` lifecycle commands (start/stop/restart/enable/disable/retry).
13. `anankectl` logs (+ --follow).
14. `anankectl` oneshot (submit, run, list, kill).
15. `anankectl` config (show, validate, reload).

Each step is testable in isolation and leaves the tree green.

## 11. Success criteria

- All 13 REST endpoints from the parent spec §11.1 implemented and exercised by integration tests.
- Both WebSocket streams in §11.2 implemented with integration-test coverage.
- `anankectl` binary builds + every subcommand in §14 has an integration test.
- `cargo test --workspace --features test-fakes` green across all three crates.
- `cargo clippy --all-targets --all-features -- -D warnings` clean.
- `daemon.allow_external_management = true` + non-loopback bind serves requests from the LAN as expected, with the startup warning emitted.
- Extended smoke on redline (existing scenarios) passes post-refactor.

## 12. Risks

1. **WebSocket lag on noisy services.** A verbose llama-server emitting thousands of log lines/second could saturate the broadcast channel capacity, producing frequent `dropped` frames for every tail subscriber. Mitigation: start with 256-element broadcast; if it proves insufficient, shard by `(service_id, run_id)` with separate broadcast channels.
2. **`notify` watcher + PUT race.** A PUT write + its subsequent notify-fire could race, producing two `ConfigReloaded` events for the same logical reload. Mitigated by content-hash dedup in `ConfigManager::apply` (if the new content matches the in-memory buffer, skip the reload).
3. **`ConfigManager::apply` validation cost.** Full parse + validate on every PUT is O(N) in service count. At the target deployment scale (<100 services) this is irrelevant; at 10k services it'd be worth caching. Not a v1 concern.
4. **`anankectl logs --follow` overlap dedup.** When the client drains `GET /logs` and then subscribes to the WS, a brief overlap window exists between the last REST row and the first WS frame. Dedup by remembering the last `(run_id, seq)` seen and dropping WS frames `<=` that cursor.
