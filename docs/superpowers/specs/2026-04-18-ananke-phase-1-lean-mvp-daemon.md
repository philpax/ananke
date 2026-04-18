# Phase 1: Lean MVP daemon — design

Status: accepted 2026-04-18. Parent design: `docs/spec.md` (authoritative). This document scopes the first vertical slice of Ananke.

## 1. Goal

Deliver a long-running `ananke` binary that:

- Loads a TOML config from the standard search path.
- Starts one or more persistent `llama-cpp` services as child processes at boot.
- Transparently HTTP-proxies traffic to each service on its declared port, preserving SSE.
- Captures stdout and stderr to SQLite with the spec's retention policy.
- Survives its own restart without orphaning or double-spawning children.
- Shuts down cleanly on SIGTERM, SIGINT, and SIGQUIT.

Non-goals for this phase: scheduler logic beyond honouring `placement_override`, VRAM estimation, on-demand lifecycle, unified OpenAI endpoint, `command` template, dynamic services, oneshots, the management API surface (beyond what's needed to serve the per-service proxy), the `anankectl` CLI, the frontend, and the Nix module. These are phases 2–7.

## 2. Roadmap context

This phase is step 1 of 7. The remaining steps exist in the parent spec and are summarised here so scope decisions stay honest:

1. **← this** — Lean MVP daemon.
2. Unified OpenAI endpoint, on-demand lifecycle with `idle_timeout`, start-future coalescing, request-body filters, single-device best-fit allocator.
3. GGUF reader, architecture-aware VRAM estimator, layer-aware multi-GPU placement, rolling correction.
4. Balloon resolver, `command` template, dynamic services, oneshots, priority-based eviction.
5. Full management API surface and `anankectl` CLI.
6. Frontend (devices dashboard, services table, config editor, oneshot launcher).
7. Nix module, release polish, manual runbooks.

Each later phase gets its own brainstorm → design → plan → execute cycle.

## 3. Scope

### 3.1 In scope

**Config loading and parsing.**

- File-path resolution: `$ANANKE_CONFIG` → `--config` CLI flag → `$XDG_CONFIG_HOME/ananke/config.toml` (default `~/.config/ananke/config.toml`) → `/etc/ananke/config.toml`. First match wins. Log the resolved path at startup.
- Data directory mirror: `$XDG_DATA_HOME/ananke/` (default `~/.local/share/ananke/`) for SQLite and state.
- Parse with `toml_edit` preserving comments and formatting, so later phases can round-trip config-editor edits without rewriting the file.
- `extends` inheritance at parse time, before validation: scalars and dotted-leaves override, sub-tables deep-merge, arrays child-replace-parent, `*_append` siblings concatenate (`parent.foo ++ parent.foo_append ++ child.foo ++ child.foo_append`, with `child.foo` falling back to `parent.foo` if not specified).
- `migrate_from` chain resolution in dependency order; cycles are errors; missing source is a warning.
- Validation errors carry file + line + column spans via `toml_edit`.

**Template support.**

- `llama-cpp` only.
- Full field surface from spec §7.1 is *accepted* (including `sampling.*`, `filters.*`, `cache_type_*`, `flash_attn`, `mmproj`, `override_tensor`, `extra_args`, `env`, `metadata.*`). Most of these are rendered onto the command line or stashed for later phases; a few are inert in phase 1 (filters are unused until the unified listener lands in phase 2).
- `mmproj` is rendered as `--mmproj <path>` on the command line (added to the spec today).

**Service lifecycle.**

- `persistent` only. All configured services start at daemon boot, in `priority` descending then `name` ascending order.
- `on_demand` and `oneshot` are rejected with a clear error message pointing at phase 2+.
- `lifecycle = "oneshot"` inside a `[[service]]` block is a validation error (spec §5.1).

**Placement.**

- `devices.placement_override` is **required in phase 1** for every service. A service without one is rejected at config-load time with a validation error explaining that the estimator lands in phase 3. This gate is removed when the estimator ships.
- `devices.placement` is still honoured for validation (overriding CPU bytes with `placement = "gpu-only"` is an error, per spec §8.2.6).
- `devices.gpu_allow` is accepted and narrows the allowed device set.
- No layer-aware walker, no `-ngl` derivation, no `--tensor-split` computation. Pass everything through `extra_args` as spec §8.2.6 requires when the override is active.
- Allocation table (`Mutex<BTreeMap<ServiceName, Allocation>>`) is introduced this phase and populated from overrides. Phase 2–3 *extend* it rather than replace it.

**Device probing.**

- `GpuProbe` trait (`list`, `query`, `processes`) with an NVML-backed impl and an in-memory fake for tests (spec §15.3).
- `CUDA_VISIBLE_DEVICES` unset at daemon startup so NVML sees every GPU.
- CPU accounting via `/proc/meminfo` `MemAvailable` (spec §4.2).
- Log the NVML-reported GPU set at startup.

**Child spawning.**

- `prctl(PR_SET_PDEATHSIG, SIGTERM)` on every child (via `nix`).
- Freshly-computed `CUDA_VISIBLE_DEVICES` per spawn from the allocation; renumbered from 0.
- `cpu-only` services spawn with `CUDA_VISIBLE_DEVICES=""`.
- No ambient `CUDA_VISIBLE_DEVICES` inherited.
- Command line rendered from config fields plus `extra_args` plus `extra_args_append`.

**Proxy.**

- One `hyper` reverse proxy per service. The daemon owns the external socket on `service.port`; the child binds a private loopback port (`127.0.0.1:<private_port>`, allocated by the daemon from an internal pool). The child's `--port` is set to the private port. The proxy forwards `service.port → private_port`.
- This split is introduced in phase 1 deliberately so drain semantics work (daemon can stop accepting new connections without killing the child) and so later phases can slot in filters and start-future coalescing without moving the port layout around.
- SSE passthrough validated in tests — no buffering, prompt flushes.
- No filters applied in phase 1. The unified listener and its filter pipeline come in phase 2.

**Health checks.**

- HTTP probe against `health.http` every `health.probe_interval` (default 5 s).
- Probe timeout → accumulate; wall-clock from launch > `health.timeout` (default 180 s for llama-cpp) transitions the service to `disabled(health_timeout)` with SIGTERM + drain.

**State machine.**

- Subset of spec §5.3 states: `idle`, `starting`, `warming`, `running`, `draining`, `stopped`, `failed`, `disabled`.
- Valid transitions enforced by a `state_machine::transition(from, event) -> Result<To, InvalidTransition>` function.
- `warming → running` on `observed_usage ≥ 80% of reservation` *or* `warming_grace` expiry. In phase 1 "observed usage" means GPU VRAM sampled via NVML every 2 s (the infrastructure is needed for phase 4 anyway; introducing it here is cheap and gives us warming correctness).
- Launch retry backoff: 2 s → 5 s → 15 s, then `disabled(launch_failed)`.
- Crash loop: >5 exits in 10 min → `disabled(crash_loop)`.

**Log capture.**

- stdout and stderr pumped to SQLite via `db::logs::Batcher` (200 ms / 100-line flush thresholds).
- Per-service retention: 7 days or 50 000 lines (whichever smaller), plus a pre-exit 500-line buffer preserved outside the rolling window.
- Retention trim runs at 3 am local; hourly `PRAGMA incremental_vacuum(N)`.

**Clean shutdown.**

- SIGTERM and SIGINT handled identically: stop accepting new connections, drain in-flight bounded by `daemon.shutdown_timeout` (default 120 s), SIGTERM children with 10 s grace, SIGKILL stragglers, clear `running_services`, close SQLite.
- SIGQUIT: skip drain, SIGTERM children immediately with 5 s grace, then SIGKILL.

**Orphan recovery (spec §9.3, strict).**

- On startup, read `running_services` rows.
- For each row, check PID liveness (`kill(pid, 0)` / `/proc/<pid>`).
- If alive and `/proc/<pid>/cmdline` matches → adopt; mark `running`, first request triggers a health check.
- If alive and cmdline mismatches → log, clean up row, do not kill.
- If dead → clean up row.
- NVML scan: for PIDs holding memory on managed GPUs that we don't recognise, require **both** (a) port binding via `/proc/net/tcp{,6}` + `/proc/<pid>/fd` inode correlation matches one of our service ports, **and** (b) cmdline matches a recorded `command_line` value for that service, before SIGTERM. Otherwise log and leave alone.

**Logging.**

- `tracing` → stderr. journald captures under systemd; `cargo run` surfaces on the terminal. No separate file-based daemon log.

### 3.2 Out of scope (deferred)

- Unified OpenAI listener, request-body filters, `/v1/models`, `/v1/chat/completions` routing → phase 2.
- `on_demand` lifecycle, `idle_timeout`, start-future coalescing, `start_queue_depth` → phase 2.
- Single-device best-fit allocator with eviction, `no_fit` disable reason at startup → phase 2.
- GGUF header reader, tensor enumeration, sharded GGUFs, mmproj estimator contribution → phase 3.
- Architecture-aware VRAM estimator, `safety_factor`, rolling correction, drift warnings → phase 3.
- Layer-aware multi-GPU placement, `--tensor-split` derivation, compute-buffer accounting, tensor-split slop fudge → phase 3.
- Balloon resolver for dynamic services → phase 4.
- `command` template and its placeholder substitution → phase 4.
- Oneshots and their port pool → phase 4.
- Priority-based eviction with drain → phase 4.
- Management API endpoints beyond the per-service proxy listeners → phase 5.
- `/api/events` WebSocket, `/api/services/*`, `/api/config`, `/api/oneshot` → phase 5.
- `anankectl` CLI → phase 5.
- Frontend → phase 6.
- Nix module and flake output → phase 7.

## 4. Architecture

```
                          +-------------------+
                          |    config.toml    |
                          +---------+---------+
                                    |
                                    v
+----------+    +-----------------------------------+    +------------+
|  NVML    | -> |            Daemon (tokio)         | -> |  SQLite    |
|  probe   |    |                                   |    |  (toasty)  |
+----------+    |   +---------------------------+   |    +------------+
                |   |     Allocation table       |   |
                |   |   Mutex<BTreeMap<Svc,     |   |
                |   |          Allocation>>     |   |
                |   +---------------------------+   |
                |                                   |
                |   +---------------------------+   |
                |   |   Supervisor tasks        |   |
                |   |   (one per service)       |   |
                |   +---------------------------+   |
                |                                   |
                |   +---------------------------+   |
                |   |   Per-service proxies     |   |
                |   |   (hyper reverse)         |   |
                |   +---------------------------+   |
                +-----------------------------------+
                                    |
                      +-------------+-------------+
                      v             v             v
                 +--------+    +--------+    +--------+
                 |llama-  |    |llama-  |    |llama-  |
                 |server #1    |server #2    |server #N
                 +--------+    +--------+    +--------+
```

The daemon orchestrates three kinds of long-running work:

- **Supervisor tasks** — one per configured service. Each owns its child's lifetime, health loop, log pumps, state transitions, and allocation-table entry. Communicates inward via an `mpsc::Sender<SupervisorCommand>`.
- **Proxy listeners** — one per service's declared port. Forward incoming HTTP requests to the child.
- **Signal handler + retention trim** — daemon-global.

The allocation table is the single source of truth for placement. Phase 1 writes it once at boot (from `placement_override`) and reads it on spawn. Later phases add writers (scheduler, balloon resolver) that serialise through a mutex.

## 5. Module layout

```
src/
├── main.rs                 // bin: ananke
├── lib.rs                  // re-exports for integration tests
├── errors.rs               // ExpectedError + internal error types
├── config/
│   ├── mod.rs              // public: load_config, EffectiveConfig
│   ├── file.rs             // path resolution, file IO, $ANANKE_CONFIG / --config / XDG
│   ├── parse.rs            // toml_edit -> raw Config struct tree
│   ├── merge.rs            // extends + *_append, migrate_from chain resolution
│   └── validate.rs         // span-annotated ConfigError; phase-1 gate on placement_override
├── db/
│   ├── mod.rs              // toasty bootstrap, migrations
│   ├── schema.rs           // services, service_config_versions, running_services, service_logs, allocation_events
│   └── logs.rs             // 200ms / 100-line batching writer
├── devices/
│   ├── mod.rs              // Device, DeviceId, DeviceSnapshot, Allocation
│   ├── probe.rs            // GpuProbe trait + NvmlProbe
│   ├── fake.rs             // #[cfg(any(test, feature = "test-fakes"))] in-memory probe
│   └── cpu.rs              // /proc/meminfo reader
├── state.rs                // ServiceState enum + valid transition fn
├── supervise/
│   ├── mod.rs              // Supervisor task; SupervisorCommand; SupervisorHandle
│   ├── spawn.rs            // child spawn, prctl, CUDA_VISIBLE_DEVICES, argv rendering
│   ├── health.rs           // HTTP probe loop
│   ├── logs.rs             // stdout/stderr pumps -> db::logs::Batcher
│   └── orphans.rs          // startup recovery (cmdline + port cross-check)
├── proxy.rs                // per-service hyper reverse proxy (SSE-safe)
├── signals.rs              // SIGTERM/SIGINT/SIGQUIT handling, shutdown sequencing
├── retention.rs            // 3am trim + hourly incremental_vacuum
└── daemon.rs               // top-level orchestration; wires everything together
```

Public API surface for library consumers (mainly integration tests): `lib.rs` re-exports `config::load_config`, `db::Database`, `devices::GpuProbe`, `devices::fake::FakeProbe`, `state::ServiceState`, and `daemon::Daemon` with an injectable probe.

## 6. Data flow

### 6.1 Boot

1. Parse CLI args (`--config PATH`).
2. Resolve config path (env → CLI → XDG → `/etc`).
3. Read file, parse with `toml_edit`, merge inheritance, resolve `migrate_from`, validate. Fatal daemon-level errors (unparseable TOML, no devices, port collisions with daemon listeners) exit non-zero with a clear message. Per-service errors mark the offending service `disabled(config_error)` and continue.
4. Open SQLite via toasty, run migrations. `PRAGMA auto_vacuum = INCREMENTAL` must run before any `CREATE TABLE` in migration 0001.
5. Unset `CUDA_VISIBLE_DEVICES`, probe NVML, log the detected device set.
6. Load `running_services` rows from the previous session; run orphan recovery.
7. For each persistent service in priority-then-name order:
   - Insert allocation from `placement_override` into the allocation table.
   - Spawn supervisor task, which spawns the child.
8. Bind per-service proxy listeners.
9. Bind signal handler and retention loop.
10. Enter steady state.

### 6.2 Per-request

1. Client connects to `service.port` on the daemon's bind address.
2. Proxy reverse-proxies to `127.0.0.1:<service.port>` inside the child. (Phase 1 does not yet separate the external listen port from the internal child port — that's a phase-2 concern when filters land. For phase 1, the child and the proxy listen on the same port, with the proxy in front.)

**Note on port binding**: in phase 1, the daemon owns the external socket and the child binds a different internal port. Resolution: daemon allocates a private loopback port (e.g. `127.0.0.1:<40000 + hash(name) % 10000>`) for the child, and the child's `--port` argument is set to that private port; the proxy listens on `service.port` externally and forwards to the private port. This keeps phase-1 semantics honest (daemon controls the wire) without over-engineering filters that don't exist yet.

### 6.3 Shutdown

1. Signal handler fires.
2. Proxy listeners stop accepting new connections.
3. Supervisors enter `draining`: refuse new requests, wait for in-flight to complete bounded by `daemon.shutdown_timeout` (120 s default).
4. SIGTERM children. 10 s grace.
5. SIGKILL stragglers.
6. Clear `running_services`. Flush remaining log batches. Close SQLite.
7. Exit 0.

SIGQUIT skips step 3, uses 5 s grace at step 4.

## 7. Error handling

Two tiers per CONTRIBUTING.md:

- `ExpectedError`: user/external errors with semantic exit codes. Fatal daemon conditions: cannot bind management port, cannot open SQLite, config file unparseable as TOML, no devices available.
- Internal errors: programming errors that panic or surface via internal enum types.

No `thiserror`. Each error type implements `std::fmt::Display` manually with lowercase sentence fragments ("failed to bind 127.0.0.1:7777"). `ErrorKind` enums group categories.

Per-service errors never crash the daemon; they flow through the `disabled` state per spec §5.4 (`config_error`, `launch_failed`, `health_timeout`, `oom` — the last only via fallback rules since there's no estimator yet, `crash_loop`, `no_fit` — deferred, `user_disabled` — via API, deferred).

## 8. SQLite schema (phase 1 subset)

Migration 0001 creates the full set of tables from spec §12 up front — even those we don't write to yet — so later phases don't need schema migrations for tables that were always going to exist:

- `services` (service_id, name, created_at, deleted_at)
- `service_config_versions` (service_id, version, effective_config JSON, recorded_at)
- `running_services` (service_id, run_id, pid, spawned_at, command_line, allocation JSON, state)
- `service_logs` (service_id, run_id, timestamp_ms, seq, stream, line)
- `allocation_events` (event_id, service_id, run_id, event_type, device, bytes, at)
- `oneshots` (id, service_id, submitted_at, started_at, ended_at, exit_code, ttl_ms) — created empty; first writer is phase 4.

Indexes: `service_logs(service_id, run_id, timestamp_ms)`.

Pragmas: `journal_mode = WAL`, `synchronous = NORMAL`, `auto_vacuum = INCREMENTAL` (set pre-table-creation), `foreign_keys = ON`.

## 9. Concrete key design calls

- **Supervisor loop shape**: each supervisor is a single `async fn run(mut rx: mpsc::Receiver<SupervisorCommand>, ...)` with a `select!` over `rx.recv()`, the health tick, the NVML-observation tick, the child exit future, and the log pumps. This keeps the state machine local to one task.
- **Health + observation sampling**: 2 s NVML sample cadence is shared between warming detection (phase 1) and the balloon resolver (phase 4). Introducing the sampler here means phase 4 inherits it.
- **Log batcher contract**: `Batcher::push(ServiceId, RunId, Stream, line)` is fire-and-forget; the batcher owns a single writer task that commits every 200 ms or every 100 lines, whichever first. Batcher must flush before daemon exit.
- **Orphan adoption has no health probe on adoption**: spec §9.3 says "first request triggers a health check". Phase 1 inherits that rather than eagerly probing.
- **Allocation-table lock discipline**: take the mutex, read/write, drop. Never await while holding. Phase 4's scheduler will keep this invariant.
- **`priority` accepted but latent**: the field is parsed and stored; startup ordering uses it; there's no eviction yet to enforce it. This avoids a config-format churn when phase 4 lands.

## 10. Testing

### 10.1 Unit

- Config parse + merge: known-good fixtures produce expected effective config; `extends` cycles rejected; `migrate_from` chains resolved in order; `*_append` associativity (proptest); child-replace-parent arrays (proptest); dotted-leaf override.
- Validation: each error variant from spec §6.5 (file-level) produces a `ConfigError` with a span. `placement_override` gate rejects services without one (phase-1 only gate).
- State transitions: `valid_transition(from, event) -> Option<To>` exhaustively tested for every pair.
- Log batcher: flushes at 200 ms; flushes at 100 lines; flushes on shutdown; respects per-service ordering with the `seq` counter.
- `CUDA_VISIBLE_DEVICES` renumbering: given a `[1, 3]` allocation on a 4-GPU host, child sees `CUDA_VISIBLE_DEVICES=1,3` (NVML indices) and llama.cpp sees GPUs renumbered to 0, 1. `cpu-only` gets `CUDA_VISIBLE_DEVICES=""`.

### 10.2 Property

- Config merge invariants.
- State-machine reachability: from any reachable state, every spec-permitted event transitions to a valid next state.

### 10.3 Integration

Uses a **toy HTTP echo server** as a stand-in for llama-server. Drives the daemon through the in-memory `GpuProbe` fake:

- End-to-end: load config declaring one echo service, daemon boots, proxy request round-trips.
- SSE passthrough: echo server emits `text/event-stream`; proxy forwards without buffering; test asserts chunk arrival timestamps.
- Child crash: echo server exits nonzero; daemon backs off 2 s / 5 s / 15 s and then `disabled(launch_failed)`.
- Health timeout: echo server never becomes healthy; daemon SIGTERMs after `health.timeout` with `disabled(health_timeout)`.
- Orphan recovery: start daemon, wait for spawn, kill daemon with SIGKILL (leaves child alive), restart daemon with same config, assert adoption (new `run_id` is *not* allocated; existing row remains).
- Unknown orphan: pre-spawn a stranger on a managed GPU via the `FakeProbe`; restart daemon; assert stranger not SIGTERM'd, log line emitted.
- Clean shutdown: SIGTERM daemon with an in-flight streaming request; request completes (or `shutdown_timeout` fires); child SIGTERM'd; SQLite flushed.

No real `llama.cpp` in CI. A manual runbook in `tests/manual/phase-1-smoke.md` covers the first real launch.

### 10.4 Test fakes

- `devices::fake::FakeProbe`: in-memory `GpuProbe` with a controllable ledger of pretend GPUs, observed free bytes, and processes. Used by all integration tests.
- `tests/support/echo_server.rs`: minimal `hyper` server for integration tests; spawned as a child to exercise the spawn path.

## 11. Tooling

- **`justfile`** (first cross-cutting recipe; CONTRIBUTING.md budgets for it): `just lint` = `cargo fmt --all --check`, both clippy invocations with `-D warnings`, `cargo test --workspace` (and `--no-default-features`), and `npm run lint` (even though the frontend doesn't have meaningful content yet — keeps the command honest).
- Pre-commit: none in phase 1. Add in a later phase if contributor flow demands it.

## 12. Risks and open items

1. **Toasty `auto_vacuum = INCREMENTAL` timing** (spec §18). If toasty can't issue the pragma before table creation, fall back to a raw-SQL migration 0001 and let toasty manage subsequent migrations. Validated during implementation, not design.
2. **Hyper SSE passthrough correctness**. Buffering bugs here are silent until a real llama-server stream is tested. Integration test uses a chunked-emitting echo server to catch regressions.
3. **Orphan cmdline matching is strict by design**. A slight command-line difference after a config reload could cause an adopted child to be missed and a stranger-killed. Phase 1 errs on "don't kill strangers"; the user sees a port-bind failure and intervenes. This is the safe side per spec §9.3.
4. **`warming → running` without an estimator**. Phase 1 has no estimate of "80% of reservation" for a hybrid model because there's no reservation computation, only the `placement_override` numbers. Those *are* the reservation, so 80% of those is the threshold — fine.
5. **`[[persistent_service]]` alternative syntax** (spec §6.2). Accept at parse time, translate to `lifecycle = "persistent"` before validation. Low complexity; keeps the parent spec honest.

## 13. What success looks like

At the end of phase 1:

- `ananke --config path/to/config.toml` runs, loads the config, starts one or more persistent `llama-cpp` services, and proxies HTTP traffic to each.
- `systemctl restart ananke.service` (or equivalent SIGTERM/SIGKILL cycle) does not orphan children.
- Kill the daemon ungracefully, restart it: children are adopted.
- `journalctl -u ananke -f` shows structured `tracing` output.
- `sqlite3 ~/.local/share/ananke/ananke.sqlite "select count(*) from service_logs"` returns a growing number.
- All CI gates pass (`just lint`, `cargo test --workspace` with and without default features, `npm run lint`).

The user can point this at their redline for a subset of models that don't need the scheduler — anything with a known static footprint and a `placement_override`. Migration of the full lmp config waits for phases 2–4.
