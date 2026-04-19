# Toasty migration — design

Status: accepted 2026-04-19. Scope: single-phase refactor. Parent design `docs/spec.md` §12 specifies toasty as the database layer; phase 1's plan took the spec-sanctioned rusqlite fallback, which never got revisited. Phases 1-4 now migrate to toasty wholesale.

## 1. Goal

Replace the hand-rolled rusqlite surface (`src/db/`, plus ~15 direct rusqlite callers across supervise, retention, oneshots, orphans, management API) with a toasty-backed `Database` type whose models carry the schema, query paths use the toasty builder DSL, and migrations are owned by the toasty CLI. The user's direction: all-or-nothing; burn existing data if it helps; revert only if toasty provably cannot meet a real requirement.

## 2. Scope

### 2.1 In scope

**Dependencies**
- Add `toasty = "0.4"` with `sqlite` feature. Remove top-level `rusqlite` dependency on the application side. Keep `rusqlite` transitively via `toasty-driver-sqlite` and pull it in explicitly for the pragma side-channel (§5.1).
- Install `toasty-cli` as a development dependency; include a `just db-migrate` recipe that wraps `toasty migrate generate`.

**Schema**
- Six models via `#[derive(toasty::Model)]`: `Service`, `ServiceConfigVersion`, `RunningService`, `ServiceLog`, `AllocationEvent`, `Oneshot`. Field set and uniqueness constraints mirror the SQL schema in today's `src/db/schema.rs::MIGRATION_0001`.
- `RunningService` and `ServiceLog` use composite keys (`#[key(partition=service_id, local=run_id)]` and `#[key(partition=service_id, local=seq)]` respectively).
- `service_logs` gets an index on `(service_id, timestamp_ms)` via `#[index]` to support the management-API "most recent 200" query.
- `services.name` gets `#[unique]` to support `upsert` by name.

**Pragma side-channel**
- A thin `pragma` module that opens a short-lived `rusqlite::Connection` when needed. Used for:
  - `PRAGMA auto_vacuum = INCREMENTAL` on freshly-created DB files (must run before any `CREATE TABLE`; toasty creates tables during `push_schema`).
  - `PRAGMA incremental_vacuum(N)` during retention runs.
- Workflow on daemon start: create/open the file with rusqlite, set auto_vacuum if this is the first-open, close. Then `toasty::Db::builder()...connect(url)`.

**Database type**
- `src/db/mod.rs::Database` becomes a thin wrapper over `toasty::Db`. Keeps the clone-able, Arc-like shape so existing consumers (supervisors, handlers, snapshotter) need no structural change.
- Current methods (`open`, `schema_version`, `upsert_service`, `reparent`, `with_conn`, `with_conn_mut`, `path`) become toasty-backed. The `with_conn*` escape hatches go away — all callers move to toasty builders.

**Query migrations** (by call site)
- `db::logs::Batcher` — bulk insert of `ServiceLog` rows via toasty's `Vec<CreateStatement>` batching inside a single transaction. Preserves 200 ms / 100-line flush thresholds and per-(service, run) `seq` counter.
- `retention::trim_logs_once` — two `Delete` queries: (1) `ServiceLog` where `timestamp_ms < cutoff`; (2) per-service, `Delete` via `path.in_query(subquery)` selecting the oldest-N rows over the cap.
- `supervise::orphans::reconcile` — `RunningService::all()` → iterate PIDs, cleanup with `delete()` where mismatched/dead.
- `supervise::mod::run` — `RunningService` insert on spawn, delete on drain; uses toasty builders instead of `db.with_conn(|c| c.execute(...))`.
- `oneshot::handlers::list_oneshots` / `get_oneshot` — `Oneshot` queries with `order_by` + `limit`.
- `management_api::handlers::service_detail` — `ServiceLog::filter(...).order_by(...).limit(200).exec(...)`.
- `app_state::spawn_oneshot` — `Oneshot` insert.

**Migrations**
- Generate initial migration `0000_migration.sql` via `toasty migrate generate`. Commit it under `toasty/migrations/`.
- `Toasty.toml` at the repo root configures the migration directory (`path = "toasty"`, `prefix_style = "Sequential"`, `checksums = false`, `statement_breakpoints = true`).
- Database bootstrap on `Database::open`: apply migrations via `toasty-cli` crate programmatically, or via raw SQL execution of committed migration files. Decision made during implementation (Task 3) based on what `toasty-cli` exposes as a library.

### 2.2 Out of scope

- Persisting rolling correction across daemon restarts (still phase-5+).
- Persisting oneshot history beyond the existing `oneshots` table.
- Backfilling existing sqlite DB files to the new schema. Per user direction: burn the data.
- Non-SQLite backends. Toasty supports Postgres/MySQL/DynamoDB; we stay on SQLite.

## 3. Architecture

```
             +-------------------+
             |  Database (new)   |
             |                   |
             |  toasty::Db       |
             |   pool            |
             |   schema          |
             +--------+----------+
                      |
            +---------+----------+----------------+
            |                    |                |
            v                    v                v
       model queries        transactions     PRAGMA side-channel
       (supervise,          (log batcher)    (rusqlite::Connection
        retention,                             opened ad-hoc for
        orphans,                               auto_vacuum +
        management)                            incremental_vacuum)
```

Toasty owns schema + migrations + most queries. The rusqlite side-channel exists only for PRAGMAs toasty can't issue.

### 3.1 Module layout (changed)

```
src/
├── db/
│   ├── mod.rs             // REWRITE: Database wraps toasty::Db
│   ├── models.rs          // NEW: #[derive(Model)] structs for the 6 tables
│   ├── pragma.rs          // NEW: thin rusqlite side-channel for PRAGMA
│   ├── logs.rs            // REWRITE: Batcher uses toasty batched inserts
│   └── migrations.rs      // NEW: load + apply SQL migrations from toasty/migrations
├── retention.rs           // MODIFY: toasty delete + in_query; keep pragma call
├── supervise/
│   ├── mod.rs             // MODIFY: RunningService toasty queries
│   └── orphans.rs         // MODIFY: toasty queries
├── oneshot/
│   ├── handlers.rs        // MODIFY: Oneshot toasty queries
│   └── ...
├── management_api/
│   └── handlers.rs        // MODIFY: ServiceLog, services detail queries
├── app_state.rs           // (unchanged; still holds Database)
└── daemon.rs              // MODIFY: bootstrap sequence

toasty/
├── migrations/
│   └── 0000_initial.sql   // NEW: generated via toasty-cli
├── snapshots/
│   └── 0000_snapshot.toml // NEW: generated
└── history.toml           // NEW: generated
Toasty.toml                 // NEW: migration config
```

### 3.2 Error handling

Toasty has its own `toasty::Error`. At module boundaries, map to `ExpectedError::database_open_failed` (startup) or log + degrade (runtime). The database-open error surface stays the same; the internal errors change. Soft failures in log batcher / retention stay soft (`warn!` and continue).

## 4. Key design calls

- **Bridge for pragmas, not a replacement**: The side-channel is limited to two pragmas (`auto_vacuum`, `incremental_vacuum`). No other SQL goes through rusqlite. Prevents dual-path drift.
- **Batcher stays a tokio task**: Same 200 ms / 100 line thresholds, same mpsc channel shape, same RAII guard. The inner SQL changes, the public interface doesn't.
- **Composite keys**: `RunningService` is keyed `(service_id, run_id)`; `ServiceLog` keyed `(service_id, seq)`. Toasty's `#[key(partition, local)]` supports this.
- **No transitive `rusqlite` in Cargo.toml re-export**: Application code doesn't touch rusqlite types directly (only the pragma module does, internally). Callers see only toasty types and our models.
- **Integration tests use the same `Database::open`**: Toasty's `sqlite::memory:` works for tests; tests use tempdir-backed files so migrations are actually exercised.
- **Migrations are committed to repo**: The generated SQL files in `toasty/migrations/` are source-controlled, not generated at daemon start. Matches toasty's own example project layout.

## 5. Risks

1. **Pragma side-channel ergonomics**: opening a second connection type, even briefly, is a minor abstraction leak. Accepted because the alternative (upstream PR to toasty) is out of scope.
2. **Retention `DELETE … WHERE rowid IN (SELECT rowid … ORDER BY … LIMIT N)`** requires either toasty's `.in_query()` + ordered subquery, or two-step (select then delete-by-id). Either works; two-step is simpler if the first path has quirks.
3. **Schema changes across phases**: today's schema is frozen in `MIGRATION_0001`. Toasty will regenerate `0000_initial.sql` from model derives; the shape must match what existing phase 1-4 code expects. Hand-verify the generated SQL against `db/schema.rs` before deleting the old.
4. **Toasty version churn**: 0.4.0 is pre-1.0. A minor version bump during implementation could break the build. Pin to exact version (`toasty = "=0.4.0"`) for the duration.
5. **Performance**: log batcher latency under load — the rusqlite path runs a prepared statement in a tight loop. Toasty's builder DSL may add per-row allocation overhead. Measure after migration; if it exceeds 500 µs per flush cycle, revisit.
6. **Database file format incompatible**: toasty-migrated schema may differ slightly from rusqlite's (e.g., text vs integer timestamps). Per user direction, we delete any existing DB files before running; no backfill.

## 6. Revert criteria

If any of the following hit during implementation, stop and open a revert PR:
- Toasty can't issue the pragmas via any mechanism including the side-channel (extremely unlikely — we control the side-channel).
- Toasty's `Vec<CreateStatement>` batching costs > 5 ms for a 100-row insert (measure).
- Toasty's query DSL lacks a primitive we need (composite-key range scans, subquery filtering, `ORDER BY … LIMIT`). Evidence of a gap that can't be worked around within one day of effort.
- Test suite cannot be green after migration.

Revert = restore the rusqlite path and **amend `docs/spec.md` §12 to record that the rusqlite fallback is the chosen path**, with specific reasons captured. No silent deviation.

## 7. Success criteria

- All 138 existing unit + 15 existing integration tests pass against toasty-backed Database.
- Extended smoke (Gemma 27B, Qwen3-VL-30B, 70B, 235B hybrid with override_tensor, 70B+4B dual-service) re-runs and responds correctly end-to-end.
- `cargo clippy --all-targets --features test-fakes -- -D warnings` clean.
- No code outside `src/db/pragma.rs` imports `rusqlite`.
- `toasty/migrations/0000_initial.sql` + `Toasty.toml` committed; `just db-migrate` recipe works.
