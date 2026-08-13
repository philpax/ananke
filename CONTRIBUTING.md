<!-- contributing-templates: files=general.md,rust.md,typescript.md,react.md,tailwind.md @ 57e7c55 -->

This file is a project-specific rewrite of the [contributing-templates](https://github.com/philpax/contributing-templates) conventions (`general.md`, `rust.md`, `typescript.md`, `react.md`, `tailwind.md`). The sync marker above pins the upstream commit the template-derived sections were last refreshed against; a future refresh with the `contributing-update` skill compares against it. Upstream rules this project deliberately overrides are noted where they differ; anything several projects converge on belongs back upstream.

## Project layout

The repository contains two main components:

- The Rust backend, a workspace producing two binaries: `ananke` (the daemon) and `anankectl` (the CLI).
- The frontend at `frontend/`, which is the web UI for this project. It is a Vite-based React 19 application written in TypeScript, styled with Tailwind CSS 4, and built with the React Compiler enabled.

Both components share the general conventions below. The Rust- and TypeScript-specific sections that follow apply to their respective trees.

The backend's crates, leaves first. Package names keep the `ananke-` prefix; the directories do not.

| crate | path | holds | depends on |
|---|---|---|---|
| `ananke-fs` | `crates/fs` | the `Fs` trait with its local and in-memory implementations | `parking_lot` |
| `ananke-errors` | `crates/errors` | `ExpectedError`, the user-facing error type with semantic exit codes | — |
| `ananke-spawn` | `crates/spawn` | `SpawnConfig`, the resolved child command line + env | — |
| `ananke-time` | `crates/time` | the wall-clock millis helpers | — |
| `ananke-gguf` | `crates/gguf` | the GGUF reader, including sharded models; the `Architecture` enum and every metadata key the workspace reads; `dump-gguf` | `ananke-fs` |
| `ananke-tuning-schema` | `crates/tuning-schema` | the type of `tuning.json`, shared by everything that reads or writes it | `serde` |
| `ananke-tuning` | `crates/tuning` | `tuning.json` and the build script that turns it into constants | `ananke-tuning-schema` (build) |
| `ananke-events` | `crates/events` | the daemon-wide `EventBus` and the generic estimate cache | `ananke-api`, `tokio` |
| `ananke-config` | `crates/config` | config defaults, the descriptor table the docs are generated from, the placement vocabulary (`SplitMode`, `DeviceSlot`), the parse → merge → validate pipeline, the config manager, and the input distillers | `ananke-api`, `ananke-errors`, `ananke-events`, `ananke-fs`, `ananke-gguf`, `ananke-time` |
| `ananke-estimate` | `crates/estimate` | the VRAM estimator and the design-column contract the fitter shares | the four above |
| `ananke-placement` | `crates/placement` | the packer, the device snapshot types, `ServiceRegistry<T>`, `KillHandle`, `DrainReason`, and the `estimate` example | `ananke-config`, `ananke-estimate`, `ananke-gguf` |
| `ananke-api` | `crates/api` | the DTOs that cross the wire to the frontend | — |
| `ananke-system` | `crates/system` | the outside-world traits (`Fs` re-export, `ProcFs`, `ProcessSpawner`), `SystemDeps`, and the test fakes | `ananke-errors`, `ananke-fs`, `ananke-spawn` |
| `ananke-db` | `crates/db` | the SQLite store, migrations, and log retention | `ananke-api`, `ananke-errors`, `ananke-time`, `rusqlite` |
| `ananke-observation` | `crates/observation` | `SharedSnapshot`, `ObservationTable`, and the per-service attribution helpers | `ananke-placement`, `ananke-system` |
| `ananke-devices` | `crates/devices` | the `GpuProbe`/NVML probe, cpu, cuda-env, and the device snapshotter | `ananke-observation`, `ananke-placement`, `ananke-system` |
| `ananke-tracking` | `crates/tracking` | activity, in-flight, progress, rolling, and the device-sample writer | `ananke-api`, `ananke-db`, `ananke-events`, `ananke-observation`, `ananke-placement`, `ananke-time` |
| `ananke-templates` | `crates/templates` | the placeholder substitution engine | `ananke-devices` |
| `ananke-proxy` | `crates/proxy` | the hyper reverse-proxy data plane and `ApiErrorCode` | `ananke-api`, `ananke-errors`, `axum`, `hyper` |
| `ananke-allocator` | `crates/allocator` | the feasibility check, eviction planner, and balloon resolver | `ananke-config`, `ananke-events`, `ananke-observation`, `ananke-placement` |
| `ananke-supervise` | `crates/supervise` | the supervisor state machine, child lifecycle, health, provision + reconcile | `ananke-config`, `ananke-system`, `ananke-devices`, `ananke-tracking`, `ananke-db`, `ananke-allocator`, `ananke-templates`, `ananke-api`, `ananke-events`, `ananke-proxy` |
| `ananke` | `ananke` | the daemon shell: HTTP surface (`api`), composition root (`daemon`), oneshot services, the input distillers re-exported at `config`, and the placeholder checker | all of the above |
| `anankectl` | `anankectl` | the CLI | `ananke-api` |

Three more live under `calibration/crates/`, because nothing shipped links them — see [`calibration/README.md`](calibration/README.md):

| crate | path | holds | depends on |
|---|---|---|---|
| `ananke-dataset` | `calibration/crates/dataset` | the one schema `calibration/data/measurements.ndjson` is written and read with, and the JSON writer that is part of that format | `serde`, `serde_json` |
| `ananke-measure` | `calibration/crates/measure` | the measurement harness and its log parser | `ananke-dataset`, `regex`, `nix` |
| `ananke-calibrate` | `calibration/crates/calibrate` | the sweep generator and campaign driver, deriving the tuned constants, fitting the compute model, `validate`, `scoreboard`, `emit` | `ananke-dataset`, `ananke-measure`, `ananke-estimate`, `ananke-placement`, `ananke-gguf`, `ananke-tuning-schema` |

That boundary is the useful one to hold in mind: `crates/tuning/tuning.json` is the entire interface between the two halves. Delete `calibration/` and the daemon still builds, runs, and estimates — what is lost is the ability to re-derive that file and the evidence for why each number in it is what it is. One exception, and it is a test: `ananke/tests/estimator_matches_measurements.rs` holds the shipped estimator against the campaign's own cells, so it takes `ananke-dataset` as a dev-dependency, reads `calibration/data` directly, and `cargo test --workspace` wants the directory present. That is the point of the test — a fixture copy would drift from the dataset the constants are derived from. The arrow only points inward: `ananke-calibrate` runs the real estimator and packer in-process, which is what makes `validate` and `scoreboard` mean anything, but nothing shipped links back.

`xtask` sits at the root beside the products it builds.

The split is for compile times as much as for structure. `ananke`'s build script runs the frontend's `npm run build`, so anything sharing that script pays for a UI rebuild on every change — which is why `tuning.json` lives in its own crate: regenerating the estimator's constants during a calibration campaign is the inner loop of that work, and it now costs a few seconds instead of a UI build. The estimator and then the packer followed for the same reason, and with them the `estimate` example — so no part of the calibration loop builds the UI any more. No part of the calibration loop needs `ANANKE_SKIP_FRONTEND_BUILD`.

`ananke-tuning-schema` is a leaf for a different reason: a build script cannot depend on the crate it builds, so the type of `tuning.json` — read by `crates/tuning`'s `build.rs`, by the derivers, by the emitter, and by the compute-model fitter — has to sit below all four. It is `serde` and nothing else.

The daemon imports the split crates directly (`ananke_gguf`, `ananke_estimate`, `ananke_system`, `ananke_tracking`, …). The leftover path shims are gone; only the kept facade modules (`config`, `supervise`, the `api` error facade) still re-export for path stability.

`ananke-measure` deliberately depends on neither `ananke-estimate` nor `ananke-placement`: measurement and estimation stay apart so that nothing on the estimation side ever links a process spawner.

### Calibration

The tuned constants in `crates/tuning/tuning.json` are derived from a measurement dataset, not chosen. Every one carries its evidence in its own doc comment, and CI regenerates the document and compares, so a value cannot drift from the data that justifies it without the drift showing up as a diff.

Two documents, and neither is duplicated here. [`calibration/README.md`](calibration/README.md) is the workflow — how to add a model, run the campaign, refit, and decide whether to trust the result. [`calibration/docs/design.md`](calibration/docs/design.md) is why the calibration code is shaped the way it is: the binaries and what each is for, the fixed `emit`-then-`fit` order, what `validate` and `crossval` do and do not tell you, and the rule that a derivation's key must pin every factor that could differ.

Getting the estimator and the packer out took a real decoupling rather than a file move. Both had taken a whole `ServiceConfig`; both now take a distilled input struct — `EstimatorInputs` and `PlacementInputs` — built by free functions in `ananke_estimate::service_inputs` and `ananke_placement::service_inputs`, re-exported at `ananke::config`. Reading a service config is the daemon's business; estimating and packing are pure functions over the fields they actually need. Prefer that shape for anything else that wants to come out.

### Platform scope

v1 targets Linux only — the daemon depends on NVML, `/proc`, and `prctl`, none of which have direct equivalents elsewhere. Linux-specific code is fine; don't invent cross-platform shims on speculation.

The one thing that does sit behind a trait is **every outside-world capability the daemon reaches for** (filesystem, child-process spawner, `/proc` reader, GPU probe, …). That isn't for portability — it's for testability: tests substitute in-memory or fake implementations so the suite stays deterministic. See the `crate::system` module and the testing section below. When a second platform does land, the same traits absorb the second implementation behind `#[cfg(target_os = …)]`; until then, the Linux impl is the only one.

### Dev shell

A `shell.nix` at the repo root wires up the toolchain (rustc, cargo, clippy, rustfmt, rust-analyzer, uv, Python 3.12) and — importantly — exports `LD_LIBRARY_PATH=/run/opengl-driver/lib` so `nvml-wrapper` can `dlopen` the driver. Enter with `nix-shell`. Without the shell (or an equivalent env export) on NixOS, the daemon logs "NVML init failed", falls back to CPU-only, and every GPU-bound service fails placement.

This shell is for local development only. Packaging + a systemd unit live in a separate NixOS module.

### Task automation

There is no top-level task runner today — `cargo …` for the Rust side and `npm run …` inside `frontend/` are invoked directly. Add one (e.g. a `justfile` or equivalent) only if a cross-cutting recipe emerges that genuinely spans both halves or encodes a multi-step flow; don't reach for one just to alias single-command invocations.

### Releasing

Releases are cut with the `xtask release` subcommand. The script bumps the workspace version across `Cargo.toml`, `Cargo.lock`, `frontend/package.json`, and `frontend/package-lock.json`, commits with `chore(release): v<version>`, and creates an annotated tag. It stops before any remote operation so the operator can inspect before publishing.

```sh
cargo xtask release 0.2.0          # bump, commit, tag
git push --follow-tags             # push when ready
```

Pushing the tag triggers the release workflow in CI, which builds binaries and creates a GitHub release with auto-generated notes. Edit the release notes on GitHub after it publishes.

Use `--dry-run` to preview, `--allow-dirty` to override a dirty working tree, and `--allow-branch-mismatch` to release from a branch other than `main`.

### Reading child-process logs

ananke captures every supervised child's stdout and stderr into its local SQLite store — they are not surfaced through the daemon's own log output. Use `anankectl logs <service>` to read them back:

```sh
# Last N lines from disk for a service:
anankectl logs <service> --limit 50

# Live tail (Ctrl-C to stop):
anankectl logs <service> --follow

# Stderr only, scoped to one specific run:
anankectl logs <service> --stream stderr --run <run-id>
```

Pass `--endpoint <url>` (or set `ANANKE_ENDPOINT`) when the management API isn't on the default `http://127.0.0.1:7071`. The same data is reachable over HTTP at `GET /api/services/{name}/logs` (paginated history) and `GET /api/services/{name}/logs/stream` (WebSocket live tail). Reach for these whenever a service reports "child exited during starting" — the daemon log only tells you the exit status; the captured child logs tell you why.

### The Rust ↔ TypeScript boundary

All types that cross the wire between the Rust backend and the TypeScript frontend should be **generated, not hand-written**. Hand-maintained duplicate type definitions are the single biggest source of silent drift in two-language projects.

Today the shared DTOs live in the `ananke-api` crate (hand-written, consumed by the Rust daemon directly). Handlers are annotated with `utoipa` and the daemon serves the live schema at `/api/openapi.json`. `frontend/src/api/types.ts` is generated by `openapi-typescript` — run `npm run gen-types` in `frontend/` after any change to a wire type, and never hand-edit that file (the next generator run silently reverts you). The `orval` half, which would produce typed React Query hooks in `frontend/src/api/client.ts`, is not yet wired; `client.ts` is hand-written today and re-exports the generated schemas. CI enforces that the generated output is current with `cargo xtask gen-api-types --check`, which reruns the same npm script and compares. The frontend should never declare an inline TypeScript type to describe an API payload — always import from the generated module.

Regenerating makes new fields *typed*; it doesn't make them *visible*. When a backend change adds or restructures state that the frontend displays, update the display code in the same change — a field that arrives typed but unrendered is a gap, where the UI shows stale behaviour while the backend has moved on.

## General conventions

### Correctness over convenience

- Model the full error space—no shortcuts or simplified error handling.
- Handle all edge cases, including race conditions, signal timing, and platform differences.
- Use the type system to encode correctness constraints.
- Prefer compile-time guarantees over runtime checks where possible.

### User experience as a primary driver

- Provide structured, helpful error messages that can be rendered with an appropriate library at a later stage.
- Make progress reporting responsive and informative.
- Maintain consistency across platforms even when underlying OS capabilities differ. Use OS-native logic rather than trying to emulate Unix on Windows (or vice versa).
- Write user-facing messages in clear, present tense: "Frobnicator now supports..." not "Frobnicator now supported..."

### Pragmatic incrementalism

- "Not overly generic"—prefer specific, composable logic over abstract frameworks.
- Evolve the design incrementally rather than attempting perfect upfront architecture.

### Production-grade engineering

- Use type system extensively: newtypes, builder patterns, type states, lifetimes.
- Use message passing or the actor model to avoid data races in concurrent code.
- Test comprehensively, including edge cases, race conditions, and stress tests.
- Pay attention to what facilities already exist for testing, and aim to reuse them.
- Prefer a **functional core, imperative shell**: keep decision logic in pure functions that take data in and return data out, and keep I/O, concurrency primitives, and orchestration in a thin shell at the edges.
- The shape is gather, then process, then persist: the shell collects the inputs, the core decides, the shell writes the result. A core function that reaches out to read something mid-decision is the thing this is meant to prevent.
- Isolate coupling to the outside world — filesystem, clock, network, subprocesses, devices — behind a small seam: a trait, an interface, a dependency struct, with a production implementation and a test fake. Tests are deterministic because they substitute the fake, not because they clean up after the real thing.
- The payoff is testability. A pure core needs no fakes at all, and a thin shell has little logic left worth mocking. When a test needs elaborate setup to reach the behaviour it's checking, that's usually the code's shape talking, not the test's.
- Getting the details right is really important!

### Documentation

- Use inline comments to explain "why," not just "what".
- Don't add narrative comments in function bodies. Only add a comment if what you're doing is non-obvious or special in some way, or if something needs a deeper "why" explanation.
- Module-level documentation should explain purpose and responsibilities.
- Item docs are three lines at most, unless the comment is a guardrail. The test: if it vanished, could someone reintroduce a bug it was warning about, or fail to find something they'd need? If so, write as much as it takes. Otherwise cut it — don't restate the identifier, don't describe fields that have names, and don't narrate what the code plainly does.
- **Always** use periods at the end of code comments.
- **Never** use title case in headings and titles. Always use sentence case.
- Always use the Oxford comma.
- Don't omit articles ("a", "an", "the"). Write "the file has a newer version" not "file has newer version".
- Comments describe the present state. Reserve past-tense narration for the rare case where history explains a standing "why".
- Keep the user-facing docs in sync with code changes. Generate the document from the code and check it in CI, rather than relying on anyone remembering. The source of truth for config defaults is the `DEFAULT_*` constants in `crates/config/src/docs/` and `crates/config/src/defaults.rs`; the source of truth for config struct fields is `ananke/src/config/parse/` and `ananke/src/config/validate/`. When you add, remove, rename, or change the default of a config field, update the descriptor table in `ananke_config::docs::all_sections()` and run `cargo xtask gen-config-docs` to regenerate `docs/configuration.md`. CI enforces this with `--check`. Likewise, changes to service states or the management/OpenAI API surface should be reflected in `docs/api.md` (run `cargo xtask gen-api-docs` to regenerate). Treat a code change that touches these areas as incomplete until the docs are updated.

### Code organization

This applies to both trees; the Rust- and TypeScript-specific sections below build on it.

- **Keep files under the size threshold.** Split a file into multiple files within a folder when it exceeds around 1000 lines (Rust) or 400 lines (TypeScript/TSX). Use `mod.rs` (Rust) or an index/re-export pattern (frontend) to re-export public items so consumers keep seeing a stable API. Measure the whole file, inline `#[cfg(test)]` module included — a test module that has outgrown its subject is itself a signal to split by concern group, with shared helpers alongside.
- Large inline test modules count toward the total; extracting one to a sibling file is often the whole fix. Existing oversized files are grandfathered — split one when a change touches it substantially, not in drive-by churn.
- **Split by concern, not by size alone.** A file should be split along natural seams — distinct data types, feature groups, or functional areas — not arbitrarily at the line limit. A cohesive single-concern file that slightly exceeds the threshold is preferable to a fragmented one. When splitting a frontend file, keep one main component per file with co-located sub-components, and put non-component utilities (hooks, constants, pure functions) in separate files so HMR boundaries stay clean.
- **Generated files are exempt.** `frontend/src/api/types.ts` and anything else produced by a generator is bound by its generator, not by this threshold.
- **Organize wide folders into subfolders.** When a folder accumulates many direct children, group them by domain or role. A flat folder of 20+ files is a signal that subfolders are wanted.

## Rust code style

### Rust edition and linting

- Use Rust 2024 edition.
- Format with **nightly rustfmt**: run `cargo +nightly fmt --all` before committing. The `rustfmt.toml` opts into `imports_granularity = "Crate"` and `group_imports = "StdExternalCrate"`, which are nightly-only features — stable rustfmt prints a warning about each and then silently skips them, so a stable-formatted file is *not* equivalent to a nightly-formatted one. See the Imports section below for details.
- Ensure the following checks pass at the end of each complete task (you do not need to do this for intermediate steps):
  - `cargo +nightly fmt --all -- --check`
  - `cargo clippy --workspace --all-targets --all-features -- -D warnings`
  - `cargo clippy --workspace --all-targets --no-default-features -- -D warnings`
  - `cargo test --workspace --all-features`
  - `cargo test --workspace --no-default-features --lib`
- Pass `--workspace` to every check. `default-members` deliberately excludes `xtask` so a bare `cargo build` doesn't build it, but that also drops it from any check that omits the flag — which is how it accumulated eight unlinted warnings before CI was widened.
- Integration tests live under `ananke/tests/` and depend on the `test-fakes` feature (for `FakeSpawner` etc.). They run under `--all-features`. The no-default-features pass is scoped to `--lib` to verify the non-feature build still compiles; integration-test failures under no-default-features are expected.
- No `unwrap()` or `expect()` in production code; tests are fine. The `unwrap_used` and `expect_used` restriction lints that would enforce this are allow-by-default and not in the check commands above — the rule is convention rather than CI-enforced.
- Never silence a lint without a concrete reason documented in a comment above it. In almost all cases, the right move is to restructure the code.
- When you do suppress one, prefer `#[expect(...)]` to `#[allow(...)]`: `expect` warns once the suppression is no longer needed, so stale suppressions can't quietly accumulate as the code changes around them.

### Build profile

- Use `cargo build` (debug) for local iteration — don't reach for `cargo build --release` unless you're specifically benchmarking or packaging. The daemon's hot paths are either I/O (child stdout piping, HTTP proxying) or already-optimised native libraries (NVML, SQLite, hyper), so the extra compile time rarely pays off and the iteration cost is real.

### Type system patterns

- **Builder patterns** for complex construction (e.g., `TestRunnerBuilder`)
- **Type states** encoded in generics when state transitions matter
- **Lifetimes** used extensively to avoid cloning (e.g., `TestInstance<'a>`)
- **Restricted visibility**: Use `pub(crate)` and `pub(super)` liberally

### Error handling

- Do not use `thiserror`. Instead, manually implement `std::fmt::Error` for a given error `struct` or `enum`.
- Group errors by category with an `ErrorKind` enum when appropriate.
- Provide rich error context using structured error types.
- Two-tier error model:
  - `ExpectedError`: User/external errors with semantic exit codes.
  - Internal errors: Programming errors that may panic or use internal error types.
- Error display messages lead with a `<context>:` prefix naming the subsystem or operation they belong to, then the cause — `failed to bind {addr}: {cause}`, `failed to parse config at {path}: {cause}` — and stay lowercase sentence fragments suitable for "failed to {error}". An aggregating error prefixes its own layer's context and delegates to the inner error, so a chained error reads as nested context. Add resource context (a path, an id) at the layer that has it.

### Async patterns

- Do not introduce async to a project without async.
- Use `tokio` for async runtime (multi-threaded).
- Use async for I/O and concurrency, keep other code synchronous.
- Use `parking_lot::Mutex`/`RwLock` for synchronous locks (the default); the guard is non-poisoning and must never be held across an `.await`. Reserve `tokio::sync::Mutex` for the rare guard that must survive an `.await`, since most locks are acquired, used, and dropped within a synchronous span.

### Module organization

- Use `mod.rs` files to re-export public items.
- Keep module boundaries strict with restricted visibility, but prefer `pub(crate)` and `pub(super)` over `pub(in <path>)`. The `pub(in …)` form scopes to a named ancestor, which is precise but reads as a smell; reach for it only when neither `pub(crate)` nor `pub(super)` expresses the intended scope.
- Use `#[cfg(unix)]` and `#[cfg(windows)]` for conditional compilation.
- **Always** import types or functions at the very top of the module, with the one exception being `cfg()`-gated functions. Never import types or modules within function contexts, other than this `cfg()`-gated exception.
- It is okay to import enum variants for pattern matching, though.
- Re-exports follow the same rule: a `pub use` belongs at the top of the module with the imports, not beside the item it re-exports. In a `mod.rs`, the `mod` declarations come first, then the `pub use` block, so the module's public surface reads as one list.
- **Always** anchor intra-crate paths at `crate::`, never `super::`. Write `crate::estimator::compute_buffer::default_for`, not `super::compute_buffer::default_for` or `super::super::…` — this holds for `use` statements, inline paths, and intra-doc links alike. The one exception is a test module, where `use super::*;` inside the `#[cfg(test)]` block is the idiomatic form and stays.
- When a path is used more than once in a module, import the specific items at the top of the module rather than repeating the fully-qualified path at each call site. A path used only once may stay fully-qualified — unless it is unwieldy (more than three module segments deep, like `crate::supervise::restart::history::Window`), in which case import it regardless of use count. And when the module already imports a sibling from the same parent, import the new item alongside it rather than writing it inline.

Within each module, organize code as follows:
1. **Public API first** - all `pub` structs, enums, and functions at the top
2. **Private implementation below** - constants, helper functions, and internal types
3. **Order by use** - private items should appear in the order they're called/used by the public API (topological order)

### File hierarchy

- The file hierarchy is the architecture diagram. A newcomer should be able to intuit what the project does by reading the directory listing.
- Avoid top-level single-file modules where a natural folder grouping exists. If several files are semantically related — or if file A is only consumed by file B — prefer merging into a folder module or into the consumer.
- Mirror the public → private structure at the tree level too: subsystems with public entry points (e.g. `api/`, `db/`, `supervise/`) live as folder modules with a `mod.rs` that states the boundary, and their internals are private submodules.
- Only `lib.rs`, `main.rs`, and genuinely cross-cutting types (e.g. `errors.rs`) should remain as top-level single files.

### Imports

- Prefer a single grouped `use` statement per crate/module rather than several siblings under the same root. Write `use crate::db::{Database, logs::BatcherHandle, models::ServiceLog};`, not three separate lines.
- Group imports into three blocks separated by blank lines, in this order: `std`, external crates, then `crate`/`super`/`self`.
- `use` brace expansion should collapse shared prefixes. `use axum::{extract::State, http::StatusCode, routing::get};` is correct; three separate lines under `axum::` is not.
- `rustfmt.toml` sets `imports_granularity = "Crate"` and `group_imports = "StdExternalCrate"` to enforce these rules automatically. Both options are nightly-only — see the linting section above for why `cargo +nightly fmt --all` is the canonical formatter and why stable `cargo fmt` is not a substitute.

### Function arguments and state

- If a function takes more than ~5 arguments, that's a signal to group related ones into a struct rather than suppressing `clippy::too_many_arguments`. Suppressing that lint is almost never right — the lint firing means a struct is wanted. The one exception is a signature you don't own (an FFI shim), where the `allow` is the honest annotation.
- Never use `#[allow(clippy::...)]` to silence a lint without a concrete reason. If clippy is wrong for a case, document why in a comment above the allow. In almost all cases the right move is to restructure the code.
- Prefer a **functional core, imperative shell**: keep decision logic in pure functions that take data in and return data out, and keep the `tokio::select!` / `tokio::spawn` / I/O at the edges. This makes the core testable without test-fakes and keeps rightward drift out of the core.
- Avoid rightward drift. If a function is nesting three `tokio::select!` blocks or four levels of `match`/`if let`, extract each arm into a named function that takes a context struct. The control flow at the top level should read like an outline.

### State machines

- Model each state machine as an explicit `enum` with named variants, even if only one field differs between them. Favour an exhaustive `match` + a `transition` helper over scattered `if let` chains.
- Where a subsystem transitions through phases that own different local state (e.g. `Idle`, `Starting`, `Running`, `Draining`), extract each phase body into its own async function and pass a typed context struct. This is the pragmatic version of the typestate pattern for actor-style loops; it keeps invariants local without requiring full type-parameterised phases.
- Invalid transitions should be unrepresentable at the boundary where they're consumed. If `transition()` returns `Option<State>`, the caller should never `.unwrap()` it in production — either enumerate the legal inputs ahead of time or make the caller total.

### Platform coupling

- Files that depend on Linux-specific facilities (NVML, `/proc`, `prctl`, `signal`) must say so in their module-level docstring on the first line: `//! Linux-only: reads /proc/{pid}/cmdline.` The convention is explicit enough that a second-platform port — or a reviewer — knows exactly what the contract is.
- Isolate outside-world coupling behind a small trait on `crate::system` (as `devices::GpuProbe` does for NVML, `system::Fs` for the filesystem, `system::ProcessSpawner` for child lifecycle, `system::ProcFs` for `/proc` reads). The trait is there so tests can substitute an in-memory or fake implementation; a second OS implementation is a nice-to-have that falls out of the same shape.
- When a second platform does land, gate the Linux impl with `#[cfg(target_os = "linux")]` and add the alternative under a sibling gate; the trait definition stays platform-neutral.

### Memory and performance

- Use `Arc` or borrows for shared immutable data.
- Use `smol_str` for efficient small string storage.
- Use `smallvec` for collections that are usually small, to avoid heap allocations in the common case. No site in the daemon or calibration harness currently meets the bar: request and config collections are built at request or reload rate, the balloon window is a `VecDeque` ring, and the proxy data plane streams via `Bytes`. Reconsider where a per-token or per-chunk `Vec` appears.
- Careful attention to cloning referencing. Avoid cloning if code has a natural tree structure.
- Stream data (e.g. iterators) where possible rather than buffering.
- To borrow the value inside a lock guard, a `Box`, or an `Arc`, prefer `.as_ref()`/`.as_mut()` over a manual double-deref: write `state.config.read().as_ref()`, not `&**state.config.read()`. The named form reads as "borrow the config" rather than as deref bookkeeping. The same applies to an `Arc<dyn Trait>`: `probe.as_ref()`, not `&**probe`.

### Serde

- Use `#[serde(deny_unknown_fields)]` on config types, and `#[serde(default)]` on new fields so they stay backwards-compatible.
- Avoid `#[serde(untagged)]` when deserialising — the error messages it produces are useless, and the inner type's `deny_unknown_fields` is silently dropped. Write a custom visitor instead.
- Reserve `#[serde(flatten)]` for the case it is genuinely for: extending a shared struct with local fields, or a `toml::Table` catch-all that preserves unknown fields across a round-trip. Note that it interacts badly with `deny_unknown_fields`.

### Chosen dependencies

The Rust stack is chosen; don't silently introduce alternatives when one of these already covers the need. Most are not yet added — pull them in when the corresponding subsystem is first implemented, and prefer these over comparable crates unless there's a concrete reason not to.

- **Async runtime**: `tokio` (multi-threaded).
- **HTTP**: `hyper` for the proxy data plane; `axum` for the management and OpenAI-compatible routing surface.
- **OpenAPI generation**: `utoipa` annotations on handlers, served at `/api/openapi.json`.
- **GPU probing**: `nvml-wrapper` behind a `GpuProbe` trait — `FakeGpuProbe` in tests, with space for ROCm/XPU impls if someone wants them.
- **Config watching**: `notify`.
- **TOML**: `toml_edit` for parse-preserving read/write (needed so the config editor keeps comments and formatting).
- **Database**: direct `rusqlite` with versioned SQL migrations under `ananke/src/db/migrations/` applied at boot.
- **Logging**: `tracing` to stderr; journald captures it under systemd.
- **Child supervision**: `nix` for `prctl(PR_SET_PDEATHSIG)` and related Linux-specific calls.
- **GGUF**: start with the `gguf` crate; fall back to a small custom reader if it can't enumerate the tensor table or handle sharded files.

### Adding a new model architecture

When a new model family ships with a `general.architecture` value that ananke does not yet recognise, the estimator refuses it with `UnknownArchitecture`. The operator's only recourse is to declare the reservation explicitly — `mode` plus `reserve_gb`, the same way a `command` service does — which skips the estimator and the packer entirely. Adding proper support takes three steps:

1. **Dump the GGUF metadata.** Run the `dump-gguf` example against the first shard:

   ```bash
   cargo run -p ananke-gguf --example dump-gguf -- /path/to/model-00001-of-NNNN.gguf
   ```

   The output shows the architecture name, block count, tensor categories, and every attention-related metadata key. This is the ground truth for what the estimator needs to read.

2. **Choose the right family module.** The estimator dispatches on `Architecture` through the family modules in `crates/estimate/src/`. Each module's `*_FAMILY` constant lists the architectures it covers, and the module docs and per-entry comments describe the tensor layouts and metadata quirks already handled. Read those alongside the dump from step 1, and pick the module whose expectations the new architecture's tensors and metadata actually match — the existing entries are worked examples of what "matching" looks like.

3. **Register and test.**

   a. Add a variant to `Architecture` in `crates/gguf/src/architecture.rs`, and its `general.architecture` spelling to `as_str` and `known()`. The tests there hold every variant to a round-trip and to distinct names, so a half-registered one fails immediately.

   b. Add that variant to the chosen family's `*_FAMILY` constant, with a comment describing the quirks it brings.

   c. Measure the architecture into the calibration dataset and re-run the campaign, so the compute model picks up coefficients for it rather than falling to the pooled default. [`calibration/README.md`](calibration/README.md) is that loop.

   d. Add a unit test in the family module that exercises the key behaviour (KV computation, layer collection, expert detection, etc.). Use `synth_gguf::Builder` from `ananke/tests/common/mod.rs` to construct a fake GGUF summary, or write one inline with the same pattern.

   e. Run the full test suite: `cargo test --workspace --all-features` and `cargo clippy --all-targets --all-features -- -D warnings`.

[`docs/memory-model.md`](docs/memory-model.md) is what those estimators are modelling — the VRAM and host-side terms, and where each one comes from.

**Reference implementation:** the `dump-gguf` example at `crates/gguf/examples/dump-gguf.rs` is the canonical tool for gathering GGUF metadata. The llama.cpp source (ask the operator where it lives) is the ground truth for tensor naming, metadata keys, and architecture classification. When in doubt about how a tensor is routed at runtime, check `llama-arch.cpp` (`LLM_TENSOR_NAMES`, `LLM_ARCH_NAMES`, `llm_arch_is_hybrid`), `llama-model.cpp` (hparams loading), and `llama-memory*.cpp` (KV cache vs recurrent state).

## TypeScript code style

The same correctness-first mindset that governs the Rust side applies here: TypeScript's type system is strong enough to encode most of the same invariants, and it should be pushed to do so. "Just cast it" is not an acceptable answer.

### Tooling and workflow

- `npm run lint` is the single check command. It covers formatting, linting, and type correctness in one pass.
- `npm run format` applies formatting in write mode. Use it to fix formatting issues surfaced by lint.
- Run `npm run lint` frequently during development, not just at the end of a task. A clean lint is cheap to maintain and expensive to recover.
- Ensure `npm run lint` passes at the end of each complete task.

### TypeScript compiler settings

- Keep the project on the strictest practical settings. `tsconfig.app.json` already enables `noUnusedLocals`, `noUnusedParameters`, `noFallthroughCasesInSwitch`, `erasableSyntaxOnly`, and `verbatimModuleSyntax`; do not relax these.
- Prefer `import type { ... }` for type-only imports, as required by `verbatimModuleSyntax`.
- Do not disable rules or flags to make a specific piece of code compile. Fix the code instead.

### Type system patterns

Treat these as the TypeScript analogues of the Rust patterns above. The goal is the same: make illegal states unrepresentable.

- **Discriminated unions** for modelling state machines and result types — the equivalent of Rust enums. Always include a `kind` (or similar) tag and narrow on it.
- **Exhaustiveness checking** via a `never`-typed default branch in switches and `if`/`else` chains, so adding a new variant becomes a compile error everywhere it is handled.
- **Branded (nominal) types** for values that share a representation but not a meaning (e.g. `UserId` vs. `ProjectId` both being `string`). This is the TypeScript parallel of Rust newtypes.
- **`readonly`** on arrays, tuples, and object properties by default. Reach for mutability only when it is genuinely needed.
- **`as const`** for literal data that should be inferred as narrowly as possible, and **`satisfies`** to check a value against a type without widening its inferred type.
- **Template literal types** and mapped/conditional types to encode constraints at the type level where it pays off.
- **Prefer `unknown` over `any`**. If you reach for `any`, stop and reconsider; if it is truly unavoidable, isolate it behind a narrow boundary and document why.
- **Avoid type assertions** (`as SomeType`) and non-null assertions (`!`). Use type guards, discriminated unions, or restructured code instead. A type assertion is a claim the compiler cannot verify, so it is a liability.
- **Validate at boundaries**. Data coming from the network, `localStorage`, URL parameters, or any other untyped source must be parsed and validated before being treated as typed. Do not trust a `JSON.parse` result.
- **Builder-style or fluent APIs** for complex construction, and **phantom/branded types** to encode state transitions when they matter.

### Errors

- Model the full error space, same as on the Rust side. Prefer a discriminated union result type (`{ kind: "ok"; value: T } | { kind: "err"; error: E }`) or similar over throwing for expected failure modes.
- Exceptions are for genuinely exceptional, programmer-error situations.
- User-facing error messages follow the same rules as the rest of the project: present tense, sentence case, with periods.

### React

- Write function components and hooks. No class components.
- The React Compiler is enabled via `babel-plugin-react-compiler`, so manual memoization (`useMemo`, `useCallback`, `React.memo`) is generally unnecessary and should not be added preemptively. Reach for it only when the compiler demonstrably cannot handle a case.
- Keep components small and focused. Lift state only as far as it needs to go.
- Follow the rules of hooks strictly, and keep `eslint-plugin-react-hooks` warnings at zero.
- That's why the hooks lint is load-bearing rather than advisory: the compiler's guarantees hold only while the code stays within the Rules of React.
- Type component props explicitly. Do not rely on inference for the public shape of a component.
- Prefer composition over configuration — a few focused components beat one component with a dozen boolean props.

### Styling: Tailwind first, CSS last

- This is a Tailwind CSS 4 project. Styling should be done with Tailwind utility classes in JSX.
- **Do not write custom CSS unless it truly, genuinely cannot be expressed in Tailwind.** This is a hard rule, not a soft preference. "It would be slightly cleaner in CSS" is not sufficient justification; neither is "I'm more comfortable with CSS". If you think you need custom CSS, first check whether an arbitrary value (`[...]`), a variant, a Tailwind theme extension, or a small component abstraction solves it.
- When custom CSS is genuinely required (e.g. a keyframe animation or a selector Tailwind cannot express), keep it minimal, colocated, and leave a comment explaining why Tailwind was not sufficient.
- Use Tailwind's theme tokens for colours, spacing, and typography rather than hard-coded values, so design changes stay centralised.

#### Linting

Class order, duplicate and conflicting utilities, typo'd class names, and shorthand collapsing are all machine-checkable — let the linter own them rather than spending review on them. Use [`eslint-plugin-better-tailwindcss`](https://github.com/schoero/eslint-plugin-better-tailwindcss) with its `recommended` config, include it in the `npm run lint` check, and treat its warnings as errors so lint stays a binary signal. Point it at the theme so custom tokens aren't reported as unknown classes (`entryPoint` for a v4 CSS-based config, `tailwindConfig` for v3).

Compose conditional classes with `clsx` (plus `tailwind-merge` where later classes must override earlier ones), or `cva` for a component with variants. Never build a class name by interpolating fragments into a string — it defeats both the linter and Tailwind's own class extraction.

### Module organization

- Import types and values at the top of the file. No inline `require`/`import()` inside function bodies except for genuinely dynamic imports (code-splitting).
- Use named exports by default. Reserve default exports for cases where a framework or tool requires them (e.g. route modules, some Vite entry points).
- Keep file layout consistent with the Rust convention: public API first, private helpers below, ordered by use.

### Chosen dependencies

Same principle as the Rust side: the frontend stack is chosen, and most of these will be added only when first needed. Prefer them over alternatives unless there's a concrete reason.

- **Build tool**: Vite (already in place).
- **UI**: React 19 with the React Compiler; Tailwind CSS 4 (already in place).
- **Server state**: TanStack Query, with typed hooks generated by `orval` from the daemon's OpenAPI schema.
- **API types**: `openapi-typescript` generates raw types; `orval` generates the hooks on top. See "The Rust ↔ TypeScript boundary" above.
- **Code editor component**: CodeMirror 6 (for the in-app TOML config editor).

## Testing

### Tests are pure; the outside world goes through `system::SystemDeps`

Tests must be deterministic. They must not spawn real processes, probe real pids, read real `/proc`, touch disk, sleep on wall-clock, or depend on any state the daemon didn't hand them. The way we enforce this is that every capability the daemon takes from the outside world lives behind a trait in `crate::system`:

- `Fs` — filesystem. `LocalFs` in production, `InMemoryFs` in tests.
- `ProcessSpawner` + `ManagedChild` — child-process lifecycle. `LocalSpawner` in production (uses `tokio::process` + `nix` for signals); `FakeSpawner` in tests (virtual pids, no OS processes, state inspectable for assertions).

These are bundled into `system::SystemDeps`. Production code calls `SystemDeps::local()`; tests call `SystemDeps::fake()` which also returns the concrete fakes so assertions can inspect state (e.g. "which children were SIGTERM'd, which were SIGKILL'd"). The `SupervisorDeps` and `AppState` structs carry a `system: SystemDeps` field — they never hold `LocalFs` or `LocalSpawner` directly.

**When adding a new outside-world dependency (clock, network, `/proc` readers, etc.):**

1. Define a trait in `ananke/src/system/<name>.rs` with a production impl and a test fake. Gate the fake behind `#[cfg(any(test, feature = "test-fakes"))]`.
2. Re-export from `system::mod.rs` and add it as a field on `SystemDeps`. Update `SystemDeps::local()` and `SystemDeps::fake()`.
3. Route every caller through `deps.system.<field>`; never use `std::fs::*`, `tokio::process::Command`, `SystemTime::now`, etc. directly outside the trait's production impl.

Time is the narrow exception: supervisors already run on `tokio::time` so `start_paused = true` gives tests virtual time without another trait. `tracking::now_unix_ms` (wall-clock) is used for event timestamps and DB rows; tests don't assert on its values.

**Anti-patterns that should not appear in tests:**

- `nix::sys::signal::kill(pid, 0)` to probe a real pid — use `FakeSpawner::children()` and assert on `FakeProcessState`.
- `tokio::process::Command` to spawn a shell sleep — use `FakeSpawner`.
- `tokio::time::sleep(Duration::from_millis(N))` to let real wall-clock time pass — use `start_paused = true` + `tokio::time::advance` or `wait_for(predicate)` on explicit state.
- `std::fs::*` or `tempfile::*` — use `InMemoryFs`.
- Real TCP sockets to a real service — the `TestHarness` echo server is the single permitted loopback listener and exists only because routing the hyper proxy data-plane through a trait would obscure its semantics.

### Testing conventions

- Do not write a test that only exercises serde or a derive. A round-trip earns its place only when it guards a real wire: the management or OpenAI-compatible API surface, a persisted DB row, or the on-disk config.

### Rust testing tools

- **test-case**: For parameterized tests.
- **proptest**: For property-based testing.
- **insta**: For snapshot testing.
- **libtest-mimic**: For custom test harnesses.
- **pretty_assertions**: For better assertion output.

### Frontend testing tools

No frontend tests exist yet, and none are required until the UI has logic worth testing on its own terms. When the first test lands, use these rather than reaching for alternatives:

- **Vitest**: Unit and component tests. Natural fit alongside Vite; shares the same config surface.
- **React Testing Library**: For component tests — query by user-visible semantics, not implementation details.
- **Playwright**: For end-to-end flows against a running daemon + frontend, if and when one is justified. Do not reach for this for what a component test can cover.
