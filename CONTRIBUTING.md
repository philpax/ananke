## Project layout

The repository contains two main components:

- The Rust backend at the crate root (`src/`, `Cargo.toml`), which produces two binaries: `ananke` (the daemon) and `anankectl` (the CLI).
- The frontend at `frontend/`, which is the web UI for this project. It is a Vite-based React 19 application written in TypeScript, styled with Tailwind CSS 4, and built with the React Compiler enabled.

Both components share the general conventions below. The Rust- and TypeScript-specific sections that follow apply to their respective trees.

### Platform scope

v1 targets Linux only — the daemon depends on NVML, `/proc`, and `prctl`, none of which have direct equivalents elsewhere. Linux-specific code is fine today; write it directly rather than introducing cross-platform shims that don't yet have a second platform to justify them. When a second platform does land, follow the cross-platform guidance under "User experience as a primary driver" and keep OS-specific logic behind `#[cfg(...)]` boundaries.

### Dev shell

A `shell.nix` at the repo root wires up the toolchain (rustc, cargo,
clippy, rustfmt, rust-analyzer, uv, Python 3.12) and — importantly —
exports `LD_LIBRARY_PATH=/run/opengl-driver/lib` so `nvml-wrapper` can
`dlopen` the driver. Enter with `nix-shell`. Without the shell (or an
equivalent env export) on NixOS, the daemon logs "NVML init failed",
falls back to CPU-only, and every GPU-bound service fails placement.

This shell is for local development only. Packaging + a systemd unit
live in a separate NixOS module.

### Task automation

A top-level `justfile` is the canonical entry point for project-wide tasks: regenerating types, running both linters together, release flows, etc. It is not yet present — add it when the first cross-cutting recipe appears, and put future cross-cutting recipes there rather than inventing parallel scripts. Component-local invocations (`cargo …`, `npm run …`) stay where they are; `just` is for things that span the two halves or encode a multi-step flow.

### The Rust ↔ TypeScript boundary

All types that cross the wire between the Rust backend and the TypeScript frontend are **generated, not hand-written**. This is a hard rule, because hand-maintained duplicate type definitions are the single biggest source of silent drift in two-language projects.

The pipeline is not yet wired up — implement it when the first API handler lands, and keep it as the only way cross-boundary types enter the frontend:

- Rust handlers are annotated with `utoipa`. The daemon serves the live schema at `/api/openapi.json`.
- `just gen-types` runs `openapi-typescript` to produce `frontend/src/api/types.ts` and `orval` to produce typed React Query hooks in `frontend/src/api/client.ts`.
- CI enforces that the generated files are up to date. A PR that changes an API handler without regenerating fails.
- The frontend never declares an inline TypeScript type to describe an API payload; always import from the generated module.

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
- Getting the details right is really important!

### Documentation

- Use inline comments to explain "why," not just "what".
- Don't add narrative comments in function bodies. Only add a comment if what you're doing is non-obvious or special in some way, or if something needs a deeper "why" explanation.
- Module-level documentation should explain purpose and responsibilities.
- **Always** use periods at the end of code comments.
- **Never** use title case in headings and titles. Always use sentence case.
- Always use the Oxford comma.
- Don't omit articles ("a", "an", "the"). Write "the file has a newer version" not "file has newer version".

## Rust code style

### Rust edition and linting

- Use Rust 2024 edition.
- Ensure the following checks pass at the end of each complete task (you do not need to do this for intermediate steps):
  - `cargo fmt --all -- --check`
  - `cargo clippy --all-targets --all-features -- -D warnings`
  - `cargo clippy --all-targets --no-default-features -- -D warnings`
  - `cargo test --workspace --all-features`
  - `cargo test --workspace --no-default-features --lib`
- Integration tests live under `ananke/tests/` and depend on the `test-fakes` feature (for `FakeSpawner` etc.). They run under `--all-features`. The no-default-features pass is scoped to `--lib` to verify the non-feature build still compiles; integration-test failures under no-default-features are expected.

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
- Error display messages should be lowercase sentence fragments suitable for "failed to {error}".

### Async patterns

- Do not introduce async to a project without async.
- Use `tokio` for async runtime (multi-threaded).
- Use async for I/O and concurrency, keep other code synchronous.

### Module organization

- Use `mod.rs` files to re-export public items.
- Keep module boundaries strict with restricted visibility.
- Use `#[cfg(unix)]` and `#[cfg(windows)]` for conditional compilation.
- **Always** import types or functions at the very top of the module, with the one exception being `cfg()`-gated functions. Never import types or modules within function contexts, other than this `cfg()`-gated exception.
- It is okay to import enum variants for pattern matching, though.

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
- `rustfmt.toml` sets `imports_granularity = "Crate"` and `group_imports = "StdExternalCrate"` to enforce this automatically. Both options are nightly-only, so the authoritative formatter is `cargo +nightly fmt --all`. Stable `cargo fmt --all -- --check` prints warnings about the unstable options and otherwise passes — CI runs on stable, and human formatting is the nightly pass.

### Function arguments and state

- If a function takes more than ~5 arguments, that's a signal to group related ones into a struct rather than suppressing `clippy::too_many_arguments`. Suppressing that lint is almost never right.
- Never use `#[allow(clippy::...)]` to silence a lint without a concrete reason. If clippy is wrong for a case, document why in a comment above the allow. In almost all cases the right move is to restructure the code.
- Prefer a **functional core, imperative shell**: keep decision logic in pure functions that take data in and return data out, and keep the `tokio::select!` / `tokio::spawn` / I/O at the edges. This makes the core testable without test-fakes and keeps rightward drift out of the core.
- Avoid rightward drift. If a function is nesting three `tokio::select!` blocks or four levels of `match`/`if let`, extract each arm into a named function that takes a context struct. The control flow at the top level should read like an outline.

### State machines

- Model each state machine as an explicit `enum` with named variants, even if only one field differs between them. Favour an exhaustive `match` + a `transition` helper over scattered `if let` chains.
- Where a subsystem transitions through phases that own different local state (e.g. `Idle`, `Starting`, `Warming`, `Running`), extract each phase body into its own async function and pass a typed context struct. This is the pragmatic version of the typestate pattern for actor-style loops; it keeps invariants local without requiring full type-parameterised phases.
- Invalid transitions should be unrepresentable at the boundary where they're consumed. If `transition()` returns `Option<State>`, the caller should never `.unwrap()` it in production — either enumerate the legal inputs ahead of time or make the caller total.

### Platform coupling

- Files that depend on Linux-specific facilities (NVML, `/proc`, `prctl`, `signal`) must say so in their module-level docstring on the first line: `//! Linux-only: reads /proc/{pid}/cmdline.` The convention is explicit enough that a second-platform port knows exactly what to replace.
- Prefer isolating platform coupling behind a small trait with a Linux impl (as `devices::GpuProbe` does for NVML) rather than threading `#[cfg(linux)]` through business logic.
- When the second platform lands, gate the Linux impl with `#[cfg(target_os = "linux")]` and add the alternative under a sibling gate; keep the trait definition platform-neutral.

### Memory and performance

- Use `Arc` or borrows for shared immutable data.
- Use `smol_str` for efficient small string storage.
- Careful attention to cloning referencing. Avoid cloning if code has a natural tree structure.
- Stream data (e.g. iterators) where possible rather than buffering.

### Chosen dependencies

The Rust stack is chosen; don't silently introduce alternatives when one of these already covers the need. Most are not yet added — pull them in when the corresponding subsystem is first implemented, and prefer these over comparable crates unless there's a concrete reason not to.

- **Async runtime**: `tokio` (multi-threaded).
- **HTTP**: `hyper` for the proxy data plane; `axum` for the management and OpenAI-compatible routing surface.
- **OpenAPI generation**: `utoipa` annotations on handlers, served at `/api/openapi.json`.
- **GPU probing**: `nvml-wrapper` (behind a `GpuProbe` trait so ROCm/XPU can slot in later).
- **Config watching**: `notify`.
- **TOML**: `toml_edit` for parse-preserving read/write (needed so the config editor keeps comments and formatting).
- **Database**: `toasty` over SQLite; keep a raw-SQL migration fallback path.
- **Logging**: `tracing` to stderr; journald captures it under systemd.
- **Child supervision**: `nix` for `prctl(PR_SET_PDEATHSIG)` and related Linux-specific calls.
- **GGUF**: start with the `gguf` crate; fall back to a small custom reader if it can't enumerate the tensor table or handle sharded files.

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
- Type component props explicitly. Do not rely on inference for the public shape of a component.
- Prefer composition over configuration — a few focused components beat one component with a dozen boolean props.

### Styling: Tailwind first, CSS last

- This is a Tailwind CSS 4 project. Styling should be done with Tailwind utility classes in JSX.
- **Do not write custom CSS unless it truly, genuinely cannot be expressed in Tailwind.** This is a hard rule, not a soft preference. "It would be slightly cleaner in CSS" is not sufficient justification; neither is "I'm more comfortable with CSS". If you think you need custom CSS, first check whether an arbitrary value (`[...]`), a variant, a Tailwind theme extension, or a small component abstraction solves it.
- When custom CSS is genuinely required (e.g. a keyframe animation or a selector Tailwind cannot express), keep it minimal, colocated, and leave a comment explaining why Tailwind was not sufficient.
- Use Tailwind's theme tokens for colours, spacing, and typography rather than hard-coded values, so design changes stay centralised.

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

Tests must be deterministic. They must not spawn real processes, probe real
pids, read real `/proc`, touch disk, sleep on wall-clock, or depend on any
state the daemon didn't hand them. The way we enforce this is that every
capability the daemon takes from the outside world lives behind a trait in
`crate::system`:

- `Fs` — filesystem. `LocalFs` in production, `InMemoryFs` in tests.
- `ProcessSpawner` + `ManagedChild` — child-process lifecycle. `LocalSpawner`
  in production (uses `tokio::process` + `nix` for signals); `FakeSpawner`
  in tests (virtual pids, no OS processes, state inspectable for assertions).

These are bundled into `system::SystemDeps`. Production code calls
`SystemDeps::local()`; tests call `SystemDeps::fake()` which also returns
the concrete fakes so assertions can inspect state (e.g. "which children
were SIGTERM'd, which were SIGKILL'd"). The `SupervisorDeps` and `AppState`
structs carry a `system: SystemDeps` field — they never hold `LocalFs`
or `LocalSpawner` directly.

**When adding a new outside-world dependency (clock, network, `/proc`
readers, etc.):**

1. Define a trait in `ananke/src/system/<name>.rs` with a production impl
   and a test fake. Gate the fake behind `#[cfg(any(test, feature = "test-fakes"))]`.
2. Re-export from `system::mod.rs` and add it as a field on `SystemDeps`.
   Update `SystemDeps::local()` and `SystemDeps::fake()`.
3. Route every caller through `deps.system.<field>`; never use
   `std::fs::*`, `tokio::process::Command`, `SystemTime::now`, etc.
   directly outside the trait's production impl.

Time is the narrow exception: supervisors already run on `tokio::time` so
`start_paused = true` gives tests virtual time without another trait.
`tracking::now_unix_ms` (wall-clock) is used for event timestamps and DB
rows; tests don't assert on its values.

**Anti-patterns that should not appear in tests:**

- `nix::sys::signal::kill(pid, 0)` to probe a real pid — use
  `FakeSpawner::children()` and assert on `FakeProcessState`.
- `tokio::process::Command` to spawn a shell sleep — use `FakeSpawner`.
- `tokio::time::sleep(Duration::from_millis(N))` to let real wall-clock
  time pass — use `start_paused = true` + `tokio::time::advance` or
  `wait_for(predicate)` on explicit state.
- `std::fs::*` or `tempfile::*` — use `InMemoryFs`.
- Real TCP sockets to a real service — the `TestHarness` echo server is the
  single permitted loopback listener and exists only because routing the
  hyper proxy data-plane through a trait would obscure its semantics.

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
