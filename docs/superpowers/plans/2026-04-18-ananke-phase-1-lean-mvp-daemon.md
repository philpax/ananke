# Ananke Phase 1 — Lean MVP Daemon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a running `ananke` binary that loads a TOML config, launches one or more persistent `llama-cpp` child processes, proxies HTTP to each on its declared port, captures logs to SQLite, survives its own restart without orphaning children, and shuts down cleanly.

**Architecture:** Tokio-based daemon. One supervisor task per service owns its child lifetime, health loop, log pumps, and state transitions. A `hyper` reverse proxy per service sits between clients and the child on a private loopback port. Allocation table (`Mutex<BTreeMap<ServiceName, Allocation>>`) is populated from mandatory `devices.placement_override` (no estimator in phase 1). SQLite via `toasty` with `rusqlite` fallback if toasty's migrations misbehave.

**Tech Stack:** Rust 2024, tokio, hyper 1.x + hyper-util, toml_edit, toasty (SQLite), nvml-wrapper, nix (`prctl`), tracing.

**Parent design:** `docs/superpowers/specs/2026-04-18-ananke-phase-1-lean-mvp-daemon.md`.

---

## File Structure

```
src/
├── main.rs                 // bin: ananke
├── lib.rs                  // re-exports for tests
├── errors.rs               // ExpectedError + internal error types
├── config/
│   ├── mod.rs              // public API
│   ├── file.rs             // path resolution, file IO
│   ├── parse.rs            // toml_edit -> raw Config
│   ├── merge.rs            // extends + *_append, migrate_from
│   └── validate.rs         // span-annotated ConfigError
├── db/
│   ├── mod.rs              // bootstrap, migrations
│   ├── schema.rs           // entity definitions
│   └── logs.rs             // batching writer
├── devices/
│   ├── mod.rs              // Device, DeviceId, Allocation
│   ├── probe.rs            // GpuProbe trait
│   ├── nvml.rs             // NvmlProbe impl
│   ├── fake.rs             // FakeProbe (tests + feature gate)
│   ├── cpu.rs              // /proc/meminfo
│   └── cuda_env.rs         // CUDA_VISIBLE_DEVICES rendering
├── state.rs                // ServiceState + valid_transition
├── supervise/
│   ├── mod.rs              // Supervisor / SupervisorHandle / SupervisorCommand
│   ├── spawn.rs            // argv rendering + child spawn with prctl
│   ├── health.rs           // HTTP health probe loop
│   ├── logs.rs             // stdout/stderr pumps
│   └── orphans.rs          // startup recovery
├── proxy.rs                // per-service hyper reverse proxy
├── signals.rs              // SIGTERM/SIGINT/SIGQUIT
├── retention.rs            // 3am trim + hourly incremental_vacuum
└── daemon.rs               // top-level orchestration

tests/
├── common/
│   ├── mod.rs              // shared helpers
│   └── echo_server.rs      // toy HTTP server
├── daemon_end_to_end.rs
├── sse_passthrough.rs
├── orphan_recovery.rs
└── config_integration.rs

justfile                    // just lint (cross-cutting)
tests/manual/
└── phase-1-smoke.md        // manual runbook
```

---

## Task 0: Project scaffolding, dependencies, and tooling

**Files:**
- Modify: `Cargo.toml`
- Create: `src/lib.rs`
- Modify: `src/main.rs`
- Create: `justfile`
- Create: `rustfmt.toml`
- Create: `clippy.toml`
- Modify: `.gitignore`

- [ ] **Step 1: Add workspace dependencies to `Cargo.toml`**

```toml
[package]
name = "ananke"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "ananke"
path = "src/main.rs"

[lib]
name = "ananke"
path = "src/lib.rs"

[dependencies]
tokio = { version = "1", features = ["rt-multi-thread", "macros", "net", "io-util", "signal", "process", "time", "fs", "sync"] }
hyper = { version = "1", features = ["server", "client", "http1"] }
hyper-util = { version = "0.1", features = ["tokio", "server", "server-auto", "client", "client-legacy", "http1"] }
http = "1"
http-body-util = "0.1"
bytes = "1"
toml_edit = { version = "0.22", features = ["serde"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "fmt"] }
nvml-wrapper = "0.10"
nix = { version = "0.29", features = ["signal", "fs", "process"] }
toasty = "0.1"
rusqlite = { version = "0.32", features = ["bundled"] }
reqwest = { version = "0.12", default-features = false, features = ["rustls-tls", "stream"] }
futures = "0.3"
smol_str = { version = "0.3", features = ["serde"] }
parking_lot = "0.12"
libc = "0.2"

[dev-dependencies]
proptest = "1"
pretty_assertions = "1"
tokio = { version = "1", features = ["full", "test-util"] }
tempfile = "3"

[features]
default = []
test-fakes = []
```

- [ ] **Step 2: Create `src/lib.rs` with module declarations**

```rust
//! Ananke — GPU/CPU-aware model proxy daemon.

pub mod config;
pub mod daemon;
pub mod db;
pub mod devices;
pub mod errors;
pub mod proxy;
pub mod retention;
pub mod signals;
pub mod state;
pub mod supervise;
```

- [ ] **Step 3: Replace `src/main.rs` with a bin that delegates to the library**

```rust
use ananke::daemon::run;
use ananke::errors::ExpectedError;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> std::process::ExitCode {
    match run().await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("ananke: {err}");
            std::process::ExitCode::from(err.exit_code())
        }
    }
}

// Re-export so the binary compiles before daemon::run exists.
// This will be removed in Task 23 when daemon::run is implemented.
#[allow(dead_code)]
fn _ensure_expected_error_in_scope() {
    let _: Option<ExpectedError> = None;
}
```

- [ ] **Step 4: Create `justfile` with the `lint` recipe**

```
default:
    @just --list

# Run all linters and tests across Rust and TypeScript.
lint: lint-rust lint-frontend

lint-rust:
    cargo fmt --all -- --check
    cargo clippy --all-targets --all-features -- -D warnings
    cargo clippy --all-targets --no-default-features -- -D warnings
    cargo test --workspace
    cargo test --workspace --no-default-features

lint-frontend:
    cd frontend && npm run lint

fmt:
    cargo fmt --all
    cd frontend && npm run format
```

- [ ] **Step 5: Create `rustfmt.toml`**

```
edition = "2024"
```

- [ ] **Step 6: Create empty `clippy.toml`**

```
# Intentionally minimal; repo-wide clippy settings live in CI invocations.
```

- [ ] **Step 7: Add `/target` and SQLite files to `.gitignore`**

Read current `.gitignore` and add if not present:

```
/target
*.sqlite
*.sqlite-journal
*.sqlite-wal
*.sqlite-shm
```

- [ ] **Step 8: Verify it compiles**

Run: `cargo check`
Expected: Compiles. `src/main.rs` will warn about the unresolved `daemon::run` and `ExpectedError` until Task 1 and Task 23 — that is expected. If `daemon::run` errors prevent compilation, stub `src/daemon.rs` with:

```rust
use crate::errors::ExpectedError;

pub async fn run() -> Result<(), ExpectedError> {
    unimplemented!("daemon::run wired up in Task 23")
}
```

and `src/errors.rs` with:

```rust
#[derive(Debug)]
pub struct ExpectedError;

impl ExpectedError {
    pub fn exit_code(&self) -> u8 { 1 }
}

impl std::fmt::Display for ExpectedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "placeholder error")
    }
}

impl std::error::Error for ExpectedError {}
```

Stub all other `pub mod` referenced in `lib.rs` with an empty file (`touch src/config/mod.rs` etc., with a single `//! Placeholder.` line). These stubs will be filled in subsequent tasks.

- [ ] **Step 9: Commit**

```bash
git add Cargo.toml Cargo.lock src/ justfile rustfmt.toml clippy.toml .gitignore
git commit -m "chore: add Cargo deps, justfile, and module skeleton"
```

---

## Task 1: Errors — `ExpectedError` and `ConfigError`

**Files:**
- Replace: `src/errors.rs`
- Test: `src/errors.rs` (inline `#[cfg(test)]`)

The error model per `CONTRIBUTING.md`: no `thiserror`, manual `Display` + `Error` impls, lowercase sentence fragments, semantic exit codes for user-facing errors.

- [ ] **Step 1: Write the failing test**

Append to `src/errors.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_error_display_is_lowercase_fragment() {
        let err = ExpectedError::bind_failed("127.0.0.1:7777".into(), "permission denied".into());
        let msg = format!("{err}");
        assert_eq!(msg, "failed to bind 127.0.0.1:7777: permission denied");
        assert_eq!(err.exit_code(), 2);
    }

    #[test]
    fn config_error_kinds_distinguished() {
        let err = ExpectedError::config_unparseable("/tmp/x.toml".into(), "unexpected token".into());
        assert!(format!("{err}").contains("/tmp/x.toml"));
        assert_eq!(err.exit_code(), 3);
    }

    #[test]
    fn no_devices_exit_code_is_stable() {
        assert_eq!(ExpectedError::no_devices().exit_code(), 4);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib errors::tests`
Expected: FAIL — methods `bind_failed`, `config_unparseable`, `no_devices` do not exist.

- [ ] **Step 3: Implement `ExpectedError`**

Replace `src/errors.rs`:

```rust
//! User-facing daemon errors with semantic exit codes.
//!
//! Internal programming errors use `anyhow`-free ad-hoc enums or panic.
//! `ExpectedError` is reserved for conditions where the daemon exits
//! non-zero and the user needs a clear message.

use std::fmt;
use std::path::PathBuf;

#[derive(Debug)]
pub struct ExpectedError {
    kind: ExpectedErrorKind,
}

#[derive(Debug)]
enum ExpectedErrorKind {
    BindFailed { addr: String, cause: String },
    ConfigUnparseable { path: PathBuf, cause: String },
    ConfigFileMissing { path: PathBuf },
    DatabaseOpenFailed { path: PathBuf, cause: String },
    NoDevices,
}

impl ExpectedError {
    pub fn bind_failed(addr: String, cause: String) -> Self {
        Self { kind: ExpectedErrorKind::BindFailed { addr, cause } }
    }

    pub fn config_unparseable(path: PathBuf, cause: String) -> Self {
        Self { kind: ExpectedErrorKind::ConfigUnparseable { path, cause } }
    }

    pub fn config_file_missing(path: PathBuf) -> Self {
        Self { kind: ExpectedErrorKind::ConfigFileMissing { path } }
    }

    pub fn database_open_failed(path: PathBuf, cause: String) -> Self {
        Self { kind: ExpectedErrorKind::DatabaseOpenFailed { path, cause } }
    }

    pub fn no_devices() -> Self {
        Self { kind: ExpectedErrorKind::NoDevices }
    }

    pub fn exit_code(&self) -> u8 {
        match self.kind {
            ExpectedErrorKind::BindFailed { .. } => 2,
            ExpectedErrorKind::ConfigUnparseable { .. } => 3,
            ExpectedErrorKind::ConfigFileMissing { .. } => 3,
            ExpectedErrorKind::DatabaseOpenFailed { .. } => 5,
            ExpectedErrorKind::NoDevices => 4,
        }
    }
}

impl fmt::Display for ExpectedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ExpectedErrorKind::BindFailed { addr, cause } => {
                write!(f, "failed to bind {addr}: {cause}")
            }
            ExpectedErrorKind::ConfigUnparseable { path, cause } => {
                write!(f, "failed to parse config at {}: {cause}", path.display())
            }
            ExpectedErrorKind::ConfigFileMissing { path } => {
                write!(f, "config file not found at {}", path.display())
            }
            ExpectedErrorKind::DatabaseOpenFailed { path, cause } => {
                write!(f, "failed to open database at {}: {cause}", path.display())
            }
            ExpectedErrorKind::NoDevices => {
                write!(f, "no devices available: NVML and CPU probes both failed")
            }
        }
    }
}

impl std::error::Error for ExpectedError {}
```

Re-include the `#[cfg(test)] mod tests` block from Step 1 at the bottom of the file.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib errors::tests`
Expected: PASS — 3 tests green.

- [ ] **Step 5: Commit**

```bash
git add src/errors.rs
git commit -m "feat(errors): add ExpectedError with semantic exit codes"
```

---

## Task 2: Config path resolution

**Files:**
- Create: `src/config/mod.rs` (replace stub)
- Create: `src/config/file.rs`

Path resolution order per spec §6.1: `$ANANKE_CONFIG` → `--config` → `$XDG_CONFIG_HOME/ananke/config.toml` → `/etc/ananke/config.toml`.

- [ ] **Step 1: Write the failing test**

Create `src/config/file.rs` with the test module:

```rust
//! Config path resolution.

use std::path::{Path, PathBuf};

use crate::errors::ExpectedError;

/// Sources checked for the config file, in priority order.
pub struct PathSources<'a> {
    pub env_ananke_config: Option<&'a str>,
    pub cli_config: Option<&'a Path>,
    pub xdg_config_home: Option<&'a Path>,
    pub home: Option<&'a Path>,
}

pub fn resolve_config_path(sources: PathSources<'_>) -> Result<PathBuf, ExpectedError> {
    unimplemented!("resolve_config_path")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn env_wins_over_cli() {
        let path = resolve_config_path(PathSources {
            env_ananke_config: Some("/a/env.toml"),
            cli_config: Some(Path::new("/b/cli.toml")),
            xdg_config_home: None,
            home: None,
        }).unwrap();
        assert_eq!(path, PathBuf::from("/a/env.toml"));
    }

    #[test]
    fn cli_wins_over_xdg() {
        let path = resolve_config_path(PathSources {
            env_ananke_config: None,
            cli_config: Some(Path::new("/b/cli.toml")),
            xdg_config_home: Some(Path::new("/home/u/.config")),
            home: None,
        }).unwrap();
        assert_eq!(path, PathBuf::from("/b/cli.toml"));
    }

    #[test]
    fn xdg_default() {
        let path = resolve_config_path(PathSources {
            env_ananke_config: None,
            cli_config: None,
            xdg_config_home: Some(Path::new("/home/u/.config")),
            home: None,
        }).unwrap();
        assert_eq!(path, PathBuf::from("/home/u/.config/ananke/config.toml"));
    }

    #[test]
    fn xdg_falls_back_to_home_dot_config() {
        let path = resolve_config_path(PathSources {
            env_ananke_config: None,
            cli_config: None,
            xdg_config_home: None,
            home: Some(Path::new("/home/u")),
        }).unwrap();
        assert_eq!(path, PathBuf::from("/home/u/.config/ananke/config.toml"));
    }

    #[test]
    fn etc_fallback() {
        let path = resolve_config_path(PathSources {
            env_ananke_config: None,
            cli_config: None,
            xdg_config_home: None,
            home: None,
        }).unwrap();
        assert_eq!(path, PathBuf::from("/etc/ananke/config.toml"));
    }
}
```

Replace `src/config/mod.rs`:

```rust
//! Configuration loading, parsing, inheritance merging, and validation.

pub mod file;

pub use file::{resolve_config_path, PathSources};
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib config::file::tests`
Expected: FAIL — `unimplemented!`.

- [ ] **Step 3: Implement `resolve_config_path`**

Replace the `unimplemented!` body with:

```rust
pub fn resolve_config_path(sources: PathSources<'_>) -> Result<PathBuf, ExpectedError> {
    if let Some(p) = sources.env_ananke_config {
        return Ok(PathBuf::from(p));
    }
    if let Some(p) = sources.cli_config {
        return Ok(p.to_path_buf());
    }
    if let Some(xdg) = sources.xdg_config_home {
        return Ok(xdg.join("ananke").join("config.toml"));
    }
    if let Some(home) = sources.home {
        return Ok(home.join(".config").join("ananke").join("config.toml"));
    }
    Ok(PathBuf::from("/etc/ananke/config.toml"))
}

/// Resolve config path from the live process environment.
/// Called from `main` via the library. Tests inject via [`resolve_config_path`].
pub fn resolve_from_env(cli_config: Option<&Path>) -> Result<PathBuf, ExpectedError> {
    let env = std::env::var("ANANKE_CONFIG").ok();
    let xdg = std::env::var("XDG_CONFIG_HOME").ok().map(PathBuf::from);
    let home = std::env::var("HOME").ok().map(PathBuf::from);
    resolve_config_path(PathSources {
        env_ananke_config: env.as_deref(),
        cli_config,
        xdg_config_home: xdg.as_deref(),
        home: home.as_deref(),
    })
}
```

Update the `pub use` in `src/config/mod.rs`:

```rust
pub use file::{resolve_config_path, resolve_from_env, PathSources};
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib config::file`
Expected: PASS — 5 tests green.

- [ ] **Step 5: Commit**

```bash
git add src/config/
git commit -m "feat(config): add config path resolution"
```

---

## Task 3: Config parse — raw TOML tree

**Files:**
- Create: `src/config/parse.rs`
- Modify: `src/config/mod.rs`

This task parses the TOML string into a typed `RawConfig` tree **without** applying inheritance or validation. Inheritance comes in Task 4, validation in Task 6.

- [ ] **Step 1: Write the failing test**

Create `src/config/parse.rs`:

```rust
//! Parse a TOML string into a `RawConfig` typed tree (pre-merge, pre-validation).

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Deserialize;
use smol_str::SmolStr;

use crate::errors::ExpectedError;

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct RawConfig {
    #[serde(default)]
    pub daemon: DaemonConfig,
    #[serde(default)]
    pub devices: DevicesConfig,
    #[serde(default)]
    pub openai_api: OpenAiApiConfig,
    #[serde(default)]
    pub defaults: DefaultsConfig,
    #[serde(default, rename = "service")]
    pub services: Vec<RawService>,
    #[serde(default, rename = "persistent_service")]
    pub persistent_services: Vec<RawService>,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct DaemonConfig {
    #[serde(default = "default_management_listen")]
    pub management_listen: String,
    pub data_dir: Option<PathBuf>,
    #[serde(default = "default_shutdown_timeout")]
    pub shutdown_timeout: String,
}

fn default_management_listen() -> String { "127.0.0.1:7777".into() }
fn default_shutdown_timeout() -> String { "120s".into() }

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct DevicesConfig {
    #[serde(default)]
    pub gpu_ids: Option<Vec<u32>>,
    #[serde(default)]
    pub gpu_reserved_mb: BTreeMap<String, u64>,
    #[serde(default)]
    pub default_gpu_reserved_mb: Option<u64>,
    #[serde(default)]
    pub cpu: CpuConfig,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct CpuConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub reserved_gb: Option<u64>,
}

fn default_true() -> bool { true }

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct OpenAiApiConfig {
    pub listen: Option<String>,
    #[serde(default)]
    pub enabled: Option<bool>,
    pub max_request_duration: Option<String>,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct DefaultsConfig {
    pub idle_timeout: Option<String>,
    pub priority: Option<u8>,
    pub warming_grace: Option<String>,
    pub start_queue_depth: Option<u32>,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(default)]
pub struct RawService {
    pub name: Option<SmolStr>,
    pub template: Option<SmolStr>,
    pub extends: Option<SmolStr>,
    pub migrate_from: Option<SmolStr>,
    pub port: Option<u16>,
    pub model: Option<PathBuf>,
    pub mmproj: Option<PathBuf>,
    pub context: Option<u32>,
    pub lifecycle: Option<SmolStr>,
    pub priority: Option<u8>,
    pub idle_timeout: Option<String>,
    pub warming_grace: Option<String>,
    pub description: Option<String>,
    pub n_gpu_layers: Option<i32>,
    pub n_cpu_moe: Option<u32>,
    pub flash_attn: Option<bool>,
    pub cache_type_k: Option<SmolStr>,
    pub cache_type_v: Option<SmolStr>,
    pub mmap: Option<bool>,
    pub mlock: Option<bool>,
    pub parallel: Option<u32>,
    pub batch_size: Option<u32>,
    pub ubatch_size: Option<u32>,
    pub threads: Option<u32>,
    pub threads_batch: Option<u32>,
    pub jinja: Option<bool>,
    pub chat_template_file: Option<PathBuf>,
    pub override_tensor: Option<Vec<String>>,
    pub sampling: Option<BTreeMap<String, toml_edit::easy::Value>>,
    pub filters: Option<RawFilters>,
    pub metadata: Option<BTreeMap<String, toml_edit::easy::Value>>,
    pub devices: Option<RawServiceDevices>,
    pub estimation: Option<RawEstimation>,
    pub extra_args: Option<Vec<String>>,
    pub extra_args_append: Option<Vec<String>>,
    pub env: Option<BTreeMap<String, String>>,
    pub health: Option<RawHealth>,
    pub drain_timeout: Option<String>,
    pub extended_stream_drain: Option<String>,
    pub max_request_duration: Option<String>,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawFilters {
    pub strip_params: Option<Vec<String>>,
    pub set_params: Option<BTreeMap<String, toml_edit::easy::Value>>,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawServiceDevices {
    pub placement: Option<SmolStr>,
    pub gpu_allow: Option<Vec<u32>>,
    pub placement_override: Option<BTreeMap<String, u64>>,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawEstimation {
    pub compute_buffer_mb: Option<u32>,
    pub safety_factor: Option<f32>,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, default)]
pub struct RawHealth {
    pub http: Option<String>,
    pub timeout: Option<String>,
    pub probe_interval: Option<String>,
}

pub fn parse_toml(source: &str, origin_path: &std::path::Path) -> Result<RawConfig, ExpectedError> {
    unimplemented!("parse_toml")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn parses_minimal() {
        let toml = r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
"#;
        let cfg = parse_toml(toml, Path::new("/tmp/c.toml")).unwrap();
        assert_eq!(cfg.services.len(), 1);
        assert_eq!(cfg.services[0].name.as_deref(), Some("demo"));
        assert_eq!(cfg.services[0].port, Some(11435));
    }

    #[test]
    fn parses_persistent_service_alias() {
        let toml = r#"
[[persistent_service]]
name = "big"
template = "llama-cpp"
model = "/m/b.gguf"
port = 11500
"#;
        let cfg = parse_toml(toml, Path::new("/tmp/c.toml")).unwrap();
        assert_eq!(cfg.persistent_services.len(), 1);
        assert_eq!(cfg.persistent_services[0].name.as_deref(), Some("big"));
    }

    #[test]
    fn parses_dotted_keys() {
        let toml = r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
sampling.temperature = 0.7
devices.placement = "gpu-only"
devices.placement_override = { "gpu:0" = 18944 }
"#;
        let cfg = parse_toml(toml, Path::new("/tmp/c.toml")).unwrap();
        let s = &cfg.services[0];
        assert_eq!(s.devices.as_ref().unwrap().placement.as_deref(), Some("gpu-only"));
        assert_eq!(s.devices.as_ref().unwrap().placement_override.as_ref().unwrap()["gpu:0"], 18944);
    }

    #[test]
    fn rejects_unparseable() {
        let toml = "this is not valid toml [[[";
        let err = parse_toml(toml, Path::new("/tmp/c.toml")).unwrap_err();
        assert!(format!("{err}").contains("parse"));
    }
}
```

Update `src/config/mod.rs`:

```rust
//! Configuration loading, parsing, inheritance merging, and validation.

pub mod file;
pub mod parse;

pub use file::{resolve_config_path, resolve_from_env, PathSources};
pub use parse::{parse_toml, RawConfig, RawService};
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib config::parse`
Expected: FAIL — `unimplemented!`.

- [ ] **Step 3: Implement `parse_toml`**

Replace the `unimplemented!` body in `parse.rs`:

```rust
pub fn parse_toml(source: &str, origin_path: &std::path::Path) -> Result<RawConfig, ExpectedError> {
    let doc = source.parse::<toml_edit::DocumentMut>().map_err(|e| {
        ExpectedError::config_unparseable(origin_path.to_path_buf(), e.to_string())
    })?;
    let de = toml_edit::de::ValueDeserializer::new(doc.as_item().clone().into_value().map_err(|_| {
        ExpectedError::config_unparseable(origin_path.to_path_buf(), "config root must be a table".into())
    })?);
    RawConfig::deserialize(de).map_err(|e| {
        ExpectedError::config_unparseable(origin_path.to_path_buf(), e.to_string())
    })
}
```

**Note to implementer:** If `toml_edit::de::ValueDeserializer` does not expose the needed API at version 0.22, use `toml_edit::de::from_str` directly:

```rust
pub fn parse_toml(source: &str, origin_path: &std::path::Path) -> Result<RawConfig, ExpectedError> {
    toml_edit::de::from_str::<RawConfig>(source).map_err(|e| {
        ExpectedError::config_unparseable(origin_path.to_path_buf(), e.to_string())
    })
}
```

Prefer the second form unless span-aware errors are needed here (they surface in `validate.rs` via re-parsing with `toml_edit::DocumentMut`).

Also: `toml_edit::easy::Value` may not exist in 0.22. Replace uses of `toml_edit::easy::Value` with `toml::Value` — add `toml = "0.8"` to `Cargo.toml` if not present for this purpose only. Re-run `cargo check` and fix any type mismatches before running tests.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib config::parse`
Expected: PASS — 4 tests green.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/config/
git commit -m "feat(config): parse TOML into RawConfig tree"
```

---

## Task 4: Config merge — `extends` and `*_append`

**Files:**
- Create: `src/config/merge.rs`
- Modify: `src/config/mod.rs`

- [ ] **Step 1: Write the failing test**

Create `src/config/merge.rs`:

```rust
//! Resolve `extends` inheritance and `*_append` concatenation before validation.
//!
//! Rules (spec §6.3):
//! - Scalars: child overrides parent.
//! - Sub-tables: deep-merge field-by-field.
//! - Arrays: child replaces parent outright.
//! - `*_append` siblings: `parent.foo ++ parent.foo_append ++ child.foo ++ child.foo_append`;
//!   `child.foo` falls back to `parent.foo` if not specified.
//! - `extends` is transitive; cycles are errors.
//! - `name` and `port` must be overridden; inheriting either is an error.
//! - `extends` and `migrate_from` are not themselves inherited.

use std::collections::{BTreeMap, BTreeSet};

use smol_str::SmolStr;

use crate::config::parse::{RawConfig, RawService};
use crate::errors::ExpectedError;

#[derive(Debug, thiserror::Error)]
pub enum MergeError { /* placeholder — real type below */ }

pub fn resolve_inheritance(cfg: &mut RawConfig) -> Result<(), ExpectedError> {
    unimplemented!("resolve_inheritance")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::parse::parse_toml;
    use std::path::Path;

    fn parse(src: &str) -> RawConfig {
        parse_toml(src, Path::new("/t")).unwrap()
    }

    #[test]
    fn child_scalar_overrides_parent() {
        let mut cfg = parse(r#"
[[service]]
name = "base"
template = "llama-cpp"
model = "/m/a.gguf"
port = 11000
context = 8192

[[service]]
name = "child"
extends = "base"
port = 11001
context = 16384
"#);
        resolve_inheritance(&mut cfg).unwrap();
        let c = cfg.services.iter().find(|s| s.name.as_deref() == Some("child")).unwrap();
        assert_eq!(c.context, Some(16384));
        assert_eq!(c.model.as_ref().unwrap().to_str(), Some("/m/a.gguf"));
    }

    #[test]
    fn extra_args_append_concatenates() {
        let mut cfg = parse(r#"
[[service]]
name = "base"
template = "llama-cpp"
model = "/m/a.gguf"
port = 11000
extra_args = ["--metrics"]
extra_args_append = ["--flash"]

[[service]]
name = "child"
extends = "base"
port = 11001
extra_args_append = ["--verbose"]
"#);
        resolve_inheritance(&mut cfg).unwrap();
        let c = cfg.services.iter().find(|s| s.name.as_deref() == Some("child")).unwrap();
        // Effective: parent.extra_args ++ parent.extra_args_append ++ child.extra_args(falls back to parent) ++ child.extra_args_append
        // Since child.extra_args is not specified, it falls back to parent.extra_args ["--metrics"]
        // So effective = ["--metrics", "--flash", "--metrics", "--verbose"]? That duplicates.
        // Per spec: "child.foo either replaces (if specified) or falls back to parent.foo (if not)".
        // Fallback means child contributes parent's value once; combined with parent's own contribution
        // this yields doubled entries. The intended rule is that the *_append chain only runs once;
        // implement as: effective = merged_base.extra_args ++ merged_base.extra_args_append ++ child.extra_args_append
        // where merged_base is parent's already-resolved extra_args list after its own *_append.
        // Test the simpler invariant: result contains both "--flash" and "--verbose" in order.
        let args = c.extra_args.clone().unwrap_or_default();
        let idx_flash = args.iter().position(|a| a == "--flash");
        let idx_verbose = args.iter().position(|a| a == "--verbose");
        assert!(idx_flash.is_some(), "missing --flash in {args:?}");
        assert!(idx_verbose.is_some(), "missing --verbose in {args:?}");
        assert!(idx_flash.unwrap() < idx_verbose.unwrap());
    }

    #[test]
    fn transitive_extends() {
        let mut cfg = parse(r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/a.gguf"
port = 11000
context = 4096

[[service]]
name = "b"
extends = "a"
port = 11001

[[service]]
name = "c"
extends = "b"
port = 11002
context = 32768
"#);
        resolve_inheritance(&mut cfg).unwrap();
        let c = cfg.services.iter().find(|s| s.name.as_deref() == Some("c")).unwrap();
        assert_eq!(c.context, Some(32768));
        assert_eq!(c.model.as_ref().unwrap().to_str(), Some("/m/a.gguf"));
    }

    #[test]
    fn cycle_is_error() {
        let mut cfg = parse(r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/a.gguf"
port = 11000
extends = "b"

[[service]]
name = "b"
template = "llama-cpp"
model = "/m/a.gguf"
port = 11001
extends = "a"
"#);
        let err = resolve_inheritance(&mut cfg).unwrap_err();
        assert!(format!("{err}").contains("cycle"));
    }

    #[test]
    fn inheriting_port_is_error() {
        let mut cfg = parse(r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/a.gguf"
port = 11000

[[service]]
name = "b"
extends = "a"
"#);
        let err = resolve_inheritance(&mut cfg).unwrap_err();
        assert!(format!("{err}").contains("port"), "error: {err}");
    }

    #[test]
    fn missing_extends_target_is_error() {
        let mut cfg = parse(r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/a.gguf"
port = 11000
extends = "does-not-exist"
"#);
        let err = resolve_inheritance(&mut cfg).unwrap_err();
        assert!(format!("{err}").contains("does-not-exist"));
    }
}
```

Update `src/config/mod.rs`:

```rust
pub mod file;
pub mod merge;
pub mod parse;

pub use file::{resolve_config_path, resolve_from_env, PathSources};
pub use merge::resolve_inheritance;
pub use parse::{parse_toml, RawConfig, RawService};
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib config::merge`
Expected: FAIL — `unimplemented!`.

- [ ] **Step 3: Implement inheritance resolution**

Replace `src/config/merge.rs` (keep the test module at bottom):

```rust
//! Resolve `extends` inheritance and `*_append` concatenation before validation.

use std::collections::{BTreeMap, BTreeSet};

use smol_str::SmolStr;

use crate::config::parse::{RawConfig, RawService};
use crate::errors::ExpectedError;

pub fn resolve_inheritance(cfg: &mut RawConfig) -> Result<(), ExpectedError> {
    // 1. Fold [[persistent_service]] into [[service]] with lifecycle=persistent default.
    for mut ps in std::mem::take(&mut cfg.persistent_services) {
        if ps.lifecycle.is_none() {
            ps.lifecycle = Some(SmolStr::new("persistent"));
        }
        cfg.services.push(ps);
    }

    // 2. Index services by name; require names.
    let mut by_name: BTreeMap<SmolStr, RawService> = BTreeMap::new();
    for s in std::mem::take(&mut cfg.services) {
        let name = s.name.clone().ok_or_else(|| {
            ExpectedError::config_unparseable(
                std::path::PathBuf::from("<config>"),
                "service block missing name".into(),
            )
        })?;
        if by_name.insert(name.clone(), s).is_some() {
            return Err(ExpectedError::config_unparseable(
                std::path::PathBuf::from("<config>"),
                format!("duplicate service name: {name}"),
            ));
        }
    }

    // 3. Topologically resolve each service's chain.
    let mut resolved: BTreeMap<SmolStr, RawService> = BTreeMap::new();
    let names: Vec<SmolStr> = by_name.keys().cloned().collect();
    for name in &names {
        resolve_one(name, &by_name, &mut resolved, &mut BTreeSet::new())?;
    }

    cfg.services = resolved.into_values().collect();
    Ok(())
}

fn resolve_one(
    name: &SmolStr,
    source: &BTreeMap<SmolStr, RawService>,
    resolved: &mut BTreeMap<SmolStr, RawService>,
    stack: &mut BTreeSet<SmolStr>,
) -> Result<(), ExpectedError> {
    if resolved.contains_key(name) {
        return Ok(());
    }
    if !stack.insert(name.clone()) {
        return Err(ExpectedError::config_unparseable(
            std::path::PathBuf::from("<config>"),
            format!("extends cycle involving service {name}"),
        ));
    }

    let raw = source.get(name).cloned().ok_or_else(|| {
        ExpectedError::config_unparseable(
            std::path::PathBuf::from("<config>"),
            format!("service {name} not found during extends resolution"),
        )
    })?;

    let merged = match raw.extends.clone() {
        None => raw,
        Some(parent_name) => {
            if !source.contains_key(&parent_name) {
                return Err(ExpectedError::config_unparseable(
                    std::path::PathBuf::from("<config>"),
                    format!("service {name} extends {parent_name} which does not exist"),
                ));
            }
            resolve_one(&parent_name, source, resolved, stack)?;
            let parent = resolved.get(&parent_name).unwrap().clone();
            merge_service(&parent, &raw, name)?
        }
    };

    stack.remove(name);
    resolved.insert(name.clone(), merged);
    Ok(())
}

fn merge_service(parent: &RawService, child: &RawService, child_name: &SmolStr) -> Result<RawService, ExpectedError> {
    // Child must override name and port.
    if child.port.is_none() {
        return Err(ExpectedError::config_unparseable(
            std::path::PathBuf::from("<config>"),
            format!("service {child_name} must override port from parent"),
        ));
    }

    let mut merged = parent.clone();

    // Scalars and paths: child overrides if present.
    macro_rules! take {
        ($field:ident) => { if child.$field.is_some() { merged.$field = child.$field.clone(); } };
    }

    merged.name = child.name.clone();
    merged.port = child.port;
    // `extends` and `migrate_from` are not inherited.
    merged.extends = None;
    merged.migrate_from = None;

    take!(template);
    take!(model);
    take!(mmproj);
    take!(context);
    take!(lifecycle);
    take!(priority);
    take!(idle_timeout);
    take!(warming_grace);
    take!(description);
    take!(n_gpu_layers);
    take!(n_cpu_moe);
    take!(flash_attn);
    take!(cache_type_k);
    take!(cache_type_v);
    take!(mmap);
    take!(mlock);
    take!(parallel);
    take!(batch_size);
    take!(ubatch_size);
    take!(threads);
    take!(threads_batch);
    take!(jinja);
    take!(chat_template_file);
    take!(override_tensor);
    take!(drain_timeout);
    take!(extended_stream_drain);
    take!(max_request_duration);

    // Nested tables deep-merge.
    merged.sampling = deep_merge_map(parent.sampling.clone(), child.sampling.clone());
    merged.metadata = deep_merge_map(parent.metadata.clone(), child.metadata.clone());
    merged.env = deep_merge_strs(parent.env.clone(), child.env.clone());

    merged.filters = match (parent.filters.clone(), child.filters.clone()) {
        (None, x) => x,
        (x, None) => x,
        (Some(p), Some(c)) => Some(crate::config::parse::RawFilters {
            strip_params: c.strip_params.or(p.strip_params),
            set_params: deep_merge_map(p.set_params, c.set_params),
        }),
    };

    merged.devices = match (parent.devices.clone(), child.devices.clone()) {
        (None, x) => x,
        (x, None) => x,
        (Some(p), Some(c)) => Some(crate::config::parse::RawServiceDevices {
            placement: c.placement.or(p.placement),
            gpu_allow: c.gpu_allow.or(p.gpu_allow),
            placement_override: c.placement_override.or(p.placement_override),
        }),
    };

    merged.estimation = match (parent.estimation.clone(), child.estimation.clone()) {
        (None, x) => x,
        (x, None) => x,
        (Some(p), Some(c)) => Some(crate::config::parse::RawEstimation {
            compute_buffer_mb: c.compute_buffer_mb.or(p.compute_buffer_mb),
            safety_factor: c.safety_factor.or(p.safety_factor),
        }),
    };

    merged.health = match (parent.health.clone(), child.health.clone()) {
        (None, x) => x,
        (x, None) => x,
        (Some(p), Some(c)) => Some(crate::config::parse::RawHealth {
            http: c.http.or(p.http),
            timeout: c.timeout.or(p.timeout),
            probe_interval: c.probe_interval.or(p.probe_interval),
        }),
    };

    // extra_args: child replaces parent if present; otherwise inherit parent.
    merged.extra_args = child.extra_args.clone().or_else(|| parent.extra_args.clone());

    // *_append: concatenate parent's fully-resolved extra_args_append with child's.
    let mut append = parent.extra_args_append.clone().unwrap_or_default();
    if let Some(child_append) = &child.extra_args_append {
        append.extend(child_append.iter().cloned());
    }
    merged.extra_args_append = if append.is_empty() { None } else { Some(append) };

    Ok(merged)
}

fn deep_merge_map<V: Clone>(
    parent: Option<BTreeMap<String, V>>,
    child: Option<BTreeMap<String, V>>,
) -> Option<BTreeMap<String, V>> {
    match (parent, child) {
        (None, x) => x,
        (x, None) => x,
        (Some(mut p), Some(c)) => {
            for (k, v) in c {
                p.insert(k, v);
            }
            Some(p)
        }
    }
}

fn deep_merge_strs(
    parent: Option<BTreeMap<String, String>>,
    child: Option<BTreeMap<String, String>>,
) -> Option<BTreeMap<String, String>> {
    deep_merge_map(parent, child)
}
```

Re-include the `#[cfg(test)] mod tests` block at the bottom.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib config::merge`
Expected: PASS — 6 tests green.

- [ ] **Step 5: Commit**

```bash
git add src/config/
git commit -m "feat(config): resolve extends inheritance and *_append concatenation"
```

---

## Task 5: Config `migrate_from` chain resolution

**Files:**
- Modify: `src/config/merge.rs` (add `resolve_migrations`)

Spec §6.4: `migrate_from` reparents a prior service's `service_id` to the new name. Phase 1 records migrations in the returned list; DB reparenting happens in Task 7 (database layer).

- [ ] **Step 1: Add failing test to `src/config/merge.rs` tests module**

```rust
    #[test]
    fn migrate_from_chain_resolved_in_order() {
        let mut cfg = parse(r#"
[[service]]
name = "c"
template = "llama-cpp"
model = "/m/x.gguf"
port = 12002
migrate_from = "b"

[[service]]
name = "b"
template = "llama-cpp"
model = "/m/x.gguf"
port = 12001
migrate_from = "a"
"#);
        resolve_inheritance(&mut cfg).unwrap();
        let migrations = resolve_migrations(&mut cfg).unwrap();
        // b must be resolved before c since c depends on b.
        let b_idx = migrations.iter().position(|m| m.new_name == "b").unwrap();
        let c_idx = migrations.iter().position(|m| m.new_name == "c").unwrap();
        assert!(b_idx < c_idx);
        assert_eq!(migrations[b_idx].old_name, "a");
        assert_eq!(migrations[c_idx].old_name, "b");
    }

    #[test]
    fn migrate_from_missing_source_is_warning_not_error() {
        let mut cfg = parse(r#"
[[service]]
name = "b"
template = "llama-cpp"
model = "/m/x.gguf"
port = 12001
migrate_from = "does-not-exist"
"#);
        resolve_inheritance(&mut cfg).unwrap();
        let migrations = resolve_migrations(&mut cfg).unwrap();
        // Missing source is a warning; the migration is recorded anyway for the DB
        // layer to treat as a no-op (see spec §6.4).
        assert_eq!(migrations.len(), 1);
        assert_eq!(migrations[0].old_name, "does-not-exist");
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib config::merge::tests::migrate_from`
Expected: FAIL — `resolve_migrations` does not exist.

- [ ] **Step 3: Implement `resolve_migrations`**

Append to `src/config/merge.rs`:

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Migration {
    pub old_name: SmolStr,
    pub new_name: SmolStr,
}

/// Resolve `migrate_from` chains into an ordered list of (old, new) pairs.
///
/// Returns pairs in topological order (sources before dependents) so the
/// database layer can reparent sequentially. Cycles are errors.
pub fn resolve_migrations(cfg: &mut RawConfig) -> Result<Vec<Migration>, ExpectedError> {
    let mut out: Vec<Migration> = Vec::new();
    let by_name: BTreeMap<SmolStr, &RawService> = cfg
        .services
        .iter()
        .map(|s| (s.name.clone().unwrap(), s))
        .collect();

    let mut visiting: BTreeSet<SmolStr> = BTreeSet::new();
    let mut visited: BTreeSet<SmolStr> = BTreeSet::new();

    fn visit(
        name: &SmolStr,
        by_name: &BTreeMap<SmolStr, &RawService>,
        visiting: &mut BTreeSet<SmolStr>,
        visited: &mut BTreeSet<SmolStr>,
        out: &mut Vec<Migration>,
    ) -> Result<(), ExpectedError> {
        if visited.contains(name) { return Ok(()); }
        if !visiting.insert(name.clone()) {
            return Err(ExpectedError::config_unparseable(
                std::path::PathBuf::from("<config>"),
                format!("migrate_from cycle involving {name}"),
            ));
        }
        if let Some(svc) = by_name.get(name) {
            if let Some(old) = &svc.migrate_from {
                if by_name.contains_key(old) {
                    visit(old, by_name, visiting, visited, out)?;
                }
                out.push(Migration { old_name: old.clone(), new_name: name.clone() });
            }
        }
        visiting.remove(name);
        visited.insert(name.clone());
        Ok(())
    }

    let names: Vec<SmolStr> = by_name.keys().cloned().collect();
    for n in &names {
        visit(n, &by_name, &mut visiting, &mut visited, &mut out)?;
    }

    // Clear the migrate_from field on services so downstream code doesn't re-process.
    for svc in cfg.services.iter_mut() {
        svc.migrate_from = None;
    }

    Ok(out)
}
```

Also re-export in `src/config/mod.rs`:

```rust
pub use merge::{resolve_inheritance, resolve_migrations, Migration};
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib config::merge`
Expected: PASS — 8 tests green.

- [ ] **Step 5: Commit**

```bash
git add src/config/
git commit -m "feat(config): resolve migrate_from chains topologically"
```

---

## Task 6: Config validation with span-aware errors

**Files:**
- Create: `src/config/validate.rs`
- Modify: `src/config/mod.rs`

Phase 1 validation surface (spec §6.5 subset + phase-1 gate):
- Required fields (`template`, `model`, `port`) after merge.
- Duplicate `name` or `port`.
- Unknown template (only `llama-cpp` allowed in phase 1; `command` validates as "deferred").
- `port` equal to `daemon.management_listen`.
- `cache_type_k != "f16"` or `cache_type_v != "f16"` with `flash_attn = false`.
- `devices.placement = "cpu-only"` with `n_gpu_layers != 0`.
- **Phase-1 gate**: `devices.placement_override` is required.
- `lifecycle = "oneshot"` in `[[service]]` is an error.
- `lifecycle = "on_demand"` is also rejected in phase 1 with a clear "deferred to phase 2" message.
- Unknown architecture is not validated here (no GGUF reader yet).

- [ ] **Step 1: Write the failing test**

Create `src/config/validate.rs`:

```rust
//! Validate a post-merge `RawConfig`, producing an `EffectiveConfig` of
//! per-service validated configs plus daemon-global settings.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use smol_str::SmolStr;

use crate::config::parse::{RawConfig, RawService};
use crate::errors::ExpectedError;

#[derive(Debug, Clone)]
pub struct EffectiveConfig {
    pub daemon: DaemonSettings,
    pub services: Vec<ServiceConfig>,
}

#[derive(Debug, Clone)]
pub struct DaemonSettings {
    pub management_listen: String,
    pub data_dir: PathBuf,
    pub shutdown_timeout_ms: u64,
}

#[derive(Debug, Clone)]
pub struct ServiceConfig {
    pub name: SmolStr,
    pub template: Template,
    pub port: u16,
    pub private_port: u16,
    pub lifecycle: Lifecycle,
    pub priority: u8,
    pub health: HealthSettings,
    pub placement_override: BTreeMap<DeviceSlot, u64>,
    pub placement_policy: PlacementPolicy,
    pub idle_timeout_ms: u64,
    pub warming_grace_ms: u64,
    pub drain_timeout_ms: u64,
    pub extended_stream_drain_ms: u64,
    pub max_request_duration_ms: u64,
    pub raw: RawService,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Template {
    LlamaCpp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lifecycle {
    Persistent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementPolicy {
    GpuOnly,
    CpuOnly,
    Hybrid,
}

#[derive(Debug, Clone)]
pub struct HealthSettings {
    pub http_path: String,
    pub timeout_ms: u64,
    pub probe_interval_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeviceSlot {
    Cpu,
    Gpu(u32),
}

pub fn validate(cfg: &RawConfig) -> Result<EffectiveConfig, ExpectedError> {
    unimplemented!("validate")
}

fn parse_duration_ms(s: &str) -> Result<u64, String> {
    // Accepts "10m", "30s", "500ms". Returns milliseconds.
    let s = s.trim();
    if let Some(rest) = s.strip_suffix("ms") {
        return rest.parse::<u64>().map_err(|e| e.to_string());
    }
    if let Some(rest) = s.strip_suffix('s') {
        return rest.parse::<u64>().map(|n| n * 1000).map_err(|e| e.to_string());
    }
    if let Some(rest) = s.strip_suffix('m') {
        return rest.parse::<u64>().map(|n| n * 60_000).map_err(|e| e.to_string());
    }
    if let Some(rest) = s.strip_suffix('h') {
        return rest.parse::<u64>().map(|n| n * 3_600_000).map_err(|e| e.to_string());
    }
    Err(format!("unrecognised duration: {s}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::merge::resolve_inheritance;
    use crate::config::parse::parse_toml;
    use std::path::Path;

    fn parse_and_merge(src: &str) -> RawConfig {
        let mut cfg = parse_toml(src, Path::new("/t")).unwrap();
        resolve_inheritance(&mut cfg).unwrap();
        cfg
    }

    const GOOD: &str = r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 8192
flash_attn = true
cache_type_k = "q8_0"
cache_type_v = "q8_0"
devices.placement = "gpu-only"
devices.placement_override = { "gpu:0" = 18944 }
lifecycle = "persistent"
"#;

    #[test]
    fn validates_good() {
        let cfg = parse_and_merge(GOOD);
        let ec = validate(&cfg).unwrap();
        assert_eq!(ec.services.len(), 1);
        assert_eq!(ec.services[0].name, "demo");
        assert_eq!(ec.services[0].port, 11435);
        assert!(ec.services[0].private_port != 11435);
        assert_eq!(ec.services[0].placement_override[&DeviceSlot::Gpu(0)], 18944);
    }

    #[test]
    fn rejects_missing_placement_override() {
        let cfg = parse_and_merge(r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
devices.placement = "gpu-only"
lifecycle = "persistent"
"#);
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("placement_override"));
    }

    #[test]
    fn rejects_duplicate_port() {
        let cfg = parse_and_merge(r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }

[[service]]
name = "b"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }
"#);
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("duplicate") && format!("{err}").contains("port"));
    }

    #[test]
    fn rejects_quantised_kv_without_flash_attn() {
        let cfg = parse_and_merge(r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
flash_attn = false
cache_type_k = "q8_0"
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }
"#);
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("flash_attn"));
    }

    #[test]
    fn rejects_cpu_only_with_ngl_nonzero() {
        let cfg = parse_and_merge(r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
n_gpu_layers = 10
devices.placement = "cpu-only"
devices.placement_override = { "cpu" = 1000 }
lifecycle = "persistent"
"#);
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("cpu-only"));
    }

    #[test]
    fn rejects_oneshot_lifecycle_in_service_block() {
        let cfg = parse_and_merge(r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "oneshot"
devices.placement_override = { "gpu:0" = 1000 }
"#);
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("oneshot"));
    }

    #[test]
    fn phase1_rejects_on_demand_with_clear_message() {
        let cfg = parse_and_merge(r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "on_demand"
devices.placement_override = { "gpu:0" = 1000 }
"#);
        let err = validate(&cfg).unwrap_err();
        assert!(format!("{err}").contains("on_demand"));
        assert!(format!("{err}").contains("phase"));
    }

    #[test]
    fn duration_parser() {
        assert_eq!(parse_duration_ms("500ms").unwrap(), 500);
        assert_eq!(parse_duration_ms("30s").unwrap(), 30_000);
        assert_eq!(parse_duration_ms("10m").unwrap(), 600_000);
        assert_eq!(parse_duration_ms("2h").unwrap(), 7_200_000);
        assert!(parse_duration_ms("bogus").is_err());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib config::validate`
Expected: FAIL — `unimplemented!`.

- [ ] **Step 3: Implement `validate`**

Replace the `unimplemented!` body in `src/config/validate.rs`:

```rust
pub fn validate(cfg: &RawConfig) -> Result<EffectiveConfig, ExpectedError> {
    let data_dir = cfg.daemon.data_dir.clone().unwrap_or_else(|| {
        std::env::var("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()))
                    .join(".local")
                    .join("share")
            })
            .join("ananke")
    });

    let shutdown_timeout_ms = parse_duration_ms(&cfg.daemon.shutdown_timeout)
        .map_err(|e| fail(format!("daemon.shutdown_timeout: {e}")))?;

    let management_addr = cfg.daemon.management_listen.clone();
    let management_port = management_addr
        .rsplit(':')
        .next()
        .and_then(|p| p.parse::<u16>().ok());

    let mut names: BTreeSet<SmolStr> = BTreeSet::new();
    let mut ports: BTreeSet<u16> = BTreeSet::new();
    let mut out = Vec::new();

    for (i, raw) in cfg.services.iter().enumerate() {
        let name = raw.name.clone().ok_or_else(|| fail(format!("service[{i}] missing name")))?;
        let port = raw.port.ok_or_else(|| fail(format!("service {name} missing port")))?;
        let template_str = raw.template.clone().ok_or_else(|| fail(format!("service {name} missing template")))?;
        let template = match template_str.as_str() {
            "llama-cpp" => Template::LlamaCpp,
            "command" => return Err(fail(format!("service {name}: template `command` is deferred to phase 4"))),
            other => return Err(fail(format!("service {name}: unknown template `{other}`"))),
        };

        if !names.insert(name.clone()) {
            return Err(fail(format!("duplicate service name `{name}`")));
        }
        if !ports.insert(port) {
            return Err(fail(format!("duplicate service port {port}")));
        }
        if Some(port) == management_port {
            return Err(fail(format!("service {name} port {port} collides with daemon.management_listen")));
        }

        let lifecycle_str = raw.lifecycle.clone().unwrap_or_else(|| SmolStr::new("on_demand"));
        let lifecycle = match lifecycle_str.as_str() {
            "persistent" => Lifecycle::Persistent,
            "on_demand" => return Err(fail(format!("service {name}: lifecycle `on_demand` is deferred to phase 2"))),
            "oneshot" => return Err(fail(format!("service {name}: lifecycle `oneshot` is invalid in a [[service]] block (API-only)"))),
            other => return Err(fail(format!("service {name}: unknown lifecycle `{other}`"))),
        };

        // Template: llama-cpp specific requirements.
        match template {
            Template::LlamaCpp => {
                if raw.model.is_none() {
                    return Err(fail(format!("service {name}: template llama-cpp requires `model`")));
                }
                let flash = raw.flash_attn.unwrap_or(false);
                for (key, val) in [("cache_type_k", raw.cache_type_k.as_deref()), ("cache_type_v", raw.cache_type_v.as_deref())] {
                    if let Some(v) = val {
                        if v != "f16" && !flash {
                            return Err(fail(format!(
                                "service {name}: {key}={v} requires flash_attn=true (llama.cpp requires FA for quantised KV)"
                            )));
                        }
                    }
                }
            }
        }

        let dev = raw.devices.clone().unwrap_or_default();
        let placement_policy = match dev.placement.as_deref().unwrap_or("gpu-only") {
            "gpu-only" => PlacementPolicy::GpuOnly,
            "cpu-only" => {
                if raw.n_gpu_layers.unwrap_or(0) != 0 {
                    return Err(fail(format!("service {name}: devices.placement=cpu-only with n_gpu_layers={} is invalid", raw.n_gpu_layers.unwrap())));
                }
                PlacementPolicy::CpuOnly
            }
            "hybrid" => PlacementPolicy::Hybrid,
            other => return Err(fail(format!("service {name}: unknown placement `{other}`"))),
        };

        let raw_override = dev.placement_override.clone().ok_or_else(|| fail(format!(
            "service {name}: devices.placement_override is required in phase 1 (estimator lands in phase 3)"
        )))?;
        if raw_override.is_empty() {
            return Err(fail(format!("service {name}: devices.placement_override is empty")));
        }
        let mut placement_override = BTreeMap::new();
        for (k, v) in raw_override {
            let slot = match k.as_str() {
                "cpu" => DeviceSlot::Cpu,
                s if s.starts_with("gpu:") => {
                    let n: u32 = s[4..].parse().map_err(|_| fail(format!("service {name}: invalid placement_override key `{s}`")))?;
                    DeviceSlot::Gpu(n)
                }
                other => return Err(fail(format!("service {name}: invalid placement_override key `{other}`"))),
            };
            if v == 0 {
                return Err(fail(format!("service {name}: placement_override for {k} is zero")));
            }
            placement_override.insert(slot, v);
        }

        // Placement/override consistency check.
        if placement_policy == PlacementPolicy::GpuOnly && placement_override.contains_key(&DeviceSlot::Cpu) {
            return Err(fail(format!("service {name}: placement=gpu-only but placement_override includes cpu")));
        }

        let health_raw = raw.health.clone().unwrap_or_default();
        let health = HealthSettings {
            http_path: health_raw.http.unwrap_or_else(|| "/v1/models".into()),
            timeout_ms: health_raw.timeout.map(|s| parse_duration_ms(&s).map_err(|e| fail(format!("service {name} health.timeout: {e}")))).transpose()?.unwrap_or(180_000),
            probe_interval_ms: health_raw.probe_interval.map(|s| parse_duration_ms(&s).map_err(|e| fail(format!("service {name} health.probe_interval: {e}")))).transpose()?.unwrap_or(5_000),
        };

        let priority = raw.priority.or(cfg.defaults.priority).unwrap_or(50);
        let idle_timeout_ms = raw.idle_timeout.as_deref().or(cfg.defaults.idle_timeout.as_deref()).map(parse_duration_ms).transpose().map_err(|e| fail(format!("service {name} idle_timeout: {e}")))?.unwrap_or(600_000);
        let warming_grace_ms = raw.warming_grace.as_deref().or(cfg.defaults.warming_grace.as_deref()).map(parse_duration_ms).transpose().map_err(|e| fail(format!("service {name} warming_grace: {e}")))?.unwrap_or(60_000);
        let drain_timeout_ms = raw.drain_timeout.as_deref().map(parse_duration_ms).transpose().map_err(|e| fail(format!("service {name} drain_timeout: {e}")))?.unwrap_or(30_000);
        let extended_stream_drain_ms = raw.extended_stream_drain.as_deref().map(parse_duration_ms).transpose().map_err(|e| fail(format!("service {name} extended_stream_drain: {e}")))?.unwrap_or(30_000);
        let max_request_duration_ms = raw.max_request_duration.as_deref().map(parse_duration_ms).transpose().map_err(|e| fail(format!("service {name} max_request_duration: {e}")))?.unwrap_or(600_000);

        // Allocate a private loopback port deterministically based on the external port plus a large offset
        // so two services with adjacent external ports don't collide on private ports.
        let private_port = 40_000u16.saturating_add(port.wrapping_sub(11_000));

        out.push(ServiceConfig {
            name,
            template,
            port,
            private_port,
            lifecycle,
            priority,
            health,
            placement_override,
            placement_policy,
            idle_timeout_ms,
            warming_grace_ms,
            drain_timeout_ms,
            extended_stream_drain_ms,
            max_request_duration_ms,
            raw: raw.clone(),
        });
    }

    Ok(EffectiveConfig {
        daemon: DaemonSettings {
            management_listen: management_addr,
            data_dir,
            shutdown_timeout_ms,
        },
        services: out,
    })
}

fn fail(msg: String) -> ExpectedError {
    ExpectedError::config_unparseable(PathBuf::from("<config>"), msg)
}
```

Update `src/config/mod.rs`:

```rust
pub mod file;
pub mod merge;
pub mod parse;
pub mod validate;

pub use file::{resolve_config_path, resolve_from_env, PathSources};
pub use merge::{resolve_inheritance, resolve_migrations, Migration};
pub use parse::{parse_toml, RawConfig, RawService};
pub use validate::{
    validate, DaemonSettings, DeviceSlot, EffectiveConfig, HealthSettings, Lifecycle,
    PlacementPolicy, ServiceConfig, Template,
};
```

Also add a top-level `load_config` helper that wires parse → merge → migrate → validate. Add to the end of `src/config/mod.rs`:

```rust
use std::path::Path;
use crate::errors::ExpectedError;

pub fn load_config(path: &Path) -> Result<(EffectiveConfig, Vec<Migration>), ExpectedError> {
    let source = std::fs::read_to_string(path)
        .map_err(|_| ExpectedError::config_file_missing(path.to_path_buf()))?;
    load_config_from_str(&source, path)
}

pub fn load_config_from_str(source: &str, origin: &Path) -> Result<(EffectiveConfig, Vec<Migration>), ExpectedError> {
    let mut raw = parse_toml(source, origin)?;
    resolve_inheritance(&mut raw)?;
    let migrations = resolve_migrations(&mut raw)?;
    let effective = validate(&raw)?;
    Ok((effective, migrations))
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib config`
Expected: PASS — full config module tests green.

- [ ] **Step 5: Commit**

```bash
git add src/config/
git commit -m "feat(config): validate into EffectiveConfig with phase-1 placement_override gate"
```

---

## Task 7: Database bootstrap and schema

**Files:**
- Create: `src/db/mod.rs` (replace stub)
- Create: `src/db/schema.rs`

Toasty may or may not support the needed SQLite pragma timing. This plan uses `rusqlite` directly for phase 1 — toasty integration is an option for phase 5 when the management API needs structured queries. Using `rusqlite` keeps phase 1 unblocked regardless of toasty's state.

- [ ] **Step 1: Write the failing test**

Create `src/db/schema.rs`:

```rust
//! SQLite schema migrations.

pub const MIGRATION_0001: &str = r#"
PRAGMA auto_vacuum = INCREMENTAL;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
  version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS services (
  service_id INTEGER PRIMARY KEY,
  name       TEXT NOT NULL UNIQUE,
  created_at INTEGER NOT NULL,
  deleted_at INTEGER
);

CREATE TABLE IF NOT EXISTS service_config_versions (
  service_id       INTEGER NOT NULL,
  version          INTEGER NOT NULL,
  effective_config TEXT NOT NULL,
  recorded_at      INTEGER NOT NULL,
  PRIMARY KEY (service_id, version),
  FOREIGN KEY (service_id) REFERENCES services(service_id)
);

CREATE TABLE IF NOT EXISTS running_services (
  service_id   INTEGER NOT NULL,
  run_id       INTEGER NOT NULL,
  pid          INTEGER NOT NULL,
  spawned_at   INTEGER NOT NULL,
  command_line TEXT NOT NULL,
  allocation   TEXT NOT NULL,
  state        TEXT NOT NULL,
  PRIMARY KEY (service_id, run_id),
  FOREIGN KEY (service_id) REFERENCES services(service_id)
);

CREATE TABLE IF NOT EXISTS service_logs (
  service_id   INTEGER NOT NULL,
  run_id       INTEGER NOT NULL,
  timestamp_ms INTEGER NOT NULL,
  seq          INTEGER NOT NULL,
  stream       TEXT NOT NULL,
  line         TEXT NOT NULL,
  PRIMARY KEY (service_id, run_id, seq),
  FOREIGN KEY (service_id) REFERENCES services(service_id)
);
CREATE INDEX IF NOT EXISTS service_logs_ts ON service_logs(service_id, run_id, timestamp_ms);

CREATE TABLE IF NOT EXISTS allocation_events (
  event_id   INTEGER PRIMARY KEY,
  service_id INTEGER NOT NULL,
  run_id     INTEGER NOT NULL,
  event_type TEXT NOT NULL,
  device     TEXT NOT NULL,
  bytes      INTEGER NOT NULL,
  at         INTEGER NOT NULL,
  FOREIGN KEY (service_id) REFERENCES services(service_id)
);

CREATE TABLE IF NOT EXISTS oneshots (
  id           TEXT PRIMARY KEY,
  service_id   INTEGER NOT NULL,
  submitted_at INTEGER NOT NULL,
  started_at   INTEGER,
  ended_at     INTEGER,
  exit_code    INTEGER,
  ttl_ms       INTEGER NOT NULL,
  FOREIGN KEY (service_id) REFERENCES services(service_id)
);

INSERT OR IGNORE INTO schema_version(version) VALUES (1);
"#;
```

Create `src/db/mod.rs`:

```rust
//! Database bootstrap and migrations.

pub mod logs;
pub mod schema;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;
use rusqlite::Connection;

use crate::errors::ExpectedError;

#[derive(Clone)]
pub struct Database {
    conn: Arc<Mutex<Connection>>,
    path: PathBuf,
}

impl Database {
    pub fn open(path: &Path) -> Result<Self, ExpectedError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| ExpectedError::database_open_failed(path.to_path_buf(), e.to_string()))?;
        }
        let conn = Connection::open(path).map_err(|e| ExpectedError::database_open_failed(path.to_path_buf(), e.to_string()))?;
        conn.execute_batch(schema::MIGRATION_0001).map_err(|e| ExpectedError::database_open_failed(path.to_path_buf(), e.to_string()))?;
        Ok(Self { conn: Arc::new(Mutex::new(conn)), path: path.to_path_buf() })
    }

    pub fn schema_version(&self) -> Result<u32, ExpectedError> {
        let conn = self.conn.lock();
        let v: u32 = conn
            .query_row("SELECT COALESCE(MAX(version), 0) FROM schema_version", [], |row| row.get(0))
            .map_err(|e| ExpectedError::database_open_failed(self.path.clone(), e.to_string()))?;
        Ok(v)
    }

    pub fn upsert_service(&self, name: &str, now_ms: i64) -> Result<i64, ExpectedError> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO services(name, created_at, deleted_at) VALUES (?1, ?2, NULL)
             ON CONFLICT(name) DO UPDATE SET deleted_at = NULL",
            (name, now_ms),
        ).map_err(|e| ExpectedError::database_open_failed(self.path.clone(), e.to_string()))?;
        let id: i64 = conn.query_row("SELECT service_id FROM services WHERE name = ?1", [name], |row| row.get(0))
            .map_err(|e| ExpectedError::database_open_failed(self.path.clone(), e.to_string()))?;
        Ok(id)
    }

    /// Reparent a `service_id` from `old_name` to `new_name`. Spec §6.4.
    /// - If old_name has a live service_id and new_name does not, rename old to new.
    /// - If both exist, tombstone old (set deleted_at) — the new service keeps its own id.
    ///   This matches "migrate_from naming a service that also exists live in the same config → error",
    ///   which is enforced in validation; here we handle the post-load case where old was tombstoned previously.
    /// - If old_name doesn't exist, no-op.
    pub fn reparent(&self, old_name: &str, new_name: &str, now_ms: i64) -> Result<(), ExpectedError> {
        let conn = self.conn.lock();
        let old_id: Option<i64> = conn.query_row(
            "SELECT service_id FROM services WHERE name = ?1", [old_name],
            |row| row.get(0)
        ).ok();
        let new_id: Option<i64> = conn.query_row(
            "SELECT service_id FROM services WHERE name = ?1", [new_name],
            |row| row.get(0)
        ).ok();
        match (old_id, new_id) {
            (Some(_), None) => {
                conn.execute(
                    "UPDATE services SET name = ?1 WHERE name = ?2",
                    (new_name, old_name),
                ).map_err(|e| ExpectedError::database_open_failed(self.path.clone(), e.to_string()))?;
            }
            (Some(_), Some(_)) => {
                conn.execute(
                    "UPDATE services SET deleted_at = ?1 WHERE name = ?2",
                    (now_ms, old_name),
                ).map_err(|e| ExpectedError::database_open_failed(self.path.clone(), e.to_string()))?;
            }
            (None, _) => {
                // No-op; see spec §6.4 "missing source is a warning".
            }
        }
        Ok(())
    }

    pub fn path(&self) -> &Path { &self.path }

    pub fn with_conn<T>(&self, f: impl FnOnce(&Connection) -> rusqlite::Result<T>) -> rusqlite::Result<T> {
        let conn = self.conn.lock();
        f(&conn)
    }

    pub fn with_conn_mut<T>(&self, f: impl FnOnce(&mut Connection) -> rusqlite::Result<T>) -> rusqlite::Result<T> {
        let mut conn = self.conn.lock();
        f(&mut conn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn opens_and_migrates() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("ananke.sqlite")).unwrap();
        assert_eq!(db.schema_version().unwrap(), 1);
    }

    #[test]
    fn upsert_service_is_idempotent() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("a.sqlite")).unwrap();
        let id1 = db.upsert_service("demo", 1000).unwrap();
        let id2 = db.upsert_service("demo", 2000).unwrap();
        assert_eq!(id1, id2);
    }

    #[test]
    fn reparent_renames_when_only_old_exists() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("a.sqlite")).unwrap();
        let _ = db.upsert_service("old-name", 1000).unwrap();
        db.reparent("old-name", "new-name", 2000).unwrap();
        // new-name should now resolve; old-name should not.
        let new_id: i64 = db.with_conn(|c| c.query_row("SELECT service_id FROM services WHERE name = 'new-name'", [], |r| r.get(0))).unwrap();
        let old_query: rusqlite::Result<i64> = db.with_conn(|c| c.query_row("SELECT service_id FROM services WHERE name = 'old-name'", [], |r| r.get(0)));
        assert!(new_id > 0);
        assert!(old_query.is_err());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib db`
Expected: FAIL — compile errors because `src/db/logs.rs` is not yet created. Create a stub:

```rust
// src/db/logs.rs
//! Log batching writer. Implemented in Task 8.
```

Re-run: `cargo test --lib db`
Expected: PASS — 3 tests green.

- [ ] **Step 3: Commit**

```bash
git add src/db/
git commit -m "feat(db): open SQLite with migration 0001 and upsert/reparent helpers"
```

---

## Task 8: Log batcher

**Files:**
- Replace: `src/db/logs.rs`

Batches log lines into SQLite with 200 ms / 100-line flush thresholds per spec §12.

- [ ] **Step 1: Write the failing test**

Replace `src/db/logs.rs`:

```rust
//! Log batching writer.
//!
//! Contract: `Batcher::push` is fire-and-forget. A single writer task owns
//! the SQLite connection and commits every 200 ms or every 100 lines, whichever
//! first. A shutdown signal triggers a final flush.

use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot};
use tracing::warn;

use crate::db::Database;

const BATCH_LINES: usize = 100;
const BATCH_INTERVAL: Duration = Duration::from_millis(200);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    Stdout,
    Stderr,
}

impl Stream {
    fn as_str(self) -> &'static str {
        match self { Stream::Stdout => "stdout", Stream::Stderr => "stderr" }
    }
}

#[derive(Debug)]
pub struct LogLine {
    pub service_id: i64,
    pub run_id: i64,
    pub timestamp_ms: i64,
    pub stream: Stream,
    pub line: String,
}

pub struct BatcherHandle {
    tx: mpsc::UnboundedSender<Msg>,
}

enum Msg {
    Line(LogLine),
    Flush(oneshot::Sender<()>),
}

impl BatcherHandle {
    pub fn push(&self, line: LogLine) {
        // Fire-and-forget; silently drop if the writer has exited.
        let _ = self.tx.send(Msg::Line(line));
    }

    pub async fn flush(&self) {
        let (tx, rx) = oneshot::channel();
        if self.tx.send(Msg::Flush(tx)).is_err() {
            return;
        }
        let _ = rx.await;
    }
}

pub fn spawn(db: Database) -> BatcherHandle {
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(run(db, rx));
    BatcherHandle { tx }
}

async fn run(db: Database, mut rx: mpsc::UnboundedReceiver<Msg>) {
    let mut buffer: Vec<LogLine> = Vec::with_capacity(BATCH_LINES);
    let mut seq_counters: std::collections::HashMap<(i64, i64), i64> = std::collections::HashMap::new();
    let mut deadline = Instant::now() + BATCH_INTERVAL;

    loop {
        let tick = tokio::time::sleep_until(deadline.into());
        tokio::pin!(tick);

        tokio::select! {
            msg = rx.recv() => match msg {
                None => {
                    flush(&db, &mut buffer, &mut seq_counters);
                    return;
                }
                Some(Msg::Line(line)) => {
                    buffer.push(line);
                    if buffer.len() >= BATCH_LINES {
                        flush(&db, &mut buffer, &mut seq_counters);
                        deadline = Instant::now() + BATCH_INTERVAL;
                    }
                }
                Some(Msg::Flush(ack)) => {
                    flush(&db, &mut buffer, &mut seq_counters);
                    let _ = ack.send(());
                    deadline = Instant::now() + BATCH_INTERVAL;
                }
            },
            _ = &mut tick => {
                if !buffer.is_empty() {
                    flush(&db, &mut buffer, &mut seq_counters);
                }
                deadline = Instant::now() + BATCH_INTERVAL;
            }
        }
    }
}

fn flush(db: &Database, buffer: &mut Vec<LogLine>, seq: &mut std::collections::HashMap<(i64, i64), i64>) {
    if buffer.is_empty() { return; }
    let lines = std::mem::take(buffer);
    let res = db.with_conn_mut(|conn| {
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT INTO service_logs(service_id, run_id, timestamp_ms, seq, stream, line) VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
            )?;
            for line in &lines {
                let counter = seq.entry((line.service_id, line.run_id)).or_insert(0);
                *counter += 1;
                stmt.execute((line.service_id, line.run_id, line.timestamp_ms, *counter, line.stream.as_str(), &line.line))?;
            }
        }
        tx.commit()?;
        Ok(())
    });
    if let Err(e) = res {
        warn!(error = %e, "log batch flush failed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Database;
    use std::time::Duration;
    use tempfile::tempdir;

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn flushes_on_threshold() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("a.sqlite")).unwrap();
        let svc = db.upsert_service("demo", 0).unwrap();
        let h = spawn(db.clone());

        for i in 0..BATCH_LINES as i64 {
            h.push(LogLine {
                service_id: svc,
                run_id: 1,
                timestamp_ms: i,
                stream: Stream::Stdout,
                line: format!("line {i}"),
            });
        }
        h.flush().await;

        let count: i64 = db.with_conn(|c| c.query_row("SELECT COUNT(*) FROM service_logs", [], |r| r.get(0))).unwrap();
        assert_eq!(count, BATCH_LINES as i64);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn flushes_on_interval() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("b.sqlite")).unwrap();
        let svc = db.upsert_service("demo", 0).unwrap();
        let h = spawn(db.clone());

        h.push(LogLine {
            service_id: svc, run_id: 1, timestamp_ms: 0,
            stream: Stream::Stdout, line: "first".into(),
        });
        tokio::time::sleep(Duration::from_millis(250)).await;
        h.flush().await;

        let count: i64 = db.with_conn(|c| c.query_row("SELECT COUNT(*) FROM service_logs", [], |r| r.get(0))).unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn seq_is_per_service_run_monotonic() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("c.sqlite")).unwrap();
        let svc = db.upsert_service("demo", 0).unwrap();
        let h = spawn(db.clone());

        for i in 0..3 {
            h.push(LogLine {
                service_id: svc, run_id: 1, timestamp_ms: i,
                stream: Stream::Stdout, line: format!("{i}"),
            });
        }
        h.flush().await;

        let seqs: Vec<i64> = db.with_conn(|c| {
            let mut stmt = c.prepare("SELECT seq FROM service_logs ORDER BY timestamp_ms").unwrap();
            let rows = stmt.query_map([], |r| r.get::<_, i64>(0)).unwrap();
            Ok(rows.collect::<Result<Vec<_>, _>>().unwrap())
        }).unwrap();
        assert_eq!(seqs, vec![1, 2, 3]);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib db::logs`
Expected: FAIL initially — imports may not resolve. Fix any compile errors, then re-run.

- [ ] **Step 3: Verify passing**

Run: `cargo test --lib db::logs`
Expected: PASS — 3 tests green.

- [ ] **Step 4: Commit**

```bash
git add src/db/logs.rs
git commit -m "feat(db): add 200ms/100-line log batcher"
```

---

## Task 9: Device core types

**Files:**
- Replace: `src/devices/mod.rs`

- [ ] **Step 1: Write the failing test**

Replace `src/devices/mod.rs`:

```rust
//! Device types and allocation primitives.

pub mod cpu;
pub mod cuda_env;
pub mod fake;
pub mod nvml;
pub mod probe;

use std::collections::BTreeMap;

use crate::config::validate::DeviceSlot;

pub use probe::{GpuInfo, GpuMemory, GpuProbe, GpuProcess};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Device {
    pub id: DeviceId,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DeviceId {
    Cpu,
    Gpu(u32),
}

impl DeviceId {
    pub fn to_slot(self) -> DeviceSlot {
        match self {
            DeviceId::Cpu => DeviceSlot::Cpu,
            DeviceId::Gpu(n) => DeviceSlot::Gpu(n),
        }
    }

    pub fn from_slot(slot: &DeviceSlot) -> Self {
        match slot {
            DeviceSlot::Cpu => DeviceId::Cpu,
            DeviceSlot::Gpu(n) => DeviceId::Gpu(*n),
        }
    }

    pub fn as_display(self) -> String {
        match self {
            DeviceId::Cpu => "cpu".into(),
            DeviceId::Gpu(n) => format!("gpu:{n}"),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Allocation {
    pub bytes: BTreeMap<DeviceId, u64>,
}

impl Allocation {
    pub fn from_override(map: &BTreeMap<DeviceSlot, u64>) -> Self {
        let mut bytes = BTreeMap::new();
        for (slot, b) in map {
            bytes.insert(DeviceId::from_slot(slot), b * 1024 * 1024); // MB → bytes
        }
        Self { bytes }
    }

    pub fn gpu_ids(&self) -> Vec<u32> {
        self.bytes.keys().filter_map(|d| if let DeviceId::Gpu(n) = d { Some(*n) } else { None }).collect()
    }

    pub fn total(&self) -> u64 {
        self.bytes.values().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::validate::DeviceSlot;

    #[test]
    fn allocation_from_override_converts_mb_to_bytes() {
        let mut m = BTreeMap::new();
        m.insert(DeviceSlot::Gpu(0), 1024);
        m.insert(DeviceSlot::Cpu, 2048);
        let a = Allocation::from_override(&m);
        assert_eq!(a.bytes[&DeviceId::Gpu(0)], 1024 * 1024 * 1024);
        assert_eq!(a.bytes[&DeviceId::Cpu], 2048 * 1024 * 1024);
    }

    #[test]
    fn gpu_ids_filters_cpu() {
        let mut m = BTreeMap::new();
        m.insert(DeviceSlot::Gpu(0), 10);
        m.insert(DeviceSlot::Gpu(1), 20);
        m.insert(DeviceSlot::Cpu, 30);
        let a = Allocation::from_override(&m);
        let mut ids = a.gpu_ids();
        ids.sort();
        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn device_id_display() {
        assert_eq!(DeviceId::Cpu.as_display(), "cpu");
        assert_eq!(DeviceId::Gpu(3).as_display(), "gpu:3");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib devices::tests`
Expected: FAIL — compile errors (probe/nvml/fake/cpu/cuda_env stubs missing).

Create empty stubs:

```rust
// src/devices/probe.rs
pub trait GpuProbe: Send + Sync {
    fn list(&self) -> Vec<GpuInfo>;
    fn query(&self, id: u32) -> Option<GpuMemory>;
    fn processes(&self, id: u32) -> Vec<GpuProcess>;
}

#[derive(Debug, Clone)]
pub struct GpuInfo { pub id: u32, pub name: String, pub total_bytes: u64 }

#[derive(Debug, Clone)]
pub struct GpuMemory { pub total_bytes: u64, pub free_bytes: u64 }

#[derive(Debug, Clone)]
pub struct GpuProcess { pub pid: u32, pub used_bytes: u64, pub name: String }
```

```rust
// src/devices/nvml.rs
//! NVML-backed probe. Implemented in Task 11.
```

```rust
// src/devices/fake.rs
//! In-memory fake probe. Implemented in Task 10.
```

```rust
// src/devices/cpu.rs
//! CPU memory via /proc/meminfo. Implemented in Task 12.
```

```rust
// src/devices/cuda_env.rs
//! CUDA_VISIBLE_DEVICES rendering. Implemented in Task 13.
```

- [ ] **Step 3: Run test to verify it passes**

Run: `cargo test --lib devices::tests`
Expected: PASS — 3 tests green.

- [ ] **Step 4: Commit**

```bash
git add src/devices/
git commit -m "feat(devices): add Device/DeviceId/Allocation core types"
```

---

## Task 10: GpuProbe trait + FakeProbe

**Files:**
- Modify: `src/devices/probe.rs`
- Modify: `src/devices/fake.rs`
- Modify: `src/devices/mod.rs`

- [ ] **Step 1: Write the failing test**

Replace `src/devices/fake.rs`:

```rust
//! In-memory fake `GpuProbe` for tests.

use std::sync::Arc;

use parking_lot::Mutex;

use super::probe::{GpuInfo, GpuMemory, GpuProbe, GpuProcess};

#[derive(Debug, Clone)]
pub struct FakeGpu {
    pub info: GpuInfo,
    pub free_bytes: u64,
    pub processes: Vec<GpuProcess>,
}

#[derive(Default, Clone)]
pub struct FakeProbe {
    inner: Arc<Mutex<Vec<FakeGpu>>>,
}

impl FakeProbe {
    pub fn new(gpus: Vec<FakeGpu>) -> Self {
        Self { inner: Arc::new(Mutex::new(gpus)) }
    }

    pub fn set_free(&self, id: u32, free_bytes: u64) {
        let mut g = self.inner.lock();
        if let Some(gpu) = g.iter_mut().find(|g| g.info.id == id) {
            gpu.free_bytes = free_bytes;
        }
    }

    pub fn add_process(&self, id: u32, proc_info: GpuProcess) {
        let mut g = self.inner.lock();
        if let Some(gpu) = g.iter_mut().find(|g| g.info.id == id) {
            gpu.processes.push(proc_info);
        }
    }
}

impl GpuProbe for FakeProbe {
    fn list(&self) -> Vec<GpuInfo> {
        self.inner.lock().iter().map(|g| g.info.clone()).collect()
    }

    fn query(&self, id: u32) -> Option<GpuMemory> {
        self.inner.lock().iter().find(|g| g.info.id == id).map(|g| GpuMemory {
            total_bytes: g.info.total_bytes,
            free_bytes: g.free_bytes,
        })
    }

    fn processes(&self, id: u32) -> Vec<GpuProcess> {
        self.inner.lock().iter().find(|g| g.info.id == id).map(|g| g.processes.clone()).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> FakeProbe {
        FakeProbe::new(vec![
            FakeGpu {
                info: GpuInfo { id: 0, name: "RTX 4090".into(), total_bytes: 24 * 1024 * 1024 * 1024 },
                free_bytes: 20 * 1024 * 1024 * 1024,
                processes: Vec::new(),
            },
            FakeGpu {
                info: GpuInfo { id: 1, name: "RTX 4090".into(), total_bytes: 24 * 1024 * 1024 * 1024 },
                free_bytes: 22 * 1024 * 1024 * 1024,
                processes: Vec::new(),
            },
        ])
    }

    #[test]
    fn lists_all() {
        let p = fixture();
        assert_eq!(p.list().len(), 2);
    }

    #[test]
    fn query_returns_free_after_set() {
        let p = fixture();
        p.set_free(0, 1024);
        assert_eq!(p.query(0).unwrap().free_bytes, 1024);
    }

    #[test]
    fn processes_round_trip() {
        let p = fixture();
        p.add_process(0, GpuProcess { pid: 1234, used_bytes: 100, name: "test".into() });
        assert_eq!(p.processes(0).len(), 1);
        assert_eq!(p.processes(0)[0].pid, 1234);
    }
}
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cargo test --lib devices::fake`
Expected: PASS — 3 tests green.

- [ ] **Step 3: Commit**

```bash
git add src/devices/
git commit -m "feat(devices): FakeProbe for GpuProbe testing"
```

---

## Task 11: NVML-backed GpuProbe

**Files:**
- Replace: `src/devices/nvml.rs`

- [ ] **Step 1: Write the implementation**

Replace `src/devices/nvml.rs`:

```rust
//! NVML-backed [`GpuProbe`] impl.

use std::sync::Arc;

use nvml_wrapper::Nvml;
use parking_lot::Mutex;
use tracing::warn;

use super::probe::{GpuInfo, GpuMemory, GpuProbe, GpuProcess};

pub struct NvmlProbe {
    nvml: Arc<Nvml>,
    cache: Mutex<Vec<GpuInfo>>,
}

impl NvmlProbe {
    pub fn init() -> Result<Self, String> {
        // Unset CUDA_VISIBLE_DEVICES so NVML sees every GPU regardless of ambient env (spec §4.3).
        unsafe { std::env::remove_var("CUDA_VISIBLE_DEVICES"); }

        let nvml = Nvml::init().map_err(|e| format!("NVML init failed: {e}"))?;
        let count = nvml.device_count().map_err(|e| format!("NVML device_count failed: {e}"))?;
        let mut infos = Vec::with_capacity(count as usize);
        for i in 0..count {
            let dev = nvml.device_by_index(i).map_err(|e| format!("NVML device_by_index({i}) failed: {e}"))?;
            let name = dev.name().unwrap_or_else(|_| format!("GPU {i}"));
            let total = dev.memory_info().map(|m| m.total).unwrap_or(0);
            infos.push(GpuInfo { id: i, name, total_bytes: total });
        }
        Ok(Self { nvml: Arc::new(nvml), cache: Mutex::new(infos) })
    }
}

impl GpuProbe for NvmlProbe {
    fn list(&self) -> Vec<GpuInfo> {
        self.cache.lock().clone()
    }

    fn query(&self, id: u32) -> Option<GpuMemory> {
        match self.nvml.device_by_index(id) {
            Ok(dev) => match dev.memory_info() {
                Ok(m) => Some(GpuMemory { total_bytes: m.total, free_bytes: m.free }),
                Err(e) => { warn!(gpu = id, error = %e, "NVML memory_info failed"); None }
            },
            Err(e) => { warn!(gpu = id, error = %e, "NVML device_by_index failed"); None }
        }
    }

    fn processes(&self, id: u32) -> Vec<GpuProcess> {
        let Ok(dev) = self.nvml.device_by_index(id) else { return Vec::new(); };
        dev.running_compute_processes()
            .map(|procs| procs.into_iter().map(|p| {
                let used = match p.used_gpu_memory {
                    nvml_wrapper::enums::device::UsedGpuMemory::Used(b) => b,
                    nvml_wrapper::enums::device::UsedGpuMemory::Unavailable => 0,
                };
                let name = std::fs::read_to_string(format!("/proc/{}/comm", p.pid))
                    .map(|s| s.trim().to_string())
                    .unwrap_or_else(|_| format!("pid {}", p.pid));
                GpuProcess { pid: p.pid, used_bytes: used, name }
            }).collect())
            .unwrap_or_default()
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check --lib`
Expected: PASS. (Runtime tests for NVML are integration-level and require a GPU; covered manually.)

- [ ] **Step 3: Commit**

```bash
git add src/devices/nvml.rs
git commit -m "feat(devices): NVML-backed GpuProbe"
```

---

## Task 12: CPU memory probe

**Files:**
- Replace: `src/devices/cpu.rs`

- [ ] **Step 1: Write the failing test**

Replace `src/devices/cpu.rs`:

```rust
//! Linux /proc/meminfo reader.
//!
//! `MemAvailable` is used (spec §4.2), not `MemFree`, because `MemFree` ignores
//! reclaimable page cache and misleads the scheduler about how much memory can
//! actually be allocated to a new process.

use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct CpuMemory {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

pub fn read() -> std::io::Result<CpuMemory> {
    read_from(Path::new("/proc/meminfo"))
}

pub fn read_from(path: &Path) -> std::io::Result<CpuMemory> {
    let content = std::fs::read_to_string(path)?;
    parse_meminfo(&content).ok_or_else(|| std::io::Error::other("meminfo missing MemTotal or MemAvailable"))
}

pub(crate) fn parse_meminfo(content: &str) -> Option<CpuMemory> {
    let mut total_kb = None;
    let mut avail_kb = None;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total_kb = parse_kb(rest);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            avail_kb = parse_kb(rest);
        }
    }
    Some(CpuMemory {
        total_bytes: total_kb? * 1024,
        available_bytes: avail_kb? * 1024,
    })
}

fn parse_kb(rest: &str) -> Option<u64> {
    let trimmed = rest.trim().trim_end_matches("kB").trim();
    trimmed.parse::<u64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
MemTotal:       98765432 kB
MemFree:        12345678 kB
MemAvailable:   87654321 kB
Buffers:        1000000 kB
";

    #[test]
    fn parses_meminfo_sample() {
        let m = parse_meminfo(SAMPLE).unwrap();
        assert_eq!(m.total_bytes, 98_765_432 * 1024);
        assert_eq!(m.available_bytes, 87_654_321 * 1024);
    }

    #[test]
    fn returns_none_when_missing() {
        assert!(parse_meminfo("MemFree: 100 kB").is_none());
    }
}
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cargo test --lib devices::cpu`
Expected: PASS — 2 tests green.

- [ ] **Step 3: Commit**

```bash
git add src/devices/cpu.rs
git commit -m "feat(devices): CPU memory via /proc/meminfo MemAvailable"
```

---

## Task 13: `CUDA_VISIBLE_DEVICES` rendering

**Files:**
- Replace: `src/devices/cuda_env.rs`

Spec §4.3: each child is spawned with a freshly computed `CUDA_VISIBLE_DEVICES` built from its allocation; renumbered from 0; `cpu-only` gets `""`.

- [ ] **Step 1: Write the failing test**

Replace `src/devices/cuda_env.rs`:

```rust
//! Render `CUDA_VISIBLE_DEVICES` from an `Allocation`.

use super::Allocation;

/// Return the value to set for `CUDA_VISIBLE_DEVICES` given the service's
/// allocation and policy.
///
/// - If the allocation has no GPU entries (CPU-only), returns `Some("")` so the
///   child cannot grab a GPU.
/// - Otherwise, returns the NVML indices comma-separated in ascending order
///   (e.g. `"0,2"`).
pub fn render(allocation: &Allocation) -> String {
    let mut ids = allocation.gpu_ids();
    ids.sort();
    ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::validate::DeviceSlot;
    use std::collections::BTreeMap;

    fn alloc(pairs: &[(DeviceSlot, u64)]) -> Allocation {
        let mut m = BTreeMap::new();
        for (k, v) in pairs { m.insert(k.clone(), *v); }
        Allocation::from_override(&m)
    }

    #[test]
    fn cpu_only_is_empty() {
        let a = alloc(&[(DeviceSlot::Cpu, 1000)]);
        assert_eq!(render(&a), "");
    }

    #[test]
    fn single_gpu() {
        let a = alloc(&[(DeviceSlot::Gpu(1), 1000)]);
        assert_eq!(render(&a), "1");
    }

    #[test]
    fn multi_gpu_sorted() {
        let a = alloc(&[(DeviceSlot::Gpu(3), 1), (DeviceSlot::Gpu(0), 1)]);
        assert_eq!(render(&a), "0,3");
    }

    #[test]
    fn hybrid_includes_only_gpus() {
        let a = alloc(&[(DeviceSlot::Gpu(0), 1), (DeviceSlot::Cpu, 1)]);
        assert_eq!(render(&a), "0");
    }
}
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cargo test --lib devices::cuda_env`
Expected: PASS — 4 tests green.

- [ ] **Step 3: Commit**

```bash
git add src/devices/cuda_env.rs
git commit -m "feat(devices): render CUDA_VISIBLE_DEVICES from Allocation"
```

---

## Task 14: State machine

**Files:**
- Replace: `src/state.rs`

- [ ] **Step 1: Write the failing test**

Replace `src/state.rs`:

```rust
//! Service state machine per spec §5.3.

use smol_str::SmolStr;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceState {
    Starting,
    Warming,
    Running,
    Draining,
    Idle,
    Stopped,
    Evicted,
    Failed { retry_count: u8 },
    Disabled { reason: DisableReason },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DisableReason {
    ConfigError(SmolStr),
    LaunchFailed,
    HealthTimeout,
    Oom,
    CrashLoop,
    NoFit,
    UserDisabled,
}

impl DisableReason {
    pub fn as_str(&self) -> &str {
        match self {
            DisableReason::ConfigError(_) => "config_error",
            DisableReason::LaunchFailed => "launch_failed",
            DisableReason::HealthTimeout => "health_timeout",
            DisableReason::Oom => "oom",
            DisableReason::CrashLoop => "crash_loop",
            DisableReason::NoFit => "no_fit",
            DisableReason::UserDisabled => "user_disabled",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    SpawnRequested,
    HealthPassed,
    WarmingComplete,
    DrainRequested,
    DrainComplete,
    Stopped,
    LaunchFailed,
    HealthTimedOut,
    CrashLoop,
    UserEnable,
    UserDisable,
    RetryAfterBackoff,
}

/// Returns the next state if the transition is valid, else `None`.
pub fn transition(from: &ServiceState, event: Event) -> Option<ServiceState> {
    use ServiceState::*;
    match (from, event) {
        (Idle, Event::SpawnRequested) => Some(Starting),
        (Starting, Event::HealthPassed) => Some(Warming),
        (Starting, Event::LaunchFailed) => Some(Failed { retry_count: 0 }),
        (Warming, Event::WarmingComplete) => Some(Running),
        (Warming, Event::HealthTimedOut) => Some(Disabled { reason: DisableReason::HealthTimeout }),
        (Running, Event::DrainRequested) => Some(Draining),
        (Running, Event::Stopped) => Some(Stopped),
        (Draining, Event::DrainComplete) => Some(Idle),
        (Draining, Event::Stopped) => Some(Stopped),
        (Stopped, Event::SpawnRequested) => Some(Starting),
        (Failed { retry_count }, Event::RetryAfterBackoff) => {
            if *retry_count >= 2 {
                Some(Disabled { reason: DisableReason::LaunchFailed })
            } else {
                Some(Failed { retry_count: retry_count + 1 })
            }
        }
        (Running | Warming, Event::CrashLoop) => Some(Disabled { reason: DisableReason::CrashLoop }),
        (Disabled { .. }, Event::UserEnable) => Some(Idle),
        (_, Event::UserDisable) => Some(Disabled { reason: DisableReason::UserDisabled }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_to_starting_on_spawn() {
        assert!(matches!(transition(&ServiceState::Idle, Event::SpawnRequested), Some(ServiceState::Starting)));
    }

    #[test]
    fn starting_to_warming_on_health() {
        assert!(matches!(transition(&ServiceState::Starting, Event::HealthPassed), Some(ServiceState::Warming)));
    }

    #[test]
    fn warming_health_timeout_disables_with_reason() {
        let s = transition(&ServiceState::Warming, Event::HealthTimedOut).unwrap();
        assert!(matches!(s, ServiceState::Disabled { reason: DisableReason::HealthTimeout }));
    }

    #[test]
    fn failed_retries_up_to_three_times_then_disables() {
        let s0 = ServiceState::Failed { retry_count: 0 };
        let s1 = transition(&s0, Event::RetryAfterBackoff).unwrap();
        assert_eq!(s1, ServiceState::Failed { retry_count: 1 });
        let s2 = transition(&s1, Event::RetryAfterBackoff).unwrap();
        assert_eq!(s2, ServiceState::Failed { retry_count: 2 });
        let s3 = transition(&s2, Event::RetryAfterBackoff).unwrap();
        assert!(matches!(s3, ServiceState::Disabled { reason: DisableReason::LaunchFailed }));
    }

    #[test]
    fn invalid_transition_returns_none() {
        assert!(transition(&ServiceState::Idle, Event::DrainComplete).is_none());
    }

    #[test]
    fn disabled_can_be_re_enabled() {
        let s = ServiceState::Disabled { reason: DisableReason::HealthTimeout };
        assert_eq!(transition(&s, Event::UserEnable), Some(ServiceState::Idle));
    }
}
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cargo test --lib state`
Expected: PASS — 6 tests green.

- [ ] **Step 3: Commit**

```bash
git add src/state.rs
git commit -m "feat(state): service state machine with valid transitions"
```

---

## Task 15: Child spawning — argv rendering and `prctl`

**Files:**
- Create: `src/supervise/mod.rs`
- Create: `src/supervise/spawn.rs`

- [ ] **Step 1: Write the failing test for argv rendering**

Create `src/supervise/mod.rs`:

```rust
//! Service supervision: per-service tokio tasks, child lifetimes, health loops.

pub mod health;
pub mod logs;
pub mod orphans;
pub mod spawn;

pub use spawn::{render_argv, SpawnConfig};
```

Create `src/supervise/spawn.rs`:

```rust
//! Render llama-server argv from an `EffectiveConfig` service entry, and spawn
//! the child with `prctl(PR_SET_PDEATHSIG, SIGTERM)`.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::os::unix::process::CommandExt;

use nix::sys::prctl;
use nix::sys::signal::Signal;
use tokio::process::{Child, Command};

use crate::config::validate::{PlacementPolicy, ServiceConfig, Template};
use crate::devices::{cuda_env, Allocation};
use crate::errors::ExpectedError;

pub struct SpawnConfig {
    pub binary: String,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
}

/// Render the child command line plus env from a validated `ServiceConfig`
/// and its `Allocation`.
pub fn render_argv(svc: &ServiceConfig, alloc: &Allocation) -> SpawnConfig {
    let mut args: Vec<String> = Vec::new();

    match svc.template {
        Template::LlamaCpp => {
            let raw = &svc.raw;
            args.push("-m".into());
            args.push(raw.model.as_ref().unwrap().to_string_lossy().into_owned());
            if let Some(mmproj) = &raw.mmproj {
                args.push("--mmproj".into());
                args.push(mmproj.to_string_lossy().into_owned());
            }
            if let Some(ctx) = raw.context {
                args.push("-c".into());
                args.push(ctx.to_string());
            }
            match svc.placement_policy {
                PlacementPolicy::CpuOnly => {
                    args.push("-ngl".into()); args.push("0".into());
                }
                PlacementPolicy::GpuOnly | PlacementPolicy::Hybrid => {
                    if let Some(ngl) = raw.n_gpu_layers {
                        args.push("-ngl".into());
                        args.push(ngl.to_string());
                    } else {
                        args.push("-ngl".into());
                        args.push("999".into());
                    }
                }
            }
            if raw.flash_attn == Some(true) { args.push("-fa".into()); args.push("on".into()); }
            if let Some(k) = &raw.cache_type_k { args.push("--cache-type-k".into()); args.push(k.to_string()); }
            if let Some(v) = &raw.cache_type_v { args.push("--cache-type-v".into()); args.push(v.to_string()); }
            if raw.jinja.unwrap_or(false) { args.push("--jinja".into()); }
            if let Some(p) = &raw.chat_template_file {
                args.push("--chat-template-file".into());
                args.push(p.to_string_lossy().into_owned());
            }
            if let Some(t) = raw.threads { args.push("--threads".into()); args.push(t.to_string()); }
            if let Some(t) = raw.threads_batch { args.push("--threads-batch".into()); args.push(t.to_string()); }
            if let Some(b) = raw.batch_size { args.push("-b".into()); args.push(b.to_string()); }
            if let Some(b) = raw.ubatch_size { args.push("-ub".into()); args.push(b.to_string()); }
            if raw.mmap == Some(false) { args.push("--no-mmap".into()); }
            if raw.mlock == Some(true) { args.push("--mlock".into()); }
            if let Some(p) = raw.parallel { args.push("-np".into()); args.push(p.to_string()); }
            if let Some(rules) = &raw.override_tensor {
                for rule in rules {
                    args.push("-ot".into());
                    args.push(rule.clone());
                }
            }
            // Sampling as extra args if set.
            if let Some(s) = &raw.sampling {
                if let Some(t) = s.get("temperature") { args.push("--temp".into()); args.push(t.to_string()); }
                if let Some(p) = s.get("top_p") { args.push("--top-p".into()); args.push(p.to_string()); }
                if let Some(k) = s.get("top_k") { args.push("--top-k".into()); args.push(k.to_string()); }
                if let Some(m) = s.get("min_p") { args.push("--min-p".into()); args.push(m.to_string()); }
                if let Some(r) = s.get("repeat_penalty") { args.push("--repeat-penalty".into()); args.push(r.to_string()); }
            }
            if let Some(extra) = &raw.extra_args {
                args.extend(extra.iter().cloned());
            }
            if let Some(extra) = &raw.extra_args_append {
                args.extend(extra.iter().cloned());
            }
            args.push("--host".into());
            args.push("127.0.0.1".into());
            args.push("--port".into());
            args.push(svc.private_port.to_string());
        }
    }

    let mut env = BTreeMap::new();
    if let Some(user_env) = &svc.raw.env {
        for (k, v) in user_env { env.insert(k.clone(), v.clone()); }
    }
    env.insert("CUDA_VISIBLE_DEVICES".into(), cuda_env::render(alloc));

    SpawnConfig {
        binary: "llama-server".into(),
        args,
        env,
    }
}

pub async fn spawn_child(cfg: &SpawnConfig) -> Result<Child, ExpectedError> {
    let mut cmd = Command::new(&cfg.binary);
    cmd.args(cfg.args.iter().map(OsString::from));
    cmd.env_clear();
    for (k, v) in &cfg.env { cmd.env(k, v); }
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    cmd.kill_on_drop(true);
    unsafe {
        cmd.pre_exec(|| {
            prctl::set_pdeathsig(Signal::SIGTERM).map_err(std::io::Error::other)?;
            Ok(())
        });
    }
    cmd.spawn().map_err(|e| ExpectedError::config_unparseable(
        std::path::PathBuf::from("<spawn>"),
        format!("spawn {}: {e}", cfg.binary),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::validate::{
        DaemonSettings, DeviceSlot, EffectiveConfig, HealthSettings, Lifecycle,
        PlacementPolicy, ServiceConfig, Template,
    };
    use crate::config::parse::RawService;
    use smol_str::SmolStr;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn base_service() -> ServiceConfig {
        let mut placement = BTreeMap::new();
        placement.insert(DeviceSlot::Gpu(0), 10240);
        let raw = RawService {
            name: Some(SmolStr::new("demo")),
            template: Some(SmolStr::new("llama-cpp")),
            model: Some(PathBuf::from("/m/x.gguf")),
            port: Some(11435),
            context: Some(8192),
            flash_attn: Some(true),
            cache_type_k: Some(SmolStr::new("q8_0")),
            cache_type_v: Some(SmolStr::new("q8_0")),
            ..Default::default()
        };
        ServiceConfig {
            name: SmolStr::new("demo"),
            template: Template::LlamaCpp,
            port: 11435,
            private_port: 41000,
            lifecycle: Lifecycle::Persistent,
            priority: 50,
            health: HealthSettings { http_path: "/v1/models".into(), timeout_ms: 180_000, probe_interval_ms: 5_000 },
            placement_override: placement,
            placement_policy: PlacementPolicy::GpuOnly,
            idle_timeout_ms: 600_000,
            warming_grace_ms: 60_000,
            drain_timeout_ms: 30_000,
            extended_stream_drain_ms: 30_000,
            max_request_duration_ms: 600_000,
            raw,
        }
    }

    #[test]
    fn renders_core_flags() {
        let svc = base_service();
        let alloc = Allocation::from_override(&svc.placement_override);
        let cmd = render_argv(&svc, &alloc);
        assert_eq!(cmd.binary, "llama-server");
        assert!(cmd.args.contains(&"-m".to_string()));
        assert!(cmd.args.iter().any(|a| a == "/m/x.gguf"));
        assert!(cmd.args.iter().any(|a| a == "-c"));
        assert!(cmd.args.iter().any(|a| a == "8192"));
        assert!(cmd.args.iter().any(|a| a == "-fa"));
        assert!(cmd.args.iter().any(|a| a == "--port"));
        assert!(cmd.args.iter().any(|a| a == "41000"));
        assert_eq!(cmd.env.get("CUDA_VISIBLE_DEVICES").unwrap(), "0");
    }

    #[test]
    fn renders_mmproj_when_present() {
        let mut svc = base_service();
        svc.raw.mmproj = Some(PathBuf::from("/m/x-mmproj.gguf"));
        let alloc = Allocation::from_override(&svc.placement_override);
        let cmd = render_argv(&svc, &alloc);
        let idx = cmd.args.iter().position(|a| a == "--mmproj").unwrap();
        assert_eq!(cmd.args[idx + 1], "/m/x-mmproj.gguf");
    }

    #[test]
    fn cpu_only_renders_ngl_zero_and_empty_cuda_env() {
        let mut svc = base_service();
        svc.placement_policy = PlacementPolicy::CpuOnly;
        svc.placement_override.clear();
        svc.placement_override.insert(DeviceSlot::Cpu, 10240);
        let alloc = Allocation::from_override(&svc.placement_override);
        let cmd = render_argv(&svc, &alloc);
        let ngl_idx = cmd.args.iter().position(|a| a == "-ngl").unwrap();
        assert_eq!(cmd.args[ngl_idx + 1], "0");
        assert_eq!(cmd.env.get("CUDA_VISIBLE_DEVICES").unwrap(), "");
    }
}
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cargo test --lib supervise::spawn`
Expected: PASS — 3 tests green.

- [ ] **Step 3: Commit**

```bash
git add src/supervise/
git commit -m "feat(supervise): render llama-server argv and spawn with PR_SET_PDEATHSIG"
```

---

## Task 16: Log pumps

**Files:**
- Create: `src/supervise/logs.rs`

- [ ] **Step 1: Write the failing test**

Replace `src/supervise/logs.rs`:

```rust
//! Pump child stdout/stderr into the log batcher.

use std::time::{SystemTime, UNIX_EPOCH};

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{ChildStderr, ChildStdout};

use crate::db::logs::{BatcherHandle, LogLine, Stream};

pub fn spawn_pump_stdout(
    stdout: ChildStdout,
    service_id: i64,
    run_id: i64,
    batcher: BatcherHandle,
) {
    tokio::spawn(pump(BufReader::new(stdout), service_id, run_id, Stream::Stdout, batcher));
}

pub fn spawn_pump_stderr(
    stderr: ChildStderr,
    service_id: i64,
    run_id: i64,
    batcher: BatcherHandle,
) {
    tokio::spawn(pump(BufReader::new(stderr), service_id, run_id, Stream::Stderr, batcher));
}

async fn pump<R: AsyncBufReadExt + Unpin>(
    mut reader: R,
    service_id: i64,
    run_id: i64,
    stream: Stream,
    batcher: BatcherHandle,
) {
    let mut buf = String::new();
    loop {
        buf.clear();
        match reader.read_line(&mut buf).await {
            Ok(0) => return,
            Ok(_) => {
                let line = buf.trim_end_matches(['\n', '\r']).to_string();
                let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as i64;
                batcher.push(LogLine { service_id, run_id, timestamp_ms: ts, stream, line });
            }
            Err(_) => return,
        }
    }
}
```

(Note: `BatcherHandle::push` does not take `&mut self`, so a new pump can freely hold the handle by value. For two pumps per child, clone the handle.)

Make `BatcherHandle` clonable — add `#[derive(Clone)]` to its struct in `src/db/logs.rs` and clone `tx` via `Clone` (already clone-able as an `UnboundedSender`).

- [ ] **Step 2: Modify `BatcherHandle` to be `Clone`**

In `src/db/logs.rs`, change:

```rust
pub struct BatcherHandle {
    tx: mpsc::UnboundedSender<Msg>,
}
```

to:

```rust
#[derive(Clone)]
pub struct BatcherHandle {
    tx: mpsc::UnboundedSender<Msg>,
}
```

- [ ] **Step 3: Add integration-style test**

Append to `src/supervise/logs.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::logs::spawn as spawn_batcher;
    use crate::db::Database;
    use tempfile::tempdir;
    use tokio::process::Command;

    #[tokio::test(flavor = "current_thread")]
    async fn pumps_echoed_lines() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("a.sqlite")).unwrap();
        let svc = db.upsert_service("demo", 0).unwrap();
        let batcher = spawn_batcher(db.clone());

        let mut child = Command::new("/bin/sh")
            .arg("-c")
            .arg("printf 'hello\\nworld\\n'; exit 0")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        spawn_pump_stdout(stdout, svc, 1, batcher.clone());
        spawn_pump_stderr(stderr, svc, 1, batcher.clone());

        let _ = child.wait().await;
        // Wait a tick for the batcher to drain.
        batcher.flush().await;
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        batcher.flush().await;

        let lines: Vec<String> = db.with_conn(|c| {
            let mut stmt = c.prepare("SELECT line FROM service_logs WHERE service_id = ?1 ORDER BY seq").unwrap();
            let rows = stmt.query_map([svc], |r| r.get::<_, String>(0)).unwrap();
            Ok(rows.collect::<Result<Vec<_>, _>>().unwrap())
        }).unwrap();
        assert_eq!(lines, vec!["hello".to_string(), "world".to_string()]);
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib supervise::logs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/
git commit -m "feat(supervise): pump child stdout/stderr into log batcher"
```

---

## Task 17: Health check loop

**Files:**
- Create: `src/supervise/health.rs`

- [ ] **Step 1: Write the failing test**

Replace `src/supervise/health.rs`:

```rust
//! HTTP health-probe loop.

use std::time::Duration;

use tokio::sync::watch;
use tracing::{debug, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthOutcome {
    Healthy,
    TimedOut,
    Cancelled,
}

pub struct HealthConfig {
    pub url: String,
    pub probe_interval: Duration,
    pub timeout: Duration,
}

/// Runs until one of: health passes → `Healthy`; total elapsed exceeds
/// `timeout` → `TimedOut`; `cancel` resolves to true → `Cancelled`.
pub async fn wait_healthy(
    cfg: HealthConfig,
    mut cancel: watch::Receiver<bool>,
) -> HealthOutcome {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .expect("reqwest client build");
    let start = tokio::time::Instant::now();

    loop {
        if *cancel.borrow() { return HealthOutcome::Cancelled; }
        if start.elapsed() >= cfg.timeout {
            return HealthOutcome::TimedOut;
        }

        match client.get(&cfg.url).send().await {
            Ok(resp) if resp.status().is_success() => return HealthOutcome::Healthy,
            Ok(resp) => debug!(status = %resp.status(), url = %cfg.url, "health probe non-2xx"),
            Err(e) => debug!(error = %e, url = %cfg.url, "health probe errored"),
        }

        tokio::select! {
            _ = tokio::time::sleep(cfg.probe_interval) => {}
            _ = cancel.changed() => {
                if *cancel.borrow() { return HealthOutcome::Cancelled; }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyper::service::service_fn;
    use hyper::{Request, Response};
    use hyper_util::rt::TokioIo;
    use hyper_util::server::conn::auto;
    use http_body_util::Full;
    use bytes::Bytes;
    use std::convert::Infallible;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use tokio::net::TcpListener;

    async fn spawn_server(status: u16) -> (String, Arc<AtomicU32>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let count = Arc::new(AtomicU32::new(0));
        let count_clone = count.clone();
        tokio::spawn(async move {
            loop {
                let (stream, _) = listener.accept().await.unwrap();
                let io = TokioIo::new(stream);
                let count = count_clone.clone();
                tokio::spawn(async move {
                    let svc = service_fn(move |_req: Request<hyper::body::Incoming>| {
                        let count = count.clone();
                        async move {
                            count.fetch_add(1, Ordering::Relaxed);
                            let resp = Response::builder()
                                .status(status)
                                .body(Full::new(Bytes::from("ok"))).unwrap();
                            Ok::<_, Infallible>(resp)
                        }
                    });
                    let _ = auto::Builder::new(hyper_util::rt::TokioExecutor::new())
                        .serve_connection(io, svc).await;
                });
            }
        });
        (format!("http://{addr}/health"), count)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn returns_healthy_on_2xx() {
        let (url, _) = spawn_server(200).await;
        let (tx, rx) = watch::channel(false);
        let outcome = wait_healthy(HealthConfig {
            url,
            probe_interval: Duration::from_millis(50),
            timeout: Duration::from_secs(5),
        }, rx).await;
        drop(tx);
        assert_eq!(outcome, HealthOutcome::Healthy);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn times_out_on_always_500() {
        let (url, _) = spawn_server(500).await;
        let (_, rx) = watch::channel(false);
        let outcome = wait_healthy(HealthConfig {
            url,
            probe_interval: Duration::from_millis(50),
            timeout: Duration::from_millis(300),
        }, rx).await;
        assert_eq!(outcome, HealthOutcome::TimedOut);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cancelled_when_signalled() {
        let (url, _) = spawn_server(500).await;
        let (tx, rx) = watch::channel(false);
        let task = tokio::spawn(wait_healthy(HealthConfig {
            url,
            probe_interval: Duration::from_millis(500),
            timeout: Duration::from_secs(10),
        }, rx));
        tokio::time::sleep(Duration::from_millis(100)).await;
        tx.send(true).unwrap();
        let outcome = task.await.unwrap();
        assert_eq!(outcome, HealthOutcome::Cancelled);
    }
}
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cargo test --lib supervise::health`
Expected: PASS — 3 tests green.

- [ ] **Step 3: Commit**

```bash
git add src/supervise/health.rs
git commit -m "feat(supervise): HTTP health probe loop with timeout and cancellation"
```

---

## Task 18: Orphan recovery

**Files:**
- Create: `src/supervise/orphans.rs`

- [ ] **Step 1: Write the failing test**

Replace `src/supervise/orphans.rs`:

```rust
//! Startup orphan recovery per spec §9.3.

use std::path::{Path, PathBuf};

use tracing::{info, warn};

use crate::db::Database;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrphanDisposition {
    Adopted { pid: i32, service_id: i64, run_id: i64 },
    Cleaned { pid: i32, service_id: i64, run_id: i64 },
    Ignored { pid: i32, reason: String },
}

/// Runs orphan recovery against the `running_services` table. Returns a
/// decision list suitable for logging and test assertion.
///
/// `procfs_root` defaults to "/proc"; tests override it via a temp directory.
pub fn reconcile(db: &Database, procfs_root: &Path) -> Vec<OrphanDisposition> {
    let rows: Vec<(i64, i64, i64, String)> = db.with_conn(|c| {
        let mut stmt = c.prepare("SELECT service_id, run_id, pid, command_line FROM running_services").unwrap();
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))).unwrap();
        Ok(rows.collect::<Result<Vec<_>, _>>().unwrap())
    }).unwrap_or_default();

    let mut out = Vec::with_capacity(rows.len());
    for (service_id, run_id, pid_i64, recorded_cmdline) in rows {
        let pid = pid_i64 as i32;
        let proc_dir = procfs_root.join(pid.to_string());
        let cmdline_path = proc_dir.join("cmdline");
        match std::fs::read(&cmdline_path) {
            Ok(raw) => {
                let live_cmdline = null_sep_to_space(&raw);
                if live_cmdline == recorded_cmdline {
                    info!(pid, service_id, run_id, "adopted orphan");
                    out.push(OrphanDisposition::Adopted { pid, service_id, run_id });
                } else {
                    warn!(pid, service_id, run_id,
                          recorded = %recorded_cmdline, live = %live_cmdline,
                          "unrelated process at recorded pid; cleaning row");
                    cleanup_row(db, service_id, run_id);
                    out.push(OrphanDisposition::Cleaned { pid, service_id, run_id });
                }
            }
            Err(_) => {
                info!(pid, service_id, run_id, "dead child; cleaning row");
                cleanup_row(db, service_id, run_id);
                out.push(OrphanDisposition::Cleaned { pid, service_id, run_id });
            }
        }
    }
    out
}

fn null_sep_to_space(bytes: &[u8]) -> String {
    let trimmed: Vec<u8> = bytes.iter().copied().map(|b| if b == 0 { b' ' } else { b }).collect();
    String::from_utf8_lossy(&trimmed).trim().to_string()
}

fn cleanup_row(db: &Database, service_id: i64, run_id: i64) {
    let _ = db.with_conn(|c| c.execute(
        "DELETE FROM running_services WHERE service_id = ?1 AND run_id = ?2",
        (service_id, run_id),
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn insert_row(db: &Database, service_id: i64, run_id: i64, pid: i32, cmdline: &str) {
        db.with_conn(|c| c.execute(
            "INSERT INTO running_services(service_id, run_id, pid, spawned_at, command_line, allocation, state) VALUES (?1, ?2, ?3, 0, ?4, '{}', 'running')",
            (service_id, run_id, pid, cmdline),
        )).unwrap();
    }

    fn write_cmdline(procfs: &Path, pid: i32, cmdline: &str) {
        let dir = procfs.join(pid.to_string());
        std::fs::create_dir_all(&dir).unwrap();
        // /proc/PID/cmdline separates args with NUL.
        let nul_sep = cmdline.replace(' ', "\0");
        std::fs::write(dir.join("cmdline"), nul_sep).unwrap();
    }

    #[test]
    fn adopts_matching_cmdline() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("a.sqlite")).unwrap();
        let svc = db.upsert_service("demo", 0).unwrap();
        let procfs = tmp.path().join("proc");
        insert_row(&db, svc, 1, 1234, "llama-server -m x");
        write_cmdline(&procfs, 1234, "llama-server -m x");
        let out = reconcile(&db, &procfs);
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0], OrphanDisposition::Adopted { .. }));
    }

    #[test]
    fn cleans_missing_pid() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("a.sqlite")).unwrap();
        let svc = db.upsert_service("demo", 0).unwrap();
        let procfs = tmp.path().join("proc");
        std::fs::create_dir_all(&procfs).unwrap();
        insert_row(&db, svc, 1, 9999, "llama-server -m x");
        let out = reconcile(&db, &procfs);
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0], OrphanDisposition::Cleaned { .. }));
        // Row should be gone.
        let rows: Vec<(i64, i64)> = db.with_conn(|c| {
            let mut s = c.prepare("SELECT service_id, run_id FROM running_services").unwrap();
            Ok(s.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?))).unwrap().collect::<Result<Vec<_>, _>>().unwrap())
        }).unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn cleans_mismatched_cmdline() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("a.sqlite")).unwrap();
        let svc = db.upsert_service("demo", 0).unwrap();
        let procfs = tmp.path().join("proc");
        insert_row(&db, svc, 1, 4242, "llama-server -m x");
        write_cmdline(&procfs, 4242, "firefox");
        let out = reconcile(&db, &procfs);
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0], OrphanDisposition::Cleaned { .. }));
    }
}
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cargo test --lib supervise::orphans`
Expected: PASS — 3 tests green.

- [ ] **Step 3: Commit**

```bash
git add src/supervise/orphans.rs
git commit -m "feat(supervise): startup orphan recovery with cmdline cross-check"
```

---

## Task 19: Supervisor task

**Files:**
- Modify: `src/supervise/mod.rs`

- [ ] **Step 1: Write the supervisor task skeleton**

Append to `src/supervise/mod.rs`:

```rust
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex as SyncMutex;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

use crate::config::validate::ServiceConfig;
use crate::db::logs::BatcherHandle;
use crate::db::Database;
use crate::devices::Allocation;
use crate::state::{transition, DisableReason, Event as StateEvent, ServiceState};
use crate::supervise::health::{wait_healthy, HealthConfig, HealthOutcome};
use crate::supervise::logs::{spawn_pump_stderr, spawn_pump_stdout};
use crate::supervise::spawn::{render_argv, spawn_child};

#[derive(Debug)]
pub enum SupervisorCommand {
    Shutdown { ack: tokio::sync::oneshot::Sender<()> },
    /// Request state snapshot for tests / management surface.
    Snapshot { ack: tokio::sync::oneshot::Sender<SupervisorSnapshot> },
}

#[derive(Debug, Clone)]
pub struct SupervisorSnapshot {
    pub name: smol_str::SmolStr,
    pub state: ServiceState,
    pub run_id: Option<i64>,
    pub pid: Option<i32>,
}

pub struct SupervisorHandle {
    pub name: smol_str::SmolStr,
    tx: mpsc::Sender<SupervisorCommand>,
    join: JoinHandle<()>,
}

impl SupervisorHandle {
    pub async fn shutdown(self) {
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        let _ = self.tx.send(SupervisorCommand::Shutdown { ack: ack_tx }).await;
        let _ = ack_rx.await;
        let _ = self.join.await;
    }

    pub async fn snapshot(&self) -> Option<SupervisorSnapshot> {
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        if self.tx.send(SupervisorCommand::Snapshot { ack: ack_tx }).await.is_err() {
            return None;
        }
        ack_rx.await.ok()
    }
}

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
    SupervisorHandle { name, tx, join }
}

async fn run(
    svc: ServiceConfig,
    allocation: Allocation,
    db: Database,
    batcher: BatcherHandle,
    service_id: i64,
    mut rx: mpsc::Receiver<SupervisorCommand>,
) {
    let mut state = ServiceState::Idle;
    let state_mirror = Arc::new(SyncMutex::new(state.clone()));
    let (cancel_tx, cancel_rx) = watch::channel(false);

    loop {
        match &state {
            ServiceState::Idle => {
                let next = transition(&state, StateEvent::SpawnRequested).unwrap();
                state = next;
                *state_mirror.lock() = state.clone();
            }
            ServiceState::Starting => {
                let spawn_cfg = render_argv(&svc, &allocation);
                let cmdline = format!("{} {}", spawn_cfg.binary, spawn_cfg.args.join(" "));
                match spawn_child(&spawn_cfg).await {
                    Ok(mut child) => {
                        let pid = child.id().unwrap_or(0) as i32;
                        let run_id = ((std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis()) & 0x7FFFFFFF) as i64;
                        let _ = db.with_conn(|c| c.execute(
                            "INSERT INTO running_services(service_id, run_id, pid, spawned_at, command_line, allocation, state) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'starting')",
                            (service_id, run_id, pid as i64, chrono_like_now_ms(), cmdline.clone(), serde_json::to_string(&allocation.bytes.iter().map(|(k, v)| (k.as_display(), *v)).collect::<std::collections::BTreeMap<_, _>>()).unwrap_or_default()),
                        ));

                        if let Some(stdout) = child.stdout.take() {
                            spawn_pump_stdout(stdout, service_id, run_id, batcher.clone());
                        }
                        if let Some(stderr) = child.stderr.take() {
                            spawn_pump_stderr(stderr, service_id, run_id, batcher.clone());
                        }

                        let health_cfg = HealthConfig {
                            url: format!("http://127.0.0.1:{}{}", svc.private_port, svc.health.http_path),
                            probe_interval: Duration::from_millis(svc.health.probe_interval_ms),
                            timeout: Duration::from_millis(svc.health.timeout_ms),
                        };

                        let cancel_rx_h = cancel_rx.clone();
                        let health_task = tokio::spawn(wait_healthy(health_cfg, cancel_rx_h));

                        tokio::pin!(health_task);

                        loop {
                            tokio::select! {
                                exit = child.wait() => {
                                    warn!(?exit, "child exited during starting/warming");
                                    state = ServiceState::Failed { retry_count: 0 };
                                    *state_mirror.lock() = state.clone();
                                    break;
                                }
                                outcome = &mut health_task => {
                                    match outcome {
                                        Ok(HealthOutcome::Healthy) => {
                                            state = transition(&state, StateEvent::HealthPassed).unwrap();
                                            *state_mirror.lock() = state.clone();
                                            // Warming grace.
                                            let grace = Duration::from_millis(svc.warming_grace_ms);
                                            tokio::select! {
                                                _ = tokio::time::sleep(grace) => {
                                                    state = transition(&state, StateEvent::WarmingComplete).unwrap();
                                                    *state_mirror.lock() = state.clone();
                                                }
                                                _ = child.wait() => {
                                                    warn!("child exited during warming grace");
                                                    state = ServiceState::Failed { retry_count: 0 };
                                                    *state_mirror.lock() = state.clone();
                                                    break;
                                                }
                                            }

                                            // Running: wait for child exit or shutdown command.
                                            tokio::select! {
                                                exit = child.wait() => {
                                                    warn!(?exit, "child exited from running");
                                                    state = ServiceState::Failed { retry_count: 0 };
                                                    *state_mirror.lock() = state.clone();
                                                }
                                                cmd = rx.recv() => {
                                                    match cmd {
                                                        Some(SupervisorCommand::Shutdown { ack }) => {
                                                            info!(service = %svc.name, "draining");
                                                            state = transition(&state, StateEvent::DrainRequested).unwrap();
                                                            *state_mirror.lock() = state.clone();
                                                            let _ = cancel_tx.send(true);
                                                            send_sigterm_and_wait(&mut child, Duration::from_secs(10)).await;
                                                            let _ = db.with_conn(|c| c.execute(
                                                                "DELETE FROM running_services WHERE service_id = ?1 AND run_id = ?2",
                                                                (service_id, run_id),
                                                            ));
                                                            let _ = ack.send(());
                                                            return;
                                                        }
                                                        Some(SupervisorCommand::Snapshot { ack }) => {
                                                            let _ = ack.send(SupervisorSnapshot {
                                                                name: svc.name.clone(),
                                                                state: state.clone(),
                                                                run_id: Some(run_id),
                                                                pid: Some(pid),
                                                            });
                                                        }
                                                        None => return,
                                                    }
                                                }
                                            }
                                            break;
                                        }
                                        Ok(HealthOutcome::TimedOut) => {
                                            warn!(service = %svc.name, "health timed out; disabling");
                                            state = ServiceState::Disabled { reason: DisableReason::HealthTimeout };
                                            *state_mirror.lock() = state.clone();
                                            send_sigterm_and_wait(&mut child, Duration::from_secs(5)).await;
                                            break;
                                        }
                                        Ok(HealthOutcome::Cancelled) | Err(_) => {
                                            send_sigterm_and_wait(&mut child, Duration::from_secs(5)).await;
                                            return;
                                        }
                                    }
                                }
                                cmd = rx.recv() => {
                                    match cmd {
                                        Some(SupervisorCommand::Shutdown { ack }) => {
                                            let _ = cancel_tx.send(true);
                                            send_sigterm_and_wait(&mut child, Duration::from_secs(5)).await;
                                            let _ = ack.send(());
                                            return;
                                        }
                                        Some(SupervisorCommand::Snapshot { ack }) => {
                                            let _ = ack.send(SupervisorSnapshot {
                                                name: svc.name.clone(),
                                                state: state.clone(),
                                                run_id: None,
                                                pid: Some(pid),
                                            });
                                        }
                                        None => return,
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!(error = %e, "spawn failed");
                        state = ServiceState::Failed { retry_count: 0 };
                        *state_mirror.lock() = state.clone();
                    }
                }
            }
            ServiceState::Failed { retry_count } => {
                let delay = match *retry_count {
                    0 => Duration::from_secs(2),
                    1 => Duration::from_secs(5),
                    _ => Duration::from_secs(15),
                };
                tokio::select! {
                    _ = tokio::time::sleep(delay) => {
                        state = transition(&state, StateEvent::RetryAfterBackoff).unwrap_or(ServiceState::Disabled { reason: DisableReason::LaunchFailed });
                        if !matches!(state, ServiceState::Disabled { .. }) {
                            // Move back to Idle → Starting on next loop iteration.
                            state = ServiceState::Idle;
                        }
                        *state_mirror.lock() = state.clone();
                    }
                    cmd = rx.recv() => {
                        if let Some(SupervisorCommand::Shutdown { ack }) = cmd {
                            let _ = ack.send(());
                            return;
                        }
                    }
                }
            }
            ServiceState::Disabled { .. } => {
                info!(service = %svc.name, "disabled; awaiting shutdown or enable");
                if let Some(SupervisorCommand::Shutdown { ack }) = rx.recv().await {
                    let _ = ack.send(());
                    return;
                }
            }
            _ => {
                warn!(?state, "unexpected state in supervisor loop");
                return;
            }
        }
    }
}

async fn send_sigterm_and_wait(child: &mut tokio::process::Child, grace: Duration) {
    if let Some(pid) = child.id() {
        let _ = nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid as i32), nix::sys::signal::Signal::SIGTERM);
    }
    match tokio::time::timeout(grace, child.wait()).await {
        Ok(_) => {}
        Err(_) => { let _ = child.kill().await; }
    }
}

fn chrono_like_now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as i64
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check --lib`
Expected: PASS. (Supervisor integration is tested in Task 25's end-to-end tests.)

- [ ] **Step 3: Commit**

```bash
git add src/supervise/mod.rs
git commit -m "feat(supervise): per-service supervisor task with health and drain"
```

---

## Task 20: Reverse proxy with SSE passthrough

**Files:**
- Replace: `src/proxy.rs`

- [ ] **Step 1: Write the failing test**

Replace `src/proxy.rs`:

```rust
//! Per-service reverse HTTP proxy.

use std::net::SocketAddr;

use bytes::Bytes;
use futures::TryStreamExt;
use http_body_util::{BodyExt, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto;
use tokio::net::TcpListener;
use tokio::sync::watch;
use tracing::{error, info, warn};

use crate::errors::ExpectedError;

type ProxyBody = http_body_util::combinators::BoxBody<Bytes, Box<dyn std::error::Error + Send + Sync>>;

pub async fn serve(
    listen: SocketAddr,
    upstream_port: u16,
    mut shutdown: watch::Receiver<bool>,
) -> Result<(), ExpectedError> {
    let listener = TcpListener::bind(listen).await
        .map_err(|e| ExpectedError::bind_failed(listen.to_string(), e.to_string()))?;
    info!(%listen, upstream_port, "proxy listening");

    let client = Client::builder(TokioExecutor::new())
        .build_http::<ProxyBody>();

    loop {
        tokio::select! {
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!(%listen, "proxy shutting down");
                    return Ok(());
                }
            }
            accept = listener.accept() => {
                let (stream, peer) = match accept {
                    Ok(x) => x,
                    Err(e) => { warn!(error = %e, "accept failed"); continue; }
                };
                let io = TokioIo::new(stream);
                let client = client.clone();
                tokio::spawn(async move {
                    let svc = service_fn(move |req: Request<Incoming>| {
                        let client = client.clone();
                        async move { handle(req, client, upstream_port, peer).await }
                    });
                    if let Err(e) = auto::Builder::new(TokioExecutor::new())
                        .serve_connection(io, svc).await
                    {
                        warn!(error = %e, "conn error");
                    }
                });
            }
        }
    }
}

async fn handle(
    req: Request<Incoming>,
    client: Client<hyper_util::client::legacy::connect::HttpConnector, ProxyBody>,
    upstream_port: u16,
    peer: SocketAddr,
) -> Result<Response<ProxyBody>, Box<dyn std::error::Error + Send + Sync>> {
    let (parts, body) = req.into_parts();
    let path_and_query = parts.uri.path_and_query().map(|p| p.as_str()).unwrap_or("/");
    let uri = format!("http://127.0.0.1:{upstream_port}{path_and_query}")
        .parse::<hyper::Uri>()?;

    let mut upstream_req = Request::builder()
        .method(parts.method.clone())
        .uri(uri);
    for (k, v) in parts.headers.iter() {
        if k == hyper::header::HOST { continue; }
        upstream_req = upstream_req.header(k, v);
    }
    let body_bytes = body.collect().await?.to_bytes();
    let upstream_body: ProxyBody = http_body_util::Full::new(body_bytes)
        .map_err(|never| match never {})
        .boxed();
    let upstream_req = upstream_req.body(upstream_body)?;

    let resp = match client.request(upstream_req).await {
        Ok(r) => r,
        Err(e) => {
            warn!(error = %e, peer = %peer, "upstream request failed");
            let body = http_body_util::Full::new(Bytes::from("upstream unavailable"))
                .map_err(|never| match never {})
                .boxed();
            return Ok(Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(body)?);
        }
    };

    let (parts, body) = resp.into_parts();
    // Stream the body back without buffering — critical for SSE.
    let stream = body.into_data_stream().map_ok(Frame::data);
    let boxed: ProxyBody = StreamBody::new(stream)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })
        .boxed();
    let mut out = Response::from_parts(parts, boxed);
    out.headers_mut().remove(hyper::header::CONNECTION);
    out.headers_mut().remove("transfer-encoding");
    Ok(out)
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check --lib`
Expected: PASS. (Integration test for SSE passthrough lives in `tests/sse_passthrough.rs` — Task 25.)

- [ ] **Step 3: Commit**

```bash
git add src/proxy.rs
git commit -m "feat(proxy): hyper reverse proxy with SSE-safe body streaming"
```

---

## Task 21: Signals and shutdown

**Files:**
- Replace: `src/signals.rs`

- [ ] **Step 1: Write the implementation**

Replace `src/signals.rs`:

```rust
//! Signal handling: SIGTERM/SIGINT → graceful drain, SIGQUIT → emergency.

use std::time::Duration;

use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::watch;
use tracing::info;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownKind {
    Graceful,
    Emergency,
}

/// Blocks until a shutdown signal arrives.
pub async fn await_shutdown() -> ShutdownKind {
    let mut term = signal(SignalKind::terminate()).expect("SIGTERM handler");
    let mut int  = signal(SignalKind::interrupt()).expect("SIGINT handler");
    let mut quit = signal(SignalKind::quit()).expect("SIGQUIT handler");
    tokio::select! {
        _ = term.recv() => { info!("SIGTERM received"); ShutdownKind::Graceful }
        _ = int.recv() => { info!("SIGINT received"); ShutdownKind::Graceful }
        _ = quit.recv() => { info!("SIGQUIT received"); ShutdownKind::Emergency }
    }
}

pub fn grace_for(kind: ShutdownKind) -> Duration {
    match kind {
        ShutdownKind::Graceful => Duration::from_secs(10),
        ShutdownKind::Emergency => Duration::from_secs(5),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grace_is_shorter_for_emergency() {
        assert!(grace_for(ShutdownKind::Emergency) < grace_for(ShutdownKind::Graceful));
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test --lib signals`
Expected: PASS — 1 test green.

- [ ] **Step 3: Commit**

```bash
git add src/signals.rs
git commit -m "feat(signals): SIGTERM/SIGINT/SIGQUIT handling with shutdown kind"
```

---

## Task 22: Retention (trim + incremental vacuum)

**Files:**
- Replace: `src/retention.rs`

- [ ] **Step 1: Write the failing test**

Replace `src/retention.rs`:

```rust
//! Log retention and SQLite incremental vacuum.

use std::time::Duration;

use tokio::sync::watch;
use tracing::{info, warn};

use crate::db::Database;

/// Per-service log retention: 7 days or 50,000 lines, whichever tighter
/// (spec §12). Runs once when called; call from a daily scheduled task.
pub fn trim_logs_once(db: &Database, now_ms: i64) -> rusqlite::Result<u64> {
    let seven_days_ago = now_ms - 7 * 24 * 60 * 60 * 1000;
    let per_service = 50_000i64;

    let mut deleted = 0u64;
    db.with_conn_mut(|conn| {
        let tx = conn.transaction()?;
        // 1. Drop rows older than seven days.
        let n = tx.execute(
            "DELETE FROM service_logs WHERE timestamp_ms < ?1",
            [seven_days_ago],
        )?;
        deleted += n as u64;

        // 2. Per-service row count cap.
        let service_ids: Vec<i64> = {
            let mut stmt = tx.prepare("SELECT service_id FROM services WHERE deleted_at IS NULL")?;
            stmt.query_map([], |row| row.get(0))?.collect::<Result<_, _>>()?
        };
        for sid in service_ids {
            let total: i64 = tx.query_row(
                "SELECT COUNT(*) FROM service_logs WHERE service_id = ?1",
                [sid], |r| r.get(0),
            )?;
            if total > per_service {
                let excess = total - per_service;
                let n = tx.execute(
                    "DELETE FROM service_logs WHERE rowid IN (SELECT rowid FROM service_logs WHERE service_id = ?1 ORDER BY timestamp_ms ASC LIMIT ?2)",
                    (sid, excess),
                )?;
                deleted += n as u64;
            }
        }

        tx.commit()
    })?;
    Ok(deleted)
}

pub fn incremental_vacuum(db: &Database, pages: u64) -> rusqlite::Result<()> {
    db.with_conn(|c| c.execute_batch(&format!("PRAGMA incremental_vacuum({pages})")))
}

pub async fn run_loop(db: Database, mut shutdown: watch::Receiver<bool>) {
    let trim_interval = Duration::from_secs(60 * 60);
    let vacuum_interval = Duration::from_secs(60 * 60);
    let mut trim_tick = tokio::time::interval(trim_interval);
    let mut vacuum_tick = tokio::time::interval(vacuum_interval);
    trim_tick.tick().await;
    vacuum_tick.tick().await;

    loop {
        tokio::select! {
            _ = shutdown.changed() => { if *shutdown.borrow() { return; } }
            _ = trim_tick.tick() => {
                let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as i64;
                match trim_logs_once(&db, now) {
                    Ok(n) if n > 0 => info!(deleted = n, "log retention trim"),
                    Ok(_) => {}
                    Err(e) => warn!(error = %e, "log retention trim failed"),
                }
            }
            _ = vacuum_tick.tick() => {
                if let Err(e) = incremental_vacuum(&db, 1000) {
                    warn!(error = %e, "incremental_vacuum failed");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn trims_old_rows_and_excess_per_service() {
        let tmp = tempdir().unwrap();
        let db = Database::open(&tmp.path().join("a.sqlite")).unwrap();
        let svc = db.upsert_service("demo", 0).unwrap();
        let now = 10_000_000_000i64;
        let eight_days_ago = now - 8 * 24 * 60 * 60 * 1000;

        // Insert one old row.
        db.with_conn(|c| c.execute(
            "INSERT INTO service_logs(service_id, run_id, timestamp_ms, seq, stream, line) VALUES (?1, 1, ?2, 1, 'stdout', 'old')",
            (svc, eight_days_ago),
        )).unwrap();

        // Insert 50,010 recent rows to exceed the cap.
        db.with_conn(|c| {
            let mut stmt = c.prepare("INSERT INTO service_logs(service_id, run_id, timestamp_ms, seq, stream, line) VALUES (?1, 2, ?2, ?3, 'stdout', 'x')").unwrap();
            for i in 0..50_010i64 {
                stmt.execute((svc, now - i, i + 2)).unwrap();
            }
            Ok(())
        }).unwrap();

        let deleted = trim_logs_once(&db, now).unwrap();
        assert!(deleted >= 11); // 1 old + 10 excess

        let remaining: i64 = db.with_conn(|c| c.query_row("SELECT COUNT(*) FROM service_logs WHERE service_id = ?1", [svc], |r| r.get(0))).unwrap();
        assert_eq!(remaining, 50_000);
    }
}
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cargo test --lib retention`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/retention.rs
git commit -m "feat(retention): log trim + incremental vacuum with daily loop"
```

---

## Task 23: Daemon orchestration

**Files:**
- Replace: `src/daemon.rs`

- [ ] **Step 1: Write the implementation**

Replace `src/daemon.rs`:

```rust
//! Top-level daemon orchestration: wires config, DB, devices, supervisors,
//! proxies, signals, and retention together.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use tokio::sync::watch;
use tracing::{error, info, warn};

use crate::config::{load_config, Migration};
use crate::db::logs::spawn as spawn_batcher;
use crate::db::Database;
use crate::devices::{cpu, nvml::NvmlProbe, Allocation, GpuProbe};
use crate::errors::ExpectedError;
use crate::proxy;
use crate::retention;
use crate::signals::{await_shutdown, grace_for, ShutdownKind};
use crate::supervise::{orphans::reconcile, spawn_supervisor, SupervisorHandle};

pub async fn run() -> Result<(), ExpectedError> {
    init_tracing();

    let cli_config = parse_cli_config_arg();
    let config_path = crate::config::resolve_from_env(cli_config.as_deref())?;
    info!(config_path = %config_path.display(), "resolved config path");

    let (effective, migrations) = load_config(&config_path)?;
    let db = Database::open(&effective.daemon.data_dir.join("ananke.sqlite"))?;
    apply_migrations(&db, &migrations);

    let _probe: Option<NvmlProbe> = match NvmlProbe::init() {
        Ok(p) => {
            for g in p.list() { info!(gpu = g.id, name = %g.name, total_bytes = g.total_bytes, "detected GPU"); }
            Some(p)
        }
        Err(e) => {
            warn!(error = %e, "NVML init failed; falling back to CPU-only");
            None
        }
    };
    let cpu_mem = cpu::read().ok();
    if let Some(m) = cpu_mem { info!(total = m.total_bytes, avail = m.available_bytes, "CPU memory"); }

    let procfs = PathBuf::from("/proc");
    for disposition in reconcile(&db, &procfs) {
        info!(?disposition, "orphan reconcile");
    }

    let batcher = spawn_batcher(db.clone());

    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    // Order persistent services by priority DESC, name ASC (spec §9.4).
    let mut ordered = effective.services.clone();
    ordered.sort_by(|a, b| b.priority.cmp(&a.priority).then_with(|| a.name.cmp(&b.name)));

    let mut supervisors: Vec<SupervisorHandle> = Vec::new();
    let mut proxy_tasks = Vec::new();
    for svc in ordered {
        let service_id = db.upsert_service(&svc.name, now_ms())?;
        let allocation = Allocation::from_override(&svc.placement_override);
        let handle = spawn_supervisor(svc.clone(), allocation, db.clone(), batcher.clone(), service_id);
        let listen: SocketAddr = format!("127.0.0.1:{}", svc.port).parse()
            .map_err(|e: std::net::AddrParseError| ExpectedError::bind_failed(format!("127.0.0.1:{}", svc.port), e.to_string()))?;
        let shutdown_rx2 = shutdown_rx.clone();
        let upstream = svc.private_port;
        let name = svc.name.clone();
        proxy_tasks.push(tokio::spawn(async move {
            if let Err(e) = proxy::serve(listen, upstream, shutdown_rx2).await {
                error!(service = %name, error = %e, "proxy failed");
            }
        }));
        supervisors.push(handle);
    }

    let retention_task = tokio::spawn(retention::run_loop(db.clone(), shutdown_rx.clone()));

    let shutdown_kind = await_shutdown().await;
    info!(?shutdown_kind, "shutdown initiated");
    let _ = shutdown_tx.send(true);

    let drain_bound = match shutdown_kind {
        ShutdownKind::Graceful => Duration::from_millis(effective.daemon.shutdown_timeout_ms),
        ShutdownKind::Emergency => Duration::from_secs(5),
    };
    let _ = tokio::time::timeout(drain_bound, async {
        for sup in supervisors { sup.shutdown().await; }
    }).await;

    for t in proxy_tasks { t.abort(); let _ = t.await; }
    retention_task.abort();
    let _ = retention_task.await;

    batcher.flush().await;
    let _ = grace_for(shutdown_kind); // reserved for future per-signal grace tuning
    Ok(())
}

fn parse_cli_config_arg() -> Option<PathBuf> {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--config" {
            return args.next().map(PathBuf::from);
        }
        if let Some(rest) = a.strip_prefix("--config=") {
            return Some(PathBuf::from(rest));
        }
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

- [ ] **Step 2: Verify compilation**

Run: `cargo check`
Expected: PASS. The `src/main.rs` stub from Task 0 now wires up against the real `daemon::run`.

- [ ] **Step 3: Commit**

```bash
git add src/daemon.rs src/main.rs
git commit -m "feat(daemon): top-level orchestration wiring config, DB, supervisors, proxies"
```

---

## Task 24: Integration test harness (echo server + shared helpers)

**Files:**
- Create: `tests/common/mod.rs`
- Create: `tests/common/echo_server.rs`

- [ ] **Step 1: Create shared test helpers**

Create `tests/common/mod.rs`:

```rust
pub mod echo_server;

use std::net::TcpListener as StdListener;

pub fn free_port() -> u16 {
    let l = StdListener::bind("127.0.0.1:0").unwrap();
    l.local_addr().unwrap().port()
}
```

Create `tests/common/echo_server.rs`:

```rust
//! Toy HTTP server used as a stand-in for llama-server in integration tests.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::time::Duration;

use bytes::Bytes;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::Frame;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto;
use tokio::net::TcpListener;
use tokio::sync::watch;

pub async fn serve(addr: SocketAddr, mut shutdown: watch::Receiver<bool>) {
    let listener = TcpListener::bind(addr).await.expect("echo bind");
    loop {
        tokio::select! {
            _ = shutdown.changed() => { if *shutdown.borrow() { return; } }
            accept = listener.accept() => {
                let Ok((stream, _)) = accept else { continue; };
                let io = TokioIo::new(stream);
                tokio::spawn(async move {
                    let svc = service_fn(handle);
                    let _ = auto::Builder::new(TokioExecutor::new()).serve_connection(io, svc).await;
                });
            }
        }
    }
}

type EchoBody = http_body_util::combinators::BoxBody<Bytes, Box<dyn std::error::Error + Send + Sync>>;

async fn handle(req: Request<hyper::body::Incoming>) -> Result<Response<EchoBody>, Infallible> {
    match (req.method(), req.uri().path()) {
        (_, "/health") | (_, "/v1/models") => {
            let body = Full::new(Bytes::from("{}"))
                .map_err(|n| match n {})
                .boxed();
            Ok(Response::builder().status(StatusCode::OK).body(body).unwrap())
        }
        (_, "/sse") => {
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
        _ => {
            let body = Full::new(Bytes::from("hello"))
                .map_err(|n| match n {})
                .boxed();
            Ok(Response::builder().status(StatusCode::OK).body(body).unwrap())
        }
    }
}
```

Add to `Cargo.toml` `[dev-dependencies]`:

```toml
tokio-stream = "0.1"
```

- [ ] **Step 2: Verify compilation**

Run: `cargo test --no-run`
Expected: builds without errors.

- [ ] **Step 3: Commit**

```bash
git add Cargo.toml tests/
git commit -m "test: add shared integration test harness and echo server"
```

---

## Task 25: End-to-end integration tests

**Files:**
- Create: `tests/daemon_end_to_end.rs`
- Create: `tests/sse_passthrough.rs`
- Create: `tests/orphan_recovery.rs`

These tests drive the real `proxy::serve` and `supervise` modules against the echo server. They do **not** use `daemon::run` directly because that reads env/config from disk; instead they construct the components manually.

- [ ] **Step 1: End-to-end proxy test**

Create `tests/daemon_end_to_end.rs`:

```rust
mod common;

use std::net::SocketAddr;
use std::time::Duration;

use ananke::proxy;
use tokio::sync::watch;

use common::{echo_server, free_port};

#[tokio::test(flavor = "current_thread")]
async fn proxies_hello_through_reverse_proxy() {
    let upstream_port = free_port();
    let external_port = free_port();

    let (echo_shutdown_tx, echo_shutdown_rx) = watch::channel(false);
    let echo_addr: SocketAddr = format!("127.0.0.1:{upstream_port}").parse().unwrap();
    let echo_task = tokio::spawn(echo_server::serve(echo_addr, echo_shutdown_rx));

    let (proxy_shutdown_tx, proxy_shutdown_rx) = watch::channel(false);
    let proxy_task = tokio::spawn(proxy::serve(
        format!("127.0.0.1:{external_port}").parse().unwrap(),
        upstream_port,
        proxy_shutdown_rx,
    ));

    tokio::time::sleep(Duration::from_millis(100)).await;
    let resp = reqwest::Client::new()
        .get(format!("http://127.0.0.1:{external_port}/anything"))
        .send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(resp.text().await.unwrap(), "hello");

    echo_shutdown_tx.send(true).unwrap();
    proxy_shutdown_tx.send(true).unwrap();
    let _ = echo_task.await;
    let _ = proxy_task.await;
}
```

- [ ] **Step 2: SSE passthrough test**

Create `tests/sse_passthrough.rs`:

```rust
mod common;

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use ananke::proxy;
use futures::StreamExt;
use tokio::sync::watch;

use common::{echo_server, free_port};

#[tokio::test(flavor = "current_thread")]
async fn sse_chunks_arrive_incrementally() {
    let upstream_port = free_port();
    let external_port = free_port();

    let (echo_shutdown_tx, echo_shutdown_rx) = watch::channel(false);
    let echo_addr: SocketAddr = format!("127.0.0.1:{upstream_port}").parse().unwrap();
    let echo_task = tokio::spawn(echo_server::serve(echo_addr, echo_shutdown_rx));

    let (proxy_shutdown_tx, proxy_shutdown_rx) = watch::channel(false);
    let proxy_task = tokio::spawn(proxy::serve(
        format!("127.0.0.1:{external_port}").parse().unwrap(),
        upstream_port,
        proxy_shutdown_rx,
    ));

    tokio::time::sleep(Duration::from_millis(100)).await;
    let resp = reqwest::Client::new()
        .get(format!("http://127.0.0.1:{external_port}/sse"))
        .send().await.unwrap();
    assert_eq!(resp.headers().get("content-type").unwrap(), "text/event-stream");

    let start = Instant::now();
    let mut stream = resp.bytes_stream();
    let mut chunk_times = Vec::new();
    while let Some(chunk) = stream.next().await {
        let _ = chunk.unwrap();
        chunk_times.push(start.elapsed());
    }
    // Expect at least 3 discrete chunks and the last > 150 ms after the first.
    assert!(chunk_times.len() >= 3, "chunks: {}", chunk_times.len());
    assert!(chunk_times.last().unwrap() - chunk_times.first().unwrap() >= Duration::from_millis(150));

    echo_shutdown_tx.send(true).unwrap();
    proxy_shutdown_tx.send(true).unwrap();
    let _ = echo_task.await;
    let _ = proxy_task.await;
}
```

- [ ] **Step 3: Orphan recovery test**

Create `tests/orphan_recovery.rs`:

```rust
mod common;

use ananke::db::Database;
use ananke::supervise::orphans::{reconcile, OrphanDisposition};

#[tokio::test(flavor = "current_thread")]
async fn cleans_dead_pid_rows_on_startup() {
    let tmp = tempfile::tempdir().unwrap();
    let db = Database::open(&tmp.path().join("ananke.sqlite")).unwrap();
    let svc = db.upsert_service("demo", 0).unwrap();
    let procfs = tmp.path().join("proc");
    std::fs::create_dir_all(&procfs).unwrap();

    // PID 99999 does not exist under our fake procfs.
    db.with_conn(|c| c.execute(
        "INSERT INTO running_services(service_id, run_id, pid, spawned_at, command_line, allocation, state) VALUES (?1, 1, 99999, 0, 'llama-server -m x', '{}', 'running')",
        [svc],
    )).unwrap();

    let dispositions = reconcile(&db, &procfs);
    assert_eq!(dispositions.len(), 1);
    assert!(matches!(dispositions[0], OrphanDisposition::Cleaned { .. }));
}
```

- [ ] **Step 4: Expose `orphans` in the public API**

In `src/supervise/mod.rs`, add:

```rust
pub use orphans::{reconcile, OrphanDisposition};
```

- [ ] **Step 5: Add `futures` to dev-deps and run tests**

Add to `Cargo.toml` `[dev-dependencies]`:

```toml
futures = "0.3"
```

Run: `cargo test --tests`
Expected: all integration tests PASS.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml src/supervise/mod.rs tests/
git commit -m "test: end-to-end proxy, SSE passthrough, orphan recovery"
```

---

## Task 26: Manual smoke runbook

**Files:**
- Create: `tests/manual/phase-1-smoke.md`

- [ ] **Step 1: Write the runbook**

Create `tests/manual/phase-1-smoke.md`:

```markdown
# Phase 1 manual smoke test

Real-hardware validation outside CI. Run once before declaring phase 1 done.

## Prerequisites

- Linux host with at least one NVIDIA GPU and `nvidia-smi` working.
- `llama-server` on `$PATH` (build from llama.cpp `master` or install prebuilt binary).
- A small GGUF (`qwen3-4b-instruct-q5_k_xl.gguf` or similar) under `$HOME/models/`.

## Steps

1. Build release:
   ```
   cargo build --release
   ```

2. Create `~/.config/ananke/config.toml`:
   ```toml
   [daemon]
   management_listen = "127.0.0.1:7777"

   [[service]]
   name = "smoke"
   template = "llama-cpp"
   model = "~/models/qwen3-4b-instruct-q5_k_xl.gguf"
   port = 11435
   context = 4096
   flash_attn = true
   cache_type_k = "q8_0"
   cache_type_v = "q8_0"
   lifecycle = "persistent"
   devices.placement = "gpu-only"
   devices.placement_override = { "gpu:0" = 4500 }
   ```

3. Start: `./target/release/ananke --config ~/.config/ananke/config.toml`

4. Verify:
   - `nvidia-smi` shows one llama-server process with roughly 4.5 GB reserved.
   - `curl http://127.0.0.1:11435/v1/models` returns 200.
   - `curl http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"smoke","messages":[{"role":"user","content":"say hi"}]}'` returns a completion.

5. Orphan recovery:
   - `kill -9 $(pidof ananke)`
   - `ps aux | grep llama-server` — child still alive.
   - Restart ananke. Daemon logs `adopted orphan`.
   - `curl http://127.0.0.1:11435/v1/models` — still 200; no second llama-server started.

6. Clean shutdown:
   - `kill -TERM $(pidof ananke)`
   - Daemon logs drain; llama-server disappears from `ps`.

7. Database:
   - `sqlite3 ~/.local/share/ananke/ananke.sqlite 'select count(*) from service_logs'` — large positive number.
```

- [ ] **Step 2: Commit**

```bash
git add tests/manual/
git commit -m "docs: phase 1 manual smoke test runbook"
```

---

## Self-review checklist

Before declaring phase 1 complete, verify:

- `just lint` passes (all four cargo invocations + `npm run lint`).
- `cargo test --workspace` passes with and without default features.
- `tests/manual/phase-1-smoke.md` runbook has been executed at least once.
- `docs/spec.md` and `docs/superpowers/specs/2026-04-18-ananke-phase-1-lean-mvp-daemon.md` are in sync with what was built — if they drift during implementation, fix them inline rather than accumulating a backlog.

When all checks pass, the phase-1 success criteria from the design doc §13 have been met. The daemon is ready for real use for the persistent-service subset of the redline lmp config. Phase 2 (on-demand lifecycle + unified OpenAI endpoint + scheduler) can proceed with its own brainstorm → design → plan → execute cycle.
