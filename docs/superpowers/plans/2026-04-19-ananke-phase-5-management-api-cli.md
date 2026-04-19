# Phase 5 Implementation Plan — management API + WebSocket streams + `anankectl` CLI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the full management API + WebSocket streams + `anankectl` CLI per `docs/superpowers/specs/2026-04-19-ananke-phase-5-management-api-cli.md`.

**Architecture:** Workspace split into three crates (`ananke` daemon, `anankectl` CLI, `ananke-api` shared DTOs). New subsystems: `ConfigManager` (owns raw+parsed config + notify watcher + disk I/O), `EventBus` (broadcast-channel wrapper), log-stream fan-out (broadcast channel in the log batcher). Nine new REST endpoints + two WebSocket streams layered on the existing axum router. CLI is clap-based, speaks HTTP via reqwest, depends only on `ananke-api` for types.

**Tech Stack:** Rust 2024, tokio, axum with `ws` feature, `clap` v4 derive, `reqwest`, `arc_swap`, `sha2`, `notify` (already present), `toml_edit` (already present), `broadcast::channel`.

**Parent spec:** `docs/superpowers/specs/2026-04-19-ananke-phase-5-management-api-cli.md`.

---

## Revert criteria

Stop and open a revert if any of:
- Workspace restructure breaks `cargo test --workspace` for reasons that can't be resolved within one day.
- `ConfigManager` introduces a measurable latency regression on request-path handlers (they call `manager.effective()` per request; the arc_swap guard should be O(1)).
- WebSocket framing doesn't survive a real client reconnect loop (integration test proves it).

---

## File structure (end-state)

```
ananke/                            # workspace root (was the daemon's package root)
├── Cargo.toml                     # [workspace] + [workspace.dependencies]
├── rustfmt.toml                   # unchanged
├── clippy.toml                    # unchanged
├── ananke/                        # daemon crate (everything under the previous src/ moves here)
│   ├── Cargo.toml                 # uses workspace deps; bin = "ananke"
│   ├── src/                       # unchanged contents, plus new modules below
│   │   ├── config/manager.rs      # NEW: ConfigManager
│   │   ├── daemon/events.rs       # NEW: EventBus
│   │   ├── api/management/
│   │   │   ├── lifecycle.rs       # NEW: start/stop/restart/enable/disable handlers
│   │   │   ├── logs.rs            # NEW: GET /api/services/{name}/logs
│   │   │   ├── config.rs          # NEW: GET/PUT/validate config
│   │   │   └── events_ws.rs       # NEW: /api/events WebSocket
│   │   └── api/logs_ws.rs         # NEW: /api/services/{name}/logs/stream WebSocket
│   └── tests/                     # existing integration tests move here unchanged
├── ananke-api/                    # NEW: shared HTTP DTOs
│   ├── Cargo.toml
│   └── src/lib.rs
└── anankectl/                     # NEW: CLI
    ├── Cargo.toml
    └── src/
        ├── main.rs                # clap dispatch
        ├── client.rs              # reqwest client + shared error mapping
        ├── output.rs              # text / JSON formatters
        └── commands/              # one module per subcommand
            ├── mod.rs
            ├── devices.rs
            ├── services.rs
            ├── show.rs
            ├── lifecycle.rs       # start/stop/restart/enable/disable/retry
            ├── logs.rs
            ├── oneshot.rs
            ├── config.rs
            └── reload.rs
```

---

## Task 1: Workspace restructure

**Files:**
- Create: `Cargo.toml` (new root), `ananke/Cargo.toml` (moved+trimmed), `ananke-api/Cargo.toml` (new), `anankectl/Cargo.toml` (new)
- Move: everything under `src/`, `tests/`, `examples/` to `ananke/src/`, `ananke/tests/`, `ananke/examples/`
- Modify: `Cargo.lock` (regenerated)

- [ ] **Step 1: Move the daemon sources**

Use `git mv` so history is preserved. The moves, from the repo root:

```bash
mkdir -p ananke/src ananke/tests ananke/examples
git mv Cargo.toml ananke/Cargo.toml
git mv src ananke/src
git mv tests ananke/tests
git mv examples ananke/examples
git mv rustfmt.toml ananke/rustfmt.toml 2>/dev/null || true   # keep one copy at root; rustfmt.toml at the workspace root is fine
git mv clippy.toml ananke/clippy.toml 2>/dev/null || true
```

Then restore `rustfmt.toml` and `clippy.toml` at the workspace root (rustfmt's `imports_granularity` and `group_imports` options need to be applied at workspace scope):

```bash
cp ananke/rustfmt.toml rustfmt.toml
git rm ananke/rustfmt.toml
cp ananke/clippy.toml clippy.toml 2>/dev/null || true
git rm ananke/clippy.toml 2>/dev/null || true
git add rustfmt.toml clippy.toml
```

- [ ] **Step 2: Write the workspace root `Cargo.toml`**

```toml
[workspace]
members = ["ananke", "anankectl", "ananke-api"]
resolver = "2"

[workspace.package]
edition = "2024"
version = "0.1.0"

[workspace.dependencies]
# Async + HTTP
tokio = { version = "1", features = ["rt-multi-thread", "macros", "net", "io-util", "signal", "process", "time", "fs", "sync"] }
hyper = { version = "1", features = ["server", "client", "http1"] }
hyper-util = { version = "0.1", features = ["tokio", "server", "server-auto", "client", "client-legacy", "http1"] }
http = "1"
http-body-util = "0.1"
bytes = "1"
axum = { version = "0.7", features = ["macros", "json", "ws"] }
tower = "0.5"
tower-http = { version = "0.6", features = ["trace"] }
reqwest = { version = "0.12", default-features = false, features = ["rustls-tls", "stream", "json"] }

# Serde + formats
serde = { version = "1", features = ["derive"] }
serde_json = "1"
toml = "0.8"
toml_edit = { version = "0.22", features = ["serde"] }

# OpenAPI
utoipa = { version = "5", features = ["axum_extras", "smallvec", "uuid"] }
utoipa-axum = "0.2"

# Other shared
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "fmt"] }
smol_str = { version = "0.3", features = ["serde"] }
parking_lot = "0.12"
arc-swap = "1"
sha2 = "0.10"
base64 = "0.22"
ulid = "1"
clap = { version = "4", features = ["derive", "env"] }
async-trait = "0.1"
pin-project-lite = "0.2"
futures = "0.3"
regex = "1"
libc = "0.2"
notify = "6"

# Linux-only
nvml-wrapper = "0.10"
nix = { version = "0.29", features = ["signal", "fs", "process"] }

# Database
toasty = { version = "=0.4.0", features = ["sqlite"] }
rusqlite = { version = "0.39", features = ["bundled"] }

[workspace.dev-dependencies]
proptest = "1"
pretty_assertions = "1"
tempfile = "3"
tokio = { version = "1", features = ["full", "test-util"] }
tokio-stream = "0.1"
```

- [ ] **Step 3: Rewrite `ananke/Cargo.toml`**

```toml
[package]
name = "ananke"
version.workspace = true
edition.workspace = true

[[bin]]
name = "ananke"
path = "src/main.rs"

[lib]
name = "ananke"
path = "src/lib.rs"

[dependencies]
ananke-api.workspace = true

tokio.workspace = true
hyper.workspace = true
hyper-util.workspace = true
http.workspace = true
http-body-util.workspace = true
bytes.workspace = true
axum.workspace = true
tower.workspace = true
tower-http.workspace = true
reqwest.workspace = true

serde.workspace = true
serde_json.workspace = true
toml.workspace = true
toml_edit.workspace = true

utoipa.workspace = true
utoipa-axum.workspace = true

tracing.workspace = true
tracing-subscriber.workspace = true
smol_str.workspace = true
parking_lot.workspace = true
arc-swap.workspace = true
sha2.workspace = true
base64.workspace = true
ulid.workspace = true
async-trait.workspace = true
pin-project-lite.workspace = true
futures.workspace = true
regex.workspace = true
libc.workspace = true
notify.workspace = true

nvml-wrapper.workspace = true
nix.workspace = true

toasty.workspace = true
rusqlite.workspace = true

[dev-dependencies]
proptest.workspace = true
pretty_assertions.workspace = true
tempfile.workspace = true
tokio = { workspace = true, features = ["full", "test-util"] }
tokio-stream.workspace = true
futures.workspace = true

[features]
default = []
test-fakes = []
```

- [ ] **Step 4: Write `ananke-api/Cargo.toml`**

```toml
[package]
name = "ananke-api"
version.workspace = true
edition.workspace = true

[lib]
name = "ananke_api"
path = "src/lib.rs"

[dependencies]
serde.workspace = true
serde_json.workspace = true
smol_str.workspace = true

[dev-dependencies]
pretty_assertions.workspace = true
```

- [ ] **Step 5: Write `anankectl/Cargo.toml`**

```toml
[package]
name = "anankectl"
version.workspace = true
edition.workspace = true

[[bin]]
name = "anankectl"
path = "src/main.rs"

[dependencies]
ananke-api.workspace = true

tokio = { workspace = true, features = ["rt-multi-thread", "macros", "net", "time", "signal"] }
reqwest.workspace = true
clap.workspace = true
serde.workspace = true
serde_json.workspace = true
toml.workspace = true
smol_str.workspace = true
tracing.workspace = true
tracing-subscriber.workspace = true
futures.workspace = true
tokio-tungstenite = { version = "0.26", default-features = false, features = ["rustls-tls-native-roots"] }

[dev-dependencies]
pretty_assertions.workspace = true
tempfile.workspace = true
tokio = { workspace = true, features = ["full", "test-util"] }
```

- [ ] **Step 6: Create `ananke-api/src/lib.rs` with a placeholder**

```rust
//! Shared HTTP DTOs between the `ananke` daemon and the `anankectl` CLI.
//!
//! No business logic lives here — only serde-friendly types that appear on
//! the wire.

#![deny(missing_docs)]

/// Placeholder so the crate compiles before Task 2 fleshes it out.
pub const _CRATE_READY: () = ();
```

Ignore the `missing_docs` attribute — the placeholder is temporary; Task 2 replaces the lib with the real DTOs.

- [ ] **Step 7: Create a stub `anankectl/src/main.rs`**

```rust
fn main() {
    eprintln!("anankectl: not yet implemented");
    std::process::exit(2);
}
```

- [ ] **Step 8: Fix path references inside `ananke/`**

Everything under `ananke/src/` still references internal modules correctly (imports used `crate::`). The only paths that change are:

- `examples/dump-gguf.rs` path reference in `ananke/Cargo.toml` (if it has an explicit path) — verify by running `cargo check --examples -p ananke`.
- Integration tests under `ananke/tests/` use `ananke::...` imports, which still work.

- [ ] **Step 9: Build + lint + test**

```bash
cargo check --workspace
cargo test --workspace --features test-fakes
cargo +nightly fmt --all
cargo clippy --all-targets --features test-fakes -- -D warnings
```

All green, no behaviour changes.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "chore: split into three-crate workspace (ananke, anankectl, ananke-api)"
```

---

## Task 2: `ananke-api` DTOs

**Files:**
- Create: `ananke-api/src/lib.rs`, `ananke-api/src/services.rs`, `ananke-api/src/devices.rs`, `ananke-api/src/logs.rs`, `ananke-api/src/config.rs`, `ananke-api/src/oneshot.rs`, `ananke-api/src/events.rs`, `ananke-api/src/lifecycle.rs`, `ananke-api/src/errors.rs`
- Test: `ananke-api/tests/roundtrip.rs`

- [ ] **Step 1: Write `ananke-api/src/lib.rs`**

```rust
//! Shared HTTP DTOs between the `ananke` daemon and the `anankectl` CLI.

#![deny(missing_docs)]

pub mod config;
pub mod devices;
pub mod errors;
pub mod events;
pub mod lifecycle;
pub mod logs;
pub mod oneshot;
pub mod services;

pub use config::{ConfigResponse, ConfigValidateRequest, ConfigValidateResponse, ValidationError};
pub use devices::{DeviceReservation, DeviceSummary};
pub use errors::ApiError;
pub use events::Event;
pub use lifecycle::{DisableResponse, EnableResponse, StartResponse, StopResponse};
pub use logs::{LogLine, LogsResponse};
pub use oneshot::{OneshotRequest, OneshotResponse, OneshotStatus};
pub use services::{ServiceDetail, ServiceSummary};
```

- [ ] **Step 2: Write `ananke-api/src/errors.rs`**

```rust
//! OpenAI-shaped error envelope used by every `/api/*` error response.

use serde::{Deserialize, Serialize};

/// `{"error": {"code", "message", "type"}}` with `type` fixed to `"server_error"`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ApiError {
    pub error: ApiErrorBody,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[allow(missing_docs)]
pub struct ApiErrorBody {
    pub code: String,
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
}

impl ApiError {
    /// Build an error with `type: "server_error"`.
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            error: ApiErrorBody {
                code: code.into(),
                message: message.into(),
                kind: "server_error".to_string(),
            },
        }
    }
}
```

- [ ] **Step 3: Write `ananke-api/src/services.rs`**

```rust
//! Service summary and detail views.

use serde::{Deserialize, Serialize};

use crate::logs::LogLine;

/// One entry in `GET /api/services`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ServiceSummary {
    pub name: String,
    pub state: String,
    pub lifecycle: String,
    pub priority: u8,
    pub port: u16,
    pub run_id: Option<i64>,
    pub pid: Option<i32>,
    pub elastic_borrower: Option<String>,
}

/// `GET /api/services/{name}` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
    pub rolling_mean: Option<f32>,
    pub rolling_samples: u64,
    pub observed_peak_bytes: u64,
    pub elastic_borrower: Option<String>,
}
```

- [ ] **Step 4: Write `ananke-api/src/devices.rs`**

```rust
//! Device + reservation views.

use serde::{Deserialize, Serialize};

/// One entry in `GET /api/devices`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeviceSummary {
    pub id: String,
    pub name: String,
    pub total_bytes: u64,
    pub free_bytes: u64,
    pub reservations: Vec<DeviceReservation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[allow(missing_docs)]
pub struct DeviceReservation {
    pub service: String,
    pub bytes: u64,
    pub elastic: bool,
}
```

- [ ] **Step 5: Write `ananke-api/src/logs.rs`**

```rust
//! Log line + paginated logs response.

use serde::{Deserialize, Serialize};

/// One captured stdout/stderr line.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LogLine {
    pub timestamp_ms: i64,
    pub stream: String,
    pub line: String,
    pub run_id: i64,
    pub seq: i64,
}

/// `GET /api/services/{name}/logs` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LogsResponse {
    pub logs: Vec<LogLine>,
    pub next_cursor: Option<String>,
}
```

- [ ] **Step 6: Write `ananke-api/src/config.rs`**

```rust
//! Config GET/PUT/validate payloads.

use serde::{Deserialize, Serialize};

/// `GET /api/config` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConfigResponse {
    pub content: String,
    pub hash: String,
}

/// `POST /api/config/validate` request body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConfigValidateRequest {
    pub content: String,
}

/// `POST /api/config/validate` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConfigValidateResponse {
    pub valid: bool,
    pub errors: Vec<ValidationError>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[allow(missing_docs)]
pub struct ValidationError {
    pub line: u32,
    pub column: u32,
    pub message: String,
}
```

- [ ] **Step 7: Write `ananke-api/src/oneshot.rs`**

```rust
//! Oneshot request + response bodies.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// `POST /api/oneshot` request body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OneshotRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub template: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workdir: Option<String>,
    pub allocation: OneshotAllocation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub devices: Option<OneshotDevices>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct OneshotAllocation {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_gb: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_vram_gb: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_vram_gb: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[allow(missing_docs)]
pub struct OneshotDevices {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement: Option<String>,
}

/// `POST /api/oneshot` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OneshotResponse {
    pub id: String,
    pub name: String,
    pub port: u16,
    pub logs_url: String,
}

/// `GET /api/oneshot/{id}` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OneshotStatus {
    pub id: String,
    pub name: String,
    pub state: String,
    pub port: u16,
    pub submitted_at_ms: i64,
    pub started_at_ms: Option<i64>,
    pub ended_at_ms: Option<i64>,
    pub exit_code: Option<i32>,
    pub logs_url: String,
}
```

- [ ] **Step 8: Write `ananke-api/src/lifecycle.rs`**

```rust
//! Service lifecycle POST response bodies.

use serde::{Deserialize, Serialize};

/// `POST /api/services/{name}/start` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum StartResponse {
    AlreadyRunning,
    Started { run_id: i64 },
    QueueFull,
    Unavailable { reason: String },
}

/// `POST /api/services/{name}/stop` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum StopResponse {
    NotRunning,
    Drained,
}

/// `POST /api/services/{name}/enable` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum EnableResponse {
    Enabled,
    NotDisabled,
    AlreadyEnabled,
}

/// `POST /api/services/{name}/disable` response body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum DisableResponse {
    Disabled,
    AlreadyDisabled,
}
```

- [ ] **Step 9: Write `ananke-api/src/events.rs`**

```rust
//! WebSocket event envelope published on `/api/events`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

/// Every variant carries `at_ms` except `Overflow`, which is an out-of-band
/// control frame.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    StateChanged {
        service: SmolStr,
        from: String,
        to: String,
        at_ms: i64,
    },
    AllocationChanged {
        service: SmolStr,
        reservations: BTreeMap<String, u64>,
        at_ms: i64,
    },
    ConfigReloaded {
        at_ms: i64,
        changed_services: Vec<SmolStr>,
    },
    EstimatorDrift {
        service: SmolStr,
        rolling_mean: f32,
        at_ms: i64,
    },
    Overflow {
        dropped: u64,
    },
}
```

- [ ] **Step 10: Write the round-trip test**

Create `ananke-api/tests/roundtrip.rs`:

```rust
use ananke_api::{
    ConfigResponse, DeviceReservation, DeviceSummary, DisableResponse, EnableResponse, Event,
    LogLine, LogsResponse, OneshotRequest, OneshotResponse, ServiceDetail, ServiceSummary,
    StartResponse, StopResponse,
    oneshot::{OneshotAllocation, OneshotDevices},
};
use pretty_assertions::assert_eq;
use smol_str::SmolStr;

fn roundtrip<T>(value: T) -> T
where
    T: serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    let json = serde_json::to_string(&value).expect("serialize");
    serde_json::from_str(&json).expect("deserialize")
}

#[test]
fn service_summary_roundtrips() {
    let v = ServiceSummary {
        name: "demo".into(),
        state: "running".into(),
        lifecycle: "persistent".into(),
        priority: 50,
        port: 11435,
        run_id: Some(1),
        pid: Some(1234),
        elastic_borrower: None,
    };
    assert_eq!(v.clone(), roundtrip(v));
}

#[test]
fn start_response_tagged_union() {
    let v = StartResponse::Unavailable { reason: "no fit".into() };
    let json = serde_json::to_value(&v).unwrap();
    assert_eq!(json, serde_json::json!({"status": "unavailable", "reason": "no fit"}));
}

#[test]
fn event_state_changed_tag() {
    let v = Event::StateChanged {
        service: SmolStr::new("demo"),
        from: "idle".into(),
        to: "starting".into(),
        at_ms: 1,
    };
    let json = serde_json::to_value(&v).unwrap();
    assert_eq!(json["type"], "state_changed");
    assert_eq!(json["service"], "demo");
}

#[test]
fn oneshot_request_optional_fields_omitted() {
    let v = OneshotRequest {
        name: None,
        template: "command".into(),
        command: Some(vec!["python".into(), "batch.py".into()]),
        workdir: None,
        allocation: OneshotAllocation {
            mode: Some("static".into()),
            vram_gb: Some(16.0),
            min_vram_gb: None,
            max_vram_gb: None,
        },
        devices: Some(OneshotDevices { placement: Some("gpu-only".into()) }),
        priority: Some(40),
        ttl: Some("2h".into()),
        port: None,
        metadata: Default::default(),
    };
    let json = serde_json::to_value(&v).unwrap();
    // Optional None/empty fields must be skipped so minimal requests stay small.
    assert!(!json.as_object().unwrap().contains_key("name"));
    assert!(!json.as_object().unwrap().contains_key("workdir"));
    assert!(!json.as_object().unwrap().contains_key("port"));
    assert!(!json.as_object().unwrap().contains_key("metadata"));
}

#[test]
fn logs_response_default_empty_cursor() {
    let v = LogsResponse {
        logs: vec![LogLine {
            timestamp_ms: 1,
            stream: "stdout".into(),
            line: "hello".into(),
            run_id: 1,
            seq: 1,
        }],
        next_cursor: None,
    };
    assert_eq!(v.clone(), roundtrip(v));
}

#[test]
fn config_response_roundtrips() {
    let v = ConfigResponse { content: "[daemon]\n".into(), hash: "abc".into() };
    assert_eq!(v.clone(), roundtrip(v));
}

#[test]
fn device_summary_roundtrips() {
    let v = DeviceSummary {
        id: "gpu:0".into(),
        name: "RTX 3090".into(),
        total_bytes: 1 << 34,
        free_bytes: 1 << 33,
        reservations: vec![DeviceReservation {
            service: "demo".into(),
            bytes: 1 << 30,
            elastic: false,
        }],
    };
    assert_eq!(v.clone(), roundtrip(v));
}

#[test]
fn service_detail_roundtrips() {
    let v = ServiceDetail {
        name: "demo".into(),
        state: "idle".into(),
        lifecycle: "persistent".into(),
        priority: 50,
        port: 11435,
        private_port: 40000,
        template: "llamacpp".into(),
        placement_override: Default::default(),
        idle_timeout_ms: 600_000,
        run_id: None,
        pid: None,
        recent_logs: vec![],
        rolling_mean: None,
        rolling_samples: 0,
        observed_peak_bytes: 0,
        elastic_borrower: None,
    };
    assert_eq!(v.clone(), roundtrip(v));
}

#[test]
fn oneshot_response_roundtrips() {
    let v = OneshotResponse {
        id: "oneshot_01H".into(),
        name: "sd-batch".into(),
        port: 18001,
        logs_url: "/api/oneshot/oneshot_01H/logs/stream".into(),
    };
    assert_eq!(v.clone(), roundtrip(v));
}

#[test]
fn stop_response_tagged_union() {
    let v = StopResponse::Drained;
    let json = serde_json::to_value(&v).unwrap();
    assert_eq!(json, serde_json::json!({"status": "drained"}));
}

#[test]
fn enable_response_tagged_union() {
    let v = EnableResponse::NotDisabled;
    let json = serde_json::to_value(&v).unwrap();
    assert_eq!(json, serde_json::json!({"status": "not_disabled"}));
}

#[test]
fn disable_response_tagged_union() {
    let v = DisableResponse::Disabled;
    let json = serde_json::to_value(&v).unwrap();
    assert_eq!(json, serde_json::json!({"status": "disabled"}));
}
```

- [ ] **Step 11: Migrate daemon handlers to use `ananke-api` types**

Replace `ananke::api::management::types` usage in `ananke/src/api/management/handlers.rs` with imports from `ananke_api`. Same for `ananke::oneshot::handlers::OneshotRecord`'s wire-facing fields.

The existing `ananke::api::management::types` module is deleted; any daemon-only types that need to stay internal (e.g., private helper structs) are inlined at the handler call site. All public JSON payloads route through `ananke-api`.

- [ ] **Step 12: Build + test**

```bash
cargo check --workspace
cargo test --workspace --features test-fakes
cargo clippy --all-targets --features test-fakes -- -D warnings
```

- [ ] **Step 13: Commit**

```bash
git add -A
git commit -m "feat(api): ananke-api crate with shared HTTP DTOs"
```

---

## Task 3: `EventBus` + publisher wiring

**Files:**
- Create: `ananke/src/daemon/events.rs`
- Modify: `ananke/src/daemon/mod.rs` (declare + construct the bus), `ananke/src/daemon/app_state.rs` (carry the bus), `ananke/src/supervise/mod.rs` (publish `StateChanged` on set_state), `ananke/src/allocator/mod.rs` (publish `AllocationChanged` on mutations)
- Test: inline unit test in `ananke/src/daemon/events.rs`

- [ ] **Step 1: Write `ananke/src/daemon/events.rs`**

```rust
//! Broadcast-based event bus. Publishers are infallible from the caller's
//! perspective; subscribers handle lag explicitly via `RecvError::Lagged`.

use ananke_api::Event;
use tokio::sync::broadcast;

/// Capacity of the per-daemon event broadcast channel. Subscribers that lag
/// beyond this buffer receive `Event::Overflow` and resume.
const EVENT_BUS_CAPACITY: usize = 1024;

/// Cheap to clone; internally Arc-backed via `broadcast::Sender`.
#[derive(Clone)]
pub struct EventBus {
    tx: broadcast::Sender<Event>,
}

impl EventBus {
    pub fn new() -> Self {
        let (tx, _rx) = broadcast::channel(EVENT_BUS_CAPACITY);
        Self { tx }
    }

    /// Publish an event. Silently drops if there are no subscribers.
    pub fn publish(&self, event: Event) {
        let _ = self.tx.send(event);
    }

    /// Subscribe to the bus. Each subscriber has its own cursor.
    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.tx.subscribe()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use ananke_api::Event;
    use smol_str::SmolStr;

    use super::*;

    #[tokio::test]
    async fn publish_then_receive() {
        let bus = EventBus::new();
        let mut rx = bus.subscribe();
        bus.publish(Event::ConfigReloaded {
            at_ms: 1,
            changed_services: vec![SmolStr::new("demo")],
        });
        match rx.recv().await.unwrap() {
            Event::ConfigReloaded { at_ms, .. } => assert_eq!(at_ms, 1),
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[tokio::test]
    async fn lag_surfaces_as_recverror() {
        let bus = EventBus::new();
        let mut rx = bus.subscribe();
        // Fill the buffer + 1 so the first slot is displaced.
        for i in 0..(EVENT_BUS_CAPACITY + 5) {
            bus.publish(Event::EstimatorDrift {
                service: SmolStr::new("demo"),
                rolling_mean: i as f32,
                at_ms: i as i64,
            });
        }
        match rx.recv().await {
            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                assert!(n >= 5, "expected at least 5 dropped, got {n}");
            }
            other => panic!("expected lag, got {other:?}"),
        }
    }
}
```

- [ ] **Step 2: Add `pub mod events;` to `ananke/src/daemon/mod.rs`**

Near the top of `ananke/src/daemon/mod.rs`, after the existing `pub mod signals;`:

```rust
pub mod events;
```

- [ ] **Step 3: Carry the bus through `AppState`**

In `ananke/src/daemon/app_state.rs`, add `events: EventBus` to the struct and update `AppState::supervisor_deps()` to propagate it (Task 4 wires it into the supervisor).

```rust
use crate::daemon::events::EventBus;

pub struct AppState {
    // ... existing fields ...
    pub events: EventBus,
}
```

- [ ] **Step 4: Instantiate in `daemon::run`**

In `ananke/src/daemon/mod.rs::run()`, early in the function (alongside the other tables):

```rust
let events = crate::daemon::events::EventBus::new();
```

Pass it into `AppState` at construction time.

- [ ] **Step 5: Publish `StateChanged` from `RunLoop::set_state`**

Find `RunLoop::set_state` in `ananke/src/supervise/mod.rs`. Add an `events: EventBus` field to `RunLoop` (threaded through `SupervisorDeps`), then:

```rust
fn set_state(&mut self, new_state: ServiceState) {
    let from = format!("{:?}", self.state).to_lowercase();
    let to = format!("{:?}", new_state).to_lowercase();
    self.state = new_state.clone();
    *self.state_mirror.lock() = new_state;
    self.deps.events.publish(ananke_api::Event::StateChanged {
        service: self.init.svc.name.clone(),
        from,
        to,
        at_ms: crate::tracking::now_unix_ms(),
    });
}
```

Add `events: EventBus` to `SupervisorDeps`. Update `AppState::supervisor_deps()` and the `daemon::run` construction of `SupervisorDeps` to pass it through.

- [ ] **Step 6: Publish `AllocationChanged` on allocation mutations**

In `ananke/src/allocator/mod.rs`, the `AllocationTable` is the central source of truth. Two paths mutate it:
- `AllocationTable::insert` on successful supervisor spawn.
- `AllocationTable::remove` on drain / evict / kill.

Rather than wiring the bus into the table itself, have each caller call `events.publish` after mutation. Specifically in `ananke/src/supervise/mod.rs` add an `emit_allocation_changed()` helper on `RunLoop`:

```rust
fn emit_allocation_changed(&self) {
    let snapshot = self
        .deps
        .allocations
        .lock()
        .get(&self.init.svc.name)
        .cloned()
        .unwrap_or_default();
    let reservations: std::collections::BTreeMap<String, u64> = snapshot
        .iter()
        .map(|(slot, mb)| (slot_key(slot), (*mb) * 1024 * 1024))
        .collect();
    self.deps.events.publish(ananke_api::Event::AllocationChanged {
        service: self.init.svc.name.clone(),
        reservations,
        at_ms: crate::tracking::now_unix_ms(),
    });
}

fn slot_key(slot: &crate::config::DeviceSlot) -> String {
    match slot {
        crate::config::DeviceSlot::Cpu => "cpu".to_string(),
        crate::config::DeviceSlot::Gpu(n) => format!("gpu:{n}"),
    }
}
```

Call `emit_allocation_changed` after: successful allocation reserve (at start), on `record_drain_complete`, and on the eviction-triggered `allocations.remove` inside `try_eviction_to_fit`.

- [ ] **Step 7: Build + test**

```bash
cargo check --workspace
cargo test --workspace --features test-fakes
```

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat(daemon): EventBus + StateChanged/AllocationChanged publishers"
```

---

## Task 4: `ConfigManager`

**Files:**
- Create: `ananke/src/config/manager.rs`
- Modify: `ananke/src/config/mod.rs` (declare `manager`), `ananke/src/daemon/app_state.rs` (hold `Arc<ConfigManager>` instead of `Arc<EffectiveConfig>`), `ananke/src/daemon/mod.rs` (build via `ConfigManager::open` + remove the inline `notify` watcher wire-up that lives here if any), all handler sites that read `state.config` → `state.config.effective()`

- [ ] **Step 1: Write `ananke/src/config/manager.rs`**

```rust
//! Owns the raw TOML + parsed `EffectiveConfig` plus all disk I/O for the
//! config file. Replaces the previous `Arc<EffectiveConfig>` held directly
//! by `AppState`.

use std::{io, path::PathBuf, sync::Arc, time::Duration};

use arc_swap::ArcSwap;
use base64::{Engine, engine::general_purpose::STANDARD as B64};
use parking_lot::RwLock;
use sha2::{Digest, Sha256};
use tracing::{info, warn};

use ananke_api::ValidationError;

use crate::{
    config::{EffectiveConfig, load_config},
    daemon::events::EventBus,
    errors::ExpectedError,
};

/// Base64-encoded SHA-256 of the raw TOML bytes. Callers treat it as opaque.
pub type ConfigHash = String;

/// Shared owner of config state. Cloned via `Arc<ConfigManager>`.
pub struct ConfigManager {
    raw: RwLock<String>,
    effective: ArcSwap<EffectiveConfig>,
    path: PathBuf,
    events: EventBus,
    _watcher: RwLock<Option<notify::RecommendedWatcher>>,
}

/// Failure modes from `ConfigManager::apply`.
pub enum ApplyError {
    HashMismatch { server_hash: ConfigHash },
    Invalid(Vec<ValidationError>),
    PersistFailed(io::Error),
}

impl ConfigManager {
    /// Load the config from disk, construct the manager, and spawn the
    /// notify watcher. The returned `Arc<ConfigManager>` is thread-safe and
    /// inexpensive to clone.
    pub async fn open(path: PathBuf, events: EventBus) -> Result<Arc<Self>, ExpectedError> {
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| ExpectedError::config_unparseable(path.clone(), e.to_string()))?;
        let (effective, _migrations) = load_config(&path)?;
        let this = Arc::new(Self {
            raw: RwLock::new(raw),
            effective: ArcSwap::from_pointee(effective),
            path: path.clone(),
            events,
            _watcher: RwLock::new(None),
        });
        this.spawn_watcher();
        Ok(this)
    }

    pub fn raw(&self) -> (String, ConfigHash) {
        let raw = self.raw.read().clone();
        let hash = hash_of(&raw);
        (raw, hash)
    }

    pub fn effective(&self) -> arc_swap::Guard<Arc<EffectiveConfig>> {
        self.effective.load()
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// Validate the given TOML without touching disk or the in-memory cache.
    pub fn validate(&self, toml: &str) -> Result<(), Vec<ValidationError>> {
        validate_toml(&self.path, toml)
    }

    /// Apply a new config: hash-check, validate, persist, update in-memory
    /// snapshot, publish `ConfigReloaded`.
    pub async fn apply(
        self: &Arc<Self>,
        new_toml: String,
        if_match: ConfigHash,
    ) -> Result<(), ApplyError> {
        {
            let (current_raw, current_hash) = self.raw();
            if current_hash != if_match {
                return Err(ApplyError::HashMismatch {
                    server_hash: current_hash,
                });
            }
            if current_raw == new_toml {
                // Noop; nothing changed.
                return Ok(());
            }
        }

        // Dry-run validate; surface span errors to the caller.
        validate_toml(&self.path, &new_toml).map_err(ApplyError::Invalid)?;

        // Persist to disk atomically (write temp → fsync → rename).
        persist_atomically(&self.path, &new_toml).map_err(ApplyError::PersistFailed)?;

        // Reload from disk (reuses the full load_config pipeline so we
        // exercise exactly the same path as the notify watcher).
        self.reload_from_disk();
        Ok(())
    }

    fn reload_from_disk(self: &Arc<Self>) {
        let raw = match std::fs::read_to_string(&self.path) {
            Ok(s) => s,
            Err(e) => {
                warn!(error = %e, "config reload: read failed");
                return;
            }
        };
        let (effective, _migs) = match load_config(&self.path) {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "config reload: validate failed; keeping live config");
                return;
            }
        };
        let changed = diff_services(&self.effective.load(), &effective);
        *self.raw.write() = raw;
        self.effective.store(Arc::new(effective));
        info!(?changed, "config reloaded");
        self.events.publish(ananke_api::Event::ConfigReloaded {
            at_ms: crate::tracking::now_unix_ms(),
            changed_services: changed,
        });
    }

    fn spawn_watcher(self: &Arc<Self>) {
        use notify::{RecursiveMode, Watcher};
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<()>();
        // Watch the file's parent directory so atomic-rename swaps fire.
        let dir = self.path.parent().map(|p| p.to_path_buf());
        let target = self.path.clone();
        let mut watcher = match notify::recommended_watcher(move |res: Result<notify::Event, _>| {
            if let Ok(ev) = res {
                if ev.paths.iter().any(|p| p == &target) {
                    let _ = tx.send(());
                }
            }
        }) {
            Ok(w) => w,
            Err(e) => {
                warn!(error = %e, "notify watcher init failed");
                return;
            }
        };
        if let Some(d) = &dir
            && let Err(e) = watcher.watch(d, RecursiveMode::NonRecursive)
        {
            warn!(error = %e, path = %d.display(), "notify watch failed");
            return;
        }
        *self._watcher.write() = Some(watcher);

        let me = Arc::clone(self);
        tokio::spawn(async move {
            let mut debounce = tokio::time::interval(Duration::from_millis(500));
            debounce.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            debounce.tick().await;
            let mut pending = false;
            loop {
                tokio::select! {
                    _ = rx.recv() => { pending = true; }
                    _ = debounce.tick(), if pending => {
                        pending = false;
                        me.reload_from_disk();
                    }
                }
            }
        });
    }
}

fn hash_of(s: &str) -> ConfigHash {
    let digest = Sha256::digest(s.as_bytes());
    B64.encode(digest)
}

fn persist_atomically(path: &std::path::Path, content: &str) -> io::Result<()> {
    use std::io::Write;
    let dir = path.parent().unwrap_or(std::path::Path::new("."));
    let tmp = tempfile::Builder::new()
        .prefix(".ananke-config-")
        .suffix(".toml")
        .tempfile_in(dir)?;
    {
        let mut f = tmp.as_file();
        f.write_all(content.as_bytes())?;
        f.sync_all()?;
    }
    tmp.persist(path).map_err(|e| e.error)?;
    Ok(())
}

fn validate_toml(path: &std::path::Path, content: &str) -> Result<(), Vec<ValidationError>> {
    // Run the full load pipeline against an ephemeral copy of the file so we
    // exercise validation end-to-end.
    let parent = path.parent().unwrap_or(std::path::Path::new("."));
    let tmp_path = parent.join(".ananke-config-validate.toml");
    if let Err(e) = std::fs::write(&tmp_path, content) {
        return Err(vec![ValidationError {
            line: 0,
            column: 0,
            message: format!("write temp file: {e}"),
        }]);
    }
    let result = load_config(&tmp_path);
    let _ = std::fs::remove_file(&tmp_path);
    match result {
        Ok(_) => Ok(()),
        Err(e) => Err(vec![ValidationError {
            line: 0,
            column: 0,
            message: e.to_string(),
        }]),
    }
}

fn diff_services(old: &EffectiveConfig, new: &EffectiveConfig) -> Vec<smol_str::SmolStr> {
    use std::collections::BTreeSet;
    let old_names: BTreeSet<_> = old.services.iter().map(|s| s.name.clone()).collect();
    let new_names: BTreeSet<_> = new.services.iter().map(|s| s.name.clone()).collect();
    let mut changed = Vec::new();
    for name in &new_names {
        if !old_names.contains(name) {
            changed.push(name.clone());
        }
    }
    for name in &old_names {
        if !new_names.contains(name) {
            changed.push(name.clone());
        }
    }
    changed.sort();
    changed.dedup();
    changed
}
```

- [ ] **Step 2: Export from `ananke/src/config/mod.rs`**

Add `pub mod manager;` near the existing `pub mod validate;` line.

- [ ] **Step 3: Replace `AppState::config: Arc<EffectiveConfig>` with `Arc<ConfigManager>`**

In `ananke/src/daemon/app_state.rs`:

```rust
use crate::config::manager::ConfigManager;

pub struct AppState {
    pub config: Arc<ConfigManager>,
    // ... rest unchanged
}
```

Update `supervisor_deps()` to pass `self.config.effective().clone()` (cloning the inner `Arc<EffectiveConfig>`).

- [ ] **Step 4: Migrate handler sites**

Grep for `state.config.services` across `ananke/src/api/`. Each site becomes `state.config.effective().services`. The `Guard` deref-coerces to `&EffectiveConfig`, so existing field accesses work unchanged after a single `.effective()` call.

- [ ] **Step 5: Boot via `ConfigManager::open`**

In `ananke/src/daemon/mod.rs::run`:

```rust
let events = crate::daemon::events::EventBus::new();
let config_path = crate::config::resolve_from_env(cli_config.as_deref())?;
let config = crate::config::manager::ConfigManager::open(config_path.clone(), events.clone()).await?;
let effective_snapshot = config.effective().clone();
```

Remove any prior inline notify watcher setup (none currently in tree; this consolidates the pattern before other modules depend on it).

- [ ] **Step 6: Test via unit tests in the manager**

Add below the existing module in `ananke/src/config/manager.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    const VALID_TOML: &str = r#"
[daemon]
management_listen = "127.0.0.1:0"
openai_listen = "127.0.0.1:0"

[[service]]
name = "demo"
template = "llama-cpp"
model = "/tmp/fake.gguf"
port = 11435
devices.placement = "cpu-only"
devices.placement_override = { cpu = 100 }
lifecycle = "on_demand"
"#;

    #[tokio::test]
    async fn apply_rejects_stale_if_match() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("ananke.toml");
        std::fs::write(&path, VALID_TOML).unwrap();
        let manager = ConfigManager::open(path.clone(), EventBus::new()).await.unwrap();
        let result = manager.apply(VALID_TOML.to_string(), "wrong-hash".to_string()).await;
        assert!(matches!(result, Err(ApplyError::HashMismatch { .. })));
    }

    #[tokio::test]
    async fn apply_writes_and_reloads_on_valid_input() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("ananke.toml");
        std::fs::write(&path, VALID_TOML).unwrap();
        let manager = ConfigManager::open(path.clone(), EventBus::new()).await.unwrap();
        let (_current, hash) = manager.raw();
        let new_toml = VALID_TOML.replace("\"demo\"", "\"demo2\"");
        let result = manager.apply(new_toml.clone(), hash).await;
        assert!(matches!(result, Ok(())));
        let (raw_after, _) = manager.raw();
        assert_eq!(raw_after, new_toml);
        let eff = manager.effective();
        assert_eq!(eff.services[0].name.as_str(), "demo2");
    }

    #[tokio::test]
    async fn apply_rejects_invalid_toml() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("ananke.toml");
        std::fs::write(&path, VALID_TOML).unwrap();
        let manager = ConfigManager::open(path.clone(), EventBus::new()).await.unwrap();
        let (_, hash) = manager.raw();
        let bad = "this is not toml";
        let result = manager.apply(bad.to_string(), hash).await;
        assert!(matches!(result, Err(ApplyError::Invalid(_))));
    }
}
```

- [ ] **Step 7: Build + test**

```bash
cargo check --workspace
cargo test --workspace --features test-fakes
cargo clippy --all-targets --features test-fakes -- -D warnings
```

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat(config): ConfigManager owns raw+parsed config and disk I/O"
```

---

## Task 5: Service lifecycle endpoints + `SupervisorCommand` additions

**Files:**
- Create: `ananke/src/api/management/lifecycle.rs`
- Modify: `ananke/src/api/management/mod.rs` (declare + route new handlers), `ananke/src/api/management/handlers.rs` (unchanged list_services + service_detail + list_devices; new handlers in a sibling module), `ananke/src/supervise/mod.rs` (new `Enable` / `Disable` commands + their handlers)
- Test: `ananke/tests/management_lifecycle.rs`

- [ ] **Step 1: Extend `SupervisorCommand`**

In `ananke/src/supervise/mod.rs`:

```rust
#[derive(Debug)]
pub enum SupervisorCommand {
    // ... existing variants ...
    Enable { ack: tokio::sync::oneshot::Sender<EnableResult> },
    Disable { ack: tokio::sync::oneshot::Sender<DisableResult> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnableResult { Enabled, NotDisabled, AlreadyEnabled }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisableResult { Disabled, AlreadyDisabled }
```

Add `SupervisorHandle::enable()` and `SupervisorHandle::disable()` methods that send the commands and await the acks, mirroring the existing `begin_drain` / `fast_kill` shape.

- [ ] **Step 2: Handle Enable/Disable in the `RunLoop`**

Add arm handlers inside `handle_idle`, `handle_disabled`, `handle_failed`, and `handle_active_lifecycle`'s `on_running_command` / `on_starting_command` dispatchers:

```rust
Some(SupervisorCommand::Enable { ack }) => {
    let result = match &self.state {
        ServiceState::Disabled { .. } => {
            let next = transition(&self.state, StateEvent::UserEnable);
            self.set_state(next);
            EnableResult::Enabled
        }
        _ => EnableResult::AlreadyEnabled,
    };
    let _ = ack.send(result);
}
Some(SupervisorCommand::Disable { ack }) => {
    match &self.state {
        ServiceState::Disabled { .. } => {
            let _ = ack.send(DisableResult::AlreadyDisabled);
        }
        ServiceState::Running => {
            // Drain first, then disable.
            send_sigterm_and_wait_logic_here();  // Use the same drain path BeginDrain uses
            let next = transition(&self.state, StateEvent::UserDisable);
            self.set_state(next);
            let _ = ack.send(DisableResult::Disabled);
        }
        _ => {
            let next = transition(&self.state, StateEvent::UserDisable);
            self.set_state(next);
            let _ = ack.send(DisableResult::Disabled);
        }
    }
}
```

The drain path for `Disable` while `Running` is complex enough that it should defer to the existing `BeginDrain` logic internally — refactor note: the existing `BeginDrain` handler body becomes a method `async fn drain_now(&mut self, reason: DrainReason)` so `Disable` can invoke it directly, then transition to Disabled instead of Idle.

- [ ] **Step 3: Create `ananke/src/api/management/lifecycle.rs`**

```rust
//! POST /api/services/{name}/{start,stop,restart,enable,disable} handlers.

use std::time::Duration;

use ananke_api::{ApiError, DisableResponse, EnableResponse, StartResponse, StopResponse};
use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};

use crate::{
    daemon::app_state::AppState,
    supervise::{
        DisableResult, EnableResult, EnsureFailure, EnsureOutcome, SupervisorHandle, await_ensure,
        drain::DrainReason,
    },
};

pub async fn post_start(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    let Some((svc, handle)) = resolve(&state, &name) else {
        return not_found(&name);
    };
    let duration = Duration::from_millis(svc.max_request_duration_ms);
    let body = match await_ensure(&handle, duration).await {
        EnsureOutcome::Ready => StartResponse::AlreadyRunning,
        EnsureOutcome::Failed(EnsureFailure::StartQueueFull) => StartResponse::QueueFull,
        EnsureOutcome::Failed(EnsureFailure::InsufficientVram(reason)) => {
            StartResponse::Unavailable { reason }
        }
        EnsureOutcome::Failed(EnsureFailure::ServiceDisabled(reason)) => {
            StartResponse::Unavailable { reason }
        }
        EnsureOutcome::Failed(EnsureFailure::StartFailed(reason)) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ApiError::new("start_failed", reason)),
            )
                .into_response();
        }
    };
    (StatusCode::ACCEPTED, Json(body)).into_response()
}

pub async fn post_stop(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    let Some((_svc, handle)) = resolve(&state, &name) else {
        return not_found(&name);
    };
    handle.begin_drain(DrainReason::UserKilled).await;
    (StatusCode::ACCEPTED, Json(StopResponse::Drained)).into_response()
}

pub async fn post_restart(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    let Some((svc, handle)) = resolve(&state, &name) else {
        return not_found(&name);
    };
    handle.begin_drain(DrainReason::UserKilled).await;
    let duration = Duration::from_millis(svc.max_request_duration_ms);
    let body = match await_ensure(&handle, duration).await {
        EnsureOutcome::Ready => StartResponse::AlreadyRunning,
        EnsureOutcome::Failed(EnsureFailure::StartQueueFull) => StartResponse::QueueFull,
        EnsureOutcome::Failed(EnsureFailure::InsufficientVram(reason)) => {
            StartResponse::Unavailable { reason }
        }
        EnsureOutcome::Failed(EnsureFailure::ServiceDisabled(reason)) => {
            StartResponse::Unavailable { reason }
        }
        EnsureOutcome::Failed(EnsureFailure::StartFailed(reason)) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ApiError::new("start_failed", reason)),
            )
                .into_response();
        }
    };
    (StatusCode::ACCEPTED, Json(body)).into_response()
}

pub async fn post_enable(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    let Some((_svc, handle)) = resolve(&state, &name) else {
        return not_found(&name);
    };
    let body = match handle.enable().await {
        EnableResult::Enabled => EnableResponse::Enabled,
        EnableResult::NotDisabled | EnableResult::AlreadyEnabled => EnableResponse::AlreadyEnabled,
    };
    (StatusCode::OK, Json(body)).into_response()
}

pub async fn post_disable(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    let Some((_svc, handle)) = resolve(&state, &name) else {
        return not_found(&name);
    };
    let body = match handle.disable().await {
        DisableResult::Disabled => DisableResponse::Disabled,
        DisableResult::AlreadyDisabled => DisableResponse::AlreadyDisabled,
    };
    (StatusCode::OK, Json(body)).into_response()
}

fn resolve(
    state: &AppState,
    name: &str,
) -> Option<(crate::config::validate::ServiceConfig, std::sync::Arc<SupervisorHandle>)> {
    let svc = state
        .config
        .effective()
        .services
        .iter()
        .find(|s| s.name == name)
        .cloned()?;
    let handle = state.registry.get(name)?;
    Some((svc, handle))
}

fn not_found(name: &str) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(ApiError::new(
            "service_not_found",
            format!("service `{name}` not found"),
        )),
    )
        .into_response()
}
```

- [ ] **Step 4: Register the routes**

In `ananke/src/api/management/mod.rs` (which re-exports the module's public router):

```rust
pub mod lifecycle;

use axum::routing::post;

pub fn register(router: axum::Router, state: crate::daemon::app_state::AppState) -> axum::Router {
    let mgmt = axum::Router::new()
        // ... existing routes (list_services, service_detail, list_devices) ...
        .route("/api/services/:name/start", post(lifecycle::post_start))
        .route("/api/services/:name/stop", post(lifecycle::post_stop))
        .route("/api/services/:name/restart", post(lifecycle::post_restart))
        .route("/api/services/:name/enable", post(lifecycle::post_enable))
        .route("/api/services/:name/disable", post(lifecycle::post_disable))
        .with_state(state);
    router.merge(mgmt)
}
```

- [ ] **Step 5: Integration test**

Create `ananke/tests/management_lifecycle.rs`:

```rust
mod common;

use std::time::Duration;

use ananke_api::{DisableResponse, EnableResponse, StartResponse};
use reqwest::Client;

#[tokio::test(flavor = "current_thread")]
async fn start_already_running_returns_already_running() {
    let harness = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let url = harness.management_url("/api/services/demo/start");
    let resp: StartResponse = Client::new()
        .post(url)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    // Test harness uses test-fakes; the service may be already running or
    // kicked off by this call.
    assert!(matches!(
        resp,
        StartResponse::AlreadyRunning | StartResponse::Started { .. }
    ));
    harness.shutdown().await;
}

#[tokio::test(flavor = "current_thread")]
async fn disable_then_enable_roundtrip() {
    let harness = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let d_url = harness.management_url("/api/services/demo/disable");
    let e_url = harness.management_url("/api/services/demo/enable");
    let d: DisableResponse = Client::new().post(d_url).send().await.unwrap().json().await.unwrap();
    assert_eq!(d, DisableResponse::Disabled);
    let e: EnableResponse = Client::new().post(e_url).send().await.unwrap().json().await.unwrap();
    assert_eq!(e, EnableResponse::Enabled);
    harness.shutdown().await;
}

#[tokio::test(flavor = "current_thread")]
async fn start_on_missing_service_404s() {
    let harness = common::build_harness(vec![]).await;
    let url = harness.management_url("/api/services/ghost/start");
    let resp = Client::new().post(url).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::NOT_FOUND);
    harness.shutdown().await;
}

fn _assert_link() {
    // Force ananke-api dependency to link when the test binary is otherwise
    // minimal, so compile errors surface early.
    let _: ananke_api::StartResponse = ananke_api::StartResponse::AlreadyRunning;
}
```

This requires the harness to expose `management_url` + `shutdown` methods. Add them to `tests/common/mod.rs` if not already present — `management_url` returns `format!("http://{}{}", self.mgmt_addr, path)`, `shutdown` drops the harness (signalling cleanup).

- [ ] **Step 6: Build + test**

```bash
cargo test --workspace --features test-fakes -- management_lifecycle
cargo clippy --all-targets --features test-fakes -- -D warnings
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(management): service lifecycle endpoints (start/stop/restart/enable/disable)"
```

---

## Task 6: Paginated logs endpoint

**Files:**
- Create: `ananke/src/api/management/logs.rs`
- Modify: `ananke/src/api/management/mod.rs`
- Test: `ananke/tests/management_logs.rs`

- [ ] **Step 1: Write `ananke/src/api/management/logs.rs`**

```rust
//! `GET /api/services/{name}/logs?since&until&run&limit&stream&before`

use std::collections::HashMap;

use ananke_api::{ApiError, LogLine, LogsResponse};
use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use base64::{Engine, engine::general_purpose::STANDARD as B64};
use serde::{Deserialize, Serialize};

use crate::{daemon::app_state::AppState, db::models::ServiceLog};

const DEFAULT_LIMIT: u32 = 200;
const MAX_LIMIT: u32 = 1000;

#[derive(Debug, Deserialize)]
pub struct LogsQuery {
    pub since: Option<i64>,
    pub until: Option<i64>,
    pub run: Option<i64>,
    pub stream: Option<String>,
    pub limit: Option<u32>,
    pub before: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct Cursor {
    run_id: i64,
    seq: i64,
}

pub async fn get_logs(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Query(q): Query<LogsQuery>,
) -> Response {
    // Resolve service name → service_id via the existing `services` model.
    let service_id = match resolve_service_id(&state, &name).await {
        Some(id) => id,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(ApiError::new(
                    "service_not_found",
                    format!("service `{name}` not found"),
                )),
            )
                .into_response();
        }
    };

    let limit = q.limit.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT) as usize;
    let cursor = match q.before.as_deref().map(decode_cursor) {
        Some(Ok(c)) => Some(c),
        Some(Err(_)) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ApiError::new("invalid_cursor", "malformed `before` cursor")),
            )
                .into_response();
        }
        None => None,
    };

    let mut db = state.db.handle();
    let mut rows: Vec<ServiceLog> =
        ServiceLog::filter(ServiceLog::fields().service_id().eq(service_id))
            .exec(&mut db)
            .await
            .unwrap_or_default();

    // Apply filters in memory. We keep the toasty filter narrow because
    // toasty's DSL does not yet compose the full predicate we need; at
    // worst we hydrate the whole service's log buffer (bounded by retention
    // to 50k lines) and filter.
    rows.retain(|r| {
        q.since.is_none_or(|s| r.timestamp_ms >= s)
            && q.until.is_none_or(|u| r.timestamp_ms <= u)
            && q.run.is_none_or(|run| r.run_id == run)
            && q.stream.as_deref().is_none_or(|s| r.stream == s)
            && cursor.as_ref().is_none_or(|c| (r.run_id, r.seq) < (c.run_id, c.seq))
    });

    // Newest-first sort by (timestamp_ms DESC, seq DESC).
    rows.sort_by(|a, b| b.timestamp_ms.cmp(&a.timestamp_ms).then(b.seq.cmp(&a.seq)));

    let truncated = rows.len() > limit;
    rows.truncate(limit);

    let next_cursor = if truncated {
        rows.last().map(|r| encode_cursor(&Cursor { run_id: r.run_id, seq: r.seq }))
    } else {
        None
    };

    let logs: Vec<LogLine> = rows
        .into_iter()
        .map(|r| LogLine {
            timestamp_ms: r.timestamp_ms,
            stream: r.stream,
            line: r.line,
            run_id: r.run_id,
            seq: r.seq,
        })
        .collect();

    (StatusCode::OK, Json(LogsResponse { logs, next_cursor })).into_response()
}

async fn resolve_service_id(state: &AppState, name: &str) -> Option<i64> {
    use crate::db::models::Service;
    let mut handle = state.db.handle();
    Service::filter_by_name(name.to_string())
        .first()
        .exec(&mut handle)
        .await
        .ok()
        .flatten()
        .map(|s| s.service_id as i64)
}

fn encode_cursor(c: &Cursor) -> String {
    let json = serde_json::to_string(c).expect("cursor serialise");
    B64.encode(json)
}

fn decode_cursor(s: &str) -> Result<Cursor, ()> {
    let bytes = B64.decode(s).map_err(|_| ())?;
    serde_json::from_slice(&bytes).map_err(|_| ())
}

fn _use_hashmap_placeholder() -> HashMap<u8, u8> {
    HashMap::new()
}
```

(Remove the `HashMap` placeholder at the bottom; it's there to silence an unused-import drift on some editor configs — actually delete it outright to keep the file clean.)

- [ ] **Step 2: Register the route**

Append to the existing `Router::merge` block in `ananke/src/api/management/mod.rs`:

```rust
.route("/api/services/:name/logs", axum::routing::get(logs::get_logs))
```

- [ ] **Step 3: Integration test**

Create `ananke/tests/management_logs.rs`:

```rust
mod common;

use ananke_api::LogsResponse;

#[tokio::test(flavor = "current_thread")]
async fn returns_empty_for_idle_service() {
    let harness = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let url = harness.management_url("/api/services/demo/logs");
    let resp: LogsResponse = reqwest::get(url).await.unwrap().json().await.unwrap();
    assert!(resp.logs.is_empty());
    assert!(resp.next_cursor.is_none());
    harness.shutdown().await;
}

#[tokio::test(flavor = "current_thread")]
async fn rejects_malformed_cursor() {
    let harness = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let url = harness.management_url("/api/services/demo/logs?before=notbase64");
    let status = reqwest::get(url).await.unwrap().status();
    assert_eq!(status, reqwest::StatusCode::BAD_REQUEST);
    harness.shutdown().await;
}
```

- [ ] **Step 4: Build + test**

```bash
cargo test --workspace --features test-fakes -- management_logs
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(management): GET /api/services/{name}/logs paginated endpoint"
```

---

## Task 7: Config endpoints

**Files:**
- Create: `ananke/src/api/management/config.rs`
- Modify: `ananke/src/api/management/mod.rs`
- Test: `ananke/tests/management_config.rs`

- [ ] **Step 1: Write `ananke/src/api/management/config.rs`**

```rust
//! GET/PUT /api/config + POST /api/config/validate

use ananke_api::{
    ApiError, ConfigResponse, ConfigValidateRequest, ConfigValidateResponse, ValidationError,
};
use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode, header::IF_MATCH},
    response::{IntoResponse, Response},
};

use crate::{
    config::manager::{ApplyError, ConfigHash},
    daemon::app_state::AppState,
};

pub async fn get_config(State(state): State<AppState>) -> Response {
    let (content, hash) = state.config.raw();
    (StatusCode::OK, Json(ConfigResponse { content, hash })).into_response()
}

pub async fn put_config(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: String,
) -> Response {
    let Some(if_match) = headers
        .get(IF_MATCH)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim_matches('"').to_string())
    else {
        return (
            StatusCode::PRECONDITION_REQUIRED,
            Json(ApiError::new(
                "if_match_required",
                "PUT /api/config requires an If-Match header with the current config hash",
            )),
        )
            .into_response();
    };
    match state.config.apply(body, if_match).await {
        Ok(()) => StatusCode::ACCEPTED.into_response(),
        Err(ApplyError::HashMismatch { server_hash }) => {
            let mut resp = (
                StatusCode::PRECONDITION_FAILED,
                Json(ApiError::new(
                    "hash_mismatch",
                    format!("config was modified since last GET; current hash is {server_hash}"),
                )),
            )
                .into_response();
            resp.headers_mut()
                .insert(axum::http::header::ETAG, server_hash.parse().unwrap());
            resp
        }
        Err(ApplyError::Invalid(errors)) => {
            let body = ConfigValidateResponse {
                valid: false,
                errors,
            };
            (StatusCode::UNPROCESSABLE_ENTITY, Json(body)).into_response()
        }
        Err(ApplyError::PersistFailed(io_err)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiError::new(
                "persist_failed",
                format!("writing config to disk failed: {io_err}"),
            )),
        )
            .into_response(),
    }
}

pub async fn post_validate(
    State(state): State<AppState>,
    Json(req): Json<ConfigValidateRequest>,
) -> Response {
    match state.config.validate(&req.content) {
        Ok(()) => Json(ConfigValidateResponse { valid: true, errors: vec![] }).into_response(),
        Err(errors) => Json(ConfigValidateResponse { valid: false, errors }).into_response(),
    }
}

#[cfg(test)]
#[allow(dead_code)]
fn _force_link() {
    let _: Vec<ValidationError> = vec![];
}
```

- [ ] **Step 2: Register the routes**

In `ananke/src/api/management/mod.rs`:

```rust
use axum::routing::{get, post, put};
// ...
.route("/api/config", get(config::get_config).put(config::put_config))
.route("/api/config/validate", post(config::post_validate))
```

- [ ] **Step 3: Integration test**

Create `ananke/tests/management_config.rs`:

```rust
mod common;

use ananke_api::ConfigResponse;
use reqwest::{Client, header::IF_MATCH};

#[tokio::test(flavor = "current_thread")]
async fn get_config_returns_content_and_hash() {
    let harness = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let url = harness.management_url("/api/config");
    let resp: ConfigResponse = reqwest::get(url).await.unwrap().json().await.unwrap();
    assert!(!resp.content.is_empty());
    assert!(!resp.hash.is_empty());
    harness.shutdown().await;
}

#[tokio::test(flavor = "current_thread")]
async fn put_without_if_match_is_428() {
    let harness = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let url = harness.management_url("/api/config");
    let resp = Client::new().put(url).body("").send().await.unwrap();
    assert_eq!(resp.status().as_u16(), 428);
    harness.shutdown().await;
}

#[tokio::test(flavor = "current_thread")]
async fn put_with_wrong_hash_is_412() {
    let harness = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let url = harness.management_url("/api/config");
    let resp = Client::new()
        .put(url)
        .header(IF_MATCH, "\"wrong\"")
        .body("")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 412);
    harness.shutdown().await;
}
```

- [ ] **Step 4: Build + test**

```bash
cargo test --workspace --features test-fakes -- management_config
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(management): config GET/PUT/validate endpoints"
```

---

## Task 8: `/api/events` WebSocket

**Files:**
- Create: `ananke/src/api/management/events_ws.rs`
- Modify: `ananke/src/api/management/mod.rs`
- Test: `ananke/tests/events_ws.rs`

- [ ] **Step 1: Write `ananke/src/api/management/events_ws.rs`**

```rust
//! `GET /api/events` WebSocket — system-wide event bus.

use std::time::Duration;

use ananke_api::Event;
use axum::{
    extract::{
        Query, State, WebSocketUpgrade,
        ws::{Message, WebSocket},
    },
    response::Response,
};
use serde::Deserialize;
use tokio::{select, sync::broadcast::error::RecvError};
use tracing::warn;

use crate::daemon::app_state::AppState;

#[derive(Debug, Deserialize)]
pub struct EventsQuery {
    pub service: Option<String>,
}

pub async fn get_events_ws(
    State(state): State<AppState>,
    Query(q): Query<EventsQuery>,
    ws: WebSocketUpgrade,
) -> Response {
    ws.on_upgrade(move |socket| serve_events(socket, state, q.service))
}

async fn serve_events(mut socket: WebSocket, state: AppState, service_filter: Option<String>) {
    let mut rx = state.events.subscribe();
    loop {
        select! {
            recv = rx.recv() => match recv {
                Ok(event) => {
                    if !passes_filter(&event, service_filter.as_deref()) { continue; }
                    let json = match serde_json::to_string(&event) {
                        Ok(s) => s,
                        Err(e) => { warn!(error = %e, "event serialise failed"); continue; }
                    };
                    if socket.send(Message::Text(json)).await.is_err() { return; }
                }
                Err(RecvError::Lagged(n)) => {
                    let overflow = Event::Overflow { dropped: n };
                    if let Ok(s) = serde_json::to_string(&overflow)
                        && socket.send(Message::Text(s)).await.is_err()
                    {
                        return;
                    }
                }
                Err(RecvError::Closed) => return,
            },
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Close(_))) | None => return,
                    Some(Ok(Message::Ping(p))) => {
                        let _ = socket.send(Message::Pong(p)).await;
                    }
                    _ => {}
                }
            }
            _ = tokio::time::sleep(Duration::from_secs(30)) => {
                // Heartbeat; keeps intermediaries from closing an idle WS.
                if socket.send(Message::Ping(vec![])).await.is_err() { return; }
            }
        }
    }
}

fn passes_filter(event: &Event, service_filter: Option<&str>) -> bool {
    let Some(filter) = service_filter else {
        return true;
    };
    match event {
        Event::StateChanged { service, .. }
        | Event::AllocationChanged { service, .. }
        | Event::EstimatorDrift { service, .. } => service.as_str() == filter,
        Event::ConfigReloaded { .. } | Event::Overflow { .. } => true,
    }
}
```

- [ ] **Step 2: Register the route**

```rust
use axum::routing::any;
// ...
.route("/api/events", any(events_ws::get_events_ws))
```

(`any` because the route accepts the WS upgrade request's GET with upgrade headers; `get` works too in axum 0.7 but `any` is more explicit about the upgrade being orthogonal to HTTP verb.)

- [ ] **Step 3: Integration test**

Create `ananke/tests/events_ws.rs`:

```rust
mod common;

use ananke_api::Event;
use futures::{SinkExt, StreamExt};
use tokio_tungstenite::{
    connect_async, tungstenite::protocol::Message,
};

#[tokio::test(flavor = "current_thread")]
async fn state_change_fires_event() {
    let harness = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let url = harness.management_url("/api/events").replace("http://", "ws://");
    let (mut ws, _) = connect_async(url).await.unwrap();
    // Trigger a state change via the start endpoint.
    let start_url = harness.management_url("/api/services/demo/start");
    tokio::spawn(async move {
        let _ = reqwest::Client::new().post(start_url).send().await;
    });
    // Expect at least one StateChanged event within a short window.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut saw_state_change = false;
    while std::time::Instant::now() < deadline {
        let recv = tokio::time::timeout(std::time::Duration::from_millis(500), ws.next()).await;
        if let Ok(Some(Ok(Message::Text(s)))) = recv {
            let event: Event = serde_json::from_str(&s).unwrap();
            if matches!(event, Event::StateChanged { .. }) {
                saw_state_change = true;
                break;
            }
        }
    }
    assert!(saw_state_change, "expected a StateChanged event");
    let _ = ws.send(Message::Close(None)).await;
    harness.shutdown().await;
}
```

Add `tokio-tungstenite` to `ananke`'s dev-deps in `ananke/Cargo.toml`:

```toml
[dev-dependencies]
# ... existing ...
tokio-tungstenite = { version = "0.26", default-features = false, features = ["connect"] }
```

- [ ] **Step 4: Build + test**

```bash
cargo test --workspace --features test-fakes -- events_ws
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(management): /api/events WebSocket event bus"
```

---

## Task 9: Log-stream WebSocket + batcher broadcast

**Files:**
- Create: `ananke/src/api/logs_ws.rs`
- Modify: `ananke/src/db/logs.rs` (add `subscribe()` to `BatcherHandle`; broadcast lines on flush), `ananke/src/api/mod.rs` (declare + register the module)
- Test: `ananke/tests/logs_ws.rs`

- [ ] **Step 1: Add `broadcast::Sender<LogLine>` to the batcher**

In `ananke/src/db/logs.rs`, carry a `broadcast::Sender<ananke_api::LogLine>` through the batcher:

```rust
use tokio::sync::broadcast;

const LOG_BROADCAST_CAPACITY: usize = 256;

pub struct BatcherHandle {
    tx: mpsc::UnboundedSender<Msg>,
    broadcast: broadcast::Sender<ananke_api::LogLine>,
}

impl BatcherHandle {
    pub fn subscribe(&self) -> broadcast::Receiver<ananke_api::LogLine> {
        self.broadcast.subscribe()
    }
}

pub fn spawn(db: Database) -> BatcherHandle {
    let (tx, rx) = mpsc::unbounded_channel();
    let (broadcast_tx, _) = broadcast::channel(LOG_BROADCAST_CAPACITY);
    tokio::spawn(run(db, rx, broadcast_tx.clone()));
    BatcherHandle { tx, broadcast: broadcast_tx }
}
```

In `run`, pass the broadcast sender into `flush`:

```rust
async fn run(
    db: Database,
    mut rx: mpsc::UnboundedReceiver<Msg>,
    broadcast: broadcast::Sender<ananke_api::LogLine>,
) { /* ...existing body, calling flush(...broadcast) on every flush path... */ }

async fn flush(
    db: &Database,
    buffer: &mut Vec<LogLine>,
    seq: &mut HashMap<(i64, i64), i64>,
    broadcast: &broadcast::Sender<ananke_api::LogLine>,
) {
    // existing pre-logic ...
    for line in &lines_with_seq {
        let api_line = ananke_api::LogLine {
            timestamp_ms: line.timestamp_ms,
            stream: line.stream.as_str().to_string(),
            line: line.line.clone(),
            run_id: line.run_id,
            seq: line.seq,
        };
        let _ = broadcast.send(api_line);
    }
    // existing DB insert ...
}
```

The existing flush logic already does a `seq.entry((...)).or_insert(0)` + increment; capture the (service_id, run_id, seq) tuple into a `lines_with_seq: Vec<LogLine>` before the DB write and iterate over it for both DB insert and broadcast.

- [ ] **Step 2: Write `ananke/src/api/logs_ws.rs`**

```rust
//! `GET /api/services/{name}/logs/stream` WebSocket — live log tail.

use std::time::Duration;

use ananke_api::{ApiError, Event};
use axum::{
    Json,
    extract::{
        Path, State, WebSocketUpgrade,
        ws::{Message, WebSocket},
    },
    http::StatusCode,
    response::{IntoResponse, Response},
};
use tokio::{select, sync::broadcast::error::RecvError};
use tracing::warn;

use crate::daemon::app_state::AppState;

pub async fn get_logs_ws(
    State(state): State<AppState>,
    Path(name): Path<String>,
    ws: WebSocketUpgrade,
) -> Response {
    let Some(service_id) = resolve_service_id(&state, &name).await else {
        return (
            StatusCode::NOT_FOUND,
            Json(ApiError::new(
                "service_not_found",
                format!("service `{name}` not found"),
            )),
        )
            .into_response();
    };
    let rx = state.batcher.subscribe();
    ws.on_upgrade(move |socket| serve(socket, service_id, rx))
}

async fn serve(
    mut socket: WebSocket,
    service_id: i64,
    mut rx: tokio::sync::broadcast::Receiver<ananke_api::LogLine>,
) {
    loop {
        select! {
            recv = rx.recv() => match recv {
                Ok(line) if line_matches(&line, service_id) => {
                    let Ok(json) = serde_json::to_string(&line) else { continue; };
                    if socket.send(Message::Text(json)).await.is_err() { return; }
                }
                Ok(_) => {}
                Err(RecvError::Lagged(n)) => {
                    let frame = serde_json::json!({"type": "overflow", "dropped": n});
                    let _ = socket.send(Message::Text(frame.to_string())).await;
                }
                Err(RecvError::Closed) => return,
            },
            msg = socket.recv() => match msg {
                Some(Ok(Message::Close(_))) | None => return,
                Some(Ok(Message::Ping(p))) => { let _ = socket.send(Message::Pong(p)).await; }
                _ => {}
            },
            _ = tokio::time::sleep(Duration::from_secs(30)) => {
                if socket.send(Message::Ping(vec![])).await.is_err() { return; }
            }
        }
    }
}

fn line_matches(line: &ananke_api::LogLine, service_id: i64) -> bool {
    // Broadcast carries LogLines for all services; filter by service_id.
    // `line` uses the ananke_api type which stores run_id + seq but not
    // service_id; we've already filtered upstream by attaching a service tag
    // during publish if needed. For phase 5, the batcher's broadcast sends
    // every line; each subscriber filters.
    //
    // Since LogLine does not carry service_id, the subscribe path takes
    // service_id as a cookie and the batcher must include it in the
    // broadcast payload. Use a wrapper type instead.
    //
    // See corresponding change in db::logs — the broadcast channel ships
    // `(service_id, LogLine)` tuples.
    let _ = service_id;
    let _ = line;
    true  // placeholder; real filter handled by wrapper type below
}
```

The placeholder at the end signals a design decision: `ananke_api::LogLine` doesn't carry `service_id`, so the broadcast channel must ship a tuple `(i64, ananke_api::LogLine)`. Update the batcher accordingly:

Revised step 1 addendum — broadcast channel type is `broadcast::Sender<(i64, ananke_api::LogLine)>`, send both the service_id and the line:

```rust
let _ = broadcast.send((line.service_id, api_line));
```

And in the WS subscriber, filter on the tuple:

```rust
Ok((sid, line)) if sid == service_id => { /* send line to ws */ }
Ok(_) => continue,
```

Update the handler accordingly.

- [ ] **Step 3: Register the route**

Add `pub mod logs_ws;` to `ananke/src/api/mod.rs`, then in the router setup function (create one if it doesn't exist — `api/mod.rs` is a natural place), register:

```rust
.route("/api/services/:name/logs/stream", axum::routing::any(logs_ws::get_logs_ws))
```

- [ ] **Step 4: Integration test**

Create `ananke/tests/logs_ws.rs` — similar shape to `events_ws.rs`, but trigger a child spawn that prints and assert the WS receives the line within a timeout.

Detailed test:

```rust
mod common;

use ananke_api::LogLine;
use futures::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

#[tokio::test(flavor = "current_thread")]
async fn live_line_arrives_on_tail() {
    let harness = common::build_harness(vec![common::minimal_llama_service("demo", 0)]).await;
    let url = harness
        .management_url("/api/services/demo/logs/stream")
        .replace("http://", "ws://");
    let (mut ws, _) = connect_async(url).await.unwrap();

    // Drive a log line by directly pushing through the batcher (the
    // test-fakes spawn_child writes "ready\n" to stdout on boot, but that's
    // timing-sensitive; we push synthetically instead).
    let batcher = harness.state.batcher.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        batcher.push(ananke::db::logs::LogLine {
            service_id: 1,
            run_id: 1,
            timestamp_ms: 0,
            stream: ananke::db::logs::Stream::Stdout,
            line: "hello from test".into(),
        });
        batcher.flush().await;
    });

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    let mut received = None;
    while std::time::Instant::now() < deadline {
        let msg = tokio::time::timeout(std::time::Duration::from_millis(500), ws.next()).await;
        if let Ok(Some(Ok(Message::Text(s)))) = msg
            && let Ok(line) = serde_json::from_str::<LogLine>(&s)
        {
            received = Some(line);
            break;
        }
    }
    assert!(received.is_some(), "expected at least one log line");
    let _ = ws.send(Message::Close(None)).await;
    harness.shutdown().await;
}
```

- [ ] **Step 5: Build + test**

```bash
cargo test --workspace --features test-fakes -- logs_ws
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(api): /api/services/{name}/logs/stream WebSocket tail"
```

---

## Task 10: External-access gate

**Files:**
- Modify: `ananke/src/config/parse.rs` (new `allow_external_management` field), `ananke/src/config/validate.rs` (validation rule), `ananke/src/daemon/mod.rs` (startup WARN when non-loopback)
- Test: `ananke/src/config/validate.rs` inline unit test

- [ ] **Step 1: Add the field to `RawDaemon`**

In `ananke/src/config/parse.rs`:

```rust
#[serde(default)]
pub allow_external_management: bool,
```

- [ ] **Step 2: Validate**

In `ananke/src/config/validate.rs`, after parsing `management_listen` into a `SocketAddr`:

```rust
let mgmt_addr: std::net::SocketAddr = effective.daemon.management_listen.parse()
    .map_err(|e: std::net::AddrParseError| {
        ExpectedError::config_unparseable(path.to_path_buf(),
            format!("daemon.management_listen: {e}"))
    })?;
if !mgmt_addr.ip().is_loopback() && !cfg.daemon.allow_external_management {
    return Err(ExpectedError::config_unparseable(
        path.to_path_buf(),
        "daemon.management_listen is non-loopback but daemon.allow_external_management is false; \
         see §11 of the spec before enabling this — the management API has no authentication".into(),
    ));
}
```

- [ ] **Step 3: Add the carrier to `DaemonSettings`**

Extend `DaemonSettings` with `pub allow_external_management: bool`.

- [ ] **Step 4: Startup warning**

In `ananke/src/daemon/mod.rs::run`, after the listener is bound:

```rust
if !mgmt_addr.ip().is_loopback() {
    warn!(
        bind = %mgmt_addr,
        "management API reachable from the network — no authentication enabled; \
         trust your network perimeter (e.g. Tailscale) or terminate TLS + auth at a reverse proxy"
    );
}
```

- [ ] **Step 5: Unit test**

In `ananke/src/config/validate.rs`'s `tests` module:

```rust
#[test]
fn non_loopback_without_flag_is_rejected() {
    let toml = r#"
[daemon]
management_listen = "0.0.0.0:17777"
openai_listen = "127.0.0.1:0"
"#;
    let raw: RawConfig = toml::from_str(toml).unwrap();
    let result = validate(&raw);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("allow_external_management"));
}

#[test]
fn non_loopback_with_flag_is_accepted() {
    let toml = r#"
[daemon]
management_listen = "0.0.0.0:17777"
openai_listen = "127.0.0.1:0"
allow_external_management = true
"#;
    let raw: RawConfig = toml::from_str(toml).unwrap();
    assert!(validate(&raw).is_ok());
}
```

(Exact validate function signature: check the current code; may need to go through the merge + validate pipeline. Adjust test construction as needed.)

- [ ] **Step 6: Build + test**

```bash
cargo test --workspace --features test-fakes
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(config): opt-in external management binding behind allow_external_management"
```

---

## Task 11: `anankectl` scaffolding + read commands

**Files:**
- Create: `anankectl/src/main.rs`, `anankectl/src/client.rs`, `anankectl/src/output.rs`, `anankectl/src/commands/mod.rs`, `anankectl/src/commands/devices.rs`, `anankectl/src/commands/services.rs`, `anankectl/src/commands/show.rs`
- Test: `anankectl/tests/smoke.rs`

- [ ] **Step 1: Write `anankectl/src/main.rs`**

```rust
use std::process::ExitCode;

use clap::{Parser, Subcommand};

mod client;
mod commands;
mod output;

#[derive(Parser)]
#[command(name = "anankectl", version)]
struct Cli {
    /// Base URL for the management API.
    #[arg(long, global = true, env = "ANANKE_ENDPOINT", default_value = "http://127.0.0.1:17777")]
    endpoint: String,

    /// Emit responses as raw JSON instead of formatted text.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List devices with reservations.
    Devices,
    /// List services.
    Services {
        /// Include disabled services.
        #[arg(long)]
        all: bool,
    },
    /// Show service detail.
    Show {
        /// Service name.
        name: String,
    },
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    let client = client::ApiClient::new(&cli.endpoint);
    let result = match cli.command {
        Command::Devices => commands::devices::run(&client, cli.json).await,
        Command::Services { all } => commands::services::run(&client, cli.json, all).await,
        Command::Show { name } => commands::show::run(&client, cli.json, &name).await,
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("anankectl: {e}");
            e.exit_code()
        }
    }
}
```

- [ ] **Step 2: Write `anankectl/src/client.rs`**

```rust
use std::process::ExitCode;

use reqwest::{StatusCode, Url};
use serde::de::DeserializeOwned;

pub struct ApiClient {
    pub endpoint: Url,
    http: reqwest::Client,
}

#[derive(Debug)]
pub enum ApiClientError {
    Connect(reqwest::Error),
    Http { status: StatusCode, body: String },
    Parse(String),
    Usage(String),
}

impl std::fmt::Display for ApiClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Connect(e) => write!(f, "connection error: {e}"),
            Self::Http { status, body } => write!(f, "HTTP {status}: {body}"),
            Self::Parse(e) => write!(f, "parse error: {e}"),
            Self::Usage(e) => write!(f, "usage error: {e}"),
        }
    }
}

impl std::error::Error for ApiClientError {}

impl ApiClientError {
    pub fn exit_code(&self) -> ExitCode {
        match self {
            Self::Usage(_) => ExitCode::from(2),
            Self::Connect(_) => ExitCode::from(3),
            _ => ExitCode::from(1),
        }
    }
}

impl ApiClient {
    pub fn new(endpoint: &str) -> Self {
        let endpoint = Url::parse(endpoint).expect("valid --endpoint URL");
        Self { endpoint, http: reqwest::Client::new() }
    }

    pub async fn get_json<T: DeserializeOwned>(&self, path: &str) -> Result<T, ApiClientError> {
        let url = self.endpoint.join(path).expect("valid path");
        let resp = self.http.get(url).send().await.map_err(ApiClientError::Connect)?;
        self.read_json(resp).await
    }

    pub async fn post_json<T: DeserializeOwned, B: serde::Serialize>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T, ApiClientError> {
        let url = self.endpoint.join(path).expect("valid path");
        let resp = self.http.post(url).json(body).send().await.map_err(ApiClientError::Connect)?;
        self.read_json(resp).await
    }

    pub async fn post_empty<T: DeserializeOwned>(&self, path: &str) -> Result<T, ApiClientError> {
        let url = self.endpoint.join(path).expect("valid path");
        let resp = self.http.post(url).send().await.map_err(ApiClientError::Connect)?;
        self.read_json(resp).await
    }

    pub async fn delete(&self, path: &str) -> Result<(), ApiClientError> {
        let url = self.endpoint.join(path).expect("valid path");
        let resp = self.http.delete(url).send().await.map_err(ApiClientError::Connect)?;
        if resp.status().is_success() {
            Ok(())
        } else {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            Err(ApiClientError::Http { status, body })
        }
    }

    pub async fn put_body(
        &self,
        path: &str,
        body: String,
        if_match: Option<&str>,
    ) -> Result<(), ApiClientError> {
        let url = self.endpoint.join(path).expect("valid path");
        let mut req = self.http.put(url).body(body);
        if let Some(h) = if_match {
            req = req.header(reqwest::header::IF_MATCH, format!("\"{h}\""));
        }
        let resp = req.send().await.map_err(ApiClientError::Connect)?;
        if resp.status().is_success() {
            Ok(())
        } else {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            Err(ApiClientError::Http { status, body })
        }
    }

    async fn read_json<T: DeserializeOwned>(&self, resp: reqwest::Response) -> Result<T, ApiClientError> {
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ApiClientError::Http { status, body });
        }
        resp.json::<T>().await.map_err(|e| ApiClientError::Parse(e.to_string()))
    }
}
```

- [ ] **Step 3: Write `anankectl/src/output.rs`**

```rust
//! Text / JSON formatters for CLI subcommands.

use ananke_api::{DeviceSummary, ServiceDetail, ServiceSummary};
use serde::Serialize;

pub fn print_json<T: Serialize>(value: &T) {
    match serde_json::to_string_pretty(value) {
        Ok(s) => println!("{s}"),
        Err(e) => eprintln!("failed to serialise response: {e}"),
    }
}

pub fn print_devices_table(devices: &[DeviceSummary]) {
    println!("{:<10} {:<28} {:>12} {:>12}   RESERVATIONS", "ID", "NAME", "TOTAL", "FREE");
    for d in devices {
        let total_gib = d.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let free_gib = d.free_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let resv = d
            .reservations
            .iter()
            .map(|r| format!("{}: {} MiB", r.service, r.bytes / (1024 * 1024)))
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "{:<10} {:<28} {:>10.1}G {:>10.1}G   {}",
            d.id, d.name, total_gib, free_gib, resv
        );
    }
}

pub fn print_services_table(services: &[ServiceSummary], show_all: bool) {
    let rows: Vec<&ServiceSummary> = services
        .iter()
        .filter(|s| show_all || !s.state.starts_with("disabled"))
        .collect();
    println!("{:<24} {:<12} {:<12} {:>4} {:>6} {:>8}", "NAME", "STATE", "LIFECYCLE", "PRI", "PORT", "PID");
    for s in rows {
        println!(
            "{:<24} {:<12} {:<12} {:>4} {:>6} {:>8}",
            s.name,
            s.state,
            s.lifecycle,
            s.priority,
            s.port,
            s.pid.map(|p| p.to_string()).unwrap_or_else(|| "—".into()),
        );
    }
}

pub fn print_service_detail(detail: &ServiceDetail) {
    println!("{}", detail.name);
    println!("  state:     {}", detail.state);
    println!("  lifecycle: {}", detail.lifecycle);
    println!("  priority:  {}", detail.priority);
    println!("  template:  {}", detail.template);
    println!("  port:      {} (private {})", detail.port, detail.private_port);
    if let Some(run_id) = detail.run_id {
        println!("  run_id:    {run_id}");
    }
    if let Some(pid) = detail.pid {
        println!("  pid:       {pid}");
    }
    if !detail.placement_override.is_empty() {
        println!("  placement_override:");
        for (k, v) in &detail.placement_override {
            println!("    {k} = {v}");
        }
    }
    if !detail.recent_logs.is_empty() {
        println!("  recent logs (last {}):", detail.recent_logs.len());
        for line in detail.recent_logs.iter().rev().take(10).rev() {
            println!("    [{}] {}", line.stream, line.line);
        }
    }
}
```

- [ ] **Step 4: Write `anankectl/src/commands/{mod,devices,services,show}.rs`**

`commands/mod.rs`:

```rust
pub mod devices;
pub mod services;
pub mod show;
```

`commands/devices.rs`:

```rust
use ananke_api::DeviceSummary;

use crate::{client::{ApiClient, ApiClientError}, output};

pub async fn run(client: &ApiClient, json: bool) -> Result<(), ApiClientError> {
    let devices: Vec<DeviceSummary> = client.get_json("/api/devices").await?;
    if json {
        output::print_json(&devices);
    } else {
        output::print_devices_table(&devices);
    }
    Ok(())
}
```

`commands/services.rs`:

```rust
use ananke_api::ServiceSummary;

use crate::{client::{ApiClient, ApiClientError}, output};

pub async fn run(client: &ApiClient, json: bool, all: bool) -> Result<(), ApiClientError> {
    let services: Vec<ServiceSummary> = client.get_json("/api/services").await?;
    if json {
        output::print_json(&services);
    } else {
        output::print_services_table(&services, all);
    }
    Ok(())
}
```

`commands/show.rs`:

```rust
use ananke_api::ServiceDetail;

use crate::{client::{ApiClient, ApiClientError}, output};

pub async fn run(client: &ApiClient, json: bool, name: &str) -> Result<(), ApiClientError> {
    let path = format!("/api/services/{}", name);
    let detail: ServiceDetail = client.get_json(&path).await?;
    if json {
        output::print_json(&detail);
    } else {
        output::print_service_detail(&detail);
    }
    Ok(())
}
```

- [ ] **Step 5: Smoke test**

Create `anankectl/tests/smoke.rs`:

```rust
// Smoke test: --help runs without error. Richer integration tests are added
// per-subcommand as those commands are implemented.

use std::process::Command;

#[test]
fn help_works() {
    let output = Command::new(env!("CARGO_BIN_EXE_anankectl"))
        .arg("--help")
        .output()
        .expect("spawn");
    assert!(output.status.success(), "--help exit: {:?}", output.status);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("devices"));
    assert!(stdout.contains("services"));
}
```

- [ ] **Step 6: Build + test**

```bash
cargo check --workspace
cargo test -p anankectl
cargo clippy --all-targets --features test-fakes -- -D warnings
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(cli): anankectl scaffolding + devices/services/show commands"
```

---

## Task 12: `anankectl` lifecycle commands

**Files:**
- Create: `anankectl/src/commands/lifecycle.rs`
- Modify: `anankectl/src/main.rs` (add enum variants), `anankectl/src/commands/mod.rs`

- [ ] **Step 1: Extend CLI enum**

In `anankectl/src/main.rs::Command`:

```rust
Start { name: String },
Stop { name: String },
Restart { name: String },
Enable { name: String },
Disable { name: String },
Retry { name: String },
```

And in the dispatch match:

```rust
Command::Start { name } => commands::lifecycle::start(&client, cli.json, &name).await,
Command::Stop { name } => commands::lifecycle::stop(&client, cli.json, &name).await,
Command::Restart { name } => commands::lifecycle::restart(&client, cli.json, &name).await,
Command::Enable { name } => commands::lifecycle::enable(&client, cli.json, &name).await,
Command::Disable { name } => commands::lifecycle::disable(&client, cli.json, &name).await,
Command::Retry { name } => commands::lifecycle::retry(&client, cli.json, &name).await,
```

- [ ] **Step 2: Write `anankectl/src/commands/lifecycle.rs`**

```rust
use ananke_api::{DisableResponse, EnableResponse, StartResponse, StopResponse};

use crate::{client::{ApiClient, ApiClientError}, output};

pub async fn start(client: &ApiClient, json: bool, name: &str) -> Result<(), ApiClientError> {
    let resp: StartResponse = client.post_empty(&format!("/api/services/{name}/start")).await?;
    report_start(&resp, json);
    Ok(())
}

pub async fn stop(client: &ApiClient, json: bool, name: &str) -> Result<(), ApiClientError> {
    let resp: StopResponse = client.post_empty(&format!("/api/services/{name}/stop")).await?;
    if json { output::print_json(&resp); } else {
        match resp {
            StopResponse::Drained => println!("ok: stopped '{name}'"),
            StopResponse::NotRunning => println!("noop: '{name}' was not running"),
        }
    }
    Ok(())
}

pub async fn restart(client: &ApiClient, json: bool, name: &str) -> Result<(), ApiClientError> {
    let resp: StartResponse = client.post_empty(&format!("/api/services/{name}/restart")).await?;
    report_start(&resp, json);
    Ok(())
}

pub async fn enable(client: &ApiClient, json: bool, name: &str) -> Result<(), ApiClientError> {
    let resp: EnableResponse = client.post_empty(&format!("/api/services/{name}/enable")).await?;
    if json { output::print_json(&resp); } else {
        match resp {
            EnableResponse::Enabled => println!("ok: enabled '{name}'"),
            EnableResponse::NotDisabled | EnableResponse::AlreadyEnabled => {
                println!("noop: '{name}' was not disabled");
            }
        }
    }
    Ok(())
}

pub async fn disable(client: &ApiClient, json: bool, name: &str) -> Result<(), ApiClientError> {
    let resp: DisableResponse = client.post_empty(&format!("/api/services/{name}/disable")).await?;
    if json { output::print_json(&resp); } else {
        match resp {
            DisableResponse::Disabled => println!("ok: disabled '{name}'"),
            DisableResponse::AlreadyDisabled => println!("noop: '{name}' was already disabled"),
        }
    }
    Ok(())
}

pub async fn retry(client: &ApiClient, json: bool, name: &str) -> Result<(), ApiClientError> {
    // Best-effort enable (idempotent if not disabled), then start.
    let _ = client.post_empty::<EnableResponse>(&format!("/api/services/{name}/enable")).await;
    start(client, json, name).await
}

fn report_start(resp: &StartResponse, json: bool) {
    if json { output::print_json(resp); return; }
    match resp {
        StartResponse::AlreadyRunning => println!("ok: already running"),
        StartResponse::Started { run_id } => println!("ok: started (run_id={run_id})"),
        StartResponse::QueueFull => println!("error: start queue full"),
        StartResponse::Unavailable { reason } => println!("error: unavailable ({reason})"),
    }
}
```

- [ ] **Step 3: Register module**

Add `pub mod lifecycle;` to `anankectl/src/commands/mod.rs`.

- [ ] **Step 4: Build + test**

```bash
cargo check --workspace
cargo clippy --all-targets --features test-fakes -- -D warnings
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(cli): anankectl lifecycle commands (start/stop/restart/enable/disable/retry)"
```

---

## Task 13: `anankectl` logs + `--follow`

**Files:**
- Create: `anankectl/src/commands/logs.rs`
- Modify: `anankectl/src/main.rs`, `anankectl/src/commands/mod.rs`

- [ ] **Step 1: Extend CLI**

```rust
/// Tail logs for a service.
Logs {
    /// Service name.
    name: String,
    /// Follow new lines as they arrive.
    #[arg(long)]
    follow: bool,
    /// Filter to a specific run id.
    #[arg(long)]
    run: Option<i64>,
    /// Minimum timestamp (ms since epoch).
    #[arg(long)]
    since: Option<i64>,
    /// Maximum timestamp (ms since epoch).
    #[arg(long)]
    until: Option<i64>,
    /// Cap on number of historical lines returned.
    #[arg(long, default_value_t = 200)]
    limit: u32,
    /// Filter to stdout or stderr.
    #[arg(long)]
    stream: Option<String>,
},
```

Dispatch:

```rust
Command::Logs { name, follow, run, since, until, limit, stream } => {
    commands::logs::run(&client, cli.json, &name, follow, run, since, until, limit, stream).await
}
```

- [ ] **Step 2: Write `anankectl/src/commands/logs.rs`**

```rust
use ananke_api::LogsResponse;
use futures::StreamExt;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

use crate::{client::{ApiClient, ApiClientError}, output};

#[allow(clippy::too_many_arguments)]
pub async fn run(
    client: &ApiClient,
    json: bool,
    name: &str,
    follow: bool,
    run: Option<i64>,
    since: Option<i64>,
    until: Option<i64>,
    limit: u32,
    stream: Option<String>,
) -> Result<(), ApiClientError> {
    let mut query = Vec::new();
    if let Some(v) = run { query.push(format!("run={v}")); }
    if let Some(v) = since { query.push(format!("since={v}")); }
    if let Some(v) = until { query.push(format!("until={v}")); }
    query.push(format!("limit={limit}"));
    if let Some(v) = stream.as_deref() { query.push(format!("stream={v}")); }
    let path = format!("/api/services/{name}/logs?{}", query.join("&"));

    let response: LogsResponse = client.get_json(&path).await?;
    let mut max_seq: Option<i64> = response.logs.first().map(|l| l.seq);
    if json {
        output::print_json(&response);
    } else {
        for line in response.logs.iter().rev() {
            println!("[{}] {}", line.stream, line.line);
        }
    }
    if !follow { return Ok(()); }

    // Upgrade to WS for the live tail.
    let ws_url = client
        .endpoint
        .join(&format!("/api/services/{name}/logs/stream"))
        .expect("valid path")
        .to_string()
        .replace("http://", "ws://")
        .replace("https://", "wss://");
    let (mut ws, _) = connect_async(ws_url)
        .await
        .map_err(|e| ApiClientError::Connect(reqwest::Error::from(std::io::Error::other(e.to_string()))))?;

    while let Some(Ok(Message::Text(s))) = ws.next().await {
        let Ok(line) = serde_json::from_str::<ananke_api::LogLine>(&s) else { continue };
        if let Some(prev) = max_seq
            && line.seq <= prev
        {
            continue;
        }
        max_seq = Some(line.seq);
        println!("[{}] {}", line.stream, line.line);
    }
    Ok(())
}
```

Note: `reqwest::Error::from` on `std::io::Error` isn't real. Replace with a proper variant of `ApiClientError` for WS connect failure — add `ApiClientError::WebSocket(String)` with its own display + exit_code arm (exit 3, treat as connection error).

- [ ] **Step 3: Register module**

Add `pub mod logs;` to `commands/mod.rs`.

- [ ] **Step 4: Build + test**

```bash
cargo check --workspace
cargo clippy --all-targets --features test-fakes -- -D warnings
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(cli): anankectl logs + --follow"
```

---

## Task 14: `anankectl` oneshot commands

**Files:**
- Create: `anankectl/src/commands/oneshot.rs`
- Modify: `anankectl/src/main.rs`, `anankectl/src/commands/mod.rs`

- [ ] **Step 1: Extend CLI**

```rust
#[derive(Subcommand)]
enum OneshotCommand {
    Submit { file: std::path::PathBuf },
    Run {
        #[arg(long)] name: Option<String>,
        #[arg(long, default_value_t = 50)] priority: u8,
        #[arg(long)] ttl: Option<String>,
        #[arg(long)] workdir: Option<std::path::PathBuf>,
        #[arg(long, default_value = "gpu-only")] placement: String,
        #[arg(long, conflicts_with_all = ["min_vram_gb", "max_vram_gb"])] vram_gb: Option<f32>,
        #[arg(long, requires = "max_vram_gb")] min_vram_gb: Option<f32>,
        #[arg(long)] max_vram_gb: Option<f32>,
        /// Command and args.
        #[arg(trailing_var_arg = true, required = true)]
        command: Vec<String>,
    },
    List,
    Kill { id: String },
}
```

`Command::Oneshot(OneshotCommand)` variant on the top-level enum.

- [ ] **Step 2: Write `anankectl/src/commands/oneshot.rs`**

```rust
use std::path::Path;

use ananke_api::{OneshotRequest, OneshotResponse, OneshotStatus, oneshot::{OneshotAllocation, OneshotDevices}};

use crate::{client::{ApiClient, ApiClientError}, output};

pub async fn submit(client: &ApiClient, json: bool, path: &Path) -> Result<(), ApiClientError> {
    let toml_str = std::fs::read_to_string(path)
        .map_err(|e| ApiClientError::Usage(format!("read {}: {e}", path.display())))?;
    let req: OneshotRequest = toml::from_str(&toml_str)
        .map_err(|e| ApiClientError::Usage(format!("parse TOML: {e}")))?;
    let resp: OneshotResponse = client.post_json("/api/oneshot", &req).await?;
    if json { output::print_json(&resp); } else {
        println!("ok: {} (port {}, logs {})", resp.id, resp.port, resp.logs_url);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn run(
    client: &ApiClient,
    json: bool,
    name: Option<String>,
    priority: u8,
    ttl: Option<String>,
    workdir: Option<std::path::PathBuf>,
    placement: String,
    vram_gb: Option<f32>,
    min_vram_gb: Option<f32>,
    max_vram_gb: Option<f32>,
    command: Vec<String>,
) -> Result<(), ApiClientError> {
    let allocation = match (vram_gb, min_vram_gb, max_vram_gb) {
        (Some(g), None, None) => OneshotAllocation {
            mode: Some("static".into()), vram_gb: Some(g), min_vram_gb: None, max_vram_gb: None,
        },
        (None, Some(lo), Some(hi)) => OneshotAllocation {
            mode: Some("dynamic".into()), vram_gb: None, min_vram_gb: Some(lo), max_vram_gb: Some(hi),
        },
        (None, None, None) => return Err(ApiClientError::Usage(
            "must specify --vram-gb or --min-vram-gb + --max-vram-gb".into(),
        )),
        _ => return Err(ApiClientError::Usage("conflicting --vram flags".into())),
    };

    let req = OneshotRequest {
        name,
        template: "command".into(),
        command: Some(command),
        workdir: workdir.map(|p| p.to_string_lossy().into_owned()),
        allocation,
        devices: Some(OneshotDevices { placement: Some(placement) }),
        priority: Some(priority),
        ttl,
        port: None,
        metadata: Default::default(),
    };
    let resp: OneshotResponse = client.post_json("/api/oneshot", &req).await?;
    if json { output::print_json(&resp); } else {
        println!("ok: {} (port {}, logs {})", resp.id, resp.port, resp.logs_url);
    }
    Ok(())
}

pub async fn list(client: &ApiClient, json: bool) -> Result<(), ApiClientError> {
    let rows: Vec<OneshotStatus> = client.get_json("/api/oneshot").await?;
    if json { output::print_json(&rows); } else {
        println!("{:<30} {:<12} {:<10} {:>6}", "ID", "NAME", "STATE", "PORT");
        for r in &rows {
            println!("{:<30} {:<12} {:<10} {:>6}", r.id, r.name, r.state, r.port);
        }
    }
    Ok(())
}

pub async fn kill(client: &ApiClient, _json: bool, id: &str) -> Result<(), ApiClientError> {
    client.delete(&format!("/api/oneshot/{id}")).await?;
    println!("ok: killed {id}");
    Ok(())
}
```

- [ ] **Step 3: Dispatch**

```rust
Command::Oneshot(o) => match o {
    OneshotCommand::Submit { file } => commands::oneshot::submit(&client, cli.json, &file).await,
    OneshotCommand::Run { name, priority, ttl, workdir, placement, vram_gb, min_vram_gb, max_vram_gb, command } =>
        commands::oneshot::run(&client, cli.json, name, priority, ttl, workdir, placement, vram_gb, min_vram_gb, max_vram_gb, command).await,
    OneshotCommand::List => commands::oneshot::list(&client, cli.json).await,
    OneshotCommand::Kill { id } => commands::oneshot::kill(&client, cli.json, &id).await,
}
```

- [ ] **Step 4: Register module**

Add `pub mod oneshot;` to `commands/mod.rs`.

- [ ] **Step 5: Build + test**

```bash
cargo check --workspace
cargo clippy --all-targets --features test-fakes -- -D warnings
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(cli): anankectl oneshot (submit, run, list, kill)"
```

---

## Task 15: `anankectl` config commands + `reload` alias

**Files:**
- Create: `anankectl/src/commands/config.rs`
- Modify: `anankectl/src/main.rs`, `anankectl/src/commands/mod.rs`
- Test: `anankectl/tests/config_cycle.rs`

- [ ] **Step 1: Extend CLI**

```rust
#[derive(Subcommand)]
enum ConfigCommand {
    Show,
    Validate { file: Option<std::path::PathBuf> },
    Reload,
}
```

Top-level enum: `Config(ConfigCommand)`, `Reload` (alias).

- [ ] **Step 2: Write `anankectl/src/commands/config.rs`**

```rust
use std::{io::Read, path::Path};

use ananke_api::{ConfigResponse, ConfigValidateRequest, ConfigValidateResponse};

use crate::{client::{ApiClient, ApiClientError}, output};

pub async fn show(client: &ApiClient, json: bool) -> Result<(), ApiClientError> {
    let resp: ConfigResponse = client.get_json("/api/config").await?;
    if json { output::print_json(&resp); } else {
        println!("{}", resp.content);
    }
    Ok(())
}

pub async fn validate(
    client: &ApiClient,
    json: bool,
    file: Option<&Path>,
) -> Result<(), ApiClientError> {
    let content = match file {
        Some(p) => std::fs::read_to_string(p)
            .map_err(|e| ApiClientError::Usage(format!("read {}: {e}", p.display())))?,
        None => {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s).ok();
            s
        }
    };
    let req = ConfigValidateRequest { content };
    let resp: ConfigValidateResponse = client.post_json("/api/config/validate", &req).await?;
    if json { output::print_json(&resp); } else if resp.valid {
        println!("ok: config is valid");
    } else {
        println!("error: config is invalid");
        for err in &resp.errors {
            println!("  line {}:{} {}", err.line, err.column, err.message);
        }
    }
    Ok(())
}

pub async fn reload(client: &ApiClient, _json: bool) -> Result<(), ApiClientError> {
    // Force-reload by PUTting the current file back to the server.
    // Read the file's current content via GET, then PUT it unchanged
    // with the matching If-Match hash.
    let resp: ConfigResponse = client.get_json("/api/config").await?;
    client.put_body("/api/config", resp.content, Some(&resp.hash)).await?;
    println!("ok: config reload requested");
    Ok(())
}
```

- [ ] **Step 3: Dispatch**

```rust
Command::Config(c) => match c {
    ConfigCommand::Show => commands::config::show(&client, cli.json).await,
    ConfigCommand::Validate { file } => {
        commands::config::validate(&client, cli.json, file.as_deref()).await
    }
    ConfigCommand::Reload => commands::config::reload(&client, cli.json).await,
},
Command::Reload => commands::config::reload(&client, cli.json).await,
```

- [ ] **Step 4: Integration test**

Create `anankectl/tests/config_cycle.rs`:

```rust
use std::process::Command;

#[test]
fn config_show_help_runs() {
    let output = Command::new(env!("CARGO_BIN_EXE_anankectl"))
        .args(["config", "--help"])
        .output()
        .expect("spawn");
    assert!(output.status.success(), "{:?}", output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("show"));
    assert!(stdout.contains("validate"));
    assert!(stdout.contains("reload"));
}
```

- [ ] **Step 5: Build + test**

```bash
cargo test --workspace --features test-fakes
cargo clippy --all-targets --features test-fakes -- -D warnings
cargo +nightly fmt --all
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(cli): anankectl config (show, validate, reload) + reload alias"
```

---

## Final verification

- [ ] `cargo fmt --all -- --check` (stable) — clean or warning-only for the unstable import options.
- [ ] `cargo +nightly fmt --all` — applied.
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` — clean.
- [ ] `cargo clippy --all-targets --no-default-features -- -D warnings` — clean.
- [ ] `cargo test --workspace --features test-fakes` — green.
- [ ] `cargo test --workspace --no-default-features` — green.

Extended smoke replay on redline is optional for this phase; the REST + CLI surface is orthogonal to the scheduling / spawn paths that the extended smoke exercises.

---

## Self-review checklist

- All 13 REST endpoints covered by a task. Task 5 covers lifecycle (5 endpoints); Task 6 covers logs; Task 7 covers 3 config endpoints. Oneshot REST endpoints pre-date this phase and need no new work.
- WS streams covered by Tasks 8 + 9.
- CLI subcommands covered across Tasks 11-15.
- Shared DTOs defined in Task 2 and referenced by every subsequent task.
- External-access gate covered by Task 10.
- `ConfigManager` introduced in Task 4 before any handler depends on its public API.
- EventBus introduced in Task 3 before the WS handlers that subscribe to it.
- Every task includes build + test + commit steps.
