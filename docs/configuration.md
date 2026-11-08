# Configuration Guide

ananke is configured via a single TOML file discovered in this order:

1. `ANANKE_CONFIG` environment variable.
2. `--config` CLI argument.
3. `$XDG_CONFIG_HOME/ananke/config.toml`
4. `~/.config/ananke/config.toml`
5. `/etc/ananke/config.toml`

The file is hot-reloaded on save: ananke validates the new config, spawns added services and drains removed ones, and ignores failed reloads so the previous valid config stays in effect.

## Daemon Settings

```toml
[daemon]
management_listen = "0.0.0.0:7071"
allow_external_management = true # Required if management_listen is non-loopback
allow_external_services = true   # Allow public access to individual model ports
data_dir = "./data"
shutdown_timeout = "120s"        # Max time to wait for services to drain
private_port_start = 40000      # Start of loopback port range for private listeners
private_port_end = 59999        # End of loopback port range
llama_server = "/opt/llama-build/llama-server" # Default binary for every llama-cpp service
```

> **Security Note:** Both the Management API (`management_listen`) and per-service reverse proxies (`allow_external_services`) are **unauthenticated**. If you bind them to non-loopback addresses:
>
> - Trust your network perimeter (e.g., Tailscale, a private VLAN).
> - Terminate TLS and authentication at a reverse proxy in front of ananke.
> - Never expose these ports directly to the public internet.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `management_listen` | string | `127.0.0.1:7071` | Bind address for the management API. Non-loopback requires `allow_external_management = true`. |
| `allow_external_management` | bool | `false` | Must be `true` when `management_listen` is non-loopback. |
| `allow_external_services` | bool | `false` | Bind per-service reverse proxies on `0.0.0.0` instead of `127.0.0.1`. Controls only the per-service proxies, not the OpenAI multiplexer (which honours `openai_api.listen`). |
| `data_dir` | path | `$XDG_DATA_HOME/ananke` (or `~/.local/share/ananke`) | Directory for the SQLite database and runtime state. |
| `shutdown_timeout` | duration string | `120s` | Max time to wait for services to drain on daemon shutdown. |
| `private_port_start` | u16 | `40000` | Inclusive lower bound of the loopback port range handed to llama-server children for their private listener. |
| `private_port_end` | u16 | `59999` | Inclusive upper bound of the private-listener port range. Override when another process occupies the default window. |
| `llama_server` | path | `llama-server` (from `$PATH`) | Default llama-server executable for every llama-cpp service. Overridable per-service. |

## OpenAI API Settings

```toml
[openai_api]
listen = "0.0.0.0:7070"
enabled = true
max_request_duration = "10m"
allow_cors = true
max_body_mb = 64
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `listen` | string | `127.0.0.1:7070` | Bind address for the OpenAI-compatible API. |
| `enabled` | bool | `true` | Set to `false` to disable the OpenAI API entirely. |
| `max_request_duration` | duration string | `10m` | Max wall-clock duration per proxied request. |
| `allow_cors` | bool | `true` | Allow cross-origin requests from browsers. Set to `false` to block browser-based access. |
| `max_body_mb` | u64 | `64` | Max request body size in MiB. Raise for large or many images (vision payloads are base64-encoded). |

## Global Defaults
These values apply to all services unless overridden per-service:

```toml
[defaults]
idle_timeout = "10m"
priority = 50
start_queue_depth = 10
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `idle_timeout` | duration string | `10m` | Default idle timeout for on-demand services. |
| `priority` | u8 | `50` | Default eviction priority (higher wins eviction contests). |
| `start_queue_depth` | u32 | `10` | Default concurrency cap on pending start requests waiting for the same supervisor before they are rejected with `QueueFull`. |

## Device Configuration
Control which GPUs are used and how much VRAM is reserved for the system:

```toml
[devices]
gpu_ids = [0, 1]
default_gpu_reserved_mb = 2048
gpu_reserved_mb = { "0" = 4096 } # Per-GPU override (GPU 0: reserve 4GB)

[devices.cpu]
enabled = true
reserved_gb = 8
```

`default_gpu_reserved_mb` and `gpu_reserved_mb` are kept free on every GPU when the packer places a service; a per-service `gpu_headroom_mb` adds to them for one model.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `gpu_ids` | array of u32 | all visible GPUs | Only probe these GPUs. |
| `default_gpu_reserved_mb` | u64 | `0` | VRAM (MiB) kept free on every GPU that lacks a `gpu_reserved_mb` entry. |
| `gpu_reserved_mb` | map string → u64 | empty | Per-GPU VRAM reserve (MiB), keyed by GPU id string. |
| `cpu.enabled` | bool | `true` | Allow CPU placement for services. |
| `cpu.reserved_gb` | u64 | `0` | Host RAM (GiB) the daemon keeps free. Bounds how much expert weight a hybrid MoE service may offload to the CPU; a placement that would exceed it is rejected. |

## Service Configuration
Services are defined as an array of `[[service]]` blocks. Each service uses one of two templates: `llama-cpp` (for GGUF models via llama.cpp) or `command` (for arbitrary binaries). Either template can run its workload in a container; see [Container Workloads](#container-workloads).

### Common Fields

These fields appear at the top level of every `[[service]]` block, regardless of template:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | string | *required* | Unique service identifier. |
| `template` | string | *required* | `"llama-cpp"` or `"command"`. |
| `port` | u16 | *required* | Public-facing port for the service's reverse proxy. |
| `lifecycle` | string | `"on_demand"` | `"on_demand"` or `"persistent"` (see [Lifecycle](#lifecycle)). |
| `priority` | u8 | `50` (or `[defaults]` value) | Eviction priority; higher wins eviction contests. |
| `idle_timeout` | duration string | `10m` (or `[defaults]` value) | Idle timeout for on-demand services. |
| `description` | string | none | Human-readable description exposed through `/v1/models` and `/api/services`. |
| `modality` | string | `"chat"` | `"chat"` or `"embedding"` (see [Embedding Services](#embedding-services)). On `llama-cpp` services, `"embedding"` also passes `--embeddings` to llama-server. Any other string is a hard config error. |
| `extra_args` | array of string | none | Extra argv appended to the service's launch command. |
| `extra_args_append` | array of string | none | Extra argv appended to the inherited list (use with `extends`; concatenated with parent's list). |
| `env` | map string → string | none | Environment variables set on the spawned process. Accepts `${port}`, `${gpu_ids}`, `${reserve_mb}`, `${model}`, `${name}` placeholders. |
| `env_inherit` | bool | `true` | Whether the child process inherits the daemon's environment (`$PATH`, `$HOME`, locale, …). Per-service `env` entries override individual inherited keys. Set `false` to start with a clean environment containing only the variables in `env` plus `CUDA_VISIBLE_DEVICES`. |
| `drain_timeout` | duration string | `30s` | Tail grace after in-flight requests end, before SIGTERM; skipped when drain starts idle. |
| `stop_grace` | duration string | `10s` | Time from SIGTERM to SIGKILL for running-service stops, evictions, idle expiry and daemon shutdown. Startup and watchdog fast-kill grace periods are unchanged. `daemon.shutdown_timeout` remains the overall daemon shutdown bound. |
| `extended_stream_drain` | duration string | `30s` | Extra grace granted to in-flight streaming requests during drain. |
| `max_request_duration` | duration string | `10m` | Cap on wall-clock duration of a single proxied request. |
| `start_queue_depth` | u32 | `10` (or `[defaults]` value) | Concurrency cap on pending start requests before `QueueFull` rejection. |
| `extends` | string | none | Name of a parent service to inherit from. See [Service Inheritance](#service-inheritance). |
| `migrate_from` | string | none | Old service name to preserve database history from. See [Service Migration](#service-migration). |

### Lifecycle

Each service runs in one of two modes:

- **On-Demand (Default)**: Loaded only when a request arrives. Unloaded after a configurable `idle_timeout` (default: 10m) to free up VRAM.
- **Persistent**: Stays loaded in memory indefinitely, ensuring zero-latency startup for critical models.

```toml
[[service]]
name = "my-model"
template = "llama-cpp"
port = 8200
model = "/path/to/model.gguf"
lifecycle = "on_demand"   # or "persistent"
```

### Placement

Placement controls where a service's tensors live and how multi-GPU splitting works.

```toml
[service.devices]
placement = "gpu-only"   # "gpu-only" (default), "cpu-only", or "hybrid"
gpu_allow = [0, 1]        # Only use these GPUs
gpu_headroom_mb = 1024    # Keep this much extra VRAM free on each GPU for this service
placement_override = { "gpu:0" = 22000, "gpu:1" = 22000 } # Hand-pin per-slot VRAM
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `placement` | string | `"gpu-only"` | Placement policy (see below). |
| `gpu_allow` | array of u32 | all `[devices]` GPUs | Restrict the service to these GPU ids. |
| `gpu_headroom_mb` | u64 | `0` | Extra per-GPU VRAM (MiB) to keep free when placing *this* service, added on top of the global `[devices]` reserve. Lets a single model be packed more conservatively without bypassing the estimator. |
| `placement_override` | map string → u64 | none | Hand-pin VRAM (MiB) per device slot. Keys: `"cpu"` or `"gpu:N"`. Overrides the estimator's per-slot distribution. Must be non-empty if present; zero values and `cpu` keys under `gpu-only` are rejected. |
| `split` | string | `"layer"` | Multi-GPU split mode for llama.cpp services: `"layer"`, `"row"`, `"tensor"`. Maps to llama.cpp's `--split-mode`. See [Multi-GPU split modes](#multi-gpu-split-modes) for constraints. |
| `tensor_split_weights` | array of f32 | none | Optional per-GPU weights for the `--tensor-split` ratio in sharded (`row`/`tensor`) modes. One positive weight per allowed GPU, in ascending GPU-id order. Unset gives an equal `1,1,...` split. Use this for heterogeneous GPUs (e.g. weight by relative memory bandwidth). Weights are meaningful to four decimal places; additional precision is rounded when converting to the integer `--tensor-split` ratio. See [Multi-GPU split modes](#multi-gpu-split-modes). |

Placement policies:

- `gpu-only` (default): Service must reside entirely on GPU.
- `cpu-only`: Service resides entirely on CPU. `n_gpu_layers` must be `0` (or unset), otherwise config validation rejects it.
- `hybrid`: Allows a mix of GPU and CPU. The packer fills the GPUs first and spills the remainder to CPU. For MoE models with `expert_offload` enabled it spills *expert tensors* before whole layers, keeping every layer's attention and KV cache on the GPU (see [MoE Expert Offload](#moe-expert-offload)). Manual `override_tensor` rules also work here for hand-picked CPU offloading.

#### Multi-GPU split modes

When a llama.cpp service spans more than one GPU, `devices.split` selects how llama.cpp divides the model across them. It maps directly to llama.cpp's `--split-mode`:

```toml
[service.devices]
placement = "gpu-only"
split = "tensor"   # "layer" (default), "row", or "tensor"
```

- `layer` (default): pipeline parallelism - each GPU holds a contiguous range of whole layers. ananke estimates each layer's footprint and packs them across the allowed GPUs first-fit, so the split ratio follows the per-GPU layer counts. Lowest interconnect demand; the right default when the cards have no fast peer link.
- `row`: the older tensor-parallel mode (`--split-mode row`). Splits individual tensors by row. Without NVLink/P2P it is typically *slower* than `layer` because every token incurs cross-GPU traffic over PCIe; prefer `tensor` over `row` on such hosts.
- `tensor`: the newer tensor-parallel mode (`--split-mode tensor`). Shards each tensor across the GPUs and emits a balanced `--tensor-split` with `--main-gpu` set to the lowest allowed GPU. On dual identical cards this measures meaningfully faster decode than `layer` even without P2P, at the cost of a larger compute buffer and constant cross-GPU communication.

`row` and `tensor` are sharded modes and carry extra constraints, rejected at config validation:

- The service must use `placement = "gpu-only"` - a sharded model cannot spill to CPU.
- Only valid for `llama-cpp` services, not `command` services.
- Cannot be combined with `override_tensor` (manual tensor placement), since the sharded modes manage tensor placement themselves.

With a sharded mode, ananke reserves an equal share of the model weights, KV cache, and compute buffer on each allowed GPU by default, placing the non-layer remainder (output tensor, MTP overhead, …) on the main GPU. The pledge book reflects this per-GPU split, so a co-tenant (e.g. an embedding service) sees the true free capacity on each card.

For heterogeneous GPUs, set `devices.tensor_split_weights` to a weight per allowed GPU in ascending GPU-id order. The weights scale the per-GPU share and the emitted `--tensor-split` ratio. For example, an RTX 3090 paired with an RTX 3060, where the 3090 has roughly 2.6 times the memory bandwidth, can be weighted `[2.6, 1.0]` to give the faster card ~2.6 times the tensors instead of the historical equal split:

```toml
[service.devices]
placement = "gpu-only"
split = "tensor"
gpu_allow = [0, 1]
tensor_split_weights = [2.6, 1.0]
```

The weights are normalised by their sum, so only the ratio matters. The number of weights must match the number of allowed GPUs, and weights must be positive and finite. Weights are meaningful to four decimal places; additional precision is rounded when converting to the integer `--tensor-split` ratio.

### Health Checks
```toml
[service.health]
http = "/health"        # HTTP path to probe for readiness
timeout = "3m"          # Per-probe timeout
probe_interval = "5s"   # Probe cadence
```

When `[service.health]` is absent, the default `http` is `/v1/models`. Disabling health checks is useful for services that don't expose an HTTP endpoint, or when the operator knows the service is ready as soon as it starts.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `http` | string | `/v1/models` | HTTP path to probe for readiness. Set to `""` (empty string) to disable the health check entirely - the service transitions to Running immediately after spawn, with no readiness probe. |
| `timeout` | duration string | `3m` | Per-probe timeout before a health check fails. |
| `probe_interval` | duration string | `5s` | Cadence between health probes. |

### Resource Allocation
ananke oversubscribes GPU memory by dynamically managing which models are active:

- **llama.cpp Services**: VRAM usage is determined by the model size and `n_gpu_layers`. ananke uses an internal GGUF-aware estimator to track usage. No allocation mode is needed.
- **Command Services**: Support two allocation modes via `[service.allocation]`:
  - `static`: Reserves a fixed amount of memory (`reserve_gb`) — host RAM for a cpu-only service, VRAM otherwise. The pre-rename `vram_gb` spelling is still accepted.
  - `dynamic`: Operates within a range (`min_reserve_gb` to `max_reserve_gb`).

For a GPU-placed service the daemon picks, in both modes, the GPU with the most available headroom (subject to `gpu_allow`), preferring one whose free capacity satisfies the upper bound (`reserve_gb` for `static`, `max_reserve_gb` for `dynamic`) so dynamic services have room to grow. The picked GPU id is exported to the spawned child as `CUDA_VISIBLE_DEVICES`, and is also available as the `${gpu_ids}` placeholder in `command` argv. A containerized service instead sets `container.gpu_device`, and ananke injects a CDI device per picked GPU so those are the only ones the container sees. A `placement = "cpu-only"` service skips the pick entirely — its reservation is host RAM, and `${gpu_ids}` substitutes to the empty string.

```toml
[service.allocation]
mode = "dynamic"         # "static" or "dynamic" (command services only)
reserve_gb = 44              # static: fixed reservation in GiB
min_reserve_gb = 2.0         # dynamic: minimum reservation in GiB
max_reserve_gb = 12.0        # dynamic: maximum reservation in GiB
min_borrower_runtime = "60s" # dynamic: balloon resolver grace period
```

**Eviction**: When VRAM is exhausted, ananke uses a priority-based eviction system. Higher priority services can displace dormant on-demand services.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `mode` | string | *required* (command only) | `"static"` or `"dynamic"`. Required on every `command` service. Optional on a `llama-cpp` one, where it replaces the estimator entirely — the only way to place a model whose architecture ananke does not recognise. |
| `reserve_gb` | f32 | none | `static` only. Memory to reserve, in GiB — host RAM for a cpu-only service, VRAM otherwise. Required for `static`. Accepted as `vram_gb` for pre-rename configs. |
| `min_reserve_gb` | f32 | none | `dynamic` only. Minimum reservation in GiB. Required for `dynamic`. Accepted as `min_vram_gb` for pre-rename configs. |
| `max_reserve_gb` | f32 | none | `dynamic` only. Maximum reservation in GiB. Required for `dynamic`; must be > `min_reserve_gb`. Accepted as `max_vram_gb` for pre-rename configs. |
| `min_borrower_runtime` | duration string | `1m` | `dynamic` only. Balloon resolver grace period: minimum runtime a borrower must accumulate before it may be fast-killed. |

### Request Filters

Modify requests before they reach the model:

```toml
[service.filters]
strip_params = ["temperature"]          # Remove these JSON keys from the request
set_params = { max_tokens = 4096 }       # Force these JSON key/value pairs
```

> **Note:** `openai_proxy.upstream_model` (for command services) overrides any `filters.set_params.model`, because the model rewrite happens *after* filters are applied. Filters can still strip or set other JSON keys. See [OpenAI Proxy](#openai-proxy).

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `strip_params` | array of string | none | JSON keys to remove from the request body before forwarding. |
| `set_params` | map string → toml value | none | JSON key/value pairs to set on the request body before forwarding. |

### Metadata
Arbitrary key-value pairs exposed through `/v1/models` and `/api/services`:

```toml
[service.metadata]
discord_visible = true
```

These are opaque to the daemon - they exist only to be echoed back to clients (Discord rotation, residence flags, …).

### Tracking

Per-service hints that adjust how the snapshotter attributes observed VRAM/RSS to the service:

```toml
[service.tracking]
cgroup_parent = "/system.slice/ananke-comfyui.slice"
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `cgroup_parent` | string | none | Cgroup v2 path under which the service's actual workload pids live. Used by services whose workload runs in a container and is therefore reparented out of the daemon's process tree, so descendant-pid attribution can't reach it. Pids whose `/proc/<pid>/cgroup` path equals this value or sits inside its subtree are summed into the service's observed peak. Must be an absolute cgroup path (no trailing slash). |

### Auto-restart

Self-healing for a `Running` service that is alive but degraded — the process has not exited, so the crash-detection path never fires, yet every request is failing, hanging, or returning garbage. Five independent triggers all feed the existing drain → respawn cycle. The error-rate, TTFT-stall, and periodic triggers observe HTTP traffic at the proxy and apply to any service; the generation-stall and spec-collapse triggers read llama.cpp-specific surfaces and are gated to services that provide them. [docs/auto-restart.md](auto-restart.md) explains the reasoning behind each trigger, the exact engine coverage, and the incidents that motivated them.

```toml
[service.auto_restart]
# Error-rate watchdog (on by default; write `error_rate = false` to opt out):
error_rate = { window = "2m", max_error_rate = 0.5, min_requests = 20, poll_interval = "30s", error_statuses = "5xx" }
# Periodic restart (off by default; a table with an interval enables it):
periodic = { interval = "6h", mode = "on-request" }
# Speculative-decoding collapse watchdog (on by default when spec_type is set):
spec_collapse = { window = "2m", min_requests = 10, poll_interval = "30s" }
# Anti-flap guardrails, shared by all triggers:
min_uptime = "5m"
max_restarts = 3
flap_window = "30m"
```

The block is resolved as a **whole unit**: a service that sets any `auto_restart` field replaces `[defaults.auto_restart]` entirely rather than merging field-by-field. The same `[defaults.auto_restart]` block is accepted for fleet-wide defaults.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `error_rate` | table | `false` | on, with the defaults below | Error-rate watchdog. `false` disables it; a table enables it and overrides individual thresholds. |
| `periodic` | table | `false` | off | Periodic restart. Absent or `false` disables it; a table (with an `interval`) enables it. |
| `ttft_stall` | table | `false` | on, with the defaults below | Time-to-first-token stall watchdog. `false` disables it; a table enables it and overrides the timeout. Catches a wedged child that accepts a streaming request but never emits a frame — a failure the error-rate watchdog cannot see, because the request never completes. Restarts only when the whole service has gone silent, so it never fights healthy concurrent traffic. |
| `generation_stall` | table | bool | on for `llama-cpp` services, off for `command` services | Generation-stall watchdog. Polls the child's `/metrics` progress counters and restarts when they stay flat while requests are in flight — the wedge `ttft_stall` cannot see, because non-streaming requests give the proxy nothing to watch. Needs the child's `--metrics` endpoint; see the generation-stall trigger section below. |
| `spec_collapse` | table | bool | on for `llama-cpp` services, off for `command` services | Speculative-decoding collapse watchdog. Fires when a run that previously accepted draft tokens stops accepting any across a full window of drafting requests, which indicates corrupted inference state (e.g. all-NaN logits) that still returns HTTP 200 and is invisible to the other watchdogs. On by default only when `spec_type` is set; an explicit per-service enable without `spec_type` is rejected. See the spec-collapse trigger section below. |
| `min_uptime` | duration string | `5m` | Minimum uptime a fresh run must reach before an error-rate, generation-stall, or spec-collapse restart may fire — the anti-flap cooldown. |
| `max_restarts` | u32 | `3` | Watchdog restarts (error-rate, stall, generation-stall, and spec-collapse) tolerated within `flap_window` before the service is disabled with reason `auto_restart_loop` instead of restarted again. Periodic restarts are intentional and do not count toward this cap. |
| `flap_window` | duration string | `30m` | Sliding window over which `max_restarts` is counted. |

`[service.auto_restart.error_rate]` fields:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `window` | duration string | `2m` | Rolling window over which the error rate is measured. Scoped to the current run, so a fresh process starts from zero. |
| `max_error_rate` | float (0.0–1.0] | `0.5` | Fraction of requests in the window that must be errors to trigger. |
| `min_requests` | u32 | `20` | Minimum request count in the window before the ratio is trusted — stops a 2-of-2-failed service from restarting. |
| `poll_interval` | duration string | `30s` | How often the watchdog queries the metrics store. |
| `error_statuses` | `"5xx"` | `"4xx+5xx"` | `5xx` | Which HTTP statuses count as errors. `5xx` (server errors only) is the default because a 4xx is usually the client's fault, not the service's. `4xx+5xx` counts any status ≥ 400. |

`[service.auto_restart.periodic]` fields:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `interval` | duration string | required | How long a run may live before a periodic restart is due, measured from when it entered `Running`. |
| `mode` | `"immediate"` | `"on-idle"` | `"on-request"` | `on-request` | How the restart is timed once the interval elapses. `immediate` drains and respawns at once (interrupting in-flight traffic gracefully). `on-idle` waits for a quiet window with no in-flight requests, then restarts — zero disruption, but may never fire under continuous load. `on-request` marks the run stale and lets the next request drive the restart, blocking that request on the fresh process; it guarantees the restart happens even under continuous load. |

`[service.auto_restart.ttft_stall]` fields:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `timeout` | duration string | `5m` | How long a streaming request may stay in-flight with no response frame before the service is restarted. A restart fires only if the *whole service* produced no frame in that window — a request merely queued behind a healthy generation does not trip it. Only streaming requests are watched (non-streaming and embeddings are bounded by `max_request_duration` instead). Does not gate on `min_uptime`; the flap cap still applies. |

The generation-stall trigger reads llama.cpp's Prometheus `/metrics` progress counters (prompt + predicted tokens); flat counters under load are the signature of a wedged child that accepts requests but never advances. On `llama-cpp` services the daemon passes `--metrics` automatically while the trigger is on; an explicit `metrics = false` suppresses the flag and disables the trigger. On `command` services it is off by default (ananke does not build the child's argv, so it cannot enable the endpoint) — opt in with `generation_stall = true` once the wrapped server exposes a llama.cpp-compatible `/metrics` on the service's private port. If the endpoint is missing or unrecognisable, the trigger logs one warning and never fires.

`[service.auto_restart.generation_stall]` fields:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `timeout` | duration string | `5m` | How long the child's `/metrics` progress counters may stay flat, with at least one request in flight, before the service is restarted. Healthy prefill and decode both advance the counters every batch, so the default is unambiguous under load. An idle service (nothing in flight) never trips it. |
| `poll_interval` | duration string | `30s` | How often the child's `/metrics` endpoint is polled. |

The spec-collapse trigger reads the speculative-decoding fields llama.cpp reports in each response's `timings` object (`draft_n` proposed, `draft_n_accepted` accepted), which the per-service proxy records per request. A working target/draft pairing accepts a substantial fraction of drafted tokens on nearly every request. A target with corrupted inference state (degenerate or NaN logits) rejects every draft token while still returning HTTP 200 — a failure the error-rate and stall triggers cannot detect, since nothing errors and generation keeps progressing.

The trip condition requires a collapse, not just absence: the run must have accepted draft tokens at some point, and the recent window must then hold at least `min_draft_tokens` drafted tokens with zero accepted among them. The floor counts tokens rather than requests because the failure this trigger detects produces long generations that arrive slowly but draft thousands of tokens each. For streamed requests to such services, the daemon asks llama.cpp for per-chunk timings, so a generation the client aborts still records its draft counts. Workloads whose acceptance is legitimately zero from the start of a run (grammar-constrained speculative decoding, for example) never satisfy the first half and never trip the trigger. The trigger is on by default only for services that configure `spec_type`; an explicit per-service enable without `spec_type` is rejected at validation, since such a service can never produce draft counts. See [docs/auto-restart.md](auto-restart.md) for the full reasoning.

`[service.auto_restart.spec_collapse]` fields:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `window` | duration string | `2m` | Rolling window over which draft acceptance is measured, keyed on request completion time and scoped to the current run. Only requests that actually drafted (`draft_n > 0` in the engine's `timings`) count; a single accepted draft token anywhere in the window vetoes the restart. |
| `min_draft_tokens` | u64 | `200` | Minimum count of drafted tokens in the window before an all-zero acceptance is trusted. Counted in tokens rather than requests so that slow-arriving long generations — which draft thousands of tokens each — reach the floor on their own; one short unlucky generation stays under it. |
| `poll_interval` | duration string | `30s` | How often the watchdog queries the metrics store. |

When a trigger fires, the service drains (SIGTERM with grace, then SIGKILL) and returns to `Idle`; the normal ensure path respawns it — on the next request for an on-demand service, or within a few seconds for a persistent one. An auto-restart emits an `auto_restarted` event on the daemon event stream (see [the API guide](api.md)) and is persisted to the daemon's store, so the recent history is visible after the fact in the service detail view (web UI and `anankectl show`) even when nothing was listening at firing time.

## Templates
### llama-cpp
Used for GGUF models via llama.cpp. Only `name`, `template`, `port`, and `model` are required.

```toml
[[service]]
name = "gemma-4"              # required
template = "llama-cpp"        # required
port = 8200                   # required
model = "/path/to/model.gguf" # required
mmproj = "/path/to/mmproj.gguf"
context = 32768
flash_attn = true
cache_type_k = "q8_0"
cache_type_v = "q8_0"
lifecycle = "on_demand"
priority = 100

[service.sampling]
temperature = 0.7
top_p = 0.95
```

#### Field Reference

#### MoE Expert Offload

Large mixture-of-experts models often don't fit a card once their expert tensors are resident, even though the attention and KV cache do. The `expert_offload` knob lets ananke move expert tensors to the CPU - the GPU keeps every layer's attention and KV cache (latency-critical), while the bulky, sparsely-activated experts live in host RAM. ananke sizes the placement and emits the matching `-ot` rules itself, so the VRAM reservation matches what the model actually uses.

`expert_offload` accepts three values, and any value other than `"off"` requires `placement = "hybrid"`:

- `"off"` (default): no expert offload. The model packs whole layers, spilling entire layers to CPU only if a layer doesn't fit.
- `"auto"`: ananke keeps each layer's experts on the GPU while there's room and greedily offloads only the surplus that doesn't fit the GPU's live free VRAM, preferring a second GPU before the CPU on multi-GPU hosts.
- an integer `N`: offload the experts of the `N` tail-most expert layers, regardless of fit. Use this when you have measured the sweet spot and want a fixed, deterministic split.

```toml
# Auto-fit a large MoE: ananke offloads the minimum experts to fit live VRAM,
# keeps 1 GiB free on the card, and emits the matching -ot rules itself.
[[service]]
name = "qwen3-moe"
template = "llama-cpp"
port = 8300
model = "/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
context = 80000
flash_attn = true
cache_type_k = "q8_0"
cache_type_v = "q8_0"
expert_offload = "auto"

[service.devices]
placement = "hybrid"
gpu_headroom_mb = 1024   # keep 1 GiB free on the card for this service
```

```toml
# Pin an exact offload count: offload the experts of the 16 tail-most expert
# layers. Equivalent in spirit to launching llama-server with --n-cpu-moe 16.
# (Set on the [[service]] block, alongside model/context/…; needs placement = "hybrid".)
expert_offload = 16
```

```toml
# Hand-picked tensor placement instead of auto-derivation: keep expert_offload
# off and write the -ot rules yourself.
expert_offload = "off"
override_tensor = [ "blk\\.(1[6-9]|2[0-9])\\.ffn_(up|down)_exps\\.=CPU" ]
```

#### Custom llama-server Binary or Wrapper

By default, ananke spawns `llama-server` from `PATH`. Two knobs change that:

- **`llama_server`**: a path to the executable (or wrapper script) that should be invoked in place of `llama-server`. The script must accept llama-server's CLI flags. Settable at the daemon level (default for every llama-cpp service) and per-service (overrides the daemon default).
- **`launcher`**: a full argv template that replaces the default `llama-server -m <model> ...` invocation. `launcher[0]` is the executable; the remaining entries are substituted with placeholders so a wrapper can see the model path separately from the rest of the flags (useful for container volume mounts).

Placeholders in `launcher` entries:

- `${model}` - the model path. Held back from `${args}` so the wrapper can position it freely.
- `${name}` - service name.
- `${port}` - the private loopback port ananke assigned.
- `${gpu_ids}` - comma-separated NVML index list ananke picked for this service.
- `${args}` - splat: expands to every llama-server flag ananke would otherwise have emitted (everything except `-m <model>` - `--mmproj`, `-c`, placement-derived `-ngl`/`--tensor-split`/`-ot`, sampling, `--host`, `--port`, `extra_args`, …). Must occupy a launcher entry on its own; `"--foo=${args}"` is rejected at config validation.

Example: wrap llama-server in a podman container that needs a volume mount for the model.

```toml
[[service]]
name = "qwen3-podman"
template = "llama-cpp"
port = 11436
model = "/srv/models/qwen3-30b.gguf"
context = 32768
flash_attn = true
launcher = ["/opt/podman-llama.sh", "${model}", "${args}"]
```

The wrapper script receives `/srv/models/qwen3-30b.gguf` as `$1` (for the volume mount) and `$@` after `shift` contains the rest of the llama-server argv - `-c 32768 -fa on -ngl 999 ... --host 127.0.0.1 --port 41000`. With `--network host` the container's llama-server is reachable on that port without further plumbing.

If you only need to point at a non-`PATH` binary (no argv rearranging), set `llama_server` instead:

```toml
[[service]]
name = "demo"
template = "llama-cpp"
port = 11437
model = "/srv/models/x.gguf"
llama_server = "/opt/llama-cuda/llama-server"
```

`CUDA_VISIBLE_DEVICES` is set on the spawned process from the picked GPU id(s) in both cases. Wrapper scripts that launch a container should forward this so the container only sees the picked GPU - for example, `podman run --device "nvidia.com/gpu=${CUDA_VISIBLE_DEVICES:-all}" ...`.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | path | *required* | Path to the GGUF model file. |
| `mmproj` | path | none | Path to an optional vision projector GGUF. Services with an `mmproj` render a purple `vision` badge. |
| `context` | u32 | `4096` (estimator default) | Context window size. If unset, a warning is logged and the estimator defaults to 4096 tokens. |
| `n_gpu_layers` | i32 | `-1` | Number of layers to offload to GPU. `-1` (default) offloads all layers. Must be `0` under `placement = "cpu-only"`. |
| `expert_offload` | string or u32 | `"off"` | MoE expert-offload policy (see [MoE Expert Offload](#moe-expert-offload)). |
| `runtime` | table | mainline llama.cpp | Serving runtime, tagged by `kind`: `{ kind = "ik-llama", mla = 1, dsa = true, attn_max_batch = 512, runtime_repack = false }` selects the ik_llama.cpp fork and its options (`-mla`, `-dsa -fidx`, `-amb`, `-rtr`). Absent means mainline. `dsa` requires f16 KV. Point `llama_server` at a matching binary. ik services use ik's `--spec-type` dialect (`"mtp:n_max=4,p_min=0.5"`). |
| `flash_attn` | bool | `false` | Enable flash attention. Required for quantised KV cache types (`cache_type_k`/`cache_type_v` other than `f16`) on mainline llama.cpp; ik_llama handles quantised caches without this flag. |
| `cache_type_k` | string | `f16` | KV cache type for keys. Non-`f16` values require `flash_attn = true` (mainline only; ik_llama is exempt). |
| `cache_type_v` | string | `f16` | KV cache type for values. Non-`f16` values require `flash_attn = true` (mainline only; ik_llama is exempt). |
| `mmap` | bool | `true` | Memory-map the model file. |
| `mlock` | bool | `false` | Lock the model in RAM (prevents swapping). |
| `parallel` | u32 | `1` | Request parallelism (`-np`). With a non-unified KV this splits the context budget across slots, so each request caps at `context / parallel`. |
| `spec_type` | string | none | Speculative-decoding type passed to `--spec-type` (e.g. `"draft-mtp"` for multi-token prediction). |
| `spec_draft_n_max` | u32 | none | Max draft tokens per step (`--spec-draft-n-max`). Only meaningful when `spec_type` is set. |
| `draft_model` | path | none | Separate draft-model GGUF for speculative decoding (`-md` / `--model-draft`). Requires `spec_type` to be set. |
| `kv_unified` | bool | `false` | Use a single unified KV cache pool shared across all parallel slots (`-kvu` / `--kv-unified`). With `parallel > 1`, idle slots lend their share to active ones; total KV footprint is unchanged. |
| `cache_idle_slots` | bool | `true` | When `false`, pass `--no-cache-idle-slots` so idle slots' prompt-cache state is dropped (a stability mitigation). |
| `cache_ram_mb` | int (MiB) | `8192` | Host RAM cap for llama-server's prompt cache (`-cram`), which holds serialized evicted prompts so a returning conversation skips reprocessing. Always passed through explicitly, so the packer's host reservation and the runtime's cap are the same number; `0` disables the cache and frees the reservation with it. |
| `metrics` | bool | `false`, but auto-enabled while the `generation_stall` watchdog is on | Expose llama-server's Prometheus `/metrics` endpoint. The generation-stall watchdog needs it and passes `--metrics` automatically while active; an explicit `metrics = false` suppresses the flag and disables that watchdog. |
| `slots` | bool | `false` | Expose the `/slots` introspection endpoint. Note: reveals prompt contents - avoid on network-reachable ports. |
| `batch_size` | u32 | none | Context batch size (`-b`). |
| `ubatch_size` | u32 | none | Physical batch size (`-ub`). |
| `threads` | u32 | none | Number of CPU threads (`-t`). |
| `threads_batch` | u32 | none | Number of CPU threads for batch processing (`-tb`). |
| `numa` | string | none | NUMA thread-and-memory placement strategy (`--numa`): `"distribute"`, `"isolate"`, `"numactl"`. Unset leaves llama.cpp's default. |
| `jinja` | bool | `false` | Use Jinja chat templates. |
| `chat_template_file` | path | none | Path to a custom chat template file. |
| `override_tensor` | array of string | none | Manual tensor placement rules (e.g. `[ ".ffn_(up|down)_exps.=CPU" ]`). Incompatible with sharded split modes (`row`/`tensor`). |
| `sampling` | table | none | Sampling parameters (see [Sampling](#sampling)). |
| `estimation` | table | none | Estimator overrides (see [Estimation Overrides](#estimation-overrides)). |
| `llama_server` | path | daemon's `llama_server` or `$PATH` | Per-service override of the llama-server executable. Has no effect when `launcher` is set. |
| `launcher` | array of string | none | Full argv template that replaces the default `llama-server -m <model> ...` invocation (see [Custom llama-server Binary or Wrapper](#custom-llama-server-binary-or-wrapper)). |

#### Estimation Overrides
Override the internal GGUF-aware VRAM estimator's parameters:

```toml
[service.estimation]
compute_buffer_mb = 512
safety_factor = 1.1
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `compute_buffer_mb` | u32 | none | Override the estimated compute buffer size (MiB). |
| `safety_factor` | f32 | none | Multiplier applied to the estimated VRAM footprint. |

#### Sampling
Sampling parameters mapped to `llama-server` CLI flags:

```toml
[service.sampling]
temperature = 0.7
top_p = 0.95
top_k = 40
min_p = 0.05
repeat_penalty = 1.1
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `temperature` | f32 | none | Sampling temperature. |
| `top_p` | f32 | none | Nucleus sampling threshold. |
| `top_k` | u32 | none | Top-k sampling limit. |
| `min_p` | f32 | none | Minimum-p sampling threshold. |
| `repeat_penalty` | f32 | none | Repeat penalty applied to generated tokens. |

### command
Used for arbitrary binaries. Only `name`, `template`, `port`, and `command` are required. To run the command inside Docker or Podman, add a `[service.container]` block rather than wrapping it in a script; see [Container Workloads](#container-workloads).

```toml
[[service]]
name = "comfyui"            # required
template = "command"        # required
port = 8188                 # required
command = ["/bin/bash", "start_comfy.sh", "--port", "${port}"] # required
lifecycle = "on_demand"

[service.allocation]
mode = "dynamic"
min_reserve_gb = 2.0
max_reserve_gb = 12.0

[service.health]
http = "/system_stats"
timeout = "30s"
```

#### Field Reference

#### Placeholders

The following placeholders are substituted in `command` and `shutdown_command` argv entries (and in `env` values):

- `${port}` - the private loopback port assigned by ananke.
- `${gpu_ids}` - comma-separated NVML index list ananke picked for this service.
- `${reserve_mb}` - the reservation in MiB, on whichever device the service was placed. Still accepted under its former name `${vram_mb}`.
- `${model}` - model path (llama-cpp only; empty for command services).
- `${name}` - service name.
- `${listen_host}` / `${listen_port}` - the interface and port the workload should bind. For a host process these are `127.0.0.1` and the private port, so `${listen_port}` and `${port}` agree; they differ only for a bridge-networked container (see [Container Workloads](#container-workloads)).
- `${host_port}` - the host-side private port, for the rare command that needs both sides of a bridge publication.

A placeholder is `${name}`, and only `${` and `$$` mean anything to the substituter. A bare `{` never does, so an argument carrying JSON, a Jinja template, or a Python format string passes through exactly as written: vLLM's `--diffusion-config '{"canvas_length": 256}'` needs no escaping. `$$` is a literal `$`, which is how a literal `${port}` is written (`$${port}`); a `$` before anything else is itself, since arguments carry bare dollars far more often than they carry `${`.

An unknown name is an error, so a typo surfaces at config load rather than reaching the argv. So does `${` with no closing brace. A known name written bare (`{port}`) is a warning rather than an error: it cannot be one, since a JSON or format-string argument may hold `{model}` on purpose. Write `$${model}` to say that deliberately and silence the warning.

> Changed in 0.3.0. Placeholders were previously written `{name}`. That form could not be told apart from an argument's own braces without guessing, which made JSON arguments fail and silently rewrote Jinja templates. Rename each placeholder to `${name}`; nothing else needs escaping. A config still using `{name}` does not error, because the text is now literal, but it does warn at config load, naming the service and the entry. Confirm that a service's rendered command still contains the expected value, via `GET /api/services/{name}/command` or the launch-command panel.

The child also inherits `CUDA_VISIBLE_DEVICES` set to the picked GPU id(s). A containerized service does not need to forward it by hand: set `container.gpu_device` and ananke injects exactly the GPUs it picked.

The `shutdown_command` field is for external processes that cannot stop via signal alone. ananke runs it after the drain pipeline completes, ensuring a clean exit for a service that does not respond to SIGTERM. A container service has no use for it: native runtime cleanup replaces it, and the two are rejected together.

#### OpenAI Proxy

A `command`-template service that already speaks the OpenAI API (vLLM, TGI, SGLang, …) can opt into ananke's `/v1/models` and `/v1/chat/completions` multiplexer by adding an `[service.openai_proxy]` block. Without the block, command services stay invisible to the OpenAI surface and are only reachable via their per-service reverse proxy - the same as before.

```toml
[[service]]
name = "qwen3.6-27b-vllm"
template = "command"
port = 8210
command = ["/srv/vllm/qwen36_27b.sh", "${port}"]
lifecycle = "on_demand"
idle_timeout = "10m"

[service.allocation]
mode = "static"
reserve_gb = 44

[service.devices]
placement = "gpu-only"
placement_override = { "gpu:0" = 22000, "gpu:1" = 22000 }

[service.health]
http = "/health"

[service.openai_proxy]
upstream_model = "qwen3.6-27b-autoround"
```

When this is set:

- The service appears in `GET /v1/models` under its `name` (here, `qwen3.6-27b-vllm`).
- `POST /v1/chat/completions` (and `/v1/completions`, `/v1/embeddings`) addressed to `qwen3.6-27b-vllm` are routed to this service. ananke ensures the command is started, then forwards the body to the service's private loopback port.
- Before forwarding, ananke rewrites the JSON `model` field to `upstream_model` (here, `qwen3.6-27b-autoround`) - the name vLLM was started with via `--served-model-name`. Clients address ananke's name; the upstream sees its own name.

The rewrite happens *after* `[service.filters]` is applied, so `openai_proxy.upstream_model` overrides any `filters.set_params.model`. Filters can still strip or set other JSON keys (see [Request Filters](#request-filters)).

#### Embedding Services

By default every service is registered as a chat model. Pooling-only embedding models (Jina v5, BGE, E5, LFM2.5, …) opt in by setting `modality = "embedding"` on the service. The proxy itself is endpoint-agnostic - it already routes `POST /v1/embeddings` by `model` field.

What the modality does depends on the template:

- On `llama-cpp` services it passes `--embeddings` to llama-server, enabling its embeddings endpoint. The pooling strategy comes from the GGUF's `{arch}.pooling_type`.
- On `command` services the wrapped server is expected to speak `/v1/embeddings` natively; nothing extra is passed.

Beyond that the field is a typed declaration: clients filter on it through `/v1/models` and `/api/services`, and the frontend renders a teal `embedding` badge next to the service name (mirroring the purple `vision` badge for llama.cpp services with an `mmproj`).

```toml
[[service]]
name = "jina-embeddings-v5-text-small-retrieval-vllm"
template = "command"
port = 8211
modality = "embedding"
command = ["/srv/vllm/jina_embed_v5_small.sh", "${port}"]
lifecycle = "on_demand"
idle_timeout = "30m"

[service.allocation]
mode = "static"
reserve_gb = 7

[service.devices]
placement = "gpu-only"
placement_override = { "gpu:1" = 7000 }

[service.health]
http = "/health"

[service.openai_proxy]
upstream_model = "jina-embeddings-v5-text-small-retrieval"
```

Valid values are `"chat"` (the default) and `"embedding"`; any other string is a hard config error rather than a silent fall-back. The field is elided from `/v1/models` and `/api/services` JSON when it equals `"chat"`, so chat-only deployments see byte-identical wire output to what they shipped before this field landed.

Once registered, hit the endpoint as you would any OpenAI embedding API:

```sh
curl -s -X POST http://localhost:7070/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "jina-embeddings-v5-text-small-retrieval-vllm",
    "input": ["the quick brown fox", "lorem ipsum dolor sit amet"]
  }' | jq '{model, dim: (.data[0].embedding | length), n: (.data | length)}'
```

ananke ensures the upstream container is started (cold-starting it on first request if needed), rewrites `model` to `upstream_model`, and relays the embedding vectors back unchanged. The static VRAM pledge is held only while the service is running; on-demand services drain back to idle after `idle_timeout` elapses with no traffic.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `command` | array of string | *required* | argv to execute. Accepts placeholders (see below). |
| `workdir` | path | none | Working directory for the spawned process. |
| `allocation` | table | none | Memory reservation (see [Resource Allocation](#resource-allocation)). Required for command services. |
| `private_port` | u16 | auto-assigned | Upstream port ananke's reverse proxy should forward to. When absent, ananke picks one from the daemon's private-port pool and substitutes it into `command`/`env` via the `${port}` placeholder. Set explicitly when the external service binds a fixed port (e.g. a docker container exposing 18188 on the host). |
| `shutdown_command` | array of string | none | Optional argv run at drain time after SIGTERM-then-SIGKILL completes. Useful for external services that don't stop via signal - e.g. a docker-run wrapper where SIGTERM reaches the host shell but the container needs an explicit `docker stop`. Accepts the same placeholder substitutions as `command`. |
| `openai_proxy` | table | none | Opt the service into the OpenAI-compatible multiplexer (see [OpenAI Proxy](#openai-proxy)). |

### OpenAI Proxy

A `command`-template service that already speaks the OpenAI API (vLLM, TGI, SGLang, …) can opt into ananke's `/v1/models` and `/v1/chat/completions` multiplexer by adding an `[service.openai_proxy]` block. Without the block, command services stay invisible to the OpenAI surface and are only reachable via their per-service reverse proxy - the same as before.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `upstream_model` | string | none | Model name the upstream server was started with (e.g. via `--served-model-name`). ananke rewrites the JSON `model` field to this value before forwarding. |

## Container Workloads
Either template can run inside a Docker or Podman container. Add a `[service.container]` block to the service. Without the block the service runs as an ordinary host process and nothing about it changes.

The block does not replace the template; it relocates it. A `llama-cpp` service still gets its placement-derived llama-server argv, and a `command` service still supplies its own. What changes is where that argv runs: as the container's command rather than as a child of the daemon.

```toml
[[service]]
name = "qwen3.6-35b-ninfer"
template = "command"
port = 8205
command = [
  "ninfer-serve", "/artifacts/qwen3_6_35b_a3b.ninfer",
  "--host", "${listen_host}",
  "--port", "${listen_port}",
]
allocation = { mode = "static", reserve_gb = 26 }
health = { http = "/health", timeout = "10m" }

[service.container]
runtime = "docker"
image = "ninfer:local"
network = "host"
gpu_device = "nvidia.com/gpu=${id}"
mounts = [
  { source = "/home/philpax/ai/ninfer", target = "/artifacts", read_only = true },
]
```

#### Prerequisites

ananke drives the runtime's CLI. It never talks to the Engine or Libpod socket, and it neither pulls nor builds images. It starts and stops the image named by the service; the image must already be in the local store before the service starts:

```sh
docker pull vllm/vllm-openai:v0.26.0
podman build -t ninfer:local ~/src/ninfer
```

A missing image surfaces as a start failure carrying the runtime's own message, which distinguishes an unknown tag, a registry that needs credentials, and an image that was never built.

The daemon needs the selected runtime binary on its `$PATH` (or an explicit `runtime_executable`) and permission to use it.

Selecting Podman changes the `runtime` field and nothing else about the service. It does not make the two runtimes interchangeable in general: ananke owns the lifecycle-flag differences, but CDI support, rootless cgroup layouts, and SELinux relabelling all vary by runtime and version, and a capability the runtime lacks fails with an error rather than silently degrading.

#### Conflicting fields

Two existing fields are rejected alongside `container`, because each replaces the same boundary the container does:

- `launcher` (llama-cpp) wraps the host execution boundary, which is what the container now is.
- `shutdown_command` (command) exists so a wrapper script can stop something signals cannot reach. Native runtime cleanup replaces it.

#### Networking

The service endpoint has two sides. The placeholders resolve to the side the workload should bind, not the side ananke connects to.

Under `network = "bridge"`, ananke publishes `127.0.0.1:<private_port>` onto `<container_port>` and nothing else. Inside the container, `${listen_host}` resolves to `0.0.0.0` and `${listen_port}` to `<container_port>`: a workload that bound loopback inside its own network namespace would be unreachable from the host side of the publication. Because ananke can only guarantee reachability it was told about, a bridge-networked `command` service must consume both `${listen_host}` and `${listen_port}` in its argv or environment; one that hardcodes its own address is rejected at config load. Use host networking for a command whose listener cannot be parameterised.

Under `network = "host"`, there is no publication and no translation. `${listen_host}` resolves to `127.0.0.1` and `${listen_port}` to the private port, exactly as for a host process.

`${port}` remains accepted everywhere as an alias for `${listen_port}`, so pre-container command templates keep working. `${host_port}` is always the host-side private port, for the rare command that needs both numbers.

llama.cpp takes the same resolved endpoint, but as a typed renderer input rather than a placeholder: ananke emits `--host`/`--port` itself, so a containerized llama.cpp service needs no configuration for this.

#### Environment and secrets

`env_inherit` copies the daemon's whole environment into a host process. That is reasonable for a child process and wrong for a container, so it governs host processes only. A container's environment is exactly what the service declares:

- `env` on the service, and `container.env` layered over it, for explicit values.
- `container.env_passthrough` for names forwarded from the daemon's own environment, e.g. `["HF_TOKEN"]`.

Passthrough values are read at launch and never leave the runtime invocation: the management API and launch previews show the variable's name only, and nothing expands a secret into a persisted command line or a log line.

#### Mounts and path translation

Nothing inside the container can see a host path that was not mounted. For `command` services the service arranges this: write container paths in the argv, and mount whatever backs them. Paths are not guessed.

For `llama-cpp`, ananke generates the argv, so it also translates the four path fields it owns (`model`, `mmproj`, `draft_model`, and `chat_template_file`) through the configured mounts. Matching is lexical and component-aware, longest source prefix wins, and there is no symlink resolution. A typed path no mount covers is a start failure naming the exact path and field, rather than a container that comes up and cannot find its weights. Paths embedded in `extra_args` stay opaque and must already be container paths.

#### GPUs

`gpu_device` is a CDI template expanded once per GPU that ananke's placement selected, so the container sees the same devices the allocator reserved for it. `nvidia.com/gpu=${id}` with GPUs 0 and 2 selected yields `--device nvidia.com/gpu=0 --device nvidia.com/gpu=2`.

Leaving it unset injects nothing. If the runtime cannot satisfy the requested CDI device, the start fails with that error. ananke does not fall back to bind-mounting `/dev/nvidia*`, which would expose every GPU on the machine to the container.

#### Memory attribution

ananke reads the container's cgroup back from its host PID after start and attributes every process in that cgroup to the service. This needs no configuration and holds regardless of runtime, cgroup driver, or rootless layout, because it is the path the container actually landed in rather than one predicted from a slice name. The host PID's process descendants are attributed too, so the two sources cover each other.

#### Lifecycle and recovery

Containers can outlive both ananke and the CLI process that started them, so the lifecycle is conservative and every step is recoverable:

1. A durable launch intent is written before anything is created, carrying the owner, the service and run identity, the exact runtime executable, and the generated name and labels.
2. The container is created, not started, and its ID is attached to the intent.
3. The ownership row is committed. Only then is the container started: a container is never running without a persisted row that says so.
4. Logs are followed by a separate `logs --follow` process, and the exit status comes from an independent `wait`.
5. On drain, `kill --signal TERM`, escalating to `KILL` on the configured `stop_grace`. Final logs are drained before the container is removed.

Each run gets its own container named `ananke-<service>-<run-id>`, removed along with any anonymous volumes the runtime created for it. A stable per-service name would be easier to read, but a single failed cleanup would then block the service permanently: the next `create` would fail because the name is already in use. A per-run name makes a stray container inert rather than blocking, and the startup sweep bounds how long one can persist.

`--rm` is deliberately not used. Auto-removal races the log follower and the exit-status read, and it destroys the evidence a crashed daemon needs, so removal is always an explicit final step and is idempotent.

At startup ananke reconciles what it finds against those records. Every container it creates is labelled with a stable per-installation owner UUID as well as its service and run, and cleanup requires both a persisted identity and a matching owner label, not a name prefix alone. This keeps a second ananke, or an unrelated container that shares a name, from being removed. A container whose ownership cannot be resolved blocks only its own service from reprovisioning, preserving the evidence, while every other service starts normally.

#### What happens when ananke stops

A native child dies with the daemon: the spawner sets `PR_SET_PDEATHSIG`, and the kernel honours it however the daemon died. A container has no equivalent: it belongs to the runtime, not to ananke's process tree. Its survival is therefore handled explicitly rather than assumed.

On a signal ananke can catch (`SIGTERM`, `SIGINT`, `SIGQUIT`, as sent by `systemctl stop`, a restart, a reboot, or Ctrl-C), every service drains: TERM, then KILL after `stop_grace`, then removal. Containers do not outlive it.

If that drain overruns `shutdown_timeout`, the daemon exits rather than hanging. Before it exits, it removes every container still carrying this installation's owner label. The sweep exists for this case; after a drain that finished, it finds nothing.

On `SIGKILL`, the OOM killer, or a power loss, no userspace code runs: that is what those events mean. Containers keep running and keep the VRAM they reserved. Two things bound the exposure:

- Startup reconciliation removes them when ananke comes back, so the exposure lasts as long as the daemon is down. Running under a supervisor that restarts it keeps that to seconds.
- The allocator reads real free VRAM from NVML, not just its own pledge book, so a surviving container makes ananke conservative rather than wrong: it declines to place, or evicts, instead of over-committing onto memory something else is holding.

An orphan is therefore a capacity problem until reconciliation clears it, not a correctness one. To make containers die with the daemon under `SIGKILL` as well, that guarantee has to come from outside ananke, for example a cgroup the supervisor tears down, because no process can clean up after a signal it cannot receive.

#### Logs

Neither `docker logs` nor `podman logs` documents a stream framing that survives the CLI boundary, so container output is stored and served as a single `combined` stream rather than being mislabelled as stdout. Host processes are unaffected and keep their split `stdout`/`stderr`. Filtering for `both` includes combined lines; asking for `stdout` or `stderr` will not match a container's output.

#### A containerized llama.cpp service

```toml
[[service]]
name = "muse-glimmer"
template = "llama-cpp"
port = 8202
model = "/home/philpax/ai/muse-glimmer/models/muse-glimmer-30B-kquant-dynamic.gguf"
mmproj = "/home/philpax/ai/muse-glimmer/models/mmproj-kquant.gguf"
draft_model = "/home/philpax/ai/muse-glimmer/models/dflash-kquant.gguf"
spec_type = "draft-dflash"
context = 262144
health = { http = "/health", timeout = "5m" }

[service.container]
runtime = "docker"
image = "ghcr.io/ggml-org/llama.cpp:server-cuda"
network = "host"
gpu_device = "nvidia.com/gpu=${id}"
mounts = [
  { source = "/home/philpax/ai/muse-glimmer/models", target = "/models", read_only = true },
]
```

Only the generated flags follow the image; the image's own ENTRYPOINT runs them. `llama_server` is not consulted here: it answers where llama-server lives on the host, and its `llama-server` default is a `$PATH` lookup that means nothing inside an image. The official llama.cpp image ships `/app/llama-server` with `/app` off `$PATH`, so an entrypoint derived from that default would fail to resolve. An image that needs something other than its declared ENTRYPOINT says so with `container.entrypoint`.

The three model paths are translated through the mount before creation, becoming `-m /models/muse-glimmer-30B-kquant-dynamic.gguf` and so on.

#### A containerized OpenAI-compatible server

vLLM's image declares its own entrypoint, so the command supplies arguments only:

```toml
[[service]]
name = "diffusiongemma-26b-a4b"
template = "command"
port = 8200
command = [
  "--model", "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
  "--max-model-len", "131072",
  "--gpu-memory-utilization", "0.715",
  "--host", "${listen_host}",
  "--port", "${listen_port}",
]
env = { VLLM_NO_USAGE_STATS = "1" }
allocation = { mode = "static", reserve_gb = 23 }
health = { http = "/health", timeout = "10m" }
openai_proxy = { upstream_model = "nvidia/diffusiongemma-26B-A4B-it-NVFP4" }

[service.container]
runtime = "docker"
image = "vllm/vllm-openai:v0.26.0"
network = "bridge"
container_port = 8000
ipc = "host"
env_passthrough = ["HF_TOKEN"]
gpu_device = "nvidia.com/gpu=${id}"
mounts = [
  { source = "/home/philpax/.cache/huggingface", target = "/root/.cache/huggingface" },
]
```

An image whose CMD is to be replaced instead takes the executable as its first argument (`command = ["ninfer-serve", "/artifacts/model.ninfer", ...]`), with the same block otherwise. The only field that differs between the Docker and Podman versions of either example is `runtime`.

#### Field Reference

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `runtime` | string | `docker` | Container runtime to drive. One of `"docker"`, `"podman"`. |
| `runtime_executable` | path | the runtime name | Absolute path to the runtime binary. Set this when the binary isn't on the daemon's `$PATH` (e.g. a Nix store path). The value is recorded in the launch intent, so a container stays reconcilable across a runtime change. |
| `image` | string | *required* | Image reference to run, e.g. `vllm/vllm-openai:v0.26.0`. Must already be in the runtime's local store: ananke neither pulls nor builds, and a missing image fails the start with the runtime's own error. |
| `entrypoint` | string | the image's own | Replaces the image ENTRYPOINT. Unset, the image's own applies, including for `llama-cpp` services, whose `llama_server` is a host-side path and is not used inside the container. |
| `workdir` | path | the image's own | Working directory inside the container. |
| `network` | string | `bridge` | Network mode, one of `"bridge"`, `"host"`. See [Networking](#networking) for what each resolves the service endpoint to. |
| `container_port` | u16 | *required in bridge mode* | Port the workload binds inside the container. ananke publishes `127.0.0.1:<private_port>:<container_port>`. Rejected as meaningless under host networking. |
| `ipc` | string | `private` | IPC namespace, one of `"private"`, `"host"`. `host` shares the host `/dev/shm`, which vLLM and other multi-worker runtimes need. |
| `gpu_device` | string | none | CDI device template expanded once per GPU ananke's placement picked, e.g. `nvidia.com/gpu=${id}`. Must contain `${id}` exactly once. Unset means no GPU is injected; there is no `/dev/nvidia*` fallback. |
| `env` | table | none | Explicit environment for the container, merged over the service's own `env`. Container services never inherit the daemon's environment: `env_inherit` governs host processes only. |
| `env_passthrough` | array of string | none | Names of host environment variables forwarded into the container by name, e.g. `["HF_TOKEN"]`. Values are read from the daemon's environment at launch and never rendered into the API, previews, or logs. |
| `labels` | table | none | User labels applied to the container. The whole `io.ananke.*` namespace is reserved for ananke's ownership labels and rejected here. |
| `mounts` | array of table | none | Bind mounts (see [Mounts](#mounts)). |
| `extra_publications` | array of table | none | Port publications beyond the service endpoint (see [Extra publications](#extra-publications)). Bridge mode only. |

### Mounts

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `source` | path | *required* | Absolute host path. Must exist when the container is created. |
| `target` | path | *required* | Absolute container path. Two mounts may not share a target. |
| `read_only` | bool | `false` | Mount read-only. |
| `selinux` | string | none | SELinux relabel policy, one of `"z"`, `"Z"` (`z` is shared, `Z` is private). Omit on systems without SELinux. |

### Extra publications

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `host_ip` | string | `127.0.0.1` | Host address to bind. The default keeps the port off the network, matching the service endpoint's own publication. |
| `host_port` | u16 | *required* | Host-side port. |
| `container_port` | u16 | *required* | Container-side port. |
| `protocol` | string | `tcp` | One of `"tcp"`, `"udp"`. |

## Service Inheritance
Services can inherit configuration from other services using `extends`. This is useful for sharing common settings across related models:

```toml
# Base template: shared settings for all Gemma 4 models
[[service]]
name = "gemma4-base"
template = "llama-cpp"
flash_attn = true
cache_type_k = "q8_0"
cache_type_v = "q8_0"
context = 262144

# Child: inherits flash_attn, cache types, and context; overrides name, port, and model
[[service]]
name = "gemma-4-31b"
template = "llama-cpp"
extends = "gemma4-base"
port = 8200
model = "/models/gemma-4-31B.gguf"
```

Merge rules:

- Scalars: child overrides parent.
- Sub-tables: deep-merged field-by-field.
- Arrays: child replaces parent outright.
- `*_append` fields (e.g., `extra_args_append`): concatenated with parent's list.
- `name`, `port`, `extends`, and `template` must be overridden in the child.
- Cross-template inheritance is an error.

## Service Migration

When renaming a service, use `migrate_from` to preserve database history:

```toml
[[service]]
name = "gemma-4-31b"
template = "llama-cpp"
migrate_from = "old-gemma-31b"
port = 8200
model = "/models/gemma-4-31B.gguf"
```

