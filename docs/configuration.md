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
Services are defined as an array of `[[service]]` blocks. Each service uses one of two templates: `llama-cpp` (for GGUF models via llama.cpp) or `command` (for arbitrary binaries or Docker wrappers).

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
| `modality` | string | `"chat"` | `"chat"` or `"embedding"` (see [Embedding Services](#embedding-services)). Any other string is a hard config error. |
| `extra_args` | array of string | none | Extra argv appended to the service's launch command. |
| `extra_args_append` | array of string | none | Extra argv appended to the inherited list (use with `extends`; concatenated with parent's list). |
| `env` | map string → string | none | Environment variables set on the spawned process. Accepts `{port}`, `{gpu_ids}`, `{vram_mb}`, `{model}`, `{name}` placeholders. |
| `env_inherit` | bool | `true` | Whether the child process inherits the daemon's environment (`$PATH`, `$HOME`, locale, …). Per-service `env` entries override individual inherited keys. Set `false` to start with a clean environment containing only the variables in `env` plus `CUDA_VISIBLE_DEVICES`. |
| `drain_timeout` | duration string | `30s` | Drain timeout before the supervisor escalates to SIGKILL. |
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
| `split` | string | `"layer"` | Multi-GPU split mode for llama.cpp services: `"layer"`, `"row"`, or `"tensor"`. Maps to llama.cpp's `--split-mode`. See [Multi-GPU split modes](#multi-gpu-split-modes) for constraints. |

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

With a sharded mode, ananke reserves an equal share of the model weights, KV cache, and compute buffer on each allowed GPU, placing the non-layer remainder (output tensor, MTP overhead, …) on the main GPU. The pledge book reflects this per-GPU split, so a co-tenant (e.g. an embedding service) sees the true free capacity on each card.

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
  - `static`: Reserves a fixed amount of VRAM (`vram_gb`).
  - `dynamic`: Operates within a range (`min_vram_gb` to `max_vram_gb`).

In both modes the daemon picks the GPU with the most available headroom (subject to `gpu_allow`), preferring one whose free capacity satisfies the upper bound (`vram_gb` for `static`, `max_vram_gb` for `dynamic`) so dynamic services have room to grow. The picked GPU id is exported to the spawned child as `CUDA_VISIBLE_DEVICES`, and is also available as the `{gpu_ids}` placeholder in `command` argv. Wrappers that launch containers should forward this env (e.g. `docker run --device "nvidia.com/gpu=$CUDA_VISIBLE_DEVICES"`) so the picked GPU is the only one the container sees.

```toml
[service.allocation]
mode = "dynamic"         # "static" or "dynamic" (command services only)
vram_gb = 44             # static: fixed VRAM in GiB
min_vram_gb = 2.0        # dynamic: minimum VRAM in GiB
max_vram_gb = 12.0       # dynamic: maximum VRAM in GiB
min_borrower_runtime = "60s" # dynamic: balloon resolver grace period
```

**Eviction**: When VRAM is exhausted, ananke uses a priority-based eviction system. Higher priority services can displace dormant on-demand services.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `mode` | string | *required* (command only) | `"static"` or `"dynamic"`. Rejected for llama-cpp services. Applies to `command` services only. |
| `vram_gb` | f32 | none | `static` only. VRAM to reserve, in GiB. Required for `static`. |
| `min_vram_gb` | f32 | none | `dynamic` only. Minimum VRAM in GiB. Required for `dynamic`. |
| `max_vram_gb` | f32 | none | `dynamic` only. Maximum VRAM in GiB. Required for `dynamic`; must be > `min_vram_gb`. |
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

Self-healing for a `Running` service that is alive but degraded — the process has not exited, so the crash-detection path never fires, yet every request is failing or the process has accumulated dirty internal state. Two independent triggers both feed the existing drain → respawn cycle.

```toml
[service.auto_restart]
# Error-rate watchdog (on by default; write `error_rate = false` to opt out):
error_rate = { window = "2m", max_error_rate = 0.5, min_requests = 20, poll_interval = "30s", error_statuses = "5xx" }
# Periodic restart (off by default; a table with an interval enables it):
periodic = { interval = "6h", mode = "on-request" }
# Anti-flap guardrails, shared by both triggers:
min_uptime = "5m"
max_restarts = 3
flap_window = "30m"
```

The block is resolved as a **whole unit**: a service that sets any `auto_restart` field replaces `[defaults.auto_restart]` entirely rather than merging field-by-field. The same `[defaults.auto_restart]` block is accepted for fleet-wide defaults.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `error_rate` | table | `false` | on, with the defaults below | Error-rate watchdog. `false` disables it; a table enables it and overrides individual thresholds. |
| `periodic` | table | `false` | off | Periodic restart. Absent or `false` disables it; a table (with an `interval`) enables it. |
| `min_uptime` | duration string | `5m` | Minimum uptime a fresh run must reach before an error-rate restart may fire — the anti-flap cooldown. |
| `max_restarts` | u32 | `3` | Error-rate restarts tolerated within `flap_window` before the service is disabled with reason `auto_restart_loop` instead of restarted again. Periodic restarts are intentional and do not count toward this cap. |
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

When a trigger fires, the service drains (SIGTERM with grace, then SIGKILL) and returns to `Idle`; the normal ensure path respawns it — on the next request for an on-demand service, or within a few seconds for a persistent one. An auto-restart emits an `auto_restarted` event on the daemon event stream (see [the API guide](api.md)). Oneshot services never auto-restart.

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

- `{model}` - the model path. Held back from `{args}` so the wrapper can position it freely.
- `{name}` - service name.
- `{port}` - the private loopback port ananke assigned.
- `{gpu_ids}` - comma-separated NVML index list ananke picked for this service.
- `{args}` - splat: expands to every llama-server flag ananke would otherwise have emitted (everything except `-m <model>` - `--mmproj`, `-c`, placement-derived `-ngl`/`--tensor-split`/`-ot`, sampling, `--host`, `--port`, `extra_args`, …). Must occupy a launcher entry on its own; `"--foo={args}"` is rejected at config validation.

Example: wrap llama-server in a podman container that needs a volume mount for the model.

```toml
[[service]]
name = "qwen3-podman"
template = "llama-cpp"
port = 11436
model = "/srv/models/qwen3-30b.gguf"
context = 32768
flash_attn = true
launcher = ["/opt/podman-llama.sh", "{model}", "{args}"]
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
| `flash_attn` | bool | `false` | Enable flash attention. Required for quantised KV cache types (`cache_type_k`/`cache_type_v` other than `f16`). |
| `cache_type_k` | string | `f16` | KV cache type for keys. Non-`f16` values require `flash_attn = true`. |
| `cache_type_v` | string | `f16` | KV cache type for values. Non-`f16` values require `flash_attn = true`. |
| `mmap` | bool | `true` | Memory-map the model file. |
| `mlock` | bool | `false` | Lock the model in RAM (prevents swapping). |
| `parallel` | u32 | `1` | Request parallelism (`-np`). With a non-unified KV this splits the context budget across slots, so each request caps at `context / parallel`. |
| `spec_type` | string | none | Speculative-decoding type passed to `--spec-type` (e.g. `"draft-mtp"` for multi-token prediction). |
| `spec_draft_n_max` | u32 | none | Max draft tokens per step (`--spec-draft-n-max`). Only meaningful when `spec_type` is set. |
| `draft_model` | path | none | Separate draft-model GGUF for speculative decoding (`-md` / `--model-draft`). Requires `spec_type` to be set. |
| `kv_unified` | bool | `false` | Use a single unified KV cache pool shared across all parallel slots (`-kvu` / `--kv-unified`). With `parallel > 1`, idle slots lend their share to active ones; total KV footprint is unchanged. |
| `cache_idle_slots` | bool | `true` | When `false`, pass `--no-cache-idle-slots` so idle slots' prompt-cache state is dropped (a stability mitigation). |
| `metrics` | bool | `false` | Expose llama-server's Prometheus `/metrics` endpoint. |
| `slots` | bool | `false` | Expose the `/slots` introspection endpoint. Note: reveals prompt contents - avoid on network-reachable ports. |
| `batch_size` | u32 | none | Context batch size (`-b`). |
| `ubatch_size` | u32 | none | Physical batch size (`-ub`). |
| `threads` | u32 | none | Number of CPU threads (`-t`). |
| `threads_batch` | u32 | none | Number of CPU threads for batch processing (`-tb`). |
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
allow_fallback = false
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `compute_buffer_mb` | u32 | none | Override the estimated compute buffer size (MiB). |
| `safety_factor` | f32 | none | Multiplier applied to the estimated VRAM footprint. |
| `allow_fallback` | bool | `false` | Accept the coarse fallback estimate when the GGUF's architecture isn't recognised by any per-family estimator. Unknown architectures hard-reject at config load by default so the operator either adds the arch to the right family list or explicitly opts in here. |

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
Used for arbitrary binaries or Docker wrappers. Only `name`, `template`, `port`, and `command` are required.

```toml
[[service]]
name = "comfyui"            # required
template = "command"        # required
port = 8188                 # required
command = ["/bin/bash", "start_comfy.sh", "--port", "{port}"] # required
lifecycle = "on_demand"

[service.allocation]
mode = "dynamic"
min_vram_gb = 2.0
max_vram_gb = 12.0

[service.health]
http = "/system_stats"
timeout = "30s"
```

#### Field Reference

#### Placeholders

The following placeholders are substituted in `command` and `shutdown_command` argv entries (and in `env` values):

- `{port}` - the private loopback port assigned by ananke.
- `{gpu_ids}` - comma-separated NVML index list ananke picked for this service.
- `{vram_mb}` - reserved VRAM in MiB.
- `{model}` - model path (llama-cpp only; empty for command services).
- `{name}` - service name.

The child also inherits `CUDA_VISIBLE_DEVICES` set to the picked GPU id(s). Wrapper scripts that launch a container should forward this so the container only sees the picked GPU - for example, `docker run --device "nvidia.com/gpu=${CUDA_VISIBLE_DEVICES:-all}" ...`.

The `shutdown_command` field is particularly useful for external processes (like Docker containers) that cannot stop via signal alone. ananke runs this command after the drain pipeline completes, ensuring clean exits for services that don't respond to SIGTERM.

#### OpenAI Proxy

A `command`-template service that already speaks the OpenAI API (vLLM, TGI, SGLang, …) can opt into ananke's `/v1/models` and `/v1/chat/completions` multiplexer by adding an `[service.openai_proxy]` block. Without the block, command services stay invisible to the OpenAI surface and are only reachable via their per-service reverse proxy - the same as before.

```toml
[[service]]
name = "qwen3.6-27b-vllm"
template = "command"
port = 8210
command = ["/srv/vllm/qwen36_27b.sh", "{port}"]
lifecycle = "on_demand"
idle_timeout = "10m"

[service.allocation]
mode = "static"
vram_gb = 44

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

By default every service is registered as a chat model. Pooling-only embedding models (Jina v5, BGE, E5, …) opt in by setting `modality = "embedding"` on the service. The proxy itself is endpoint-agnostic - it already routes `POST /v1/embeddings` by `model` field - so the `modality` field is purely a typed declaration: clients filter on it through `/v1/models` and `/api/services`, and the frontend renders a teal `embedding` badge next to the service name (mirroring the purple `vision` badge for llama.cpp services with an `mmproj`).

```toml
[[service]]
name = "jina-embeddings-v5-text-small-retrieval-vllm"
template = "command"
port = 8211
modality = "embedding"
command = ["/srv/vllm/jina_embed_v5_small.sh", "{port}"]
lifecycle = "on_demand"
idle_timeout = "30m"

[service.allocation]
mode = "static"
vram_gb = 7

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
| `allocation` | table | none | VRAM allocation (see [Resource Allocation](#resource-allocation)). Required for command services. |
| `private_port` | u16 | auto-assigned | Upstream port ananke's reverse proxy should forward to. When absent, ananke picks one from the daemon's private-port pool and substitutes it into `command`/`env` via the `{port}` placeholder. Set explicitly when the external service binds a fixed port (e.g. a docker container exposing 18188 on the host). |
| `shutdown_command` | array of string | none | Optional argv run at drain time after SIGTERM-then-SIGKILL completes. Useful for external services that don't stop via signal - e.g. a docker-run wrapper where SIGTERM reaches the host shell but the container needs an explicit `docker stop`. Accepts the same placeholder substitutions as `command`. |
| `openai_proxy` | table | none | Opt the service into the OpenAI-compatible multiplexer (see [OpenAI Proxy](#openai-proxy)). |

### OpenAI Proxy

A `command`-template service that already speaks the OpenAI API (vLLM, TGI, SGLang, …) can opt into ananke's `/v1/models` and `/v1/chat/completions` multiplexer by adding an `[service.openai_proxy]` block. Without the block, command services stay invisible to the OpenAI surface and are only reachable via their per-service reverse proxy - the same as before.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `upstream_model` | string | none | Model name the upstream server was started with (e.g. via `--served-model-name`). ananke rewrites the JSON `model` field to this value before forwarding. |

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

