# Spec: Ananke (`ananke`)

GPU/CPU-aware model proxy for a single-host workstation running llama.cpp and ComfyUI-style services. Written in Rust. Binaries: `ananke` (daemon), `anankectl` (CLI).

## 1. Overview

Ananke is a single-host process supervisor that knows about GPUs. It proxies requests to on-demand-launched model services (primarily llama.cpp), schedules them onto the right devices given live memory availability, and coexists with dynamic services (ComfyUI) that grow and shrink their VRAM use. It exposes an OpenAI-compatible unified API on top of the per-service proxies and a web UI for configuration and observability.

**Non-goals.** Multi-host clustering; container-runtime orchestration beyond bare processes; arbitrary ML platform features.

## 2. TOML conventions

Dotted keys inside `[[service]]` blocks (`sampling.temperature = 0.7`) — never `[service.sampling]` sub-headers in array-of-tables, which are fragile under reordering. Inline tables for small atomic values.

## 3. Core data model

- **Device:** a GPU (via NVML) or the CPU (via `/proc/meminfo`). Each has `total_bytes` and live `observed_free_bytes`.
- **Template:** recipe for launching a class of service. Built-ins: `llama-cpp`, `command`.
- **Service:** a named, persistent configuration instance of a template. Identity is the `name` string.
- **Allocation:** mapping of a running service to one or more devices with reserved bytes per device. The allocation table is the single source of truth for placement; all decisions serialise through a scheduler task holding a mutex on it.
- **Run:** one invocation of a service's process. Services may have many runs over their lifetime.
- **Oneshot:** a service that lives only for its configured TTL or until it exits, submitted via API rather than config.

## 4. Device model

### 4.1 Autodetection

No manual device totals. GPUs via `nvml-wrapper`; CPU via `/proc/meminfo`. Config declares management scope and headroom only:

```toml
[devices]
gpu_ids = [0, 1]                   # omit to manage all visible GPUs
gpu_reserved_mb = { "0" = 512 }    # per-GPU, keys are GPU id as string
default_gpu_reserved_mb = 256
cpu.enabled = true
cpu.reserved_gb = 16
```

### 4.2 CPU accounting

The CPU device uses `MemAvailable` from `/proc/meminfo`, not `MemFree`. `MemFree` ignores reclaimable page cache and badly under-reports the actual available space for a new allocation; `MemAvailable` is the kernel's estimate of how much can be allocated without swapping, which matches what matters for "can I load a 40GB model."

### 4.3 `CUDA_VISIBLE_DEVICES` handling

The daemon **unsets** `CUDA_VISIBLE_DEVICES` from its own environment at startup so NVML sees every GPU the driver exposes. Ananke performs its own device selection via `devices.gpu_ids`; masking at the CUDA-runtime layer only confuses the picture, since NVML enumerates the full set regardless of that envvar anyway.

- GPU IDs in config are **NVML indices**. These are stable; if the user wants them PCI-bus-ordered they can set `CUDA_DEVICE_ORDER=PCI_BUS_ID` upstream of the daemon and the upstream ordering will match what NVML reports. The daemon logs the NVML-reported set at startup for clarity.
- Each child process is spawned with a freshly computed `CUDA_VISIBLE_DEVICES` built from its allocation. No ambient envvar is inherited; the child sees exactly the GPUs the scheduler assigned, renumbered from 0.
- `cpu-only` services are spawned with `CUDA_VISIBLE_DEVICES=""` so CUDA-aware code can't grab a GPU.

### 4.3.1 Non-NVIDIA hardware

v1 assumes NVIDIA hardware via NVML. The device-probing layer is a first-class abstraction (see §15.3's `GpuProbe` trait) so that AMD (ROCm/`rocm-smi`) and Intel (`xpu-smi`) backends can be slotted in without scheduler changes. If NVML is not available at startup, the daemon logs a warning and falls back to CPU-only operation; any `gpu-only` or `hybrid` services disable with `no_fit` until a probe is registered.

### 4.4 Virtual allocations for untracked usage

When `observed_free < total - sum(our_reservations)`, the gap is attributed to an **external reservation**: non-evictable, inferred, labelled with PID + process name where possible via `nvmlDeviceGetComputeRunningProcesses`. Shown in the UI as a grey block.

Effective availability per device:

```
available_i = total_i
            - reserved_headroom_i
            - sum(our_reservations on i)
            - external_observed_i

external_observed_i = max(0, (total_i - free_i) - sum(our_observed_usage on i))
```

### 4.5 Per-service placement policy

Every service declares where it may run:

```toml
devices.placement = "gpu-only"     # default; layer-aware across allowed GPUs
# devices.placement = "cpu-only"   # weights and KV in RAM, -ngl 0
# devices.placement = "hybrid"     # GPUs first, spill to CPU
devices.gpu_allow = [0]            # optional subset for gpu-only / hybrid
```

(For `cpu-only` the child's envvar handling is covered in §4.3.)

## 5. Service lifecycle

### 5.1 Lifecycle modes

- **`on_demand`** (default): idle until a request arrives, start on demand, unload after `idle_timeout` (default 10m) of no traffic. Evictable per §5.2.
- **`persistent`**: starts with the daemon (see §9.4 for startup ordering), never unloads on a timer. Still evictable per §5.2 while idle; a background watcher re-ensures every idle persistent service on a short cadence, so it comes back as soon as VRAM permits without requiring a new request.
- **`oneshot`**: launched via `POST /api/oneshot` rather than config. Consumes an allocation only while running. Exits automatically on TTL expiry or process termination. Not restarted. Declaring `lifecycle = "oneshot"` in a `[[service]]` config block is a validation error — oneshots are API-only.

### 5.2 Priority and eviction

Priority is a `u8` in [0, 100], default 50. It governs **busy** services only: a service currently serving an in-flight request is evictable only by a strictly-higher-priority placement. An **idle** service (no in-flight traffic) is always an eviction candidate regardless of priority — if nothing's using it, displacing it is free.

Within the candidate set the ranking is: idle before busy, lowest priority first, smallest allocation first. Ties are then broken by least-recently-used.

Consequence: priority is only load-bearing when two requests race for the same VRAM window concurrently, or when an incoming request would have to interrupt a busy service. For single-workload setups where nothing runs concurrently, it has no observable effect.

Tier conventions surfaced in the UI:

| Range | Tier | Meaning |
|---|---|---|
| 0–20 | Background | Displaced early when busy services contend |
| 30–50 | Normal | Default workload |
| 60–80 | Important | Preferred over normal in concurrent contention |
| 90–100 | Critical | Survives contention from all but peers |

### 5.3 States

| State | Meaning |
|---|---|
| `starting` | Process spawned, waiting for health check |
| `running` | Healthy and serving requests |
| `draining` | Marked for shutdown, waiting for in-flight requests |
| `idle` | Configured, not currently running (on_demand, no traffic) |
| `stopped` | Configured, explicitly stopped via API/CLI |
| `evicted` | Recently evicted to free memory; may auto-restart on next request |
| `failed` | Last run crashed; retry backoff pending |
| `disabled` | Requires user intervention before it will start again |

Valid transitions:

```
idle → starting → running → draining → {idle, stopped}
stopped → starting (explicit start via API/CLI)
starting → failed (launch/health fail)
running → evicted (scheduler) → idle
* → disabled (auto-disable trigger or user action)
disabled → idle (re-enable)
failed → starting (retry backoff elapsed)
failed → disabled (retry limit reached)
```

### 5.4 Disable reasons

`config_error`, `launch_failed`, `health_timeout`, `oom`, `crash_loop`, `no_fit`, `user_disabled`.

Auto-disable triggers:

- **Config error** (parse, collision, missing required field): the offending service disables; rest of config loads.
- **Launch failure** (binary missing, immediate exit): backoff retry 3× (2s, 5s, 15s) then `disabled(launch_failed)`.
- **Health timeout**: SIGTERM + drain, `disabled(health_timeout)`.
- **OOM at launch or within 30s of starting**: one retry with fudge bumped 1.1 → 1.4. Second OOM → `disabled(oom)` with observed-vs-reserved delta logged.
- **Crash loop**: > 5 exits in 10 minutes → `disabled(crash_loop)`.
- **No fit at startup**: a `persistent` service couldn't be placed → `disabled(no_fit)`.

Recovery: "Re-enable" in UI, `anankectl enable <name>`. Editing the config to resolve a `config_error` auto-clears that reason on next load; other reasons require explicit re-enable to avoid silent loops.

Fatal daemon-level conditions (exits non-zero): cannot bind management or OpenAI port; cannot open SQLite DB; config file unparseable as TOML; no devices available at all.

## 6. Configuration

### 6.1 File location

Search order:

1. `$ANANKE_CONFIG` if set.
2. `--config` CLI argument.
3. `$XDG_CONFIG_HOME/ananke/config.toml` (default `~/.config/ananke/config.toml`).
4. `/etc/ananke/config.toml`.

First match wins. Log the resolved path at startup.

Data directory (SQLite DB, one-shot history) follows the same pattern under `$XDG_DATA_HOME/ananke/` (default `~/.local/share/ananke/`).

### 6.2 Top-level structure

```toml
[daemon]
management_listen = "127.0.0.1:7071"   # management API + web UI
data_dir = "~/.local/share/ananke"     # SQLite + state

[devices]
# see §4

[openai_api]
listen = "127.0.0.1:7070"
enabled = true
max_request_duration = "10m"

[defaults]
idle_timeout = "10m"
priority = 50
start_queue_depth = 10

[[service]]
# one block per service; see §7
```

### 6.3 Inheritance

```toml
[[service]]
name = "qwen3-coder-32b"
template = "llama-cpp"
model = "/models/qwen3-coder-32b-q4_k_m.gguf"
port = 11434
context = 32768
flash_attn = true
cache_type_k = "q8_0"
cache_type_v = "q8_0"
sampling.temperature = 0.7
sampling.top_p = 0.95
devices.placement = "gpu-only"

[[service]]
name = "qwen3-coder-32b-thinking"
extends = "qwen3-coder-32b"
port = 11444                             # child must override
sampling.temperature = 0.3
chat_template_file = "/templates/qwen-thinking.jinja"
extra_args_append = ["--verbose"]
```

Merge semantics, applied at parse time before validation:

- Scalars and dotted-leaves: child overrides parent.
- Sub-tables: deep-merged field-by-field.
- Arrays: child replaces parent outright. (E.g., child `extra_args = [...]` replaces the parent's `extra_args` entirely.)
- `*_append` siblings: always additive. The effective value is `parent.foo ++ parent.foo_append ++ child.foo ++ child.foo_append`, where `child.foo` either replaces (if specified) or falls back to `parent.foo` (if not). This lets a child add to the parent's arg list without knowing its contents.
- `filters.set_params`: map values replace per-key (child's `temperature` overrides parent's). Child-declared value for a key whose value is an array replaces that array; no merging.
- `name` and `port`: child must override; inheriting either is an error.
- `extends` is transitive; cycles are errors.
- `extends` and `migrate_from` are themselves not inherited.

### 6.4 `migrate_from` for renames

```toml
[[service]]
name = "qwen3-coder-v2"
migrate_from = "qwen3-coder-v1"
# ...rest
```

On load: if `qwen3-coder-v1` exists (live or tombstoned), reparent its `service_id` to `qwen3-coder-v2`; tombstone the old name. UI flags the field with "migration complete, remove this line?" — non-destructive, never auto-stripped.

Chains (A → B → C in one load) apply in dependency order. Cycles are errors. Missing source is a warning and proceeds without adoption.

### 6.5 Collision and validation

Checked before any service starts. Distinct from model-level validation (§8.3), which can only run after reading a GGUF and surfaces as `disabled(config_error)` on first start attempt.

File-level (config load):

- Duplicate `name` or `port` → error.
- `extends` target missing or cyclic → error.
- Unknown template → error.
- Required fields missing after merge → error (e.g., `llama-cpp` without `model`).
- `port` equal to `daemon.management_listen` port or `openai_api.listen` port → error.
- `cache_type_k != "f16"` or `cache_type_v != "f16"` with `flash_attn = false` → error (llama.cpp requires FA for quantised KV).
- `devices.placement = "cpu-only"` with `n_gpu_layers != 0` → error.
- `lifecycle = "oneshot"` on a `[[service]]` block → error.
- `migrate_from` naming a service that also exists live in the same config → error (resolve by removing the `migrate_from` line once the migration has taken effect on a prior load).
- Two `static` services pinned to overlapping exclusive devices exceeding total → warning.

Model-level (first start, once GGUF is read):

- `flash_attn` or `cache_type_*` on Mamba/SSM architectures → `disabled(config_error)`.
- Unknown architecture with quantised KV → warning (estimator falls back; user should validate).

Errors carry file + line + column via `toml_edit` span tracking.

### 6.6 Reload atomicity

`notify` watches the config file. If the resolved path is a symlink (standard Nix pattern, where `/etc/ananke/config.toml` points into the Nix store), the daemon watches the **parent directory** for events on that filename rather than the symlink target; otherwise a Nix switch swapping the symlink atomically would not fire a notification.

On event:

1. Debounce 500ms (editors often fire multiple events per save).
2. Atomic read. If mid-write (truncated, invalid UTF-8), skip and await the next event.
3. Parse + validate the full config. On failure, **keep the live config** and surface the error in the UI. Never blow away working state on a bad edit.
4. On success, diff against live: additions start per lifecycle, removals SIGTERM + drain, changes restart if command changed or update in place if not.

Web-editor writes: `toml_edit` rewrite preserving comments → temp file → `fsync` → atomic `rename`. File mtime + content hash checked before write to detect external edits since last read; conflict surfaced as a diff dialog.

### 6.7 NixOS integration

`pkgs.formats.toml { }` generates the file from Nix. A typed module in `nixos/module.nix` exposes options mapping 1:1 to the TOML schema; it's the ergonomic skin, the TOML is the contract. Module and daemon co-located in-repo so schema drift is a single-repo concern.

Flake output: `nixosModules.default`.

## 7. Templates

### 7.1 `llama-cpp`

Full field surface:

```toml
[[service]]
name = "llama3-70b"
template = "llama-cpp"
model = "/models/llama-3.3-70b-instruct-q4_k_m.gguf"
mmproj = "/models/llama-3.3-70b-mmproj.gguf"     # optional; vision models only. Rendered as --mmproj.
port = 11435
context = 16384

# Lifecycle
lifecycle = "on_demand"                # "on_demand" | "persistent" | "oneshot"
priority = 50
idle_timeout = "10m"
description = "Llama 3.3 70B"

# llama-server passthroughs
n_gpu_layers = -1                      # -1 = scheduler decides
n_cpu_moe = 0                          # MoE expert layers to offload to CPU
flash_attn = true
cache_type_k = "q8_0"
cache_type_v = "q8_0"
mmap = true
mlock = false
parallel = 1
batch_size = 2048
ubatch_size = 512
threads = 24                           # generation threads; default = physical cores
threads_batch = 24                     # prompt/batch threads; default = same as `threads`
jinja = true
chat_template_file = "/templates/llama3-system.jinja"

# Tensor-level placement override (llama.cpp's -ot / --override-tensor).
# Array preserves rule order; first match wins, matching llama.cpp semantics.
# Each string is `<regex>=<device>` where device is CPU, GPU0, GPU1, ...
override_tensor = [".ffn_(up|down)_exps.=CPU"]

# Sampling (surfaced for convenience; less-common samplers go in extra_args)
sampling.temperature = 0.7
sampling.top_p = 0.95
sampling.top_k = 40
sampling.min_p = 0.05
sampling.repeat_penalty = 1.1

# Request-body rewrites (applied by the unified OpenAI proxy)
filters.strip_params = ["temperature", "top_p"]
filters.set_params = { temperature = 0.7, max_tokens = 4096 }

# Arbitrary metadata exposed via /v1/models and /api/services/<name>
metadata.tags = ["general"]
metadata.description_long = "Q4_K_M quant."

# Placement
devices.placement = "gpu-only"
# Manual override when the estimator can't figure it out (see §8.2.6):
# devices.placement_override = { cpu = 87000, "gpu:0" = 18944, "gpu:1" = 18944 }  # MB

# VRAM estimator tuning (rarely needed)
estimation.compute_buffer_mb = 400     # default; bumps safety factor
estimation.safety_factor = 1.1         # default; bumped to 1.4 after OOM

# Escape hatches
extra_args = ["--metrics"]
extra_args_append = []
env = { CUDA_DEVICE_ORDER = "PCI_BUS_ID" }

# Health / shutdown
health.http = "/v1/models"             # default for llama-cpp
health.timeout = "180s"                # wall-clock from launch
health.probe_interval = "5s"           # time between probes
drain_timeout = "30s"
extended_stream_drain = "30s"
max_request_duration = "10m"           # overrides openai_api default
```

### 7.2 `command`

```toml
[[service]]
name = "comfyui"
template = "command"
command = ["python", "main.py", "--port", "{port}", "--listen", "127.0.0.1"]
workdir = "/srv/comfyui"
env = { PYTHONUNBUFFERED = "1" }
port = 8188

lifecycle = "on_demand"
priority = 70

allocation.mode = "dynamic"            # "static" | "dynamic"
allocation.min_vram_gb = 4
allocation.max_vram_gb = 20
# For static:
# allocation.mode = "static"
# allocation.vram_gb = 6

devices.placement = "gpu-only"
devices.gpu_allow = [0]

health.http = "/system_stats"
drain_timeout = "5s"
```

### 7.3 Placeholders

Substituted in `command` vector and `env` values at spawn time:

- `{port}` — the declared service port
- `{gpu_ids}` — comma-separated GPU ids from the allocation
- `{vram_mb}` — per-device reservation in MB; only valid when allocation is single-GPU and `static`. Using it on a `dynamic` allocation is a validation error (the number would be meaningless — it's only substituted once at spawn and cannot track growth).
- `{model}` — the `model` field if present
- `{name}` — service name

`CUDA_VISIBLE_DEVICES` is set automatically from the allocation; commands that respect it don't need `{gpu_ids}`.

## 8. Scheduling

### 8.1 Allocation algorithm

Per request:

1. **Compute need.** For `llama-cpp`, run the architecture-aware estimator (§8.3). For `command` static, use the declared value. For `command` dynamic, use `min_vram_gb`.
2. **Single-device best-fit.** Walk allowed devices, pick the one whose `available` is smallest while still sufficient. GPUs before CPU.
3. **Multi-GPU layer-aware split** (llama-cpp only; §8.2).
4. **GPU+CPU hybrid spill** (llama-cpp, `placement = "hybrid"`; §8.2).
5. **Eviction.** Collect candidates per §5.2: any idle service, plus busy services with strictly lower priority than the incoming request. Sort idle-first, then lowest-priority, then smallest allocation. Kill minimum set with drain (§10.3). Retry placement.
6. **Queue or reject.** Proxied requests queue up to `start_queue_depth`; other requests reject immediately with `insufficient_vram` / `no_evictable_services` / `start_queue_full`.

### 8.2 Layer-aware placement

Since GGUF enumerates tensors with names, shapes, and types at the head of the file, per-layer weight cost is cheap to compute. Group by `blk.N.*` for layer tensors; embeddings and output head are separate.

Algorithm: walk layers in index order, pack each onto the first allowed device with room, preferring GPUs. Convert to llama.cpp invocation:

- Total layers on GPUs → `-ngl N`
- Per-GPU layer count → `--tensor-split A,B,...` ratios, computed as normalised layer counts per device
- Remaining layers fall off the end → implicit CPU (llama.cpp handles this given `-ngl < total_layers`)

Unequal GPU splits are the natural output: a GPU with 18GB free takes more layers than one with 10GB. Equal split is the degenerate case.

#### 8.2.1 Non-layer tensor placement

llama.cpp places non-layer tensors deterministically when offloading: the output head (`output.weight`, often 1–2 GB on a 70B-class model) goes to GPU 0, and the token embeddings (`token_embd.weight`) stay on CPU. Ananke's per-device decomposition mirrors this:

- Output head bytes added to GPU 0's reservation (or whichever GPU is first in the allocation).
- Token embeddings added to CPU reservation — always, even in `gpu-only` mode, since llama.cpp keeps them resident on CPU regardless.
- Other non-layer tensors (norms, rope tables, etc.) are small; attributed to GPU 0 if any GPU is used, CPU otherwise.

#### 8.2.2 Compute buffer per backend

llama.cpp allocates a compute buffer per *active backend*, not per model. Ananke adds `estimation.compute_buffer_mb` (default 400) to each device that hosts any layer or non-layer tensor. A 2-GPU + CPU hybrid deployment therefore budgets three compute buffers, not one. `cpu-only` gets one (CPU); single-GPU `gpu-only` gets one (that GPU).

#### 8.2.3 Per-device decomposition

Combining the above, for hybrid placement the per-device reservation is:

```
reserve_gpu_i  = sum(weights of layers packed on gpu_i)
               + kv_bytes_per_token × context × (layers_on_gpu_i / n_layers)
               + (output_head_bytes if i == 0 else 0)
               + compute_buffer_mb × 1MB
               + one_layer_fudge                        # see §8.2.5

reserve_cpu    = sum(weights of layers that spilled to CPU)
               + kv_bytes_per_token × context × (layers_on_cpu / n_layers)
               + token_embd_bytes
               + compute_buffer_mb × 1MB
               + one_layer_fudge                        # §8.2.5
```

Each line is then multiplied by `safety_factor`. Rolling correction (§8.3) tunes the total, applied pro-rata to each device's reservation — since we can only observe a single per-service scaling factor, not per-device ones.

#### 8.2.4 Tensor-level placement override (`override_tensor`)

The llama-cpp template exposes `override_tensor` as a first-class array of `<regex>=<device>` rules (corresponding to llama.cpp's `-ot` / `--override-tensor` flag). Ananke reads this field directly; users should not put `-ot` in `extra_args`, since Ananke wouldn't see it and the estimator would mis-attribute the affected tensors.

Semantics:

- Before the layer walker runs, tensors matching any rule have their bytes attributed to the specified device (`CPU`, `GPU0`, `GPU1`, …) rather than following layer placement.
- The layer walker then packs only the residual tensors (those not matched by any rule) across allowed devices normally.
- Rules apply in array order; first match wins, matching llama.cpp's behaviour.
- Ananke renders the rules into `-ot` arguments on the llama.cpp command line automatically — the user never writes them twice.
- Invalid regex or unknown device in a rule is a `config_error` at parse time (no first-start delay needed).

Example: `override_tensor = [".ffn_(up|down)_exps.=CPU"]` on an MoE model classifies every `blk.N.ffn_(up|down)_exps.weight` tensor as CPU-resident; the layer walker then packs remaining per-layer tensors (attention, routers, gate experts, shared experts) across GPUs normally. This is the idiomatic way to run large MoE models hybrid — much more surgical than `n_cpu_moe` and worth supporting natively.

#### 8.2.5 Tensor-split slop and boundary fudge

llama.cpp uses the tensor-split ratios to do its own layer assignment with integer rounding; the result is not guaranteed to match Ananke's packing exactly, and can drift by up to one layer per GPU boundary. Ananke accounts for this by adding a **per-device fudge** equal to one layer's weight + KV cost to every device that holds at least one layer — GPUs *and* CPU when the GPU→CPU boundary is in play. Rolling correction (§8.3) converges on the true footprint over the first few runs.

#### 8.2.6 Manual placement override

For cases Ananke's estimator cannot reason about — novel architectures, complex `-ot` pipelines, models whose real footprint has been measured empirically — the user can bypass the estimator entirely:

```toml
devices.placement_override = { cpu = 87000, "gpu:0" = 18944, "gpu:1" = 18944 }  # MB
```

When set:

- The scheduler reserves exactly these values; the estimator is not run.
- `devices.placement` is still respected for validation (overriding CPU bytes with `placement = "gpu-only"` is an error).
- The user is responsible for getting llama.cpp to actually match the declared layout, typically via hand-tuned `extra_args` (`-ngl`, `--tensor-split`) combined with `override_tensor` rules.
- Rolling correction still runs and will warn if observed usage drifts meaningfully from the override (>30%) — the override is trust, not blind obedience.
- `n_gpu_layers`, `n_cpu_moe`, and automatic `--tensor-split` derivation are all disabled when override is active; pass everything explicitly via `extra_args`.

`cpu-only`: skip GPU placement entirely, `-ngl 0`.

### 8.3 VRAM estimation

Architecture-aware, dispatched on `general.architecture`:

**Llama-family** (`llama`, `qwen2`, `qwen3`, `mistral`, `gemma`, `gemma2`, `gemma3`, `phi3`, `glm4`, etc.): standard transformer. Sum per-layer tensor sizes; KV cache from `<arch>.attention.head_count_kv`, `<arch>.attention.key_length`, `<arch>.attention.value_length`:

```
kv_per_token = n_layers × n_kv_heads ×
               (key_length × bytes(cache_type_k) + value_length × bytes(cache_type_v))
```

**MoE variants** (`llama4`, `qwen3moe`, `deepseek2`, `mixtral`, `gpt-oss`): expert tensors identified by `_exps` suffix on `blk.N.ffn_{gate,up,down}_exps.weight`. All experts are memory-resident unless offloaded. If `n_cpu_moe > 0`, subtract expert-tensor sizes of the top-N expert-bearing layers from GPU cost, add to CPU cost. Non-expert tensors (shared experts via `_shexp`, router via `gate_inp`, attention) placed normally.

**MoE × hybrid composition order.** When both `n_cpu_moe` and `placement = "hybrid"` are set, expert offload applies **first** — the layer walker (§8.2) sees reduced per-layer GPU costs for the top-N layers, then packs the residual across GPUs and potentially spills more to CPU. If `override_tensor` rules are also present (§8.2.4), they apply before both: `override_tensor` → `n_cpu_moe` → layer walker. This ordering matches what the user expects when composing multiple offload mechanisms, and avoids double-counting expert bytes.

**SSM/Mamba** (`mamba`): no conventional KV cache. State cost derived from `mamba.ssm.state_size`, `mamba.ssm.conv_kernel`, `mamba.ssm.inner_size`. `flash_attn` and `cache_type_*` don't apply; passing them is a validation error for this architecture.

**Hybrid** (`jamba` and future): per-layer type read from architecture metadata where available. KV cache only for attention layers.

**Unknown architecture fallback**: `total_tensor_bytes × 1.15 + 512MB`. Log a warning so the user knows estimates are coarse. Rolling correction (below) gradually refines.

**Multimodal projector (`mmproj`).** Vision models carry a separate projector file loaded via `--mmproj`. When the `mmproj` field is set, read the projector's GGUF header the same way as the main model, sum its tensor bytes into the GPU weights budget (projectors live on GPU 0 alongside the output head; they are small — typically 0.5–1.5 GB — and llama.cpp does not offload them to CPU), and render `--mmproj <path>` on the command line automatically. The projector contributes to `weights` in the final estimate; it does not affect KV cache sizing.

**Sharded GGUFs.** llama.cpp supports multi-file models; the user points `--model` at the first shard and llama.cpp discovers the rest. Each shard is a valid GGUF file whose header carries `split.no` (0-indexed shard number), `split.count` (total shards), and `split.tensors.count` (total tensor count across the set). Ananke handles shards as follows:

- At config load, read the GGUF header of the file named in `model`. If `split.count > 1`, it's a shard; if `split.no != 0`, normalise to the shard-0 filename using the conventional pattern `{basename}-{NNNNN}-of-{MMMMM}.gguf` (shard numbers are 1-indexed in filenames, 5-digit zero-padded, `00001` is shard-0 per `split.no`) and re-read from shard 0.
- Walk shards 0 through `split.count - 1`, reading each header. Sum tensor byte totals across shards for the estimator. Architecture metadata is read from shard 0.
- If any shard is missing or unreadable, or the `split.count` values disagree between shards, error with `config_error` naming the offending file.
- Layer-aware placement (§8.2) treats the shard set as a single virtual file; layer indices are global across the shards.
- Pass the shard-0 path to llama-server unchanged; llama.cpp loads the rest itself.

KV bytes-per-element table, matching llama.cpp's accepted `--cache-type-k` / `--cache-type-v` values:

| Type | Bytes/element |
|---|---|
| f32 | 4.00 |
| f16, bf16 | 2.00 |
| q8_0 | ~1.06 |
| q5_1 | ~0.75 |
| q5_0 | ~0.69 |
| q4_1 | ~0.63 |
| q4_0, iq4_nl | ~0.56 |

KV quantisation below f16/bf16 requires `flash_attn = true`; enforced at config load.

Final estimate: `weights + kv_per_token × context + compute_buffer_mb × 1MB`, then × `safety_factor`.

**Rolling correction.** Observed peak VRAM captured after the service has been Running long enough for weights to load. Per-service rolling mean of `observed_peak / base_estimate`:

```
adjusted_estimate = base_estimate × clamp(rolling_mean, 0.8, 1.5)
```

Cap prevents anomalous samples from destabilising estimates. A persistent `rolling_mean > 1.2` raises an under-estimation warning — the architecture-specific estimator is probably missing something. A persistent `rolling_mean < 0.85` raises an over-reservation warning — the service is consistently holding less VRAM than claimed, and other services could be getting crowded out unnecessarily.

**Estimator drift signal.** Whenever `|rolling_mean - 1.0| > 0.3` persistently (sustained across ≥5 runs), emit an `estimator_drift` event on `/api/events` carrying the service's effective config, observed-vs-estimated deltas, and GGUF architecture metadata. The UI surfaces this as a diagnostic prompt: the estimator is doing a lot of corrective work for this service, which usually means the architecture-specific path is missing something model-specific. The user can paste the diagnostic into a bug report without having to reconstruct it by hand.

**CPU-side observation.** NVML only reports GPU memory; for services with a CPU portion (hybrid, cpu-only, or any service using `-ot '...=CPU'`), Ananke also samples `/proc/<pid>/status` `VmRSS` at the same 2s cadence used for GPU accounting. CPU observed usage feeds into:

- Rolling correction, summed with GPU observed usage before dividing by the total estimate.
- The balloon resolver's `observed_free` accounting for the CPU device (§4.4).
- The placement-override drift warning (§8.2.6).

Note that `VmRSS` includes mmap'd weight pages that are resident; if `mmap = true` (the default) this tracks what actually matters for "is this process taking CPU memory from other services." `mlock = true` pins everything and `VmRSS` converges on the full weight size immediately.

### 8.4 Balloon resolver (dynamic services)

A `dynamic` service declares `(min_vram_gb, max_vram_gb)`. Runtime behaviour:

- Starts with `min_vram_gb` reserved on allocated device(s).
- Daemon samples `observed_usage` every 2s, keeping a rolling 6-sample window (12s).
- Elastic region = `max_vram_gb - current_observed - margin` (margin default 512MB).

Other services may be placed into the elastic region, tagged as **elastic borrowers**. The resolver uses borrower **observed** usage (not borrower reservation) when deciding whether the dynamic service is constrained — a borrower that drifts above its reservation naturally triggers contention sooner, which is the correct incentive.

**Growth detection.** Compute slope over the 6-sample window. If slope positive and projected next-sample usage would exceed the floor currently claimed by the borrower, trigger contention resolution:

- Dynamic service priority > borrower priority → evict borrower.
- Borrower priority ≥ dynamic service priority → evict dynamic service. Rare in practice (operators typically give dynamic services higher priority), but consistent with the priority model.

**Fast-path SIGKILL.** When the dynamic service is actively allocating into contested space, the standard drain sequence in §10.3 is too slow to prevent an OOM on either side. The balloon resolver therefore uses a fast-path:

1. Mark the loser `draining` and refuse new requests.
2. Send SIGTERM immediately; start a **5-second** grace timer (not the normal `drain_timeout`).
3. If the loser hasn't exited within 5 seconds, SIGKILL. In-flight requests are severed.

This is deliberately harsh: dynamic services (ComfyUI-style) grow in seconds, and preserving in-flight requests to the loser is worth less than preventing a crash on the winner. For non-contention evictions (higher-priority scheduler placement, config reconcile), the full §10.3 drain applies.

**Minimum borrower runtime.** Do not evict a borrower that started less than `min_borrower_runtime` (default 60s, configurable per service) ago, to prevent thrashing under oscillating usage. During this grace window the dynamic service queues instead; if the queue times out the dynamic service's request fails.

**Jitter tolerance.** Raw NVML samples have ~±10% noise from CUDA context initialisation and allocator behaviour. The slope detector requires a positive trend across the majority of the window, not a single high sample.

**Exceeded `max_vram_gb`.** If observed usage exceeds the declared ceiling by more than 10% for more than 30 seconds, the dynamic service is **SIGKILL**ed (draining a leaking process would let the leak grow for another `max_request_duration`). Logged loudly.

### 8.5 OOM retry policy

If a service OOMs during launch or within the first 30 seconds of running: one automatic retry with `safety_factor` bumped from 1.1 to 1.4. If the retry also OOMs, `disabled(oom)` with the observed-vs-reserved delta logged so the user can see how much the estimator under-predicted. Rolling correction updated regardless.

## 9. Process supervision

### 9.1 Child lifetime binding

Children spawn with `prctl(PR_SET_PDEATHSIG, SIGTERM)` on Linux, so they die if the daemon dies. This is the primary defence against orphans.

### 9.2 PID tracking

On spawn, a row is inserted into `running_services`:

```sql
CREATE TABLE running_services (
  service_id    INTEGER NOT NULL,
  run_id        INTEGER NOT NULL,
  pid           INTEGER NOT NULL,
  spawned_at    INTEGER NOT NULL,        -- unix ms
  command_line  TEXT NOT NULL,           -- argv joined
  allocation    TEXT NOT NULL,           -- JSON: [{device, bytes}]
  state         TEXT NOT NULL,           -- 'starting' | 'running' | ...
  PRIMARY KEY (service_id, run_id)
);
```

Updated transactionally on state changes. Removed on clean exit.

### 9.3 Startup orphan recovery

On daemon startup, before scheduling:

1. Read `running_services` rows persisted from the last session.
2. For each row, check PID liveness (`kill(pid, 0)` or `/proc/<pid>`).
3. If alive and `/proc/<pid>/cmdline` matches the recorded command line → adopt: mark as running with its allocation. First request triggers a health check.
4. If alive and cmdline doesn't match → unrelated process reusing the PID. Log and clean up the row without killing.
5. If dead → clean up the row.
6. Additionally, scan NVML's per-process VRAM accounting for PIDs holding memory on our managed GPUs that we don't recognise. Before SIGTERM-ing any such process, require **all** of: (a) the PID is bound to one of our managed service ports (via `/proc/net/tcp{,6}` + `/proc/<pid>/fd` inode correlation), **and** (b) `/proc/<pid>/cmdline` matches a `command_line` value stored in `running_services` for that port's service, possibly from an earlier tombstoned run. Both conditions ensure we're killing our own orphan, not a legitimate process that happened to bind a conflicting port. If either check fails, log the suspected orphan and leave it alone — the user will see a port-bind failure when the service tries to start and can intervene.

### 9.4 Startup ordering for persistent services

After orphan recovery, persistent services start in order of `priority` descending, then `name` ascending for determinism. Each placement uses the normal scheduler.

If a persistent service doesn't fit, do **not** evict other persistent services — two persistents that can't coexist is a config error the user should resolve. Instead mark `disabled(no_fit)` and continue with remaining persistents. UI surfaces the shortfall.

### 9.5 Clean shutdown

SIGTERM and SIGINT are handled identically, triggering graceful shutdown. (SIGINT is Ctrl-C in the foreground; SIGTERM is what systemd sends. Treating them the same keeps development and production behaviour consistent.)

Sequence:

1. Stop accepting new requests.
2. Drain active ones, bounded by `daemon.shutdown_timeout` (default **120s**) — a daemon-wide ceiling that caps total drain regardless of per-service `max_request_duration`. A 60-minute LLM request shouldn't block `systemctl stop`.
3. SIGTERM all children with 15s grace.
4. SIGKILL stragglers.
5. Clear `running_services`. Close SQLite cleanly.

SIGQUIT (Ctrl-\\) is reserved for "emergency stop": skip drain, SIGTERM children immediately with 5s grace, then SIGKILL.

### 9.5.1 Daemon logging

Daemon `tracing` output goes to stderr. Under systemd this is captured by journald via the standard unit wiring; under `cargo run` it appears on the terminal. No file-based daemon log; per-service logs remain in SQLite (§12).

### 9.6 Weight-loading grace

A freshly-spawned service whose health probe has passed may still be paging weights into VRAM for tens of seconds — NVML observed usage lags the reservation during this window. While `(now - run.spawned_at) < weight_loading_grace` for a given run:

- Drift detection is suppressed against that device's reservation. Without this, the balloon resolver would hand the gap to a borrower that would OOM when loading completes.
- Elastic borrowing into this service's region is paused.

Once observed usage reaches ≥80% of reservation, or the grace expires, the service is treated as fully loaded for drift / elastic purposes. There is no separate user-visible `warming` state — the service is `running` throughout, and this grace is purely an internal signal to the drift / balloon subsystems.

## 10. Proxy

### 10.1 Two listeners

**Per-service ports** — each service listens on its declared port for transparent pass-through. No filters applied. Clients that want the llama-server admin UI or direct API use these.

**Unified OpenAI listener** on `openai_api.listen` — routes by the `model` field in the request body, applies filters, handles model-level 503s. OpenAI-compatible clients use this.

### 10.2 Unified endpoint behaviour

| Endpoint | Method | Behaviour |
|---|---|---|
| `/v1/models` | GET | Lists all `starting`, `running`, or `idle` services whose template produces an OpenAI-compatible server (every `llama-cpp` service; `command` services are never listed). `disabled`, `failed`, and `stopped` services hidden. `starting` is included because requests for such models are accepted (they queue on the start future, §10.4); omitting them would make the listing inconsistent with what's actually routable. Response: `{object: "list", data: [{id, object: "model", created, owned_by: "ananke", ananke_metadata: {...}}]}` — `ananke_metadata` is a passthrough of the service's `metadata.*` config entries (§7.1), elided when empty. The management API `/api/services` surface exposes full state for UI purposes. |
| `/v1/chat/completions` | POST | Look up by `model`. Apply `strip_params` then `set_params`. Trigger start if needed. Proxy transparently, preserving SSE for `stream: true`. |
| `/v1/completions` | POST | As above (legacy). |
| `/v1/embeddings` | POST | As above, routed to services whose underlying server advertises embeddings. |
| `/v1/audio/*`, `/v1/images/*`, `/v1/files/*`, `/v1/fine_tuning/*`, `/v1/batches` | any | `501 Not Implemented`. |

Error shapes follow OpenAI conventions: `{error: {code, message, type}}`.

- Unknown `model` → `404 model_not_found`.
- Model `disabled` → `503 service_disabled` with reason in message.
- Start queue full → `503 start_queue_full`.
- Start failed → `503 start_failed` with last stderr line in message.

### 10.3 Eviction with in-flight requests

When a running service is selected for eviction (balloon resolver, higher-priority placement, config reconcile):

1. Mark `draining`. New requests to this service return 503 (unified proxy retries routing to an alternate if one exists; per-service-port clients see 503 directly).
2. Wait for in-flight requests to complete, bounded by `max_request_duration` (default 10min, per-service overridable). This bounds worst-case memory hold.
3. After all complete, enter drain window: `drain_timeout` (default 30s llama-cpp, 5s command). This is a short grace for connection teardown races.
4. If streaming requests are still active at end of drain, extend by up to `extended_stream_drain` (default 30s) specifically to let SSE streams finish.
5. SIGTERM.
6. 10s grace, then SIGKILL.

Worst-case eviction latency for llama-cpp: `max_request_duration + drain_timeout + extended_stream_drain + sigterm_grace` ≈ 10m + 30s + 30s + 10s ≈ 10.7 minutes. Shorter for dynamic services. Active clients whose streams are cut on SIGTERM see a closed connection; the unified proxy closes the SSE stream gracefully when possible.

### 10.4 Concurrent requests during startup

When a request arrives for an `idle` service, a **start future** is created; the request awaits it. Subsequent requests for the same service while `starting` await the same future — no redundant spawns. The future resolves when health passes (all waiters proceed) or start fails (all waiters 503).

Waiter queue per service bounded by `start_queue_depth` (default 10, configurable globally and per-service). If the queue is full when a request arrives, the request is rejected immediately with `503 start_queue_full`. Once the service is running, requests proxy without queueing — the cap applies only during the start transition.

## 11. Management API

Served on `daemon.management_listen`. All endpoints documented via OpenAPI at `/api/openapi.json`. Full spec is generated from `utoipa` annotations on the Rust handlers; the list below is the authoritative surface for v1.

### 11.1 REST

| Path | Method | Purpose |
|---|---|---|
| `/api/openapi.json` | GET | OpenAPI schema |
| `/api/devices` | GET | List devices with current reservations, observed usage, external blocks |
| `/api/services` | GET | List all configured services + active oneshots |
| `/api/services/{name}` | GET | Full service detail (state, effective config post-merge, allocation, metadata) |
| `/api/services/{name}/start` | POST | Force start (bypass idle) |
| `/api/services/{name}/stop` | POST | Force stop with drain |
| `/api/services/{name}/restart` | POST | Stop then start |
| `/api/services/{name}/enable` | POST | Clear disabled state |
| `/api/services/{name}/disable` | POST | Set `disabled(user_disabled)` |
| `/api/services/{name}/logs` | GET | Paginated historical logs; query params `since`, `until`, `run`, `limit`, `stream` |
| `/api/config` | GET | Current config file contents |
| `/api/config` | PUT | Replace config (structured validation + `toml_edit` write); body = TOML string; optimistic locking via `If-Match` header with content hash |
| `/api/config/validate` | POST | Parse + validate without saving; returns errors with spans |
| `/api/oneshot` | POST | Launch a oneshot (see §11.3) |
| `/api/oneshot/{id}` | GET | Oneshot status |
| `/api/oneshot/{id}` | DELETE | Terminate a running oneshot |

### 11.2 WebSocket streams

| Path | Purpose |
|---|---|
| `/api/services/{name}/logs/stream` | Live tail of stdout/stderr |
| `/api/events` | Real-time state changes: service state transitions, allocation changes, external reservation updates, config reloads, estimator drift warnings. Used by the frontend to keep views live without polling |

### 11.3 Oneshot API

```http
POST /api/oneshot
Content-Type: application/json

{
  "name": "sd-batch-20260417",          // optional; auto-generated if absent
  "template": "command",
  "command": ["python", "batch.py", "--port", "{port}"],
  "workdir": "/srv/sd",
  "allocation": { "mode": "static", "vram_gb": 16 },
  "devices": { "placement": "gpu-only" },
  "priority": 40,
  "ttl": "2h",                          // max lifetime; killed on expiry
  "port": null,                         // null = daemon assigns from pool
  "metadata": { "tags": ["batch-job"] }
}
```

Response:

```json
{
  "id": "oneshot_01H...",
  "name": "sd-batch-20260417",
  "port": 18001,
  "logs_url": "/api/oneshot/oneshot_01H.../logs/stream"
}
```

Dynamic ports: a configurable pool range (default 18000–18999) from which oneshots draw when `port: null`. Exhaustion returns 503.

Oneshots are scheduled through the same allocation algorithm. TTL expiry triggers the normal drain/SIGTERM sequence.

**Eviction.** Oneshots are evictable by strictly higher priority placements, same as any other service. `ttl` is a **maximum lifetime, not a minimum guarantee** — a oneshot may be evicted before its TTL elapses. Clients that need guaranteed runtime should submit at high priority. Evicted oneshots are **not** restarted; they end with an `evicted` final status surfaced on `GET /api/oneshot/{id}`.

## 12. Database

SQLite via `toasty`. Schema:

```sql
-- Service identity, kept forever (soft-deletion via deleted_at)
CREATE TABLE services (
  service_id    INTEGER PRIMARY KEY,
  name          TEXT NOT NULL UNIQUE,
  created_at    INTEGER NOT NULL,
  deleted_at    INTEGER                  -- NULL if currently in config
);

-- Snapshot of each effective (post-merge) config whenever it changes
CREATE TABLE service_config_versions (
  service_id    INTEGER NOT NULL,
  version       INTEGER NOT NULL,
  effective_config TEXT NOT NULL,        -- JSON of merged config
  recorded_at   INTEGER NOT NULL,
  PRIMARY KEY (service_id, version)
);

-- One row per process spawn (see §9.2)
CREATE TABLE running_services (
  service_id    INTEGER NOT NULL,
  run_id        INTEGER NOT NULL,
  pid           INTEGER NOT NULL,
  spawned_at    INTEGER NOT NULL,
  command_line  TEXT NOT NULL,
  allocation    TEXT NOT NULL,
  state         TEXT NOT NULL,
  PRIMARY KEY (service_id, run_id)
);

-- Stdout/stderr capture
CREATE TABLE service_logs (
  service_id    INTEGER NOT NULL,
  run_id        INTEGER NOT NULL,
  timestamp_ms  INTEGER NOT NULL,
  seq           INTEGER NOT NULL,         -- per-(service, run) monotonic, disambiguates same-ms lines
  stream        TEXT NOT NULL,            -- 'stdout' | 'stderr'
  line          TEXT NOT NULL,
  PRIMARY KEY (service_id, run_id, seq)
);
CREATE INDEX service_logs_ts ON service_logs(service_id, run_id, timestamp_ms);

-- Audit: every allocation change
CREATE TABLE allocation_events (
  event_id      INTEGER PRIMARY KEY,
  service_id    INTEGER NOT NULL,
  run_id        INTEGER NOT NULL,
  event_type    TEXT NOT NULL,           -- 'reserved' | 'released' | 'evicted' | 'oom' | ...
  device        TEXT NOT NULL,           -- 'gpu:0', 'gpu:1', 'cpu'
  bytes         INTEGER NOT NULL,
  at            INTEGER NOT NULL
);

-- Oneshot history
CREATE TABLE oneshots (
  id            TEXT PRIMARY KEY,
  service_id    INTEGER NOT NULL,
  submitted_at  INTEGER NOT NULL,
  started_at    INTEGER,
  ended_at      INTEGER,
  exit_code     INTEGER,
  ttl_ms        INTEGER NOT NULL
);
```

Retention policies:

- `service_logs`: 7 days or 50k lines per service, whichever smaller. Plus a pre-exit 500-line buffer preserved outside the rolling window.
- `allocation_events`: 30 days.
- `oneshots`: 90 days.
- `service_config_versions`: kept indefinitely (small).

**Vacuuming.** The initial migration sets `PRAGMA auto_vacuum = INCREMENTAL` (this must be set before any tables are created; it cannot be switched on afterwards without a dump-and-reload). Retention trim runs at 3am local. After trim, and also hourly, the daemon calls `PRAGMA incremental_vacuum(N)` to reclaim pages in small chunks; this does not lock the database and is safe to run alongside active log writers. The old "daily full VACUUM" is dropped.

**Migrations** handled by toasty's schema versioning. A fallback raw-SQL migration path is retained in case toasty's migration tooling misbehaves on a schema we care about.

**Known issue: log write amplification.** A verbose llama-server can emit thousands of lines per second. The 200ms / 100-line batching mitigates but doesn't eliminate the risk of saturating the writer. Not addressed in v1; possible mitigations if it bites: per-service line-rate limiting with a "logs truncated" marker, or a separate DB file for logs so VACUUM doesn't block scheduler tables.

## 13. Frontend

Vite + React 19 + TypeScript + Tailwind 4. No SSR. Assets embedded via `rust-embed` in release builds.

### 13.1 Type sync

Rust handlers annotated with `utoipa`. Daemon serves OpenAPI at `/api/openapi.json`. `just gen-types` runs `openapi-typescript` to produce `frontend/src/api/types.ts` and `orval` to produce typed React Query hooks in `frontend/src/api/client.ts`. CI enforces generated files are up to date.

Same-repo layout: backend and frontend share version history, one recipe regenerates both.

### 13.2 Views

**Devices dashboard.** Card per device (GPUs + CPU). Stacked reservation bar, coloured by service; grey blocks for external reservations (labelled with PID/process where known); two-tone (solid min + hatched elastic) for dynamic services; distinct hatch while a service is still within the weight-loading grace (§9.6). Observed-usage thin bar below for drift visibility.

**Services table.** All services including disabled/tombstoned-but-migrating. Columns: name, template, state, priority, lifecycle, devices, last-active. Disabled rows show reason + Re-enable / View-logs. Row expand: last ~200 log lines, effective (post-merge) config, metadata, request-count and latency-percentile stats pulled from the request-metrics store.

**Config editor.** CodeMirror 6 with TOML highlighting and schema-driven autocomplete (schema from `/api/openapi.json`). Save via `toml_edit` preserving comments and formatting. Structured form mode alongside raw TOML. Diff-before-save; restart-required flags per changed field; conflict detection via mtime + hash.

**Oneshot launcher.** Modal form: template, params, priority, TTL. On submit, returns ID and opens a logs panel. Running oneshots appear in the services table until they exit.

Real-time updates from `/api/events` WebSocket; no polling.

## 14. CLI

`anankectl` speaks HTTP API only, no direct DB access.

```
anankectl devices
anankectl services [--all]                     # include disabled
anankectl show <name>
anankectl start <name>
anankectl stop <name>
anankectl restart <name>
anankectl enable <name>
anankectl disable <name>
anankectl retry <name>                         # clear failed + force start
anankectl logs <name> [--follow] [--run N] [--since DURATION]
anankectl oneshot submit <file.toml>           # submit oneshot from file
anankectl oneshot list
anankectl oneshot kill <id>
anankectl config show
anankectl config validate [<file>]             # validate without applying
anankectl config reload                        # force reload from disk
anankectl reload                               # alias
```

`--json` flag on query commands for scriptable output.

## 15. Testing

### 15.1 Unit

- **GGUF reader:** fixture files, tensor enumeration, header round-trip, truncated-file is error-not-panic.
- **VRAM estimator:** known GGUFs across llama-family, MoE, Mamba. Assert estimates within ±10% of measured footprint for several (cache_type_k, cache_type_v, context, n_cpu_moe) combinations. Canary that fails loudly if llama.cpp's allocator drifts enough to invalidate the byte table.
- **TOML config:** valid configs parse; invalid produce span-annotated errors; inheritance merge produces expected effective config; collisions detected; `migrate_from` chains resolved; `extends` cycles rejected.
- **State transitions:** assert only valid transitions happen given a sequence of events.

### 15.2 Property-based (`proptest`)

- **Allocator:** random (devices, services, priorities). Invariants: no overcommit, higher-priority-evicts-lower, single-device best-fit finds valid placement when one exists.
- **Layer-aware split:** random (per-layer costs, per-GPU capacities). Invariants: layers contiguous per device, no overcommit, CPU spill only in hybrid mode.
- **Balloon resolver:** random (usage trajectories, priorities, jitter). Invariants: right party wins per priority comparison, drain before SIGTERM, jitter tolerance holds at ±10%.

### 15.3 Integration

Mock NVML via `trait GpuProbe`:

```rust
trait GpuProbe: Send + Sync {
    fn list(&self) -> Vec<GpuInfo>;
    fn query(&self, id: u32) -> GpuMemory;
    fn processes(&self, id: u32) -> Vec<GpuProcess>;
}
```

NVML implementation and an in-memory fake. Integration tests drive the fake: GPU loses memory mid-run, external process appears, managed child crashes, asymmetric free space, weight-loading grace behaviour.

Toy HTTP echo service as a `command` template exercises start/stop/evict/route through the real scheduler, real proxy, fake GPU. Validates the pipeline without actual hardware.

Config reload: add/remove/edit/migrate services; assert DB state (tombstones, id reuse, history continuity).

OpenAI routing: various `model` values; correct routing, filter application, stream passthrough, disabled-service 503s, start-queue-full 503s, 501s for unimplemented.

Orphan recovery: kill daemon with child alive; restart; assert adoption or cleanup.

### 15.4 Manual runbooks

In `tests/manual/`: actual llama.cpp launch with a small GGUF, actual ComfyUI balloon under load, actual multi-GPU layer split. Run during release prep, not CI.

## 16. Tech stack

**Daemon:** Tokio, `hyper` (proxy data plane), Axum (management + OpenAI routing), `utoipa` (OpenAPI generation), `nvml-wrapper`, `notify` (config watch), `toml_edit`, `toasty` (SQLite), `tracing`. GGUF: start with `gguf` crate; small custom reader if it lacks tensor-table access. Child supervision: `nix` crate for `prctl(PR_SET_PDEATHSIG)`.

**Frontend:** Vite, React 19, TypeScript, Tailwind 4, TanStack Query, CodeMirror 6, `orval`, `openapi-typescript`.

**CLI:** single binary over HTTP.

**Auth:** none in v1. Management API binds to `127.0.0.1` by default. External access expected through a reverse proxy with auth at that layer.

## 17. Defaults summary

| Setting | Default | Scope |
|---|---|---|
| `daemon.management_listen` | `127.0.0.1:7071` | Global |
| `openai_api.listen` | `127.0.0.1:7070` | Global |
| Oneshot port pool | `18000–18999` | Global |
| `start_queue_depth` | 10 | Global + per-service |
| `min_borrower_runtime` | 60s | Per-service (dynamic) |
| `drain_timeout` (llama-cpp) | 30s | Per-service |
| `drain_timeout` (command) | 5s | Per-service |
| `extended_stream_drain` | 30s | Per-service |
| `max_request_duration` | 10m | Per-service |
| `daemon.shutdown_timeout` | 120s | Global |
| SIGTERM → SIGKILL grace | 10s | Global |
| Balloon fast-path SIGKILL grace | 5s | Global |
| SIGQUIT emergency-stop grace | 5s | Global |
| `idle_timeout` | 10m | Per-service |
| `health.timeout` (llama-cpp) | 180s | Per-service |
| `health.timeout` (command) | 60s | Per-service |
| Health probe interval | 5s | Global (configurable) |
| Crash loop threshold | 5 exits / 10min | Global |
| Launch retry backoff | 2s, 5s, 15s | Global |
| OOM fudge retry | 1.1 → 1.4 | Global |
| Elastic margin | 512MB | Global |
| Balloon sample interval | 2s | Global |
| Balloon window | 6 samples (12s) | Global |
| Drift warning (under) | `rolling_mean > 1.2` | Per-service |
| Drift warning (over) | `rolling_mean < 0.85` | Per-service |
| Log retention | 7 days / 50k lines | Per-service |
| Allocation event retention | 30 days | Global |
| Oneshot history retention | 90 days | Global |
| Estimator fallback safety | 1.15× + 512MB | Global |
| Estimator correction clamp | [0.8, 1.5] | Global |
| Tensor-split fudge | 1 layer per device (GPU and CPU when hybrid) | Global |
| Placement-override drift warning | > 30% deviation | Per-service |

## 18. Open implementation items

- Validate the `gguf` crate exposes tensor-table enumeration with names, types, shapes, and handles sharded files. If not, a ~200-line custom reader covering the format (including shard header sum) suffices.
- Calibrate the VRAM estimator against a matrix of (architecture × cache_type × context × n_cpu_moe) on representative GGUFs to seed the rolling-correction table; pre-release validation step.
- Calibrate the tensor-split one-layer fudge (§8.2) against real multi-GPU runs — one layer's worth is a safe upper bound but may be more than necessary once llama.cpp's rounding behaviour is characterised empirically.
- Confirm toasty's migration tooling handles the schema; retain a raw-SQL fallback.
- Confirm toasty/sqlite set `auto_vacuum = INCREMENTAL` correctly in the initial migration (pragma is only effective pre-table-creation).
- Decide on `orval` vs hand-written React Query hooks atop `openapi-typescript`; start with `orval`, reassess if generated code bloats.
- Decide Nix flake output shape — `nixosModules.default` + `packages.default` (the built daemon) is the expected surface.
- Non-NVIDIA backends (§4.3.1) are spec'd as pluggable but not implemented in v1. Concrete `GpuProbe` impls for ROCm and XPU deferred to future revisions.
