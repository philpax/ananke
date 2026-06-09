# ananke

> [!WARNING]  
> This codebase is somewhere between "copilot" and "auto" on the [ai-declaration.md](https://ai-declaration.md/en/0.1.2/) scale. It was extruded from a spec that was authored by Claude on my guidance. Since then, I have been steadily adjusting it and fixing bugs that I encounter.
>
> I would not recommend that you use this unless you are able to diagnose and fix issues yourself. It won't destroy your system, but it may also not behave as expected. This was built to solve a problem I have; it'll take some time before I can be confident in saying that it can solve your problem, too.

ananke is a GPU/CPU-aware model proxy daemon designed to manage multiple LLMs and other AI tools (like ComfyUI) efficiently. It provides an OpenAI-compatible API and a management CLI to orchestrate model loading, unloading, and resource allocation.

## Getting Started

### Prerequisites

ananke is currently Linux-only.

- **Rust toolchain** (stable) to build with `cargo`.
- **NVIDIA driver with NVML** (`libnvidia-ml.so`) on the host, for GPU detection and VRAM tracking. Without it the daemon falls back to CPU-only and any GPU-bound service fails placement.
- **`llama-server`** on `PATH`, if you plan to use the `llama-cpp` service template (the path the Quick Start below takes). Build or download it from [llama.cpp](https://github.com/ggml-org/llama.cpp). Point ananke at a non-`PATH` binary or wrap it in a container via [`daemon.llama_server`](#daemon-settings), the per-service `llama_server` field, or a full [`launcher` template](#custom-llama-server-binary-or-wrapper).

### Installation

```bash
git clone https://github.com/philpax/ananke.git
cd ananke
cargo build
```

Binaries are located in `target/debug/ananke` and `target/debug/anankectl`.

### Quick Start

Create a minimal config at `~/.config/ananke/config.toml`:

```toml
[[service]]
name = "my-model"
template = "llama-cpp"
port = 8200
model = "/path/to/model.gguf"
```

That's the whole config. Everything else has a sensible default: the OpenAI API listens on `127.0.0.1:7070`, the management API on `127.0.0.1:7071`, every visible NVIDIA GPU is probed, and the service is on-demand with a 10-minute idle timeout.

The [Configuration Guide](#configuration-guide) below covers how to override any of these.

Start the daemon:

```bash
ananke
```

Then talk to it. The shortest path uses the bundled CLI:

```bash
anankectl chat my-model "Hello!"
```

Any OpenAI-compatible client also works — for example, with `curl`:

```bash
curl http://127.0.0.1:7070/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "my-model", "messages": [{"role": "user", "content": "Hello!"}]}'
```

ananke loads the model on the first request, serves the response, then unloads it after 10 minutes of idle time to free VRAM.

### Configuration Resolution

ananke searches for its configuration file in the following priority:

1. `ANANKE_CONFIG` environment variable.
2. `--config` CLI argument.
3. `$XDG_CONFIG_HOME/ananke/config.toml`
4. `~/.config/ananke/config.toml`
5. `/etc/ananke/config.toml`

## Components

- `ananke`: The core daemon responsible for supervision, resource allocation, and proxying.
- `ananke-api`: The API layer providing the OpenAI-compatible interface and management endpoints.
- `anankectl`: A CLI tool for managing the daemon and its services.

## Configuration Guide

ananke uses a TOML configuration.

### Daemon Settings

```toml
[daemon]
management_listen = "0.0.0.0:7071"
allow_external_management = true # Required if management_listen is non-loopback
allow_external_services = true   # Allow public access to individual model ports
data_dir = "./data"
shutdown_timeout = "120s"         # Max time to wait for services to drain
llama_server = "/opt/llama-build/llama-server"  # Optional: default llama-server binary for every llama-cpp service (overridable per-service)
```

> **Security Note:** Both the Management API (`management_listen`) and per-service reverse proxies (`allow_external_services`) are **unauthenticated**. If you bind them to non-loopback addresses:
>
> - Trust your network perimeter (e.g., Tailscale, a private VLAN).
> - Terminate TLS and authentication at a reverse proxy in front of ananke.
> - Never expose these ports directly to the public internet.

### OpenAI API Settings

```toml
[openai_api]
listen = "0.0.0.0:7070"
enabled = true                   # Set to false to disable the OpenAI API
max_request_duration = "10m"     # Max wall-clock duration per proxied request
```

### Global Defaults

These values apply to all services unless overridden per-service:

```toml
[defaults]
idle_timeout = "10m"             # Default idle timeout for on-demand services
priority = 50                    # Default eviction priority
```

### Device Configuration

Control which GPUs are used and how much VRAM is reserved for the system:

```toml
[devices]
gpu_ids = [0, 1]                 # Only probe these GPUs
default_gpu_reserved_mb = 2048   # Reserve 2GB on all GPUs for system processes
gpu_reserved_mb = { "0" = 4096 } # Per-GPU override (GPU 0: reserve 4GB)

[devices.cpu]
enabled = true                   # Allow CPU placement for services
reserved_gb = 8                  # Reserve 8GB of CPU RAM
```

`default_gpu_reserved_mb` and `gpu_reserved_mb` are kept free on every GPU when the packer places a service; a per-service `gpu_headroom_mb` adds to them for one model. `[devices.cpu] reserved_gb` is host RAM the daemon keeps free — it bounds how much expert weight a hybrid MoE service may offload to the CPU, and a placement that would exceed it is rejected.

### Service Configuration

Services are defined as an array of `[[service]]` blocks.

#### Service Lifecycles

Each service runs in one of two modes:

- **On-Demand (Default)**: Loaded only when a request arrives. Unloaded after a configurable `idle_timeout` (default: 10m) to free up VRAM.
- **Persistent**: Stays loaded in memory indefinitely, ensuring zero-latency startup for critical models.

```toml
[[service]]
name = "my-model"
template = "llama-cpp"
lifecycle = "on_demand"   # or "persistent"
```

#### Resource Allocation & VRAM

ananke oversubscribes GPU memory by dynamically managing which models are active:

- **llama.cpp Services**: VRAM usage is determined by the model size and `n_gpu_layers`. ananke uses an internal GGUF-aware estimator to track usage.
- **Command Services**: Support two allocation modes:
  - `static`: Reserves a fixed amount of VRAM (`vram_gb`).
  - `dynamic`: Operates within a range (`min_vram_gb` to `max_vram_gb`).

  In both modes the daemon picks the GPU with the most available headroom (subject to `gpu_allow`), preferring one whose free capacity satisfies the upper bound (`vram_gb` for `static`, `max_vram_gb` for `dynamic`) so dynamic services have room to grow. The picked GPU id is exported to the spawned child as `CUDA_VISIBLE_DEVICES`, and is also available as the `{gpu_ids}` placeholder in `command` argv. Wrappers that launch containers should forward this env (e.g. `docker run --device "nvidia.com/gpu=$CUDA_VISIBLE_DEVICES"`) so the picked GPU is the only one the container sees.
- **Eviction**: When VRAM is exhausted, ananke uses a priority-based eviction system. Higher priority services can displace dormant on-demand services.

#### Placement Policies

- `gpu-only`: Service must reside entirely on GPU.
- `cpu-only`: Service resides entirely on CPU.
- `hybrid`: Allows a mix of GPU and CPU. The packer fills the GPUs first and spills the remainder to CPU. For MoE models with `expert_offload` enabled it spills *expert tensors* before whole layers, keeping every layer's attention and KV cache on the GPU (see [MoE expert offload](#moe-expert-offload)). Manual `override_tensor` rules also work here for hand-picked CPU offloading.

#### Multi-GPU split modes

When a llama.cpp service spans more than one GPU, `devices.split` selects how llama.cpp divides the model across them. It maps directly to llama.cpp's `--split-mode`:

```toml
[service.devices]
placement = "gpu-only"
split = "tensor"   # "layer" (default), "row", or "tensor"
```

- `layer` (default): pipeline parallelism — each GPU holds a contiguous range of whole layers. ananke estimates each layer's footprint and packs them across the allowed GPUs first-fit, so the split ratio follows the per-GPU layer counts. Lowest interconnect demand; the right default when the cards have no fast peer link.
- `row`: the older tensor-parallel mode (`--split-mode row`). Splits individual tensors by row. Without NVLink/P2P it is typically *slower* than `layer` because every token incurs cross-GPU traffic over PCIe; prefer `tensor` over `row` on such hosts.
- `tensor`: the newer tensor-parallel mode (`--split-mode tensor`). Shards each tensor across the GPUs and emits a balanced `--tensor-split` with `--main-gpu` set to the lowest allowed GPU. On dual identical cards this measures meaningfully faster decode than `layer` even without P2P, at the cost of a larger compute buffer and constant cross-GPU communication.

`row` and `tensor` are sharded modes and carry extra constraints, rejected at config validation:

- The service must use `placement = "gpu-only"` — a sharded model cannot spill to CPU.
- Only valid for `llama-cpp` services, not `command` services.
- Cannot be combined with `override_tensor` (manual tensor placement), since the sharded modes manage tensor placement themselves.

With a sharded mode, ananke reserves an equal share of the model weights, KV cache, and compute buffer on each allowed GPU, placing the non-layer remainder (output tensor, MTP overhead, …) on the main GPU. The pledge book reflects this per-GPU split, so a co-tenant (e.g. an embedding service) sees the true free capacity on each card.

#### llama.cpp Template

Used for GGUF models via llama.cpp.

```toml
[[service]]
name = "gemma-4"
template = "llama-cpp"
port = 8200
model = "/path/to/model.gguf"
mmproj = "/path/to/mmproj.gguf" # Optional vision projector
context = 32768
flash_attn = true
cache_type_k = "q8_0"
cache_type_v = "q8_0"
lifecycle = "on_demand"
priority = 100

# CPU Offloading: Move specific tensors to CPU via regex
override_tensor = [ ".ffn_(up|down)_exps.=CPU" ]

[service.sampling]
temperature = 0.7
top_p = 0.95

# Other common llama.cpp options:
# n_gpu_layers = -1       # Offload all layers to GPU (default)
# threads = 8             # Number of CPU threads
# parallel = 4            # Request parallelism (-np). With a non-unified KV
#                         # this splits the context budget across slots, so
#                         # each request caps at context / parallel.
# batch_size = 512        # Context batch size
# mmap = true             # Memory-map the model file
# jinja = true            # Use Jinja chat templates

# Speculative decoding via multi-token prediction (MTP / NextN):
# spec_type = "draft-mtp" # --spec-type. For models with an embedded MTP head
#                         # (nextn_predict_layers > 0, e.g. Qwen 3.6), this
#                         # drafts ahead using the same weights — no separate
#                         # draft model. ananke's estimator adds the MTP draft
#                         # context's VRAM (a small f16 KV over the nextn
#                         # layer(s) plus a ~1.7 GiB compute buffer).
# spec_draft_n_max = 2    # --spec-draft-n-max: max draft tokens per step.
# MTP composes with `parallel > 1` and `mmproj` (vision) on current llama.cpp.
```

#### MoE expert offload

Large mixture-of-experts models often don't fit a card once their expert tensors are resident, even though the attention and KV cache do. The `expert_offload` knob lets ananke move expert tensors to the CPU — the GPU keeps every layer's attention and KV cache (latency-critical), while the bulky, sparsely-activated experts live in host RAM. ananke sizes the placement and emits the matching `-ot` rules itself, so the VRAM reservation matches what the model actually uses.

`expert_offload` accepts three values, and any value other than `"off"` requires `placement = "hybrid"`:

- `"off"` (default): no expert offload. The model packs whole layers, spilling entire layers to CPU only if a layer doesn't fit.
- `"auto"`: ananke offloads the minimum number of layers' experts needed to fit the model in the GPU's live free VRAM, preferring a second GPU before the CPU on multi-GPU hosts.
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

- **`llama_server`** — a path to the executable (or wrapper script) that should be invoked in place of `llama-server`. The script must accept llama-server's CLI flags. Settable at the daemon level (default for every llama-cpp service) and per-service (overrides the daemon default).
- **`launcher`** — a full argv template that replaces the default `llama-server -m <model> ...` invocation. `launcher[0]` is the executable; the remaining entries are substituted with placeholders so a wrapper can see the model path separately from the rest of the flags (useful for container volume mounts).

Placeholders in `launcher` entries:

- `{model}` — the model path. Held back from `{args}` so the wrapper can position it freely.
- `{name}` — service name.
- `{port}` — the private loopback port ananke assigned.
- `{gpu_ids}` — comma-separated NVML index list ananke picked for this service.
- `{args}` — splat: expands to every llama-server flag ananke would otherwise have emitted (everything except `-m <model>` — `--mmproj`, `-c`, placement-derived `-ngl`/`--tensor-split`/`-ot`, sampling, `--host`, `--port`, `extra_args`, …). Must occupy a launcher entry on its own; `"--foo={args}"` is rejected at config validation.

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

The wrapper script receives `/srv/models/qwen3-30b.gguf` as `$1` (for the volume mount) and `$@` after `shift` contains the rest of the llama-server argv — `-c 32768 -fa on -ngl 999 ... --host 127.0.0.1 --port 41000`. With `--network host` the container's llama-server is reachable on that port without further plumbing.

If you only need to point at a non-`PATH` binary (no argv rearranging), set `llama_server` instead:

```toml
[[service]]
name = "demo"
template = "llama-cpp"
port = 11437
model = "/srv/models/x.gguf"
llama_server = "/opt/llama-cuda/llama-server"
```

`CUDA_VISIBLE_DEVICES` is set on the spawned process from the picked GPU id(s) in both cases. Wrapper scripts that launch a container should forward this so the container only sees the picked GPU — for example, `podman run --device "nvidia.com/gpu=${CUDA_VISIBLE_DEVICES:-all}" ...`.

#### Command Template

Used for arbitrary binaries or Docker wrappers.

```toml
[[service]]
name = "comfyui"
template = "command"
port = 8188
lifecycle = "on_demand"
# Placeholders substituted in argv:
#   {port}     - the private loopback port assigned by ananke
#   {gpu_ids}  - comma-separated NVML index list ananke picked for this service
#   {vram_mb}  - reserved VRAM in MiB
#   {model}    - model path (llama-cpp only; empty here)
#   {name}     - service name
command = ["/bin/bash", "start_comfy.sh", "--port", "{port}"]
shutdown_command = ["/bin/bash", "stop_comfy.sh"]

[service.allocation]
mode = "dynamic"
min_vram_gb = 2.0
max_vram_gb = 12.0

[service.health]
http = "/system_stats"
timeout = "30s"
```

The child also inherits `CUDA_VISIBLE_DEVICES` set to the picked GPU id(s). Wrapper scripts that launch a container should forward this so the container only sees the picked GPU — for example, `docker run --device "nvidia.com/gpu=${CUDA_VISIBLE_DEVICES:-all}" ...`.

The `shutdown_command` field is particularly useful for external processes (like Docker containers) that cannot stop via signal alone. ananke runs this command after the drain pipeline completes, ensuring clean exits for services that don't respond to SIGTERM.

#### Fronting an OpenAI-Compatible Upstream

A `command`-template service that already speaks the OpenAI API (vLLM, TGI, SGLang, …) can opt into ananke's `/v1/models` and `/v1/chat/completions` multiplexer by adding an `[service.openai_proxy]` block. Without the block, command services stay invisible to the OpenAI surface and are only reachable via their per-service reverse proxy — the same as before.

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
- Before forwarding, ananke rewrites the JSON `model` field to `upstream_model` (here, `qwen3.6-27b-autoround`) — the name vLLM was started with via `--served-model-name`. Clients address ananke's name; the upstream sees its own name.

The rewrite happens after `[service.filters]` is applied, so `openai_proxy.upstream_model` overrides any `filters.set_params.model`. Filters can still strip or set other JSON keys.

#### Embedding Services

By default every service is registered as a chat model. Pooling-only embedding models (Jina v5, BGE, E5, …) opt in by setting `modality = "embedding"` on the service. The proxy itself is endpoint-agnostic — it already routes `POST /v1/embeddings` by `model` field — so the `modality` field is purely a typed declaration: clients filter on it through `/v1/models` and `/api/services`, and the frontend renders a teal `embedding` badge next to the service name (mirroring the purple `vision` badge for llama.cpp services with an `mmproj`).

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

### Advanced Service Options

- **Filters**: Modify requests before they reach the model.
  ```toml
  [service.filters]
  strip_params = ["temperature"] # Remove these from the request
  set_params = { max_tokens = 4096 } # Force these values
  ```
- **Devices**: Control GPU visibility and multi-GPU splitting.
  ```toml
  [service.devices]
  placement = "hybrid"
  gpu_allow = [0, 1] # Only use these GPUs
  split = "tensor"   # Multi-GPU split mode; see "Multi-GPU split modes" above
  ```
- **Metadata**: Arbitrary key-value pairs exposed through `/v1/models` and `/api/services`.
  ```toml
  [service.metadata]
  discord_visible = true
  ```

### Service Inheritance

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

- Scalars: child overrides parent.
- Sub-tables: deep-merged field-by-field.
- Arrays: child replaces parent outright.
- `*_append` fields (e.g., `extra_args_append`): concatenated with parent's list.
- `name`, `port`, `extends`, and `template` must be overridden in the child.
- Cross-template inheritance is an error.

### Service Migration

When renaming a service, use `migrate_from` to preserve database history:

```toml
[[service]]
name = "gemma-4-31b"
template = "llama-cpp"
migrate_from = "old-gemma-31b"
port = 8200
model = "/models/gemma-4-31B.gguf"
```

### Service States

ananke tracks each service through a state machine:

| State      | Description                                                                                                                                    |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `idle`     | Service is unloaded, waiting for requests                                                                                                      |
| `starting` | Service is launching, waiting for health check                                                                                                 |
| `running`  | Service is healthy and accepting traffic                                                                                                       |
| `draining` | Service is shutting down, waiting for inflight requests                                                                                        |
| `stopped`  | Service was explicitly stopped                                                                                                                 |
| `evicted`  | Service was displaced by a higher-priority model                                                                                               |
| `failed`   | Service failed to start (with retry backoff, up to 3 attempts)                                                                                 |
| `disabled` | Service is permanently disabled (reason: `launch_failed`, `health_timeout`, `oom`, `crash_loop`, `no_fit`, `config_error`, or `user_disabled`) |

## Oneshot API

For temporary inference tasks that don't need a persistent config entry, ananke supports spawning short-lived services via the API:

```bash
# Submit from a TOML file
anankectl oneshot submit job.toml

# Run inline: allocate 4GB VRAM static, auto-terminate after 30 minutes
anankectl oneshot run --vram-gb 4 --ttl 30m --name my-job -- python inference.py

# List active oneshot jobs
anankectl oneshot list

# Cancel a running job
anankectl oneshot kill <id>
```

Oneshot services are allocated from the same resource pool as configured services and are automatically torn down when their TTL expires or when explicitly killed.

The TOML request format:

```toml
name = "my-job"
template = "command"
command = ["python", "inference.py"]
ttl = "30m"
priority = 50

[allocation]
mode = "static"
vram_gb = 4.0

[devices]
placement = "gpu-only"
```

## Config Hot-Reload

ananke watches its config file for changes and automatically reloads when modifications are detected. The reload process:

1. Parses and validates the new config.
2. Preflights GGUF models (catching dtype/shard issues before traffic hits them).
3. Spawns added services and drains removed ones.
4. Publishes a `config_reloaded` event via the WebSocket endpoint.

Failed reloads are silently ignored - the previous valid config remains in effect.

## Real-Time Events

The management API exposes a WebSocket endpoint at `/api/events` that delivers real-time state changes:

| Event                | Description                                                 |
| -------------------- | ----------------------------------------------------------- |
| `state_changed`      | Service entered a new state (e.g., `idle` → `starting`)     |
| `allocation_changed` | VRAM reservation changed for a service                      |
| `config_reloaded`    | Config was reloaded, with list of changed services          |
| `estimator_drift`    | VRAM estimator detected significant drift from actual usage |
| `overflow`           | Event buffer overflow (some events were dropped)            |

## Using `anankectl`

The CLI tool allows real-time management of the daemon. The base URL it talks to is resolved in this order: `--endpoint` flag, `ANANKE_ENDPOINT` environment variable, the `endpoint` key in `~/.config/anankectl/config.toml`, then the built-in default.

```bash
# At-a-glance health
anankectl status               # Daemon endpoint, OpenAI port, devices, services in one view

# Service management
anankectl services [--all]     # List services (include disabled with --all)
anankectl show <name>          # Show detailed service status
anankectl start <name>         # Manually trigger a service load
anankectl stop <name>          # Manually trigger a service drain
anankectl restart <name>       # Drain then reload a service
anankectl enable <name>        # Re-enable a disabled service
anankectl disable <name>       # Disable a service (stops traffic)
anankectl retry <name>         # Enable then start a service

# Monitoring
anankectl logs <name> [--follow] [-n <lines>]  # Stream service logs
anankectl devices              # View GPU/CPU resource utilization

# Daemon configuration
anankectl server-config show          # View current effective configuration
anankectl server-config validate      # Validate a TOML file without persisting
anankectl reload                      # Force a config reload

# Client (anankectl) configuration
anankectl config set endpoint <url>   # Persist a default management endpoint
anankectl config get endpoint
anankectl config list                 # Show every known key + current value
anankectl config unset endpoint
anankectl config path                 # Print path to the client config file
anankectl config edit                 # Open the client config in $EDITOR

# Talking to a model
anankectl chat <model> <prompt>...    # One-shot, streams the response to stdout
anankectl chat <model>                # Interactive TUI session
anankectl chat                        # Pick interactively from OpenAI-accessible services

# Oneshot
anankectl oneshot submit <file>       # Submit from TOML
anankectl oneshot list                # List active oneshot jobs
anankectl oneshot kill <id>           # Cancel a running oneshot
```

### Interactive chat (`anankectl chat`)

When invoked without a model, `chat` opens a small picker showing every service that `/v1/models` is willing to route to, sorted with running services at the top and the cursor pre-positioned on the hottest one. Single-candidate services auto-select; everything else is `↑/↓` (or `j/k`) and `Enter`.

The chat TUI itself supports:

- **Multi-line input** — the input box grows from 3 to 10 rows of content as you type, then scrolls internally past the cap. Press `Shift+Enter` (or `Ctrl+J` if your terminal sends a literal LF for Shift+Enter) to insert a newline; `Enter` sends. `Esc` clears the buffer.
- **Per-turn boxes** — each user/assistant/system turn renders in its own bordered block, role-coloured. Reasoning content (`delta.reasoning_content` / `delta.reasoning`, e.g. from DeepSeek) is shown in a subdued style above the visible response so you can watch the model think without it cluttering the final answer.
- **Live token-rate stats** — the assistant box title shows decoded-token count, decode rate (tok/s), time-to-first-token, and prompt-processing rate once the streamed `usage` chunk lands. Stats are live during streaming and freeze when the turn finishes.
- **Cancellation** — `Ctrl+X` cancels the in-flight turn, drops the HTTP stream, freezes the partial response with a red "· cancelled" badge in the title, and returns control to the input box.
- **Scroll** — `↑/↓` or the mouse wheel scrolls the conversation; `Ctrl+End` snaps back to the bottom.
- **Quit** — `Ctrl+C` (or `Ctrl+D` on an empty input).

`anankectl chat <model> <prompt>` (with a prompt) skips the TUI entirely and streams the completion to stdout — handy for one-off shell pipelines.

## Comparing Alternatives

If you're looking for model management tools, you may also want to consider:

### llama-swap ([mostlygeek/llama-swap](https://github.com/mostlygeek/llama-swap))

A well-established Go proxy with 3.5k+ stars. It's excellent if you want to:

- Support diverse upstream servers beyond llama.cpp (vLLM, tabbyAPI, stable-diffusion.cpp).
- Use a built-in web UI for interacting with your models.
- Define concurrent model combinations with a solver-based DSL matrix.

llama-swap uses a matrix of valid model sets with a solver that picks the cheapest eviction when a new model is requested. It does not track VRAM - you must manually ensure models fit together on your GPU.

### Large Model Proxy ([perk11/large-model-proxy](https://github.com/perk11/large-model-proxy))

A smaller Go proxy with resource pool management. It's a good fit if you:

- Want resource-aware loading with manual VRAM/RAM declarations per service into shared pools.
- Need straightforward LRU-based eviction when the resource pool is exhausted.

Like llama-swap, it requires you to declare the VRAM each model will consume ahead of time.

### How ananke Compares

| Feature             | ananke                      | llama-swap                          | Large Model Proxy                     |
| ------------------- | --------------------------- | ----------------------------------- | ------------------------------------- |
| Language            | Rust                        | Go                                  | Go                                    |
| VRAM estimation     | **Automatic** (GGUF-aware)  | None (user-managed)                 | Manual (declared per model in pools)  |
| Service templates   | `llama-cpp`, `command`      | Any upstream server                 | Any command                           |
| Eviction strategy   | Priority-based              | Solver-based (cheapest eviction)    | LRU                                   |
| Config hot-reload   | Yes, with preflight         | Yes                                 | No                                    |
| CLI tool            | `anankectl` (comprehensive) | Basic HTTP API                      | Basic HTTP API                        |
| Management API      | REST + WebSocket events     | REST + Web UI                       | REST + Web Dashboard                  |
| Service inheritance | `extends` with deep merge   | Macros                              | No                                    |
| Temporary services  | Oneshot API with TTL        | TTL per model                       | TTL per model                         |

**Choose ananke if:** You want automatic VRAM estimation (no manual declarations needed) and a CLI/API for programmatic management. ananke is designed for users who want to add models to their config and trust the daemon to figure out where they fit.

**Choose llama-swap if:** You need maximum flexibility with upstream servers, want a battle-tested solution with a large community, and prefer a solver-based matrix for defining which models can run concurrently.

**Choose Large Model Proxy if:** You want a lightweight proxy with resource pools, manual VRAM declarations, and don't need advanced features like service inheritance or real-time event streaming.

## API Reference

### OpenAI API

Available at `http://<host>:7070/v1`.

| Endpoint               | Method | Description                            |
| ---------------------- | ------ | -------------------------------------- |
| `/v1/models`           | GET    | List available models and their states |
| `/v1/chat/completions` | POST   | Chat completions (proxy to upstream)   |
| `/v1/completions`      | POST   | Legacy completions (proxy to upstream) |
| `/v1/embeddings`       | POST   | Embeddings (proxy to upstream)         |

Other standard OpenAI endpoints return appropriate HTTP 501 "not implemented" responses.

### Management API

Available at `http://<host>:7071`. Used by `anankectl` to orchestrate the daemon.

| Endpoint                          | Method   | Description                           |
| --------------------------------- | -------- | ------------------------------------- |
| `/api/services`                   | GET      | List all services                     |
| `/api/services/:name`             | GET      | Get service detail (with recent logs) |
| `/api/services/:name/start`       | POST     | Start a service                       |
| `/api/services/:name/stop`        | POST     | Stop a service                        |
| `/api/services/:name/restart`     | POST     | Restart a service                     |
| `/api/services/:name/enable`      | POST     | Enable a disabled service             |
| `/api/services/:name/disable`     | POST     | Disable a service                     |
| `/api/services/:name/logs`        | GET      | Get recent logs for a service         |
| `/api/services/:name/logs/stream` | GET (WS) | Stream logs for a service             |
| `/api/devices`                    | GET      | Device/VRAM status                    |
| `/api/config`                     | GET      | Get current config (with ETag hash)   |
| `/api/config`                     | PUT      | Atomically update config              |
| `/api/config/validate`            | POST     | Validate TOML without persisting      |
| `/api/oneshot`                    | POST     | Spawn a short-lived service           |
| `/api/oneshot`                    | GET      | List oneshot services                 |
| `/api/oneshot/:id`                | GET      | Get oneshot status                    |
| `/api/oneshot/:id`                | DELETE   | Delete a oneshot service              |
| `/api/events`                     | GET (WS) | WebSocket for real-time events        |
| `/api/openapi.json`               | GET      | OpenAPI spec                          |
