# Configuration Guide

## Daemon Settings

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

## OpenAI API Settings

```toml
[openai_api]
listen = "0.0.0.0:7070"
enabled = true                   # Set to false to disable the OpenAI API
max_request_duration = "10m"     # Max wall-clock duration per proxied request
allow_cors = true                # Allow cross-origin requests from browsers (default: true)
max_body_mb = 64                 # Max request body size in MiB (default: 64; raise for large or many images)
```

## Global Defaults

These values apply to all services unless overridden per-service:

```toml
[defaults]
idle_timeout = "10m"             # Default idle timeout for on-demand services
priority = 50                    # Default eviction priority
```

## Device Configuration

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

`default_gpu_reserved_mb` and `gpu_reserved_mb` are kept free on every GPU when the packer places a service; a per-service `gpu_headroom_mb` adds to them for one model. `[devices.cpu] reserved_gb` is host RAM the daemon keeps free - it bounds how much expert weight a hybrid MoE service may offload to the CPU, and a placement that would exceed it is rejected.

## Service Configuration

Services are defined as an array of `[[service]]` blocks.

### Service Lifecycles

Each service runs in one of two modes:

- **On-Demand (Default)**: Loaded only when a request arrives. Unloaded after a configurable `idle_timeout` (default: 10m) to free up VRAM.
- **Persistent**: Stays loaded in memory indefinitely, ensuring zero-latency startup for critical models.

```toml
[[service]]
name = "my-model"
template = "llama-cpp"
lifecycle = "on_demand"   # or "persistent"
```

### Resource Allocation & VRAM

ananke oversubscribes GPU memory by dynamically managing which models are active:

- **llama.cpp Services**: VRAM usage is determined by the model size and `n_gpu_layers`. ananke uses an internal GGUF-aware estimator to track usage.
- **Command Services**: Support two allocation modes:
  - `static`: Reserves a fixed amount of VRAM (`vram_gb`).
  - `dynamic`: Operates within a range (`min_vram_gb` to `max_vram_gb`).

  In both modes the daemon picks the GPU with the most available headroom (subject to `gpu_allow`), preferring one whose free capacity satisfies the upper bound (`vram_gb` for `static`, `max_vram_gb` for `dynamic`) so dynamic services have room to grow. The picked GPU id is exported to the spawned child as `CUDA_VISIBLE_DEVICES`, and is also available as the `{gpu_ids}` placeholder in `command` argv. Wrappers that launch containers should forward this env (e.g. `docker run --device "nvidia.com/gpu=$CUDA_VISIBLE_DEVICES"`) so the picked GPU is the only one the container sees.
- **Eviction**: When VRAM is exhausted, ananke uses a priority-based eviction system. Higher priority services can displace dormant on-demand services.

### Placement Policies

- `gpu-only`: Service must reside entirely on GPU.
- `cpu-only`: Service resides entirely on CPU.
- `hybrid`: Allows a mix of GPU and CPU. The packer fills the GPUs first and spills the remainder to CPU. For MoE models with `expert_offload` enabled it spills *expert tensors* before whole layers, keeping every layer's attention and KV cache on the GPU (see [MoE expert offload](#moe-expert-offload)). Manual `override_tensor` rules also work here for hand-picked CPU offloading.

### Multi-GPU split modes

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

### llama.cpp Template

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
# kv_unified = true       # -kvu: one shared KV pool across slots, so idle
#                         # slots lend capacity instead of a static per-slot
#                         # split. Total KV footprint is unchanged.
# batch_size = 512        # Context batch size
# mmap = true             # Memory-map the model file
# jinja = true            # Use Jinja chat templates
# cache_idle_slots = false # Pass --no-cache-idle-slots: drop idle slots'
#                         # prompt-cache state (a stability mitigation).
# metrics = true          # Expose llama-server's Prometheus /metrics endpoint
# slots = true            # Expose the /slots endpoint (note: reveals prompt
#                         # contents - avoid on network-reachable ports)

# Speculative decoding via multi-token prediction (MTP / NextN):
# spec_type = "draft-mtp" # --spec-type. ananke's estimator adds the MTP draft
#                         # context's VRAM. Two shapes:
#                         #  - Embedded head (nextn_predict_layers > 0, e.g.
#                         #    Qwen 3.6): drafts using the same weights, no
#                         #    separate file - a small f16 KV over the nextn
#                         #    layer(s) plus a ~1.7 GiB compute buffer.
#                         #  - Separate draft model (e.g. Gemma 4): set
#                         #    draft_model below. Its attention reuses the
#                         #    target's KV, so the cost is just the draft's
#                         #    weights plus a small buffer (~0.4 GiB).
# draft_model = "/path/to/mtp-head.gguf" # -md: separate MTP/draft GGUF.
#                         # Requires spec_type to be set.
# spec_draft_n_max = 2    # --spec-draft-n-max: max draft tokens per step.
# MTP composes with `parallel > 1` and `mmproj` (vision) on current llama.cpp.
```

### MoE expert offload

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

### Custom llama-server Binary or Wrapper

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

### Command Template

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

The `http` field defaults to `/v1/models` when `[service.health]` is absent. Set `http = ""` to disable the health check entirely - the service transitions to Running immediately after spawn, with no readiness probe. This is useful for services that don't expose an HTTP endpoint, or when the operator knows the service is ready as soon as it starts.

The child also inherits `CUDA_VISIBLE_DEVICES` set to the picked GPU id(s). Wrapper scripts that launch a container should forward this so the container only sees the picked GPU - for example, `docker run --device "nvidia.com/gpu=${CUDA_VISIBLE_DEVICES:-all}" ...`.

The `shutdown_command` field is particularly useful for external processes (like Docker containers) that cannot stop via signal alone. ananke runs this command after the drain pipeline completes, ensuring clean exits for services that don't respond to SIGTERM.

### Fronting an OpenAI-Compatible Upstream

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

The rewrite happens after `[service.filters]` is applied, so `openai_proxy.upstream_model` overrides any `filters.set_params.model`. Filters can still strip or set other JSON keys.

### Embedding Services

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

## Advanced Service Options

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

## Service States

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

## Config Hot-Reload

ananke watches its config file for changes and automatically reloads when modifications are detected. The reload process:

1. Parses and validates the new config.
2. Preflights GGUF models (catching dtype/shard issues before traffic hits them).
3. Spawns added services and drains removed ones.
4. Publishes a `config_reloaded` event via the WebSocket endpoint.

Failed reloads are silently ignored - the previous valid config remains in effect.
