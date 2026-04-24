# ananke

> [!WARNING]  
> This codebase is somewhere between "copilot" and "auto" on the [ai-declaration.md](https://ai-declaration.md/en/0.1.2/) scale. It was extruded from a spec that was authored by Claude on my guidance. Since then, I have been steadily adjusting it and fixing bugs that I encounter.
>
> I would not recommend that you use this unless you are able to diagnose and fix issues yourself. It won't destroy your system, but it may also not behave as expected. This was built to solve a problem I have; it'll take some time before I can be confident in saying that it can solve your problem, too.

ananke is a GPU/CPU-aware model proxy daemon designed to manage multiple LLMs and other AI tools (like ComfyUI) efficiently. It provides an OpenAI-compatible API and a management CLI to orchestrate model loading, unloading, and resource allocation.

## Getting Started

### Prerequisites

- Rust toolchain
- NVIDIA GPU (for CUDA support)
- `nvml-wrapper` (required for GPU detection and VRAM tracking)

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
[openai_api]
listen = "127.0.0.1:7070"

[daemon]
management_listen = "127.0.0.1:7071"

[devices]
gpu_ids = [0]

[[service]]
name = "my-model"
template = "llama-cpp"
port = 8200
model = "/path/to/model.gguf"
lifecycle = "on_demand"
```

Start the daemon:

```bash
ananke
```

Send a request:

```bash
curl http://127.0.0.1:7070/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "my-model", "messages": [{"role": "user", "content": "Hello!"}]}'
```

ananke will automatically load the model on first request, serve the response, then unload it after 10 minutes of idle time to free VRAM.

### Configuration Resolution

ananke searches for its configuration file in the following priority:

1. `ananke_CONFIG` environment variable.
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
- **Eviction**: When VRAM is exhausted, ananke uses a priority-based eviction system. Higher priority services can displace dormant on-demand services.

#### Placement Policies

- `gpu-only`: Service must reside entirely on GPU.
- `cpu-only`: Service resides entirely on CPU.
- `hybrid`: Allows a mix of GPU and CPU (e.g., using `override_tensor` for specific CPU offloading).

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
# parallel = 4            # Request parallelism
# batch_size = 512        # Context batch size
# mmap = true             # Memory-map the model file
# jinja = true            # Use Jinja chat templates
```

#### Command Template

Used for arbitrary binaries or Docker wrappers.

```toml
[[service]]
name = "comfyui"
template = "command"
port = 8188
lifecycle = "on_demand"
# {port} is replaced by the private loopback port assigned by ananke
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

The `shutdown_command` field is particularly useful for external processes (like Docker containers) that cannot stop via signal alone. ananke runs this command after the drain pipeline completes, ensuring clean exits for services that don't respond to SIGTERM.

### Advanced Service Options

- **Filters**: Modify requests before they reach the model.
  ```toml
  [service.filters]
  strip_params = ["temperature"] # Remove these from the request
  set_params = { max_tokens = 4096 } # Force these values
  ```
- **Devices**: Control GPU visibility.
  ```toml
  [service.devices]
  placement = "hybrid"
  gpu_allow = [0, 1] # Only use these GPUs
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

The CLI tool allows real-time management of the daemon. The base URL is set via `--endpoint` or the `ananke_ENDPOINT` environment variable.

```bash
# Service Management
anankectl services [--all]     # List all services (include disabled with --all)
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

# Configuration
anankectl config show          # View current effective configuration
anankectl config validate      # Validate a TOML file without persisting
anankectl reload               # Force a config reload

# Oneshot
anankectl oneshot submit <file> # Submit from TOML
anankectl oneshot list           # List active oneshot jobs
anankectl oneshot kill <id>     # Cancel a running oneshot
```

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
