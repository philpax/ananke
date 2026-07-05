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
