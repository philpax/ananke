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
