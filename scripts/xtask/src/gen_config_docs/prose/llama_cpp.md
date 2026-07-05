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
