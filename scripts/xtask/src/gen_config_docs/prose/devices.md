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
