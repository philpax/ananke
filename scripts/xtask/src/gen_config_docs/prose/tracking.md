Per-service hints that adjust how the snapshotter attributes observed VRAM/RSS to the service:

```toml
[service.tracking]
cgroup_parent = "/system.slice/ananke-comfyui.slice"
```
