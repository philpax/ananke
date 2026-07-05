Override the internal GGUF-aware VRAM estimator's parameters:

```toml
[service.estimation]
compute_buffer_mb = 512
safety_factor = 1.1
allow_fallback = false
```
