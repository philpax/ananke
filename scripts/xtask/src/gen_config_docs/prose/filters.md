Modify requests before they reach the model:

```toml
[service.filters]
strip_params = ["temperature"]          # Remove these JSON keys from the request
set_params = { max_tokens = 4096 }       # Force these JSON key/value pairs
```

> **Note:** `openai_proxy.upstream_model` (for command services) overrides any `filters.set_params.model`, because the model rewrite happens *after* filters are applied. Filters can still strip or set other JSON keys. See [OpenAI Proxy](#openai-proxy).
