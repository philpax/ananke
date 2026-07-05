```toml
[service.health]
http = "/health"        # HTTP path to probe for readiness
timeout = "3m"          # Per-probe timeout
probe_interval = "5s"   # Probe cadence
```

When `[service.health]` is absent, the default `http` is `/v1/models`. Disabling health checks is useful for services that don't expose an HTTP endpoint, or when the operator knows the service is ready as soon as it starts.
