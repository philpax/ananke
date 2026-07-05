Self-healing for a `Running` service that is alive but degraded — the process has not exited, so the crash-detection path never fires, yet every request is failing or the process has accumulated dirty internal state. Two independent triggers both feed the existing drain → respawn cycle.

```toml
[service.auto_restart]
# Error-rate watchdog (on by default; write `error_rate = false` to opt out):
error_rate = { window = "2m", max_error_rate = 0.5, min_requests = 20, poll_interval = "30s", error_statuses = "5xx" }
# Periodic restart (off by default; a table with an interval enables it):
periodic = { interval = "6h", mode = "on-request" }
# Anti-flap guardrails, shared by both triggers:
min_uptime = "5m"
max_restarts = 3
flap_window = "30m"
```

The block is resolved as a **whole unit**: a service that sets any `auto_restart` field replaces `[defaults.auto_restart]` entirely rather than merging field-by-field. The same `[defaults.auto_restart]` block is accepted for fleet-wide defaults.
