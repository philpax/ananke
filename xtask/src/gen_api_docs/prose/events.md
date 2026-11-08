The daemon publishes lifecycle events on an internal bus and keeps the newest 1000 in memory. `GET /api/events/history` returns that window, and the `/api/live` socket's `events` resource pushes it again whenever an event arrives. The window starts afresh when the daemon restarts.

Each event is a JSON object with a `type` tag. Every variant carries `at_ms`, a millisecond UNIX timestamp.

### Variants

#### `state_changed`

Emitted when a service transitions between states.

```json
{
  "type": "state_changed",
  "service": "demo",
  "from": "idle",
  "to": "starting",
  "at_ms": 1700000000000
}
```

#### `allocation_changed`

Emitted when a service's per-device memory pledge changes.

```json
{
  "type": "allocation_changed",
  "service": "demo",
  "reservations": {
    "gpu:0": 8589934592
  },
  "at_ms": 1700000000000
}
```

#### `config_reloaded`

Emitted when the daemon's config file is reloaded. `changed_services` lists service names whose config was modified.

```json
{
  "type": "config_reloaded",
  "at_ms": 1700000000000,
  "changed_services": ["demo", "qwen"]
}
```

#### `estimator_drift`

Emitted when one of a service's rolling estimator corrections moves by more than 5%. Each service carries one correction per memory pool, learned independently from that pool's own observation, and `class` says which this event is about: `"vram"` (the observed NVML peak over the reservation's GPU slots) or `"host"` (the observed RSS peak, less the GPU-resident share of the model mapping, over the reservation's CPU slot).

```json
{
  "type": "estimator_drift",
  "service": "demo",
  "class": "vram",
  "rolling_mean": 1.05,
  "at_ms": 1700000000000
}
```

#### `auto_restarted`

Emitted when a `Running` service is drained and respawned by its auto-restart policy (see [auto-restart](configuration.md#auto-restart)). `trigger` is `"error_rate"`, `"periodic"`, `"ttft_stall"` (the service produced no response frame within the stall timeout while a request was in flight), `"generation_stall"` (the child's `/metrics` progress counters stayed flat for the stall timeout while requests were in flight), or `"spec_collapse"` (a run that previously accepted speculative draft tokens stopped accepting any); `detail` is a human-readable reason such as the observed error rate and window.

```json
{
  "type": "auto_restarted",
  "service": "demo",
  "trigger": "error_rate",
  "detail": "error rate 100% (24/24 requests over 120s) ≥ threshold 50%",
  "at_ms": 1700000000000
}
```
