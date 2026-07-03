Each service is in exactly one state at any time. The state machine drives lifecycle transitions, idle eviction, and health-gated startup.

### States

| State | Code | Description |
| --- | --- | --- |
| `idle` | 0 | Not running, ready to start on demand. |
| `starting` | 1 | Spawn requested, awaiting health check. |
| `running` | 2 | Process is alive and healthy. |
| `draining` | 3 | Graceful shutdown in progress; in-flight requests completing. |
| `stopped` | 4 | Process exited cleanly (drain complete or stopped). |
| `evicted` | 5 | Evicted to free VRAM for a higher-priority service. |
| `failed` | 6 | Process crashed; retrying with exponential backoff. |
| `disabled` | 7 | Administratively disabled; will not start until enabled. |
| `unknown` | 8 | No supervisor handle exists (used in Prometheus only). |

### Disabled reasons

A `disabled` state carries a reason that explains why:

| Reason | Description |
| --- | --- |
| `config_error` | The service's config is invalid. |
| `launch_failed` | The process failed to launch 3 times (retry cap exceeded). |
| `health_timeout` | The health check didn't pass within the timeout. |
| `oom` | The process was killed by the OOM killer. |
| `crash_loop` | The process crashed repeatedly after becoming healthy. |
| `no_fit` | The model doesn't fit on any available device. |
| `user_disabled` | An operator disabled the service via the management API. |

### Transitions

- `idle → starting`: spawn requested (first request or persistent watcher).
- `starting → running`: health check passed.
- `starting → failed`: launch failed (retry 0–2).
- `starting → disabled`: health check timed out.
- `running → draining`: drain requested (stop, evict, or shutdown).
- `running → stopped`: process exited.
- `running → disabled`: crash loop detected.
- `draining → idle`: drain complete.
- `draining → stopped`: process exited during drain.
- `stopped → starting`: re-spawn requested.
- `failed → failed`: retry backoff elapsed (increments retry_count).
- `failed → disabled`: retry cap (3) exceeded → `launch_failed`.
- `disabled → idle`: user enable.
- `* → disabled`: user disable.
