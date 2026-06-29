# API Reference

## Real-Time Events

The management API exposes a WebSocket endpoint at `/api/events` that delivers real-time state changes:

| Event                | Description                                                 |
| -------------------- | ----------------------------------------------------------- |
| `state_changed`      | Service entered a new state (e.g., `idle` → `starting`)     |
| `allocation_changed` | VRAM reservation changed for a service                      |
| `config_reloaded`    | Config was reloaded, with list of changed services          |
| `estimator_drift`    | VRAM estimator detected significant drift from actual usage |
| `overflow`           | Event buffer overflow (some events were dropped)            |

## Service States

ananke tracks each service through a state machine. The `state_changed` event (above) fires on every transition.

| State      | Description                                                                                                                                    |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `idle`     | Service is unloaded, waiting for requests                                                                                                      |
| `starting` | Service is launching, waiting for health check                                                                                                 |
| `running`  | Service is healthy and accepting traffic                                                                                                       |
| `draining` | Service is shutting down, waiting for inflight requests                                                                                        |
| `stopped`  | Service was explicitly stopped                                                                                                                 |
| `evicted`  | Service was displaced by a higher-priority model                                                                                               |
| `failed`   | Service failed to start (with retry backoff, up to 3 attempts)                                                                                 |
| `disabled` | Service is permanently disabled (reason: `launch_failed`, `health_timeout`, `oom`, `crash_loop`, `no_fit`, `config_error`, or `user_disabled`) |

## OpenAI API

Available at `http://<host>:7070/v1`.

| Endpoint               | Method | Description                            |
| ---------------------- | ------ | -------------------------------------- |
| `/v1/models`           | GET    | List available models and their states |
| `/v1/chat/completions` | POST   | Chat completions (proxy to upstream)   |
| `/v1/completions`      | POST   | Legacy completions (proxy to upstream) |
| `/v1/embeddings`       | POST   | Embeddings (proxy to upstream)         |

Other standard OpenAI endpoints return appropriate HTTP 501 "not implemented" responses.

## Management API

Available at `http://<host>:7071`. Used by `anankectl` to orchestrate the daemon.

| Endpoint                          | Method   | Description                           |
| --------------------------------- | -------- | ------------------------------------- |
| `/api/services`                   | GET      | List all services                     |
| `/api/services/:name`             | GET      | Get service detail (with recent logs) |
| `/api/services/:name/start`       | POST     | Start a service                       |
| `/api/services/:name/stop`        | POST     | Stop a service                        |
| `/api/services/:name/restart`     | POST     | Restart a service                     |
| `/api/services/:name/enable`      | POST     | Enable a disabled service             |
| `/api/services/:name/disable`     | POST     | Disable a service                     |
| `/api/services/:name/logs`        | GET      | Get recent logs for a service         |
| `/api/services/:name/logs/stream` | GET (WS) | Stream logs for a service             |
| `/api/devices`                    | GET      | Device/VRAM status                    |
| `/api/devices/samples`            | GET      | Historical VRAM samples               |
| `/api/config`                     | GET      | Get current config (with ETag hash)   |
| `/api/config`                     | PUT      | Atomically update config              |
| `/api/config/validate`            | POST     | Validate TOML without persisting      |
| `/api/oneshot`                    | POST     | Spawn a short-lived service           |
| `/api/oneshot`                    | GET      | List oneshot services                 |
| `/api/oneshot/:id`                | GET      | Get oneshot status                    |
| `/api/oneshot/:id`                | DELETE   | Delete a oneshot service              |
| `/api/events`                     | GET (WS) | WebSocket for real-time events        |
| `/api/metrics`                    | GET      | Aggregated request metrics (JSON)     |
| `/metrics`                        | GET      | Prometheus text-format metrics        |
| `/api/openapi.json`               | GET      | OpenAPI spec                          |
