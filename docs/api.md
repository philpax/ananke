# Ananke API

## Overview

Ananke is a local-first model orchestrator: it manages GPU/CPU memory placement, supervises service lifecycles, and exposes an OpenAI-compatible API alongside a management API.

### Listeners

The daemon listens on two HTTP sockets:

- **OpenAI-compatible API** (`127.0.0.1:7070` by default): `/v1/models`,
  `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`.
- **Management API** (`127.0.0.1:7071` by default): `/api/services`,
  `/api/config`, `/api/devices`, `/api/metrics`, `/api/oneshot`, `/api/events`,
  `/api/info`, and `/api/openapi.json`.

Both addresses are configurable via `[daemon]` in the TOML config.

### Security

The daemon binds to loopback by default. Do **not** expose the management API or the OpenAI API to untrusted networks without an authenticating reverse proxy in front — there is no built-in authentication or TLS.

### Conventions

- All JSON responses use standard `application/json` with UTF-8 encoding.
- Error responses follow the OpenAI error envelope shape:
  `{"error": {"code": "...", "message": "...", "type": "..."}}`.
- Timestamps are millisecond UNIX epochs unless noted.
- Byte sizes are raw bytes (not MiB/GiB) unless the field name says otherwise.
- WebSocket endpoints use text frames containing JSON-serialised messages.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/config` | Get or update the daemon config |
| `PUT` | `/api/config` | Get or update the daemon config |
| `POST` | `/api/config/validate` | Validate TOML config without persisting |
| `GET` | `/api/devices` | List devices and reservations |
| `GET` | `/api/devices/samples` | Get device memory samples |
| `GET` | `/api/info` | Get daemon listen addresses |
| `GET` | `/api/metrics` | Get request metrics (time-bucketed) |
| `GET` | `/api/oneshot` | Create or list oneshot processes |
| `POST` | `/api/oneshot` | Create or list oneshot processes |
| `GET` | `/api/oneshot/{id}` | Get or delete a oneshot process |
| `DELETE` | `/api/oneshot/{id}` | Get or delete a oneshot process |
| `GET` | `/api/services` | List all services |
| `GET` | `/api/services/{name}` | Get service detail |
| `GET` | `/api/services/{name}/command` | Get launch command preview |
| `POST` | `/api/services/{name}/disable` | Disable a service |
| `POST` | `/api/services/{name}/enable` | Enable a disabled service |
| `GET` | `/api/services/{name}/logs` | Get service logs (paginated) |
| `POST` | `/api/services/{name}/restart` | Restart a service |
| `POST` | `/api/services/{name}/start` | Start a service |
| `POST` | `/api/services/{name}/stop` | Stop a service |
| `POST` | `/v1/chat/completions` | Chat completion (OpenAI-compatible proxy) |
| `POST` | `/v1/completions` | Text completion (OpenAI-compatible proxy) |
| `POST` | `/v1/embeddings` | Embeddings (OpenAI-compatible proxy) |
| `GET` | `/v1/models` | List available models (OpenAI-compatible) |

## OpenAI-compatible API

The OpenAI-compatible API (`/v1/*`) is the primary inference surface. Ananke acts as a smart proxy: it resolves the `model` field in each request to a configured service, ensures the service is running (starting it on-demand if needed), then forwards the request body to the upstream service's private port.

### Proxy behaviour

- The request body is parsed only enough to extract `model`; all other
  fields are passed through verbatim to the upstream.
- Filters may rewrite JSON fields before forwarding (configured per-service
  via `[[service.filters]]`).
- For `openai_proxy` command services, the `model` field is rewritten to
  the upstream's expected name (`openai_proxy.upstream_model`).
- Responses — including SSE streams — are proxied back without buffering.
- Hop-by-hop headers (`connection`, `transfer-encoding`, `keep-alive`) are
  stripped from upstream responses so the browser doesn't misinterpret
  them.

### Streaming

Streaming responses (SSE) are supported on all three POST endpoints. Set `"stream": true` in the request body. The upstream's SSE chunks are proxied to the client as they arrive — there is no buffering.

### 501 stubs

The following OpenAI endpoints return `501 Not Implemented`:

- `/v1/audio/*`
- `/v1/images/*`
- `/v1/files/*`
- `/v1/fine_tuning/*`
- `/v1/batches`

### POST /v1/chat/completions

Chat completion (OpenAI-compatible proxy)

**Request body**:

```typescript
{
  model: string
  ...any
}
```

| Status | Description | Body |
| --- | --- | --- |
| 200 | Proxied from upstream | — |
| 400 | invalid_request_error | `ApiError` |
| 404 | model_not_found | `ApiError` |
| 502 | upstream_unavailable | `ApiError` |
| 503 | service_disabled, start_queue_full, start_failed, insufficient_vram, service_blocked | `ApiError` |

### POST /v1/completions

Text completion (OpenAI-compatible proxy)

**Request body**:

```typescript
{
  model: string
  ...any
}
```

| Status | Description | Body |
| --- | --- | --- |
| 200 | Proxied from upstream | — |
| 400 | invalid_request_error | `ApiError` |
| 404 | model_not_found | `ApiError` |
| 502 | upstream_unavailable | `ApiError` |
| 503 | service_disabled, start_queue_full, start_failed, insufficient_vram, service_blocked | `ApiError` |

### POST /v1/embeddings

Embeddings (OpenAI-compatible proxy)

**Request body**:

```typescript
{
  model: string
  ...any
}
```

| Status | Description | Body |
| --- | --- | --- |
| 200 | Proxied from upstream | — |
| 400 | invalid_request_error | `ApiError` |
| 404 | model_not_found | `ApiError` |
| 502 | upstream_unavailable | `ApiError` |
| 503 | service_disabled, start_queue_full, start_failed, insufficient_vram, service_blocked | `ApiError` |

### GET /v1/models

List available models (OpenAI-compatible)

| Status | Description | Body |
| --- | --- | --- |
| 200 |  | `ModelsResponse` |

**Response (200)**:

```typescript
{
  data: {
    ananke_metadata?: Record<string, any>
    created: number
    id: string
    modality?: "chat" | "embedding"
    object: string
    owned_by: string
  }[]
  object: string
}
```

## Management API

The management API (`/api/*`) provides read-only inspection and control of the daemon's state. All endpoints return JSON unless noted.

### Resource groups

- **Services**: list, detail, launch command preview, lifecycle
  (start/stop/restart/enable/disable), logs.
- **Devices**: list with reservations, memory samples.
- **Config**: get/put/validate the TOML config.
- **Metrics**: request metrics buckets, device samples.
- **Oneshot**: create/list/get/delete ephemeral processes.
- **Events**: WebSocket stream of state changes, allocation changes, etc.
- **Info**: daemon listen addresses.

### Error responses

All management API errors use the standard OpenAI-shaped error envelope with a typed `code` slug, human-readable `message`, and taxonomy `type`. See the [Error codes](#error-codes) section for the full catalogue.

### Config

#### GET /api/config

Get or update the daemon config

| Status | Description | Body |
| --- | --- | --- |
| 200 |  | `ConfigResponse` |

**Response (200)**:

```typescript
{
  content: string
  hash: string
  writable: boolean
}
```

#### PUT /api/config

Get or update the daemon config

| Status | Description | Body |
| --- | --- | --- |
| 202 |  | — |
| 412 | hash_mismatch | `ApiError` |
| 422 |  | `ConfigValidateResponse` |
| 428 | if_match_required | `ApiError` |
| 500 | persist_failed | `ApiError` |

#### POST /api/config/validate

Validate TOML config without persisting

**Request body**:

```typescript
{
  content: string
}
```

| Status | Description | Body |
| --- | --- | --- |
| 200 |  | `ConfigValidateResponse` |

**Response (200)**:

```typescript
{
  errors: {
    column: number
    line: number
    message: string
  }[]
  valid: boolean
}
```

### Devices

#### GET /api/devices

List devices and reservations

| Status | Description | Body |
| --- | --- | --- |
| 200 |  | `DeviceSummary[]` |

**Response (200)**:

```typescript
{
  free_bytes: number
  id: string
  name: string
  reservations: {
    bytes: number
    elastic: boolean
    service: string
  }[]
  total_bytes: number
}[]
```

#### GET /api/devices/samples

Get device memory samples

| Parameter | In | Required | Description |
| --- | --- | --- | --- |
| `device` | query | no | Filter to one device (e.g. "gpu:0", "cpu") |
| `since` | query | no | Earliest timestamp_ms (default: 1h ago) |
| `until` | query | no | Latest timestamp_ms (default: now) |

| Status | Description | Body |
| --- | --- | --- |
| 200 |  | `DeviceSamplesResponse` |

**Response (200)**:

```typescript
{
  samples: {
    device: string
    free_bytes: number
    timestamp_ms: number
    total_bytes: number
    used_bytes: number
  }[]
}
```

### Info

#### GET /api/info

Get daemon listen addresses

| Status | Description | Body |
| --- | --- | --- |
| 200 |  | `DaemonInfoResponse` |

**Response (200)**:

```typescript
{
  management_listen: string
  openai_listen: string
}
```

### Metrics

#### GET /api/metrics

Get request metrics (time-bucketed)

| Parameter | In | Required | Description |
| --- | --- | --- | --- |
| `service` | query | no | Filter to one service name |
| `since` | query | no | Earliest timestamp_ms, inclusive (default: 1h ago) |
| `until` | query | no | Latest timestamp_ms, inclusive (default: now) |
| `bucket` | query | no | Bucket size: "1m", "5m", "1h" (default: "5m") |

| Status | Description | Body |
| --- | --- | --- |
| 200 |  | `MetricsResponse` |
| 400 | invalid_request_error | `ApiError` |

**Response (200)**:

```typescript
{
  buckets: {
    aggregate_tps?: number | null
    avg_duration_ms?: number | null
    avg_ttft_ms?: number | null
    bucket_start: number
    completion_tokens: number
    error_count: number
    input_tps?: number | null
    output_tps?: number | null
    prompt_tokens: number
    request_count: number
    service?: string | null
  }[]
}
```

### Oneshot

#### GET /api/oneshot

Create or list oneshot processes

| Status | Description | Body |
| --- | --- | --- |
| 200 |  | `OneshotStatus[]` |

**Response (200)**:

```typescript
{
  ended_at_ms?: number | null
  exit_code?: number | null
  id: string
  logs_url: string
  name: string
  port: number
  started_at_ms?: number | null
  state: string
  submitted_at_ms: number
}[]
```

#### POST /api/oneshot

Create or list oneshot processes

**Request body**:

```typescript
{
  allocation: {
    max_vram_gb?: number | null
    min_vram_gb?: number | null
    mode?: string | null
    vram_gb?: number | null
  }
  command?: string[] | null
  devices?: {
    placement?: string | null
  } | null
  health?: {
    http: string
    probe_interval?: string | null
    timeout?: string | null
  } | null
  metadata?: Record<string, any>
  name?: string | null
  port?: number | null
  priority?: number | null
  template: string
  ttl?: string | null
  workdir?: string | null
}
```

| Status | Description | Body |
| --- | --- | --- |
| 200 |  | `OneshotResponse` |
| 400 | invalid_request_error | `ApiError` |
| 503 | start_failed, start_queue_full | `ApiError` |

**Response (200)**:

```typescript
{
  id: string
  logs_url: string
  name: string
  port: number
}
```

#### GET /api/oneshot/{id}

Get or delete a oneshot process

| Parameter | In | Required | Description |
| --- | --- | --- | --- |
| `id` | path | yes |  |

| Status | Description | Body |
| --- | --- | --- |
| 200 |  | `OneshotStatus` |
| 404 | service_not_found | `ApiError` |

**Response (200)**:

```typescript
{
  ended_at_ms?: number | null
  exit_code?: number | null
  id: string
  logs_url: string
  name: string
  port: number
  started_at_ms?: number | null
  state: string
  submitted_at_ms: number
}
```

#### DELETE /api/oneshot/{id}

Get or delete a oneshot process

| Parameter | In | Required | Description |
| --- | --- | --- | --- |
| `id` | path | yes |  |

| Status | Description | Body |
| --- | --- | --- |
| 204 |  | — |
| 404 | service_not_found | `ApiError` |

### Services

#### GET /api/services

List all services

| Status | Description | Body |
| --- | --- | --- |
| 200 |  | `ServicesResponse` |

**Response (200)**:

```typescript
{
  openai_api_port: number
  services: {
    ananke_metadata?: Record<string, any>
    elastic_borrower?: string | null
    fit_verdict?: "fits" | "needs_eviction" | "does_not_fit" | null
    has_mmproj?: boolean | null
    inflight_count?: number
    last_used_ms?: number | null
    lifecycle: string
    modality?: "chat" | "embedding"
    name: string
    pid?: number | null
    port: number
    priority: number
    run_id?: number | null
    state: string
    vram_bytes?: number | null
  }[]
}
```

#### GET /api/services/{name}

Get service detail

| Parameter | In | Required | Description |
| --- | --- | --- | --- |
| `name` | path | yes |  |

| Status | Description | Body |
| --- | --- | --- |
| 200 |  | `ServiceDetail` |
| 404 | service_not_found | `ApiError` |

**Response (200)**:

```typescript
{
  ananke_metadata?: Record<string, any>
  current_allocation: Record<string, number>
  elastic_borrower?: string | null
  estimate?: {
    compute_buffer_bytes_per_device: number
    configured_context: number
    kv_bytes_for_context: number
    kv_per_token: number
    weights_bytes: number
  } | null
  idle_timeout_ms: number
  last_used_ms?: number | null
  lifecycle: string
  modality?: "chat" | "embedding"
  model_info?: {
    architecture: string
    block_count?: number | null
    file_name: string
    has_mmproj: boolean
    license?: string | null
    model_name?: string | null
    parameter_count?: number | null
    shard_count: number
    total_tensor_bytes: number
    trained_context_length?: number | null
  } | null
  name: string
  observed_peak_bytes: number
  pid?: number | null
  placement_override: Record<string, number>
  placement_preview?: {
    devices: {
      bytes: number
      device: string
      max_bytes: number
      total_bytes: number
      used_by_others_bytes: number
    }[]
    expert_offload_bytes: number
    expert_offload_layers: number
    verdict: "fits" | "needs_eviction" | "does_not_fit"
  } | null
  port: number
  priority: number
  private_port: number
  recent_logs: {
    line: string
    run_id: number
    seq: number
    stream: string
    timestamp_ms: number
  }[]
  rolling_mean?: number | null
  rolling_samples: number
  run_id?: number | null
  state: string
  template: string
}
```

#### GET /api/services/{name}/command

Get launch command preview

| Parameter | In | Required | Description |
| --- | --- | --- | --- |
| `name` | path | yes | Service name |

| Status | Description | Body |
| --- | --- | --- |
| 200 |  | `LaunchCommandResponse` |
| 404 | service_not_found | `ApiError` |
| 422 | insufficient_vram | `ApiError` |

**Response (200)**:

```typescript
{
  active?: {
    argv: string[]
    env: {
      key: string
      value: string
    }[]
    env_inherit: boolean
    source: "running" | "preview"
  } | null
  on_empty: {
    argv: string[]
    env: {
      key: string
      value: string
    }[]
    env_inherit: boolean
    source: "running" | "preview"
  }
}
```

#### POST /api/services/{name}/disable

Disable a service

| Parameter | In | Required | Description |
| --- | --- | --- | --- |
| `name` | path | yes | Service name |

| Status | Description | Body |
| --- | --- | --- |
| 200 |  | `DisableResponse` |
| 404 | service_not_found | `ApiError` |

**Response (200)**:

```typescript
// disabled
{ status: "disabled" }
// already_disabled
{ status: "already_disabled" }
```

#### POST /api/services/{name}/enable

Enable a disabled service

| Parameter | In | Required | Description |
| --- | --- | --- | --- |
| `name` | path | yes | Service name |

| Status | Description | Body |
| --- | --- | --- |
| 200 |  | `EnableResponse` |
| 404 | service_not_found | `ApiError` |

**Response (200)**:

```typescript
// enabled
{ status: "enabled" }
// not_disabled
{ status: "not_disabled" }
// already_enabled
{ status: "already_enabled" }
```

#### GET /api/services/{name}/logs

Get service logs (paginated)

| Parameter | In | Required | Description |
| --- | --- | --- | --- |
| `name` | path | yes | Service name |
| `since` | query | no | Earliest timestamp_ms, inclusive |
| `until` | query | no | Latest timestamp_ms, inclusive |
| `run` | query | no | Restrict to one run_id |
| `stream` | query | no | "stdout" or "stderr" |
| `limit` | query | no | Max rows to return (≤1000, default 200) |
| `before` | query | no | Opaque cursor from a prior response |

| Status | Description | Body |
| --- | --- | --- |
| 200 |  | `LogsResponse` |
| 400 | invalid_cursor | `ApiError` |
| 404 | service_not_found | `ApiError` |

**Response (200)**:

```typescript
{
  logs: {
    line: string
    run_id: number
    seq: number
    stream: string
    timestamp_ms: number
  }[]
  next_cursor?: string | null
}
```

#### POST /api/services/{name}/restart

Restart a service

| Parameter | In | Required | Description |
| --- | --- | --- | --- |
| `name` | path | yes | Service name |

| Status | Description | Body |
| --- | --- | --- |
| 202 |  | `StartResponse` |
| 404 | service_not_found | `ApiError` |
| 503 | service_disabled, start_queue_full, start_failed, insufficient_vram, service_blocked | `ApiError` |

**Response (202)**:

```typescript
// already_running
{ status: "already_running" }
// started
{ run_id: number, status: "started" }
// queue_full
{ status: "queue_full" }
// unavailable
{ error: ApiErrorBody, status: "unavailable" }
```

#### POST /api/services/{name}/start

Start a service

| Parameter | In | Required | Description |
| --- | --- | --- | --- |
| `name` | path | yes | Service name |

| Status | Description | Body |
| --- | --- | --- |
| 202 |  | `StartResponse` |
| 404 | service_not_found | `ApiError` |
| 503 | service_disabled, start_queue_full, start_failed, insufficient_vram, service_blocked | `ApiError` |

**Response (202)**:

```typescript
// already_running
{ status: "already_running" }
// started
{ run_id: number, status: "started" }
// queue_full
{ status: "queue_full" }
// unavailable
{ error: ApiErrorBody, status: "unavailable" }
```

#### POST /api/services/{name}/stop

Stop a service

| Parameter | In | Required | Description |
| --- | --- | --- | --- |
| `name` | path | yes | Service name |

| Status | Description | Body |
| --- | --- | --- |
| 202 |  | `StopResponse` |
| 404 | service_not_found | `ApiError` |

**Response (202)**:

```typescript
// not_running
{ status: "not_running" }
// drained
{ status: "drained" }
```

## Per-service reverse proxy

Each service exposes a per-service reverse proxy on its configured `port`. This proxy forwards HTTP and WebSocket requests to the service's `private_port`, enabling direct access to the upstream (e.g. llama.cpp's web UI, ComfyUI's API) without going through the OpenAI-compatible endpoints.

### Reverse proxy behaviour

- **Transparent forwarding**: the full request path, query string, and
  headers (except `Host`) are forwarded to the upstream.
- **WebSocket upgrade**: upgrade requests are handled with a raw byte
  splice after the 101 Switching Protocols response. The upstream's
  `Connection` and `Upgrade` headers are forwarded verbatim — aiohttp's
  WebSocket client validates `Connection` with exact equality after
  lowercasing, so any rewrite breaks the handshake.
- **Hop-by-hop header stripping**: `connection`, `transfer-encoding`,
  and `keep-alive` are stripped from upstream responses so the browser
  doesn't misinterpret them (e.g. llama.cpp sends `keep-alive: timeout=5`,
  which would cause the browser to close the connection prematurely).
- **On-demand start**: the proxy triggers a service ensure (start) before
  forwarding the first request. If the supervisor cannot start the service
  (VRAM shortfall, disabled, etc.), a `503` error is returned with the
  standard error envelope.
- **In-flight tracking**: each request increments a per-service counter
  that pins the service open against drain. The counter stays elevated
  for the full response body duration (including SSE streams). WebSocket
  sessions hold a long-lived guard that prevents idle eviction.
- **Activity pinging**: WebSocket sessions ping the activity stamp every
  5 seconds so the supervisor's idle-eviction loop doesn't terminate a
  quietly-active session.
- **`allow_external_services`**: when enabled in config, the proxy also
  serves external services that aren't managed by ananke's supervisor.

## Service states

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
| `auto_restart_loop` | Auto-restart fired more times than its flap cap allowed within the flap window; the service kept degrading in service (as opposed to `crash_loop`, where the process never launched), so it is disabled rather than restarted again. |

### Transitions

- `idle → starting`: spawn requested (first request or persistent watcher).
- `starting → running`: health check passed.
- `starting → failed`: launch failed (retry 0–2).
- `starting → disabled`: health check timed out.
- `running → draining`: drain requested (stop, evict, or shutdown).
- `running → stopped`: process exited.
- `running → disabled`: crash loop detected, or the auto-restart flap cap was tripped → `auto_restart_loop`.
- `draining → idle`: drain complete.
- `draining → stopped`: process exited during drain.
- `stopped → starting`: re-spawn requested.
- `failed → failed`: retry backoff elapsed (increments retry_count).
- `failed → disabled`: retry cap (3) exceeded → `launch_failed`.
- `disabled → idle`: user enable.
- `* → disabled`: user disable.

## WebSocket: /api/events

The `/api/events` WebSocket delivers a system-wide stream of daemon events. Connect with a standard WebSocket handshake to `ws://<host>:7071/api/events`.

An optional `?service=<name>` query parameter filters events to a single service. Events that don't carry a service field (`config_reloaded`, `overflow`) are always delivered.

### Frame format

Each frame is a text message containing a single JSON object with a `type` tag. The `at_ms` field (millisecond UNIX timestamp) is present on every variant except `overflow`.

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

Emitted when the rolling estimator updates its correction factor for a service.

```json
{
  "type": "estimator_drift",
  "service": "demo",
  "rolling_mean": 1.05,
  "at_ms": 1700000000000
}
```

#### `auto_restarted`

Emitted when a `Running` service is drained and respawned by its auto-restart policy (see [auto-restart](configuration.md#auto-restart)). `trigger` is `"error_rate"` or `"periodic"`; `detail` is a human-readable reason such as the observed error rate and window.

```json
{
  "type": "auto_restarted",
  "service": "demo",
  "trigger": "error_rate",
  "detail": "error rate 100% (24/24 requests over 120s) ≥ threshold 50%",
  "at_ms": 1700000000000
}
```

#### `overflow`

Emitted when the event bus drops events because a subscriber fell behind.

```json
{
  "type": "overflow",
  "dropped": 42
}
```

### Heartbeat

The server sends a WebSocket Ping frame every 30 seconds. Clients should respond with Pong to keep intermediaries from closing the connection.

## WebSocket: /api/services/{name}/logs/stream

The `/api/services/{name}/logs/stream` WebSocket delivers a live tail of captured stdout/stderr lines for a single service.

### Frame format

Each frame is a text message containing a single JSON object with a `type` tag.

### Variants

#### `line`

One captured log line.

```json
{
  "type": "line",
  "timestamp_ms": 1700000000000,
  "stream": "stdout",
  "line": "model loaded",
  "run_id": 1,
  "seq": 42
}
```

#### `overflow`

Emitted when the subscriber's broadcast buffer lagged and frames were dropped.

```json
{
  "type": "overflow",
  "dropped": 10
}
```

### Heartbeat

The server sends a WebSocket Ping frame every 30 seconds.

### Cursor pagination (REST)

For historical logs, use `GET /api/services/{name}/logs` with cursor-based pagination:

1. Request the first page with `?limit=200`.
2. If `next_cursor` is present in the response, pass it as `?before=<cursor>`
   on the next request.
3. Repeat until `next_cursor` is `null`.

The cursor is an opaque base64-encoded string encoding `(run_id, seq)`. Rows are returned newest-first. Other filters: `since`, `until`, `run`, `stream`.

## Error codes

All error responses — across both the OpenAI-compatible API and the management API — use the same OpenAI-shaped envelope:

```json
{
  "error": {
    "code": "insufficient_vram",
    "message": "service `demo` cannot fit: ...",
    "type": "server_error"
  }
}
```

### Fields

| Field | Type | Description |
| --- | --- | --- |
| `code` | `ApiErrorCodeSlug` | Stable wire slug identifying the error class. |
| `message` | `string` | Human-readable error message. |
| `type` | `ApiErrorKind` | `invalid_request_error` or `server_error`. |

### Forward compatibility

Both `code` and `type` use an `Other` fallback for deserialization. If the daemon adds a new error code before a client is updated, the client deserialises it as `Other` rather than failing. This lets the daemon evolve without breaking older clients.

### `StartResponse::Unavailable`

The `POST /api/services/{name}/start` endpoint returns `202 Accepted` even when the supervisor declines to start the service (VRAM shortfall, disabled, etc.). The body is `{"status": "unavailable", "error": {...}}` with the same `ApiErrorBody` shape a `503` error would carry. This is a "controlled outcome" of the start request, not a server-side fault.

| Slug | Description |
| --- | --- |
| `model_not_found` | Client referenced a model name that isn't configured. |
| `service_not_found` | Management caller referenced a service that isn't in the config. |
| `service_disabled` | Service is administratively disabled. |
| `start_queue_full` | Supervisor's start queue saturated. |
| `start_failed` | Spawn or health-probe failure during ensure. |
| `insufficient_vram` | Packer couldn't lay out the model on available devices. |
| `service_blocked` | Queued behind a busy non-elastic peer. |
| `upstream_unavailable` | Upstream child rejected the wire or never replied. |
| `proxy_internal` | Bug inside the proxy itself (URI parse, header build, etc.). |
| `not_implemented` | OpenAI endpoint the daemon hasn't implemented. |
| `invalid_request_error` | Client request was malformed (bad JSON, missing field, etc.). |
| `invalid_cursor` | Log-paging cursor failed to decode. |
| `if_match_required` | Config PUT arrived without an If-Match header. |
| `hash_mismatch` | Config PUT's If-Match didn't match the current hash. |
| `persist_failed` | Config write failed at the IO layer. |
| `other` | Forward-compatibility fallback for unknown codes. |

## Prometheus metrics

The `/metrics` endpoint exposes Prometheus text-format metrics for external scrapers. Prometheus stores history; the JSON `/api/metrics` endpoint serves ananke's own historical data for the dashboard charts.

### Metric families

| Family | Type | Description |
| --- | --- | --- |
| `ananke_requests_total` | counter | Total number of requests proxied. |
| `ananke_tokens_total` | counter | Total tokens processed (labelled by `type`: `prompt` or `completion`). |
| `ananke_inflight_requests` | gauge | Current number of in-flight requests. |
| `ananke_memory_bytes` | gauge | Total memory capacity in bytes (labelled by `device`). |
| `ananke_memory_free_bytes` | gauge | Free memory in bytes (labelled by `device`). |
| `ananke_memory_used_bytes` | gauge | Used memory in bytes (labelled by `device`). |
| `ananke_output_tokens_per_second` | gauge | Average output tokens/sec during decode over the last 5 minutes. |
| `ananke_input_tokens_per_second` | gauge | Average input tokens/sec during prompt processing over the last 5 minutes. |
| `ananke_aggregate_tokens_per_second` | gauge | Average end-to-end throughput (total tokens over wall-clock duration) over the last 5 minutes. Not a decode rate; the fallback when no input/output split is available (non-streaming with no engine timings). |
| `ananke_service_state` | gauge | Numeric service state code (labelled by `service`). |

### State codes

The `ananke_service_state` gauge uses numeric codes:

| Code | State |
| --- | --- |
| 0 | idle |
| 1 | starting |
| 2 | running |
| 3 | draining |
| 4 | stopped |
| 5 | evicted |
| 6 | failed |
| 7 | disabled |
| 8 | unknown |

Code 8 (`unknown`) is used when no supervisor handle exists for a configured service — the service is in the config but hasn't been registered yet.

## OpenAPI specification

The daemon serves its own OpenAPI specification at `GET /api/openapi.json`. This is the same document used to generate this file and the frontend's TypeScript types.

The spec is produced by `utoipa`'s compile-time derive from the `#[utoipa::path]` annotations on the daemon's handler functions and the `#[derive(ToSchema)]` types in the `ananke-api` crate. No daemon state or runtime introspection is involved.

## Embedded frontend

The daemon embeds the frontend SPA as a fallback handler. Any request that doesn't match an `/api/*`, `/v1/*`, `/metrics`, or WebSocket route is served from the embedded `frontend/dist` directory.

The embedded assets are built at compile time via `rust-embed`. Set the `ANANKE_SKIP_FRONTEND_BUILD=1` environment variable during `cargo build` to skip the frontend build (useful for Rust-only dev loops).

