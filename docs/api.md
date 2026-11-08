# Ananke API

## Overview

Ananke is a local-first model orchestrator: it manages GPU/CPU memory placement, supervises service lifecycles, and exposes an OpenAI-compatible API alongside a management API.

### Listeners

The daemon listens on two HTTP sockets:

- **OpenAI-compatible API** (`127.0.0.1:7070` by default): `/v1/models`,
  `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`.
- **Management API** (`127.0.0.1:7071` by default): `/api/services`,
  `/api/config`, `/api/devices`, `/api/metrics`, `/api/events/history`,
  `/api/info`, `/api/live`, and `/api/openapi.json`.

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
| `GET` | `/api/events/history` | List recent events |
| `GET` | `/api/info` | Get daemon listen addresses and uptime |
| `GET` | `/api/metrics` | Get request metrics (time-bucketed) |
| `GET` | `/api/restarts` | Get auto-restart firings |
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

### Llama.cpp-native endpoints

llama-server exposes its own native surface alongside the OpenAI ones, and those paths are served on the same listener: `/tokenize`, `/detokenize`, `/apply-template`, `/completion`, `/infill`, `/embedding`, `/embeddings`, `/rerank`, `/reranking`, `/props`, `/slots`, `/slots/{id}`, `/lora-adapters`, `/health`, `/v1/health`, and `/metrics`. A request to any of them is forwarded **verbatim** — original method, path, query, headers, and body — to the upstream of a `llama-cpp` template service, as long as the service is running (it is started on demand if not).

These routes exist only for `llama-cpp` template services; a request that names a `command`-template service is rejected. Service selection mirrors the rest of the surface:

- A `model` field in the JSON body selects the service, if present.
- Without one, a single configured llama-cpp service is the target — the
  reference single-model setup needs no extra configuration.
- Multiple llama-cpp services without a `model` field is an error naming
  the candidates and the fix.
- With no llama-cpp service at all, the request gets a structured error.

Filters do not apply to these requests (they expect OpenAI-shaped bodies), and they are not recorded for request metrics, matching the per-service proxy. In a multi-llama-cpp daemon, hit the native endpoints that take no model (the GETs, `/slots/{id}`, and model-less POSTs) on the service's own port instead — the per-service proxy forwards everything on `port` directly to that service's upstream.

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
| 503 | service_disabled, start_queue_full, start_failed, insufficient_capacity, service_blocked | `ApiError` |

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
| 503 | service_disabled, start_queue_full, start_failed, insufficient_capacity, service_blocked | `ApiError` |

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
| 503 | service_disabled, start_queue_full, start_failed, insufficient_capacity, service_blocked | `ApiError` |

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
- **Events**: the recent event history; live through `/api/live`.
- **Info**: daemon listen addresses.

### Log history

`GET /api/services/{name}/logs` pages a service's captured lines newest first. Filters are `since`, `until`, `run`, and `stream`, and `limit` caps a page at 1000 lines (200 by default). To page further back, pass a response's `next_cursor` as `before` on the next request, until `next_cursor` is `null`. The cursor is opaque. Pages are ordered by position, run id then `seq`, so paging neither skips nor repeats a line.

A page requested without `before` or `until` also carries `follow_after`, the newest position in the service's whole log when it was read. Following from it on `/api/live` continues the page without a gap; see the feeds section of the `/api/live` reference.

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
  path: string
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
| 500 | query_failed | `ApiError` |

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

Get daemon listen addresses and uptime

| Status | Description | Body |
| --- | --- | --- |
| 200 |  | `DaemonInfoResponse` |

**Response (200)**:

```typescript
{
  management_listen: string
  openai_listen: string
  uptime_ms: number
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
| 500 | query_failed | `ApiError` |

**Response (200)**:

```typescript
{
  buckets: {
    avg_duration_ms?: number | null
    avg_ttft_ms?: number | null
    bucket_start: number
    completion_tokens: number
    effective_tps?: number | null
    error_count: number
    input_tps?: number | null
    output_tps?: number | null
    prompt_tokens: number
    request_count: number
    service?: string | null
  }[]
}
```

### Other

#### GET /api/events/history

List recent events

| Status | Description | Body |
| --- | --- | --- |
| 200 |  | `EventHistory` |

**Response (200)**:

```typescript
{
  events:
    // state_changed
    | {
      at_ms: number
      disabled_reason?: "launch_failed" | "health_timeout" | "oom" | "user_disabled" | "auto_restart_loop" | null
      from: "idle" | "starting" | "running" | "draining" | "stopped" | "evicted" | "failed" | "disabled" | "unknown"
      service: string
      to: "idle" | "starting" | "running" | "draining" | "stopped" | "evicted" | "failed" | "disabled" | "unknown"
      type: "state_changed"
    }
    // allocation_changed
    | {
      at_ms: number
      reservations: Record<string, number>
      service: string
      type: "allocation_changed"
    }
    // config_reloaded
    | {
      at_ms: number
      changed_services: string[]
      type: "config_reloaded"
    }
    // estimator_drift
    | {
      at_ms: number
      class: string
      rolling_mean: number
      service: string
      type: "estimator_drift"
    }
    // auto_restarted
    | {
      at_ms: number
      detail: string
      service: string
      trigger: string
      type: "auto_restarted"
    }[]
  since_ms: number
}
```

#### GET /api/restarts

Get auto-restart firings

| Parameter | In | Required | Description |
| --- | --- | --- | --- |
| `service` | query | no | Filter to one service name |
| `since` | query | no | Earliest at_ms, inclusive (default: 1h ago) |
| `until` | query | no | Latest at_ms, inclusive (default: now) |

| Status | Description | Body |
| --- | --- | --- |
| 200 |  | `RestartsResponse` |
| 500 | query_failed | `ApiError` |

**Response (200)**:

```typescript
{
  restarts: {
    at_ms: number
    detail: string
    run_id?: number | null
    service: string
    trigger: string
  }[]
}
```

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
    description?: string | null
    disabled_reason?: "launch_failed" | "health_timeout" | "oom" | "user_disabled" | "auto_restart_loop" | null
    elastic_borrower?: string | null
    fit_verdict?:
      // fits
      | {
        kind: "fits"
      }
      // needs_eviction
      | {
        kind: "needs_eviction"
      }
      // does_not_fit
      | {
        kind: "does_not_fit"
        shortfalls: {
          available_bytes: number
          device: string
          requested_bytes: number
        }[]
      } | null
    footprint_bytes?: number | null
    footprint_devices?: {
      bytes: number
      device: string
    }[]
    has_mmproj?: boolean | null
    inflight_count?: number
    last_used_ms?: number | null
    lifecycle: "persistent" | "on_demand"
    modality?: "chat" | "embedding"
    name: string
    openai_compat: boolean
    pid?: number | null
    port: number
    priority: number
    run_id?: number | null
    state: "idle" | "starting" | "running" | "draining" | "stopped" | "evicted" | "failed" | "disabled" | "unknown"
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
  command?: {
    private_port_fixed: boolean
    shutdown_command?: string[] | null
    upstream_model?: string | null
    workdir?: string | null
  } | null
  container?: {
    container_id?: string | null
    container_name?: string | null
    image: string
    network: string
    runtime: string
  } | null
  current_allocation: Record<string, number>
  description?: string | null
  disabled_reason?: "launch_failed" | "health_timeout" | "oom" | "user_disabled" | "auto_restart_loop" | null
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
  lifecycle: "persistent" | "on_demand"
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
  openai_compat: boolean
  pid?: number | null
  placement: {
    allocation:
      // estimated
      | {
        kind: "estimated"
      }
      // static
      | {
        kind: "static"
        reserve_bytes: number
      }
      // dynamic
      | {
        kind: "dynamic"
        max_bytes: number
        min_borrower_runtime_ms: number
        min_bytes: number
      }
    gpu_allow?: number[]
    gpu_headroom_bytes: number
    policy: "gpu-only" | "cpu-only" | "hybrid"
    split?: {
      mode: "layer" | "row" | "tensor"
      tensor_split_weights?: number[] | null
    } | null
  }
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
    verdict:
      // fits
      | {
        kind: "fits"
      }
      // needs_eviction
      | {
        kind: "needs_eviction"
      }
      // does_not_fit
      | {
        kind: "does_not_fit"
        shortfalls: {
          available_bytes: number
          device: string
          requested_bytes: number
        }[]
      }
  } | null
  port: number
  priority: number
  private_port: number
  recent_restarts?: {
    at_ms: number
    detail: string
    run_id?: number | null
    trigger: string
  }[]
  rolling_mean?: number | null
  rolling_mean_host?: number | null
  rolling_samples: number
  rolling_samples_host: number
  run_id?: number | null
  runtime?: {
    ik?: {
      attn_max_batch?: number | null
      dsa: boolean
      mla?: number | null
      runtime_repack: boolean
    } | null
    kind: string
  } | null
  serving?: {
    batch_size?: number | null
    binary: string
    cache_ram_bytes: number
    cache_type_k: string
    cache_type_v: string
    context?: number | null
    draft_model?: string | null
    effective_context_per_slot?: number | null
    expert_offload: string
    flash_attn: boolean
    kv_unified: boolean
    mlock: boolean
    mmap: boolean
    n_gpu_layers?: number | null
    numa?: string | null
    parallel: number
    spec_type?: string | null
    threads?: number | null
    threads_batch?: number | null
    ubatch_size?: number | null
  } | null
  state: "idle" | "starting" | "running" | "draining" | "stopped" | "evicted" | "failed" | "disabled" | "unknown"
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
| 422 | preview_failed | `ApiError` |

**Response (200)**:

```typescript
{
  active?: {
    argv: string[]
    container?: {
      argv: string[]
      create_argv: string[]
      env: {
        key: string
        value: string
      }[]
      env_passthrough: string[]
      gpu_devices: string[]
      image: string
      ipc: string
      mounts: {
        read_only: boolean
        source: string
        target: string
      }[]
      name_pattern: string
      network: string
      publication?: string | null
      runtime: string
    } | null
    env: {
      key: string
      value: string
    }[]
    env_inherit: boolean
    source: "running" | "preview"
  } | null
  on_empty: {
    argv: string[]
    container?: {
      argv: string[]
      create_argv: string[]
      env: {
        key: string
        value: string
      }[]
      env_passthrough: string[]
      gpu_devices: string[]
      image: string
      ipc: string
      mounts: {
        read_only: boolean
        source: string
        target: string
      }[]
      name_pattern: string
      network: string
      publication?: string | null
      runtime: string
    } | null
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
  | {
    status: "disabled"
  }
  // already_disabled
  | {
    status: "already_disabled"
  }
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
  | {
    status: "enabled"
  }
  // not_disabled
  | {
    status: "not_disabled"
  }
  // already_enabled
  | {
    status: "already_enabled"
  }
```

#### GET /api/services/{name}/logs

Get service logs (paginated)

| Parameter | In | Required | Description |
| --- | --- | --- | --- |
| `name` | path | yes | Service name |
| `since` | query | no | Earliest timestamp_ms, inclusive |
| `until` | query | no | Latest timestamp_ms, inclusive |
| `run` | query | no | Restrict to one run_id |
| `stream` | query | no | "stdout", "stderr", or "combined" |
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
  follow_after?: {
    run_id: number
    seq: number
  } | null
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
| 503 | service_disabled, start_queue_full, start_failed, insufficient_capacity, service_blocked | `ApiError` |

**Response (202)**:

```typescript
  // already_running
  | {
    status: "already_running"
  }
  // started
  | {
    run_id: number
    status: "started"
  }
  // queue_full
  | {
    status: "queue_full"
  }
  // unavailable
  | {
    error: {
      code: "model_not_found" | "service_not_found" | "service_disabled" | "start_queue_full" | "start_failed" | "insufficient_capacity" | "service_blocked" | "upstream_unavailable" | "proxy_internal" | "not_implemented" | "invalid_request_error" | "invalid_cursor" | "if_match_required" | "hash_mismatch" | "persist_failed" | "query_failed" | "preview_failed" | "invalid_live_request" | "duplicate_subscription_id" | "subscription_limit" | "other"
      message: string
      type: "invalid_request_error" | "server_error" | "other"
    }
    status: "unavailable"
  }
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
| 503 | service_disabled, start_queue_full, start_failed, insufficient_capacity, service_blocked | `ApiError` |

**Response (202)**:

```typescript
  // already_running
  | {
    status: "already_running"
  }
  // started
  | {
    run_id: number
    status: "started"
  }
  // queue_full
  | {
    status: "queue_full"
  }
  // unavailable
  | {
    error: {
      code: "model_not_found" | "service_not_found" | "service_disabled" | "start_queue_full" | "start_failed" | "insufficient_capacity" | "service_blocked" | "upstream_unavailable" | "proxy_internal" | "not_implemented" | "invalid_request_error" | "invalid_cursor" | "if_match_required" | "hash_mismatch" | "persist_failed" | "query_failed" | "preview_failed" | "invalid_live_request" | "duplicate_subscription_id" | "subscription_limit" | "other"
      message: string
      type: "invalid_request_error" | "server_error" | "other"
    }
    status: "unavailable"
  }
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
  | {
    status: "not_running"
  }
  // drained
  | {
    status: "drained"
  }
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
  (insufficient capacity, disabled, etc.), a `503` error is returned with the
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

A `disabled` service says why in `disabled_reason`, which the service list, the service detail, and the `state_changed` event carry exactly when the state is `disabled`:

| Reason | Description |
| --- | --- |
| `launch_failed` | The process failed to launch 3 times (retry cap exceeded). |
| `health_timeout` | The health check didn't pass within the timeout. |
| `oom` | The process was killed by the OOM killer. |
| `user_disabled` | An operator disabled the service via the management API. |
| `auto_restart_loop` | Auto-restart fired more times than its flap cap allowed within the flap window; the service launched but kept degrading in service (as opposed to `launch_failed`, where it never launched), so it is disabled rather than restarted again. |

### Transitions

- `idle → starting`: spawn requested (first request or persistent watcher).
- `starting → running`: health check passed.
- `starting → failed`: launch failed (retry 0–2).
- `starting → disabled`: health check timed out → `health_timeout`, or the process was OOM-killed twice → `oom`.
- `running → draining`: drain requested (stop, evict, or shutdown).
- `running → stopped`: process exited.
- `running → disabled`: the auto-restart flap cap was tripped → `auto_restart_loop`.
- `draining → idle`: drain complete.
- `draining → stopped`: process exited during drain.
- `stopped → starting`: re-spawn requested.
- `failed → failed`: retry backoff elapsed (increments retry_count).
- `failed → disabled`: retry cap (3) exceeded → `launch_failed`.
- `disabled → idle`: user enable.
- `* → disabled`: user disable.

## Events

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

## WebSocket: /api/live

The `/api/live` WebSocket carries read-only, multiplexed subscriptions to the management API's live state, and follows of append-only feeds such as a service's log. The embedded frontend reads all of its live state through it, logs included. Each subscription delivers a full snapshot of its resource straight away, then another whenever something the resource depends on changes and the new body differs from the last one sent. Snapshot bodies are the same DTOs the resource's HTTP `GET` returns, so a client can load over HTTP and switch to the socket, or fall back to HTTP while the socket is down.

The socket carries no commands. Starting, stopping, and editing config stay on their HTTP endpoints; the resulting state changes arrive as snapshots.

### Requests

The client sends JSON text frames, each a `LiveRequest`. The `id` is chosen by the client, is 1 to 128 bytes long, and must not already be active on the connection.

```json
{"type": "subscribe", "id": "services", "resource": {"kind": "services"}}
{"type": "subscribe", "id": "detail:demo", "resource": {"kind": "service", "name": "demo"}}
{"type": "follow", "id": "logs:demo", "feed": {"kind": "logs", "service": "demo", "after": {"run_id": 1761000000000, "seq": 812}}}
{"type": "cancel", "id": "detail:demo"}
```

Subscriptions and follows share the id space and the limit, and both end with `cancel`. Cancelling an unknown id does nothing.

### Resources

| Resource | Body | Pushed when |
| --- | --- | --- |
| `{"kind": "services"}` | `GET /api/services` | any event, a device sample, an in-flight count change, an activity stamp, or a supervisor's state, run id, or pid changing |
| `{"kind": "service", "name": "…"}` | `GET /api/services/{name}` | the same as `services`, with events limited to that service and config reloads |
| `{"kind": "devices"}` | `GET /api/devices` | a device sample, an `allocation_changed` event, or a config reload |
| `{"kind": "events"}` | `GET /api/events/history` | any event |
| `{"kind": "info"}` | `GET /api/info` | a config reload |

"Pushed when" lists what makes the daemon recompute the body; it sends the result only if it differs byte for byte from the previous frame for that subscription. Changes within 100 ms of each other are coalesced into one recompute. A `services` subscription recomputes on every device sample (about every 2 seconds), so fit verdicts and free memory follow the hardware even when no event fires.

### Feeds

A follow has no snapshot. The daemon sends what the feed holds after the client's position, then each new entry as it lands, and the client keeps its own position so it can resume after a reconnect.

`{"kind": "logs", "service": "…"}` follows a service's captured log lines, the same lines `GET /api/services/{name}/logs` returns. Optional fields:

| Field | Meaning |
| --- | --- |
| `after` | a position `{"run_id", "seq"}`; only lines after it are sent. `null` or absent starts from the oldest retained line. |
| `run` | only lines from this run |
| `stream` | only lines from `stdout`, `stderr`, or `combined`; any other value is a 400 |

A position orders lines by run id, then by `seq` within the run, and a later run always has a larger run id. To continue a page of history without a gap, pass the page's `follow_after` as `after`: it is the newest position in the whole log when the page was read, ignoring the page's `run`, `stream`, and `since`, which the follow applies itself. `follow_after` is `null` for a request with `before` or `until`, since rows newer than that page exist, and for a service with no lines.

The daemon first sends the stored lines after `after` as `log_lines` frames, oldest first, then `log_caught_up`, then new lines as they are written. Lines are never repeated and none are skipped between the stored and the new ones. A follower that falls behind the live stream rereads the store and sends `log_caught_up` again, without losing lines. One read sends at most 1000 lines; when more match, the oldest are skipped and a `log_gap` says how many and where. A client that wants them can fetch them over HTTP with `before`.

One case can skip lines: a line from an older run written after a newer run's line has already been sent. This needs a restart and a straggling write from the old process within the same moment, and is accepted.

An unknown service is a 404 that leaves the follow active, as for the `service` resource; it starts once a config reload adds the service.

### Messages

The daemon sends JSON text frames, each a `LiveMessage`.

The first frame on every connection is a hello carrying a build identifier: the daemon's version plus a hash of its embedded frontend. A client that sees it change after reconnecting is talking to an upgraded daemon; the embedded frontend responds by offering to reload.

```json
{"type": "hello", "build": "0.2.0+3f9a1c2b7d4e"}
```

A snapshot carries the subscription id and a body tagged with its resource:

```json
{"type": "snapshot", "id": "services", "data": {"resource": "services", "body": {"services": [], "openai_api_port": 7070}}}
```

A follow's frames carry its id. `log_lines` holds one or more lines, oldest first; `log_gap` reports lines a read skipped, between `after` (`null` from the start of the log) and `before`, the first line sent after the gap; `log_caught_up` marks the end of a read of the store:

```json
{"type": "log_lines", "id": "logs:demo", "lines": [{"timestamp_ms": 1761000000123, "stream": "stderr", "line": "model loaded", "run_id": 1761000000000, "seq": 813}]}
{"type": "log_gap", "id": "logs:demo", "after": null, "before": {"run_id": 1761000000000, "seq": 501}, "skipped": 500}
{"type": "log_caught_up", "id": "logs:demo"}
```

An error carries the subscription id, the HTTP status the same error would have, and the standard error envelope:

```json
{"type": "error", "id": "detail:ghost", "status": 404, "error": {"error": {"code": "service_not_found", "message": "service `ghost` not found", "type": "invalid_request_error"}}}
```

A resource error such as `service_not_found` leaves the subscription or follow active: a config reload can make the name valid, and the next snapshot supersedes the error. Protocol errors use `invalid_live_request` (400; `id` is `null` when the frame could not be parsed), `duplicate_subscription_id` (409), and `subscription_limit` (429).

### Limits and keepalive

A connection holds at most 64 subscriptions and follows together. Inbound frames are capped at 64 KiB. Outbound frames queue in a bounded buffer; if one takes more than 10 seconds to send, the daemon closes the connection with code 1008. The daemon sends a Ping every 30 seconds and answers Pings with Pongs.

### Origin

A handshake with an `Origin` header must match the `Host` header in scheme, host, and port, or it is refused with 403. This stops pages on other sites from reading the daemon's state through a visitor's browser. Clients that send no `Origin`, such as `anankectl` and scripts, are allowed.

### Reconnecting

Subscriptions and follows do not survive a disconnect. A client reconnects with backoff and subscribes again; the first frame for each subscription is a full snapshot, so nothing needs replaying. A follow resumes by sending `follow` again with `after` set to the last line it received.

## Error codes

All error responses — across both the OpenAI-compatible API and the management API — use the same OpenAI-shaped envelope:

```json
{
  "error": {
    "code": "insufficient_capacity",
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

The `POST /api/services/{name}/start` endpoint returns `202 Accepted` even when the supervisor declines to start the service (insufficient capacity, disabled, etc.). The body is `{"status": "unavailable", "error": {...}}` with the same `ApiErrorBody` shape a `503` error would carry. This is a "controlled outcome" of the start request, not a server-side fault.

| Slug | Description |
| --- | --- |
| `model_not_found` | Client referenced a model name that isn't configured. |
| `service_not_found` | Management caller referenced a service that isn't in the config. |
| `service_disabled` | Service is administratively disabled. |
| `start_queue_full` | Supervisor's start queue saturated. |
| `start_failed` | Spawn or health-probe failure during ensure. |
| `insufficient_capacity` | Packer couldn't lay out the model on available devices. |
| `service_blocked` | Queued behind a busy non-elastic peer. |
| `upstream_unavailable` | Upstream child rejected the wire or never replied. |
| `proxy_internal` | Bug inside the proxy itself (URI parse, header build, etc.). |
| `not_implemented` | OpenAI endpoint the daemon hasn't implemented. |
| `invalid_request_error` | Client request was malformed (bad JSON, missing field, etc.). |
| `invalid_cursor` | Log-paging cursor failed to decode. |
| `if_match_required` | Config PUT arrived without an If-Match header. |
| `hash_mismatch` | Config PUT's If-Match didn't match the current hash. |
| `persist_failed` | Config write failed at the IO layer. |
| `query_failed` | The daemon could not read the metrics, restart, or device history you asked for. |
| `preview_failed` | The launch command for this service could not be built. |
| `invalid_live_request` | A `/api/live` frame could not be parsed as a request. |
| `duplicate_subscription_id` | A `/api/live` subscribe reused an id already active on the connection. |
| `subscription_limit` | A `/api/live` connection reached its subscription limit. |
| `other` | Forward-compatibility fallback for unknown codes. |

## Prometheus metrics

The `/metrics` endpoint exposes Prometheus text-format metrics for external scrapers. Prometheus stores history; the JSON `/api/metrics` endpoint serves ananke's own historical data for the dashboard charts.

### Metric families

| Family | Type | Description |
| --- | --- | --- |
| `ananke_requests_total` | counter | Total number of requests proxied. |
| `ananke_tokens_total` | counter | Total tokens processed (labelled by `type`: `prompt` or `completion`). |
| `ananke_auto_restarts_total` | counter | Auto-restart watchdog firings from the daemon's persisted history (labelled by `service` and `trigger`). Aggregate globally with `sum(ananke_auto_restarts_total)`. The backing history is capped per service, so a long-accumulated count can plateau; the rate of increase is the signal. |
| `ananke_inflight_requests` | gauge | Current number of in-flight requests. |
| `ananke_memory_bytes` | gauge | Total memory capacity in bytes (labelled by `device`). |
| `ananke_memory_free_bytes` | gauge | Free memory in bytes (labelled by `device`). |
| `ananke_memory_used_bytes` | gauge | Used memory in bytes (labelled by `device`). |
| `ananke_output_tokens_per_second` | gauge | Average output tokens/sec during decode over the last 5 minutes. |
| `ananke_input_tokens_per_second` | gauge | Average input tokens/sec during prompt processing over the last 5 minutes. |
| `ananke_effective_tokens_per_second` | gauge | Average end-to-end effective generation throughput (completion tokens over wall-clock duration) over the last 5 minutes. Always ≤ the decode rate; the fallback when no input/output split is available (non-streaming with no engine timings). |
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

The frontend reads live state (services, service detail, devices, the event history, and daemon info) through one `/api/live` WebSocket, loading over HTTP first and polling over HTTP every 5 seconds while the socket is down. Logs load their history over HTTP and then follow new lines over the same socket. Commands such as starting a service or saving the config go over HTTP.

