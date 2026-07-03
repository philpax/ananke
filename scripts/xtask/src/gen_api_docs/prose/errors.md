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
