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
