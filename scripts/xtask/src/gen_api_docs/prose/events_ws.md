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
