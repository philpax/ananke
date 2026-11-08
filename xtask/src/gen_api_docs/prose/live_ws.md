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
