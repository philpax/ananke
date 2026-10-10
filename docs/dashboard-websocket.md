# Dashboard WebSocket

The dashboard opens one same-origin `/api/dashboard` WebSocket. Initial inventory, hardware, model discovery, configuration, event capture, historical metrics, restart records, log backfill, and older log pages all travel over this connection. Service commands, configuration validation and saves, and streaming chat use it too. There are no dashboard HTTP data requests or periodic browser reads.

The protocol DTOs live in `crates/api/src/dashboard/`; the live OpenAPI schema includes `DashboardRequest`, `DashboardQuery`, `DashboardRange`, `DashboardAction`, `DashboardMessage`, and `DashboardEvents`. The frontend generates its TypeScript types and runtime validators from that schema. Response bodies use the existing endpoint DTOs. The native HTTP and legacy event/log WebSocket APIs remain available to the CLI and other clients.

## Reads and subscriptions

Send a connection-local, unique `id` with a `read` operation. `subscribe: false` reads once; `subscribe: true` returns the initial snapshot and then pushes replacements until `cancel` removes the subscription. The same id correlates every pushed result. Registering producer receivers before backfill prevents a concurrent write from being missed.

```json
{"type":"read","id":"inventory","query":{"type":"services"},"subscribe":true}
{"type":"read","id":"metrics","query":{"type":"metrics","range":{"since":0,"until":0,"live_ms":21600000},"service":null,"bucket":"5m"},"subscribe":true}
{"type":"read","id":"logs","query":{"type":"logs","name":"demo","range":{"since":0,"until":0,"live_ms":21600000},"before":null},"subscribe":true}
{"type":"cancel","id":"logs"}
```

A range with `live_ms` follows the server clock. A range with `live_ms: null` uses its explicit Unix millisecond bounds. Changing a preset or applying a custom range replaces the subscription; advancing the browser's display clock sends no request.

Snapshots follow the existing device sampler, lifecycle event bus, successful database writes, and committed log broadcasts. The dashboard has no timer that queries these resources. The existing native device probes and historical sampler continue collecting measurements. Server Ping control frames every 25 seconds keep the connection alive and do not request dashboard data.

Log snapshots contain at most 200 lines and an opaque `next_cursor`. Older pages are one-shot reads using `before`. The browser merges by `(run_id, seq)`, fills gaps after reconnect or a large log batch by following cursors, and preserves scroll position when the reader is away from the tail.

The event bus retains its newest 1,024 lifecycle events even without a connected browser. Reading `events` returns this capture buffer and `since_ms`, the beginning of the available window. Reconnecting browsers receive that buffer again. General events are retained for the current daemon instance; automatic restart records, metrics, and logs also have their existing persistent database history. A daemon restart begins a new general event capture window.

## Commands and chat

Commands call the same handlers as their HTTP equivalents, preserving lifecycle semantics, error status codes, TOML validation, and optimistic concurrency. A config save supplies the last read's `hash`; a stale hash returns status 412 and leaves the draft available for review.

```json
{"type":"action","id":"start","name":"demo","action":"start"}
{"type":"validate","id":"validate","content":"[daemon]\n"}
{"type":"save","id":"save","content":"[daemon]\n","hash":"the-last-read-hash"}
{"type":"chat","id":"chat","body":{"model":"demo","messages":[{"role":"user","content":"Hello"}],"stream_options":{"include_usage":true}}}
{"type":"cancel","id":"chat"}
```

A response is `{"type":"response","id":"…","status":200,"body":…}`. A successful command with no body returns JSON null. Errors retain the existing error payload and status. Chat forces streaming and forwards the existing OpenAI SSE text in `stream` frames, preserving UTF-8 across upstream byte fragments. An `end` frame closes the stream; the browser still requires the upstream `[DONE]` marker before treating generation as successful. Cancel or disconnect drops the stream and releases its in-flight proxy guard.

The browser reconnects with bounded backoff and restores read subscriptions, which backfill from current server state. It never replays a command or chat request after losing an acknowledgement: the operator sees an error and can inspect the current state before retrying. Each connection permits 64 subscriptions and 32 concurrent commands, uses a bounded outgoing channel, and disconnects a stalled writer. Cookie-authenticated browser sockets must have an Origin matching the request Host; clients without an Origin can connect locally.
