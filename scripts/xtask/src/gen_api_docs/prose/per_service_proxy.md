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
