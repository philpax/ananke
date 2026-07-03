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
