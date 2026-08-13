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

### Config validation diagnostics

`POST /api/config/validate` parses, merges, and semantically validates the submitted TOML without reading model files. It returns HTTP 200 with `valid` and an ordered `errors` array. An invalid `PUT /api/config` returns the same response shape with HTTP 422. GGUF model preflight remains a startup/reload operational check and is not projected into this response.

Each validation error contains:

- `code`: a stable snake-case identifier for the actionable diagnostic category. Message wording may change without changing the code.
- `message`: the centrally rendered human-readable diagnostic.
- `path`: an optional configuration field path.
- `service` and `service_index`: the owning `[[service]]` block, when the diagnostic belongs to one. `service_index` is the zero-based position of the block in the original TOML and is present even when the name is missing or duplicated, so a diagnostic can always be attributed back to the block that produced it.
- `context`: a tagged structured payload with service, field, value, count, index, placeholder, merge, or parser data.
- `location`: an optional source span with zero-based byte offsets and one-based line and column. Semantic diagnostics do not claim a source location, so clients must not render `0:0` as a fallback.

Diagnostics are ordered as parse, merge, global validation, and then services in original TOML source order. Within each service, the existing validation section order is preserved.

### Error responses

All management API errors use the standard OpenAI-shaped error envelope with a typed `code` slug, human-readable `message`, and taxonomy `type`. See the [Error codes](#error-codes) section for the full catalogue.
