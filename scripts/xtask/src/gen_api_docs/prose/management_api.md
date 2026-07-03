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
