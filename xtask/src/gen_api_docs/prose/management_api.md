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
