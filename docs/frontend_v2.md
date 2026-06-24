# ananke frontend v2 spec

A spec for the complete rebuild of the ananke web dashboard.

Resolves:

- [#5 — Service expandability is not obvious](https://github.com/philpax/ananke/issues/5): service detail moves from inline row expansion to its own dedicated route (`/services/:name`), with clear navigation affordances.
- [#8 — Track token consumption](https://github.com/philpax/ananke/issues/8): the daemon records per-request token usage from proxied OpenAI responses; the frontend surfaces it as charts and per-service breakdowns.
- [#10 — In-frontend chat](https://github.com/philpax/ananke/issues/10): a chat interface mirroring `anankectl chat`, sufficient to test a model from the browser.

## current state assessment

The existing frontend is five components in a single-column page: a devices panel, a services table with inline-expandable detail rows, a logs viewer, and a read-only config dump. It works, but it has grown organically and shows it.

### what works

- **Services table** with state badges, lifecycle actions (start/stop/restart/enable/disable), and inline detail expansion. The state ranking (running floats to top, disabled sinks) is right.
- **Placement visualisation** — per-device VRAM bars showing used-by-others vs. this-service vs. growth headroom, with fit verdicts. This is the most thoughtful part of the current UI.
- **Logs viewer** — REST seed + WebSocket live tail, deduplication by `(run_id, seq)`, auto-follow pinning, time-window presets. The architecture is sound; only the rendering layer is unbounded.
- **Estimate breakdown** — weights / KV / compute-buffer components laid out clearly, with observed-peak comparison.

### what's missing or broken

- **No virtualisation.** The logs viewer renders every line as a DOM node. Leaving the dashboard open on a chatty service for an hour produces tens of thousands of `<div>` elements; the browser seizes.
- **No chat.** `anankectl chat` exists as a full TUI (streaming, reasoning display, token stats, cancellation, `@file` attachments), but the web UI has no way to talk to a model. An operator who wants to test a model has to drop to a terminal.
- **No config editing.** The backend supports `PUT /api/config` with ETag optimistic concurrency and `POST /api/config/validate`, but the frontend only shows raw TOML in a collapsed `<details>`. The planned CodeMirror editor was never wired.
- **No oneshot management.** Full CRUD backend (`POST/GET/DELETE /api/oneshot`) with TTL, port allocation, and log streaming — entirely absent from the UI.
- **No events feed.** `/api/events` WebSocket publishes state changes, allocation shifts, config reloads, and estimator drift. The frontend polls instead, ignoring the real-time channel that already exists.
- **No token metrics.** The daemon proxies every OpenAI request but doesn't record token counts. There's no historical view of throughput, latency, or consumption.
- **No observability.** No charts, no time-series, no error rate, no request rate. An operator can see *now* but not *trend*.
- **Service detail discoverability.** The inline row expansion is not obviously clickable — issue #5 reports a user who couldn't find the detail view. Moving to a dedicated route fixes this.
- **Inconsistent design.** Ad hoc Tailwind classes per component, no shared primitives, no colour system, no typography scale. Dark mode is bolted on with `dark:` variants rather than designed.

## goals

1. **Virtualised everything.** Logs, event feeds, oneshot lists — any list that can grow unbounded uses windowed rendering.
2. **Chat built in.** An operator should be able to talk to any model from the dashboard without leaving the browser. This may require CORS on the OpenAI API endpoints (see [CORS and direct chat](#cors-and-direct-chat)).
3. **Cohesive design system.** A small set of primitives (buttons, badges, cards, bars, tables) shared across every view. Consistent spacing, typography, and colour. Dark mode as a first-class theme, not an afterthought.
4. **Token consumption metrics.** The backend records per-request token usage; the frontend surfaces it as charts and per-service breakdowns.
5. **Surface everything the backend exposes.** Oneshots, events, config editing, metadata, estimator drift — all visible in the UI.
6. **Observability.** Request rate, token throughput, latency percentiles, VRAM history, error rate — charted over time.
7. **Mobile-usable.** The dashboard works on a phone screen — not as rich as desktop, but functional for checking status and basic operations.
8. **Internationalisation-ready.** All user-facing strings go through `react-i18next` with TypeScript-checked translation keys. English is the default and only locale for now, but the infrastructure is there.

## design language

### aesthetic: dense + clean hybrid

Terminal-level data density paired with Linear-style cleanliness. The dashboard should feel like a power tool — crammed with information but not visually noisy. Think: a modern terminal emulator's aesthetic applied to a monitoring dashboard, but with the polish and restraint of a well-designed SaaS product.

Core visual properties:

- **Thin 1px borders** in `zinc-800` (dark) / `zinc-200` (light) define structure. No drop shadows. Depth is conveyed by background contrast (`zinc-950` → `zinc-900` → `zinc-800`), not elevation.
- **No rounded corners larger than `rounded-md`** (6px). Cards, buttons, badges use `rounded-md`; inputs use `rounded-sm`. No `rounded-xl` or `rounded-2xl`.
- **Compact vertical rhythm.** Rows are 28–32px tall. Spacing between sections is `gap-3` (12px), not `gap-8`. The dashboard optimises for information per screen.
- **Monospace for all numeric/data** — PIDs, ports, byte counts, token counts, timestamps, device IDs, VRAM bars. Sans-serif for prose (descriptions, help text, labels). The contrast between the two faces signals "this is data" vs. "this is explanation".
- **Restrained colour.** A neutral zinc base with semantic colours used precisely: emerald for healthy/running, amber for warning/needs-eviction, red for error/failed, blue for the primary action, purple for vision/multimodal, teal for embeddings. No gradients, no glassmorphism, no decorative colour.
- **Dark mode as default.** An operator watching a GPU rack at 2am does not want a white page. Dark is the primary theme; light mode is a toggle.
- **Motion is meaning.** A state transition animates. A new log line slides in. A placement bar grows. But nothing animates just to animate — every transition signals a state change the operator needs to perceive.
- **Keyboard-first.** The dashboard should be navigable by keyboard. Service rows, log views, chat input — all reachable and operable without a mouse.

### colour system

Tailwind CSS 4 theme tokens, centralised in `index.css` via `@theme`:

| Token | Dark value | Light value | Usage |
| --- | --- | --- | --- |
| `bg-base` | `zinc-950` | `white` | page background |
| `bg-surface` | `zinc-900` | `zinc-50` | cards, panels |
| `bg-elevated` | `zinc-800` | `zinc-100` | hover, nested |
| `border-default` | `zinc-800` | `zinc-200` | dividers, borders |
| `text-primary` | `zinc-100` | `zinc-900` | body text |
| `text-secondary` | `zinc-400` | `zinc-500` | labels, hints |
| `text-tertiary` | `zinc-600` | `zinc-400` | timestamps, de-emphasised |
| `accent` | `blue-500` | `blue-600` | primary actions, links |
| `success` | `emerald-500` | `emerald-600` | running, fits |
| `warning` | `amber-500` | `amber-600` | needs-eviction, starting |
| `danger` | `red-500` | `red-600` | failed, error |
| `vision` | `purple-500` | `purple-600` | multimodal badge |
| `embedding` | `teal-500` | `teal-600` | embedding badge |

### typography

- Sans: system stack (`-apple-system, BlinkMacSystemFont, "Segoe UI", …`), Tailwind's default.
- Mono: `"SF Mono", "Cascadia Code", "JetBrains Mono", "Fira Code", monospace`. Applied via a `font-mono` utility or a `data-mono` attribute on data-bearing elements.
- Scale: `text-xs` (12px) for dense data, `text-sm` (14px) for body, `text-base` (16px) for headings, `text-lg` for page titles. No `text-2xl` or larger — the dashboard isn't a landing page.

### shared primitives

A small component library in `src/components/ui/`:

- `Button` — variants: `primary`, `secondary`, `ghost`, `danger`; sizes: `sm`, `md`. Icon + label composition.
- `Badge` — coloured pill, used for states, modalities, fit verdicts.
- `Card` — surface container with optional header.
- `Bar` — horizontal proportion bar (the VRAM/placement bars, generalised).
- `Stat` — label + value + optional unit, monospace value.
- `Table` — sticky-header, sortable, with row expansion support.
- `VirtualList` — windowed list renderer (see [virtualisation](#virtualisation)).
- `Tooltip` — hover/focus tooltip for dense data.
- `EmptyState` — icon + message for empty lists.
- `Spinner` — small loading indicator.

### internationalisation

`react-i18next` from the start, with full TypeScript integration. Translation files live in `src/locales/en/translation.json`. The JSON shape is imported as a typed resource so that `t('services.table.actions.start')` is verified at compile time — a misspelled key is a compile error, and editors get autocomplete on translation keys.

All user-facing strings go through `t()`. This includes button labels, state names, column headers, empty-state messages, error messages, and chart axis labels. Data values (service names, PIDs, byte counts) are not translated.

English is the default and only locale for v1. Adding a second locale later is a matter of adding a `src/locales/<lang>/translation.json` file — no code changes needed.

### mobile layout

The dashboard is responsive. On mobile (`< sm`, 640px):

- The sidebar collapses to a bottom tab bar with icon-only navigation.
- Tables become card lists — each row renders as a stacked card with the key fields.
- Charts simplify to sparklines or single-value stat cards.
- The chat interface adapts to a narrow column: message boxes fill the width, the input area docks to the bottom, and the model picker collapses to a dropdown.
- The config editor remains usable (CodeMirror adapts to width), though it's a secondary use case on mobile.

Breakpoints: `sm` (640px) and `md` (768px). Below `sm` is mobile; `sm` to `md` is tablet (sidebar visible but collapsed); above `md` is desktop (full layout).

## architecture

### routing

The current single-page layout doesn't scale to the feature set. A client-side router (React Router or a minimal hash-based router) with these top-level routes:

| Route | View |
| --- | --- |
| `/` | dashboard overview |
| `/services` | services table (full-width, no inline expansion) |
| `/services/:name` | service detail page |
| `/devices` | device detail page |
| `/chat` | chat interface |
| `/oneshots` | oneshot job management |
| `/config` | config editor |
| `/events` | event feed |
| `/metrics` | observability charts |

A persistent sidebar (collapsible) navigates between these. The sidebar shows the daemon endpoint and a health dot (green if any service is running, amber if all idle, red if any failed). On mobile, the sidebar becomes a bottom tab bar.

### state management

- **Server state:** TanStack Query, leaned into hard. Specific patterns:
  - **Infinite queries** for log pagination — the existing manual multi-page fetch loop is replaced by `useInfiniteQuery` with cursor-based pagination, so the frontend can scroll back through history without bespoke fetch logic.
  - **Optimistic updates** for lifecycle actions — when the user clicks "start", the service state is updated in the cache immediately, with rollback on error. The events WebSocket confirms the real state shortly after.
  - **WebSocket-driven invalidation** — see [real-time updates](#real-time-updates).
  - **Per-data-type `staleTime`/`gcTime`** — services and devices: `0` (always refetch on focus/mount, events WebSocket keeps them fresh anyway). Config: `Infinity` (only refetch on explicit action). Logs seed: `5m`. Metrics: `30s`.
- **Local UI state:** React state + context for theme, sidebar collapse, active filters.
- **Chat state:** Local React state per chat session (messages, streaming state, stats). No persistence needed across page reloads for v1; the conversation lives in memory.

### real-time updates

The backend already exposes `/api/events` as a system-wide WebSocket. The current frontend ignores it and polls every 2s. The new frontend should:

1. Open a single `/api/events` WebSocket on app mount.
2. On `state_changed`, `allocation_changed`, `config_reloaded`, `estimator_drift` events, invalidate the relevant TanStack Query keys. This gives near-instant updates without polling.
3. Keep polling as a fallback (at a slower cadence, e.g. 10s) for WebSocket disconnection and for data the events channel doesn't cover (device free bytes, observed peaks).
4. The events feed page subscribes to the same WebSocket but renders the raw event stream.

### virtualisation

A `VirtualList` component that renders only the visible window of a list, using `ResizeObserver` to measure row heights and `IntersectionObserver` (or manual scroll math) to decide what to render. Required for:

- **Logs** — the primary case. A chatty service can emit thousands of lines per minute. The virtualiser renders ~50 visible rows, recycles DOM nodes as the user scrolls, and caps the in-memory buffer at a configurable limit (default 10k lines) with older lines evicted.
- **Event feed** — same pattern, same cap.
- **Large service lists** — unlikely to matter for v1, but the primitive is there.

Implementation: use `@tanstack/react-virtual` (already in the TanStack ecosystem, no new dependency family). It handles variable-height rows, smooth scrolling, and works with the existing React 19 + Tailwind setup.

### CORS and direct chat

For the chat feature, the frontend needs to POST to `/v1/chat/completions` on the OpenAI API port (default 7070), which is a different port than the management API/frontend (default 7071). This is a cross-origin request from the browser.

Options:

1. **CORS on the OpenAI API.** Add `Access-Control-Allow-Origin` headers to the OpenAI API handlers. The management API is already same-origin with the frontend (served from the same daemon), so no CORS needed there. The OpenAI API is the only cross-origin surface. This is the simplest approach — a `tower_http::cors` layer on the OpenAI router, configurable in `[openai_api]` with `allow_cors = true` (default `true` when the management API is non-loopback, or always — ananke is unauthenticated anyway).

2. **Proxy through the management API.** Add a `POST /api/chat/completions` management endpoint that forwards to the OpenAI port internally. Avoids CORS but adds a proxy layer for no real benefit, and the streaming semantics are more complex through axum's management router than through the existing hyper proxy.

3. **Same-port serving.** Bind the OpenAI API and management API on the same port, routing by path prefix. This is a bigger change to the daemon's listener architecture and not worth it for v1.

**Recommendation: option 1.** Add a CORS layer to the OpenAI API router. The daemon is already unauthenticated and designed for trusted-network deployment, so CORS is not a security boundary — it's a browser-compatibility fix. The `allow_cors` flag in `[openai_api]` makes it opt-in for operators who want it off.

Alternatively, the frontend can discover the OpenAI port from `/api/services` (which already returns `openai_api_port`) and construct the URL. The chat hook then posts to `http://{management_host}:{openai_api_port}/v1/chat/completions` with `stream: true` and parses the SSE response client-side — the same thing `anankectl chat` does, but in the browser.

## feature spec

### 1. dashboard overview (`/`)

The landing page. A single screen that answers "what's happening right now?"

**Layout:** a responsive grid.

- **Header strip:** daemon endpoint, OpenAI port, health dot, active/incoming request count.
- **Device summary:** one card per GPU + CPU, each with a VRAM/RAM utilisation bar and a reservation breakdown. Click-through to `/devices`.
- **Service grid:** compact cards (name, state badge, port, priority) for every service, sorted by state rank. Click-through to `/services/:name`.
- **Recent events:** last 10 events from the WebSocket, one-line summaries.
- **Quick stats:** total services, running count, total VRAM in use, total tokens served today (if metrics are available).

### 2. services view (`/services`, `/services/:name`)

The services table, rebuilt. This resolves [#5](https://github.com/philpax/ananke/issues/5) — service detail is no longer hidden behind an undiscoverable inline expansion.

**Table view (`/services`):**

- Full-width table, sticky header, sortable by any column.
- Columns: name (with badges), state, lifecycle, priority, port, PID, modality, metadata indicators, actions.
- Row click navigates to `/services/:name` (no inline expansion — the detail view gets its own page with a clear, clickable name link and a chevron affordance).
- Action buttons inline (start/stop/restart/enable/disable), with pending state.
- Filter bar: by state, by modality, by lifecycle, text search on name.

**Detail view (`/services/:name`):**

- **Header:** name, state badge, template, port (with link to proxy), private port, lifecycle, priority, run ID, PID.
- **Model section:** architecture, name, file, parameters, on-disk size, layers, trained context, shards, license, mmproj badge. As-is, but in a cleaner layout.
- **VRAM estimate section:** weights / KV / compute-buffer breakdown, with configured context, observed peak, and rolling correction factor. The rolling mean should be explained ("estimator has seen N samples; current correction is 1.05×, meaning estimates are 5% low").
- **Placement section:** current allocation + placement preview, with per-device bars. As-is, but generalised into the shared `Bar` primitive.
- **Launch command:** collapsible, with copy button. As-is.
- **Logs:** the virtualised logs viewer (see [logs viewer](#logs-viewer)).
- **Chat:** a "chat with this model" button that navigates to `/chat?model=:name`.
- **Metrics:** per-service token consumption, request rate, latency (if metrics are available — see [token metrics backend](#token-metrics-backend)).

### 3. logs viewer

The existing `LogsView` architecture (REST seed + WebSocket live tail, time-window presets, deduplication) is sound. The rendering layer is replaced with a virtualised list.

**Requirements:**

- Virtualised rendering via `@tanstack/react-virtual`. Only visible rows are in the DOM.
- In-memory buffer cap (default 10,000 lines). When the cap is reached, older lines are evicted. A "lines evicted" counter is shown in the status strip.
- Row height is variable (long lines wrap), measured via `ResizeObserver`.
- Auto-follow pinning, jump-to-latest button — as-is.
- Stream filter (stdout / stderr / both) — add a toggle.
- Run filter — if the service has multiple runs, a dropdown to filter by run ID.
- Search/filter within the visible buffer — a text input that highlights matching lines and dims non-matches.
- Per-line metadata: timestamp, stream, run ID (shown in a tooltip or expandable detail).

### 4. chat interface (`/chat`)

A web equivalent of `anankectl chat`. The operator picks a model (or arrives via `?model=`), enters a system prompt, and chats. This resolves [#10](https://github.com/philpax/ananke/issues/10).

**Features (mirroring the TUI):**

- **Model picker:** dropdown of OpenAI-accessible services, sorted with running first. Single-candidate auto-selects.
- **System prompt:** configurable, persisted to `localStorage` per model.
- **Streaming:** SSE parsing, content + reasoning display. Reasoning shown in a dimmed block above the content (as in the TUI).
- **Token stats:** per-turn display of decoded tokens, decode rate (tok/s), time-to-first-token, prompt-processing rate. Live during streaming, frozen after.
- **Cancellation:** a stop button cancels the in-flight stream.
- **Multi-line input:** textarea that grows with content. `Enter` sends, `Shift+Enter` inserts a newline (or `Ctrl+Enter` sends — pick one and document it).
- **Conversation history:** the full message list is sent with each turn. A "clear" button resets.
- **Error display:** HTTP errors, model-not-found, service-unavailable, insufficient-VRAM — all rendered as inline error messages.
- **Markdown rendering:** assistant responses are rendered as markdown (headings, code blocks, lists, tables) via `react-markdown` + `remark-gfm`. Rendered HTML is sanitised via `DOMPurify`.
- **File attachments:** drag-and-drop or file picker in the input area. Supports:
  - **Text files** (`.txt`, `.md`, `.rs`, `.py`, `.ts`, `.json`, etc.) — read via `FileReader.readAsText`, appended to the message content as `\n{filename}:\n{contents}` (same pattern as the TUI's `@file` references). A collapsible preview shows the file contents inline.
  - **Image files** (`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`) — only available when the selected model has `mmproj` (vision support). Read via `FileReader.readAsDataURL` and sent as an `image_url` content part in the OpenAI message format: `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`. A thumbnail preview is shown in the input area.
  - Attachment previews show file name, size, and a remove button. Multiple attachments per message are supported.

**Out of scope for v1:**

- Multiple concurrent conversations/tabs.
- Conversation persistence across page reloads.
- File attachments that read from the daemon's filesystem (the browser can't do this — only user-uploaded files via the file picker / drag-and-drop).

### 5. oneshot management (`/oneshots`)

Full CRUD for oneshot jobs, which currently only `anankectl oneshot` can do.

The existing oneshot backend works but has a significant limitation: the health check is hardcoded to `GET /v1/models` with a 3-minute timeout, which is wrong for arbitrary command services (e.g. ComfyUI uses `/system_stats`). A oneshot that doesn't serve `/v1/models` will sit in `starting` until the health timeout, then fail. This is fixed as part of the backend changes (see [backend changes](#backend-changes)): an optional `health` field on `OneshotRequest` lets the caller specify the health check path and timeout, with a default of no health check (the service is treated as immediately running).

**List view:**

- Table of oneshot jobs: ID, name, state, port, submitted, TTL remaining, actions (view logs, kill).
- Live updates via the events WebSocket (if oneshot state changes are published — they may not be today; if not, poll at 5s).
- Filter by state (running / ended / all).

**Submit form:**

- Template selection (llama-cpp or command). Note: llama-cpp oneshots are currently rejected by the backend (`OneshotRequest` has no `model` field); the form should grey out the llama-cpp option with a tooltip explaining why.
- For command: command argv, working directory.
- Allocation: static (vram_gb) or dynamic (min/max).
- Devices: placement policy.
- Priority, TTL, name, port (optional).
- Health check: optional HTTP path + timeout (uses the new `health` field).
- Submit creates the oneshot and navigates to its log view.

**Detail view:**

- Status, port, timing, exit code.
- Logs viewer (same virtualised component, pointed at the oneshot's log stream).

### 6. config editor (`/config`)

A real TOML editor with validation, replacing the read-only dump.

**Features:**

- **CodeMirror 6** editor with TOML syntax highlighting and line numbers.
- **Validation:** on edit (debounced), POST to `/api/config/validate`. Inline error markers at the line/column the backend reports. Validation errors shown in a panel below the editor.
- **Save:** `PUT /api/config` with `If-Match: "<hash>"` header. On `412` (hash mismatch), show a "config has changed on disk, reload?" dialog with the server's current hash.
- **Diff view:** before saving, show a diff of the current content vs. the last-loaded content. Helps catch accidental edits.
- **Reload:** fetch the current config from the daemon (discarding local changes).
- **Read-only mode:** when the config is managed externally (e.g. NixOS), the operator may want to view but not edit. A toggle.

### 7. event feed (`/events`)

A live, virtualised feed of system events.

**Features:**

- Subscribes to `/api/events` WebSocket.
- Each event rendered as a one-line summary: timestamp, event type, service name (if applicable), key fields.
- Filter bar: by event type (state_changed, allocation_changed, config_reloaded, estimator_drift), by service.
- Virtualised list (same `VirtualList` primitive as logs).
- In-memory buffer cap (default 1,000 events).
- Pause/resume button — pauses the feed without dropping the WebSocket connection; events buffer server-side (the broadcast channel).

### 8. metrics / observability (`/metrics`)

Charts and time-series. This requires backend support for recording and serving historical metrics (see [token metrics backend](#token-metrics-backend)). This resolves [#8](https://github.com/philpax/ananke/issues/8).

**Charts:**

- **Request rate:** requests per minute over the last hour / 6h / 24h, per service or aggregated.
- **Token throughput:** tokens/s (prompt + completion) over time.
- **Latency:** TTFT (time-to-first-token) and total request duration, as p50 / p90 / p99 lines.
- **VRAM utilisation:** per-device VRAM usage over time (sampled, not per-request).
- **Error rate:** 4xx / 5xx responses per minute.
- **Token consumption:** cumulative tokens served, per model, over a time range. A stacked area chart.

**Implementation:**

- Charts via `uPlot` — a thin wrapper around canvas, renders 100k+ points without breaking a sweat, and has React bindings via `uplot-react`. Chosen over `recharts` (heavy, struggles with large datasets) for the "not too fancy" goal and the need for fast time-series rendering.
- Data from the JSON `GET /api/metrics` endpoint (see below).

### 9. devices view (`/devices`)

An expanded version of the current devices panel, as its own page.

- One card per GPU: name, ID, total/free/used VRAM, utilisation bar, reservation breakdown (service + bytes + elastic flag).
- CPU card: same, with RSS instead of VRAM.
- Per-device historical VRAM chart (if metrics are available).
- Per-device reservation list, with click-through to the reserving service.

## token metrics backend

The daemon currently proxies OpenAI requests without recording anything about them. To support token consumption metrics (issue #8), the backend needs to:

### 1. record per-request metrics

When proxying `/v1/chat/completions`, `/v1/completions`, and `/v1/embeddings`, intercept the response and extract:

- `service_id`, `run_id` (from the supervisor handle)
- `timestamp` (request received)
- `endpoint` (`/v1/chat/completions`, etc.)
- `model` (the ananke service name, not the upstream model)
- `prompt_tokens`, `completion_tokens` (from the `usage` field in the response body)
- `duration_ms` (request received to response complete)
- `ttft_ms` (time to first token, for streaming responses)
- `status` (success / error status code)

For streaming responses, the `usage` field arrives in the final SSE chunk (when `stream_options.include_usage` is set). For non-streaming, it's in the response body. The proxy already reads the request body; it needs to also inspect the response body (or the final SSE chunk) to extract usage.

For streaming, TTFT is measurable: the time from request dispatch to the first `data:` frame with content.

### 2. new database table

A new migration (`0002_request_metrics.sql`):

```sql
CREATE TABLE request_metrics (
  metric_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  service_id  INTEGER NOT NULL,
  run_id      INTEGER,
  timestamp_ms INTEGER NOT NULL,
  endpoint    TEXT NOT NULL,
  model       TEXT NOT NULL,
  prompt_tokens   INTEGER,
  completion_tokens INTEGER,
  duration_ms INTEGER,
  ttft_ms     INTEGER,
  status_code INTEGER NOT NULL,
  FOREIGN KEY (service_id) REFERENCES services(service_id)
);
CREATE INDEX request_metrics_ts ON request_metrics(service_id, timestamp_ms);
CREATE INDEX request_metrics_run ON request_metrics(run_id);
```

Retention: prune to a configurable window (default 7 days) via the existing retention loop. The table is append-only and indexed by `(service_id, timestamp_ms)`, so range queries are cheap.

### 3. JSON metrics endpoint

`GET /api/metrics` with query parameters:

- `service` (optional): filter to one service.
- `since`, `until` (ms timestamps): time range.
- `bucket` (e.g. `1m`, `5m`, `1h`): aggregate into time buckets.
- `fields` (optional): which metrics to return (default all).

Response: time-series data suitable for charting — an array of `{ timestamp, bucket_start, request_count, prompt_tokens, completion_tokens, avg_ttft_ms, p50_ttft_ms, p90_ttft_ms, error_count }`.

The daemon computes aggregations in SQLite (or in-memory for small ranges) and returns pre-bucketed data. The frontend doesn't do aggregation; it just renders.

### 4. Prometheus metrics endpoint

`GET /metrics` in Prometheus text format, for external consumption (Prometheus → Grafana, etc.). This is a separate endpoint from the JSON API — Prometheus expects current-state gauges/counters/histograms, not historical time-series. Prometheus scrapes this every 15s and stores history in its own TSDB; the JSON endpoint serves ananke's own historical data for the dashboard's charts.

Exposed metrics:

- `ananke_requests_total{service,endpoint,status}` — counter
- `ananke_tokens_total{service,type}` — counter (type: `prompt` or `completion`)
- `ananke_inflight_requests{service}` — gauge
- `ananke_vram_bytes{device}` — gauge
- `ananke_vram_free_bytes{device}` — gauge
- `ananke_service_state{service}` — gauge (numeric state code)
- `ananke_request_duration_seconds{service}` — histogram (for latency percentiles)
- `ananke_ttft_seconds{service}` — histogram

Implementation: use the [`prometheus` crate](https://docs.rs/prometheus). It handles metric registration, label escaping, and histogram bucket computation. The handler reads current state from the daemon's in-memory tables (inflight, observation, allocation, snapshot) and the metrics DB (for counters), then formats as Prometheus text. The format itself is simple — `# HELP` / `# TYPE` headers followed by `metric{labels} value` lines — but the library handles edge cases and gives us histograms for free.

### 5. VRAM sampling

Separately from per-request metrics, the daemon should periodically sample per-device VRAM usage and store it for charting. A new `device_samples` table:

```sql
CREATE TABLE device_samples (
  sample_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  device      TEXT NOT NULL,
  timestamp_ms INTEGER NOT NULL,
  total_bytes INTEGER NOT NULL,
  free_bytes  INTEGER NOT NULL,
  used_bytes  INTEGER NOT NULL
);
CREATE INDEX device_samples_ts ON device_samples(device, timestamp_ms);
```

Sampled every 10s (configurable), retained for 24h (configurable). The snapshotter already reads NVML; this just persists the readings.

## additional features to discuss

These are not committed for v1, but worth considering:

### per-service request log

A list of recent requests to a service, with timestamp, token counts, duration, and status. Useful for diagnosing "why did this request take 30s?" without digging through logs. Low effort once the metrics table exists — just a different view of the same data.

### model comparison view

Side-by-side comparison of two models' metrics: token throughput, latency, VRAM. Useful when evaluating whether to switch from one quantisation to another.

### per-model pricing

Issue #8 mentions an optional extension: operators register per-model prices in the ananke config (e.g. `[service.pricing] prompt_per_1m = 0.15, completion_per_1m = 0.60`), and the dashboard displays cost alongside token consumption. The config is already the source of truth for per-service settings, so this fits naturally. Low effort once the metrics table exists — multiply token counts by the configured rate and display the result.

### alerting / thresholds

Configurable alerts: "notify if a service has been in `starting` for >60s", "notify if VRAM utilisation >95%", "notify if error rate >5%". Browser notifications or a simple toast. The events WebSocket makes this low-effort — the frontend already sees every state change.

### multi-daemon view

If an operator runs multiple ananke instances (e.g. across multiple hosts), a federated view that aggregates metrics from several daemons. This is a v2+ feature; v1 is single-daemon only.

### dark/light/system theme toggle

Three-way toggle: dark (default), light, system (follows `prefers-color-scheme`). Stored in `localStorage`.

## technical plan

### dependencies to add

| Package | Purpose |
| --- | --- |
| `@tanstack/react-virtual` | virtualised list rendering for logs, events |
| `react-router-dom` | client-side routing |
| `uplot` + `uplot-react` | time-series charts for metrics |
| `@codemirror/lang-toml` | TOML syntax support for the config editor |
| `@codemirror/view` + `@codemirror/state` | CodeMirror 6 core |
| `react-markdown` + `remark-gfm` | markdown rendering for chat responses |
| `dompurify` | sanitise rendered markdown HTML |
| `react-i18next` + `i18next` | internationalisation with TypeScript integration |
| `i18next-browser-languagedetector` | automatic locale detection from browser settings |

All are established, maintained packages with no transitive dependency conflicts.

### project structure

```
frontend/src/
  api/
    client.ts          # fetch helpers (existing, keep)
    hooks.ts           # TanStack Query hooks (existing, extend)
    types.ts           # generated OpenAPI types (existing, regenerate)
    events.ts          # WebSocket event subscription hook (new)
  components/
    ui/                 # shared primitives
      Button.tsx
      Badge.tsx
      Card.tsx
      Bar.tsx
      Stat.tsx
      Table.tsx
      VirtualList.tsx
      Tooltip.tsx
      EmptyState.tsx
      Spinner.tsx
    layout/
      Sidebar.tsx
      Header.tsx
      AppShell.tsx
      MobileNav.tsx     # bottom tab bar for mobile
    services/
      ServicesTable.tsx
      ServiceDetail.tsx
      ModelSection.tsx
      EstimateSection.tsx
      PlacementSection.tsx
      LaunchCommand.tsx
    logs/
      LogsViewer.tsx
      LogRow.tsx
      LogsToolbar.tsx
    chat/
      ChatView.tsx
      MessageList.tsx
      MessageInput.tsx
      ModelPicker.tsx
      TokenStats.tsx
      AttachmentPreview.tsx
    devices/
      DeviceCard.tsx
      DeviceGrid.tsx
    oneshot/
      OneshotList.tsx
      OneshotSubmit.tsx
      OneshotDetail.tsx
    config/
      ConfigEditor.tsx
      ValidationPanel.tsx
    events/
      EventFeed.tsx
      EventRow.tsx
    metrics/
      MetricsPage.tsx
      RequestRateChart.tsx
      TokenThroughputChart.tsx
      LatencyChart.tsx
      VramChart.tsx
  hooks/
    useTheme.ts
    useEvents.ts
    useChat.ts
  locales/
    en/
      translation.json
  routes/
    index.tsx          # route definitions
  util.ts              # formatting helpers (existing, extend)
  i18n.ts             # i18next initialization
  App.tsx              # root: AppShell + Router + I18nProvider
  main.tsx
  index.css            # @theme tokens
```

### backend changes

1. **CORS on OpenAI API** — add `tower_http::cors` layer, gated by `[openai_api] allow_cors = true`.
2. **Request metrics recording** — intercept proxied responses, extract `usage`, write to `request_metrics` table.
3. **VRAM sampling** — periodic NVML reads written to `device_samples` table.
4. **`GET /api/metrics` endpoint** — JSON aggregated time-series query.
5. **`GET /metrics` endpoint** — Prometheus text format for external scrapers (Prometheus → Grafana).
6. **`GET /api/devices/samples` endpoint** — VRAM history for the devices view.
7. **Events for oneshot state changes** — publish `state_changed` events when a oneshot transitions (currently only configured services publish events).
8. **In-flight request count in API** — surface `tracking::InflightTable` counts in `/api/services` or a new endpoint, so the dashboard can show active request counts.
9. **Oneshot health check fix** — add an optional `health` field to `OneshotRequest` (http path + timeout), defaulting to no health check (service treated as immediately running). The current hardcoded `GET /v1/models` health check breaks for non-OpenAI command services.

### migration plan

The frontend is rebuilt from scratch, but the daemon keeps running. The plan:

1. **Backend changes first.** CORS, metrics recording, metrics endpoints (JSON + Prometheus), VRAM sampling, oneshot events, oneshot health check fix. These are additive — the old frontend doesn't use them, the new one will.
2. **New frontend in parallel.** Build the new frontend in `frontend/src/` (replacing the existing files). The old frontend stays in place until the new one is ready.
3. **Cutover.** Swap the embedded frontend bundle. The old files are deleted; the new ones take over.

### testing

Per AGENTS.md, frontend tests use Vitest + React Testing Library. The first tests to write:

- `VirtualList` — renders only visible rows, handles variable heights, scrolls correctly.
- `useChat` hook — SSE parsing, cancellation, error handling, token stats, file attachment handling.
- `LogsViewer` — buffer cap eviction, auto-follow, stream filter.
- `ConfigEditor` — validation feedback, save/reload flow, hash mismatch handling.
- `i18n` — translation key resolution, missing key behaviour, locale switching.
- Formatting utilities — byte formatting, parameter counts, duration formatting.

## out of scope for v1

- Authentication / authorisation. ananke is unauthenticated; the frontend is too.
- Multi-daemon federation.
- Chat conversation persistence.
- Plugin / extension system for custom views.
- Additional locales beyond English (the infrastructure is there; only `en` ships).
