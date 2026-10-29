# Changelog

Notable changes to ananke. Entries are grouped by release; the top section collects what has landed since the last one.

The audience is an operator deciding whether to upgrade and what they will have to change. Behaviour changes and anything that touches an existing config belong here; refactors and internal tidying do not.

## Unreleased

### Breaking

- Placeholders are now written `${name}`, not `{name}`. The old form could not be distinguished from an argument's own braces without guessing, which made JSON arguments fail to launch and silently rewrote Jinja templates. Only `${` and `$$` are special now: a bare `{` never is, so JSON, Jinja, and format strings pass through exactly as written and need no escaping. `$$` is a literal `$`, which is how a literal `${port}` is written.

  Rename every placeholder in `command`, `shutdown_command`, `launcher`, `env` values, and `container.gpu_device`. A config still using `{name}` does not error, because the text is now literal, so confirm a service's rendered command still contains the expected value, through `GET /api/services/{name}/command` or the launch-command panel.

  An unknown name, or a `${` with no closing brace, is a config error. A known name written bare, such as `{port}`, is a warning naming the service and the entry, since it cannot be an error: a JSON or format-string argument may hold `{model}` on purpose. Write `$${model}` to say that and silence the warning.
- Oneshots are removed: the `/api/oneshots` endpoints, `anankectl oneshot`, and the web UI's Oneshots view. A service launched outside ananke already has its usage subtracted from the pool, so they added little. Their database table is dropped on upgrade, and `lifecycle = "oneshot"` is rejected as an unknown lifecycle.
- The per-service log socket, `/api/services/{name}/logs/stream`, is removed. Follow logs on the `/api/live` WebSocket instead, with a `follow` request whose feed is `{"kind": "logs", "service": "…"}`: pass a logs page's new `follow_after` as `after` and the follow continues exactly where the page ended, then sends new lines as they are written. A follower that falls behind rereads the store rather than dropping lines, so the old `overflow` frame has no replacement; `log_gap` reports only lines skipped when more than 1000 are pending at once.
- The `/api/events` WebSocket is removed, along with its `overflow` event and `?service=` filter. Read events through the `/api/live` WebSocket instead: subscribe to `{"kind": "events"}` for the newest 1000 events, pushed again as each one arrives, or fetch the same body from `GET /api/events/history`. Every other live resource updates on the same events.
- A service's `lifecycle` is reported as `on_demand` on the management API (`GET /api/services`, `GET /api/services/{name}`, and the live `services` and `service` resources), the way a config file spells it, instead of `ondemand`. A client that compares against `"ondemand"` needs updating; `persistent` is unchanged.
- `GET /api/services/{name}` no longer includes `recent_logs`. Building it read the service's whole retained log on every request and on every live update to the service, and the lines it carried went stale between log writes. Fetch `GET /api/services/{name}/logs?limit=N` for the newest lines instead.

### Added

- Container workloads. Either template can run its workload in a Docker or Podman container through a `[service.container]` block, without a wrapper script. ananke owns the whole lifecycle: it creates, starts, follows logs, reads the authoritative exit status, signals, removes, and reconciles leftovers after a crash. The template still decides what argv is produced; the block decides where it runs. It covers host and bridge networking with a loopback-only publication, bind mounts with automatic path translation for llama.cpp's model fields, CDI GPU injection scoped to the devices the allocator picked, explicit environment plus a name-only passthrough allowlist, and IPC mode. See [Container Workloads](docs/configuration.md#container-workloads).
- A `combined` log stream for container output. Neither runtime's CLI documents a framing that preserves stdout from stderr across the process boundary, so container output is labelled `combined` rather than being passed off as either. Native processes keep their split streams. Available through the API, the WebSocket, `anankectl logs --stream`, and the log viewer.
- Container detail in the API and UI: runtime, image, network mode, and live container identity, alongside the exact `create` argv a service would launch with. Passthrough variables appear by name only.
- An offline config validator: `cargo run -p ananke-supervise --example validate-config -- <file>` parses, validates, and renders the create argv each containerized service would launch with. `anankectl server-config validate` needs a running daemon; this does not.
- Self-healing auto-restart. Watchdogs detect and restart a degraded running service without operator action, configured under `[service.auto_restart]` with fleet-wide defaults under `[defaults.auto_restart]`. Five triggers cover distinct failure modes:
  - Error rate, on by default.
  - Time-to-first-token stall.
  - Generation stall.
  - Speculative-decoding collapse.
  - Periodic, time-based restart.
- Anti-flap guards cap auto-restarts within a sliding window and require a minimum uptime before a fresh run can trigger. A service past its restart budget is disabled with reason `auto_restart_loop` until it is re-enabled.
- Restart history is persisted and exposed:
  - `GET /api/restarts`, paginated by service and time range.
  - The `ananke_auto_restarts_total` counter, labelled by service and trigger.
  - `auto_restarted` events carrying the trigger and the observed detail.
  - A chart in the web UI.
- ik_llama.cpp runtime selection. A `llama-cpp` service selects between mainline llama.cpp (default) and the ik_llama.cpp fork through the `runtime` field, a tagged table (`{ kind = "ik-llama", mla = 1, dsa = true, attn_max_batch = 512, runtime_repack = false }`). ananke emits the matching `-mla`, `-dsa -fidx`, `-amb`, and `-rtr` flags and the ik `--spec-type` dialect, reserves whole GPUs for an ik service under `placement = "fit"`, and reads ik_llama.cpp's quantised tensor dtypes.
- Heterogeneous GPU tensor-split. `devices.tensor_split_weights` balances layer-parallel work across unequal GPUs by weight, scaling each GPU's share and the emitted `--tensor-split` ratio.
- Automatic expert offload for MoE models. `expert_offload = "auto"` offloads whole expert layers to host RAM through llama-server's `--n-cpu-moe`, keeping attention and the KV cache on the GPU, and balances the offload across GPUs with a room-based tensor split.
- NUMA placement. The `numa` field (`distribute`, `interleave`, or `local`) passes `--numa` to llama-server for explicit thread and memory placement.
- `env_inherit` (default on) copies the daemon's environment into a spawned host child; set it off to pass only the service's explicit `env` plus `CUDA_VISIBLE_DEVICES`. A container's environment is always explicit, regardless of this flag.
- `cache_ram_mb` caps the host RAM llama-server may use for its prompt cache. The value is passed through explicitly so the packer's host reservation matches the runtime's cap; `0` disables the cache.
- The VRAM estimator adds support for the laguna, glm-dsa, deepseek4, lfm2, and gemma4 architectures, and now models host RAM, quantised KV cache, sliding-window and recurrent (hybrid) KV, tensor-split compute buffers, and speculative-decoding overhead.
- Token throughput is split into input (prompt processing) and output (decode) tokens per second, with an effective-generation figure (completion tokens over wall-clock) as the end-to-end fallback. The per-service reverse proxy records per-request token counts at the HTTP layer, and the web UI shows the split alongside the effective rate.
- A `llama-cpp` service with `modality = "embedding"` now passes `--embeddings` to llama-server, enabling its embeddings endpoint.
- A responsive web UI for small screens, with layouts adapted for touch and narrow viewports. Navigation is a rail on wide screens and a scrolling tab row on narrow ones, so every section is reachable on a phone, and a band across the top shows the daemon's address, the live connection's state, and the theme toggle at every width.
- The service memory footprint is broken out per device in both the API and the UI, showing where a service's allocation lands.
- The service detail view surfaces runtime and serving configuration (binary path, KV cache types, batch, context, and thread settings, speculative decoding, and expert-offload mode) in the API and the UI.
- A shared time range across a service's history and logs and the Stats view, with 5-minute, 1-hour, 6-hour, and 24-hour rolling presets and a custom fixed range. The range is kept in the URL (`?range=6h`, or `?from=…&to=…` in epoch milliseconds), so a link or a reload keeps it, and moving between a service and Stats carries it along. A rolling range keeps moving with the clock rather than freezing when it was picked.
- Chat renders LaTeX math, markdown in reasoning traces, and correctly escaped code blocks.
- Chat has a copy button on every message and every code block, which also shows the block's language, and a header with the model, the number of exchanges, and the tokens in context. The reasoning trace is a collapsible block, and the wait while a stopped model starts is shown as such, with Stop available throughout.
- `anankectl logs` accepts human-friendly `--since` and `--until` values: RFC 3339 timestamps, local datetimes and dates, or relative ages like `2h` and `30m`, in addition to epoch milliseconds.
- Llama.cpp-native endpoints on the OpenAI listener. `/tokenize`, `/detokenize`, `/apply-template`, `/completion`, `/infill`, `/embedding`, `/embeddings`, `/rerank`, `/reranking`, `/props`, `/slots`, `/slots/{id}`, `/lora-adapters`, `/health`, `/v1/health`, and `/metrics` are now served on the main inference URL and forwarded verbatim to the upstream of a `llama-cpp` template service (started on demand). A `model` field in the body selects the service; without one, the sole llama-cpp service is the target. The OpenAI-shape aliases (`/v1/*`, `/models`, `/completions`, `/chat/completions`, …) are not proxied.
- `GET /api/info` and the live `info` resource report `uptime_ms`, the time since the daemon started.
- `GET /api/config` reports `path`, the file the daemon reads its config from.
- The service detail reports the placement constraints a service runs under (`placement`: policy, allowed GPUs, split mode and tensor-split weights, GPU headroom, and allocation mode), a command service's settings (`command`: upstream model, shutdown command, whether the private port is fixed, and working directory), and the configured `n_gpu_layers`, `context`, and resolved prompt-cache cap under `serving`. The list and detail report `openai_compat` and `description`.
- The service list and detail report why a disabled service is disabled, as `disabled_reason` (`launch_failed`, `health_timeout`, `oom`, `user_disabled`, or `auto_restart_loop`), and `state_changed` events carry it. `anankectl services` shows it beside the state, `anankectl status` counts the services disabled after a fault, and the web UI colours those services as failed and counts them on the Services tab.

### Changed

- The reservation field `vram_gb` is renamed `reserve_gb` (with `min_reserve_gb` and `max_reserve_gb` for dynamic mode), a device-neutral name since it holds host RAM for a cpu-only service and VRAM otherwise. The old spellings are still accepted.
- The VRAM estimator's tuned constants are now derived from a measurement dataset and checked in CI, so a constant cannot drift from its evidence. An architecture the estimator is not calibrated for is refused rather than under-reserved: such a service must declare an explicit reservation (`mode` plus `reserve_gb`).
- `anankectl logs --follow` follows over `/api/live`. It now honours `--run` and `--stream` while following, no longer loses the lines written between the history fetch and the start of the follow, and reports lines it had to skip as `[N lines skipped]` on stderr. With `--json`, a follow prints each line as its own JSON object, one per line, history included. `--stream` rejects values other than `stdout`, `stderr`, and `combined` instead of silently matching nothing, and `--follow` can no longer be combined with `--until`.
- The web UI is restyled: flat, square panels on neutral surfaces with a single accent colour, Work Sans for text and Iosevka for data. It has light and dark themes that follow the system setting, with a toggle in the top band that overrides it and switches back to following the system when it matches again.
- The web UI's Dashboard is replaced by Services, a list beside the selected service's detail. The list can be searched, filtered to favourites, sorted, and grouped by state, and its rows show each service's footprint per device, in-flight requests, a request-rate sparkline, and start and stop controls. On a phone the list and the detail are separate screens. The fleet totals the Dashboard showed are on Stats.
- The web UI's Events view shows the shared time range, which the navigation now carries to it, and includes the auto-restarts recorded before the daemon's live history begins. Events are listed newest first with plain-language summaries, link to their service, and filter by any combination of types; the live connection's state replaces the reconnect button.
- The web UI's Stats view opens with the totals over the range (running services, requests, errors, tokens, latency, and token rates), ranks services by output tokens per second, and charts GPU and host memory apart, so host RAM no longer flattens the GPU lines. A chart with nothing in the range says so instead of drawing a flat line, and the charts go one, two, or three across by the room they have.
- The web UI's Config view names the file it edits and when it was read, keeps unsaved edits when you leave the view and come back, confirms a save, and explains a read-only config.
- A service the operator disables stays disabled across daemon and host restarts until it is enabled again; before, a restart brought it back, and a persistent service started at once. The disable is stored in the database (a new table, added on upgrade). Automatic disables, such as after an OOM or a restart loop, still clear on restart, unless the operator disables the service again on top of one, which also changes its reason to `user_disabled`. The mark is kept when a service leaves the config, so a service added back under the same name comes back disabled. If the database write fails, `enable` and `disable` return `500 persist_failed` and leave the service as it was; that code now covers any durable write, not only the config file.
- The management API returns the documented error envelope (`{"error": {"code", "message", "type"}}`) rather than a bare string, so a client can dispatch on `error.code`.

### Fixed

- The web UI's config editor no longer replaces unsaved edits when the config is reloaded from disk. It keeps them and says the file changed, offering to keep the edits (the next save overwrites the file) or to reload the version on disk.
- A `}` with nothing to close made the placeholder scanner loop forever, hanging config load.
- A typo in an `env` or `container.env` value now fails at config load, naming the variable, rather than at the launch it breaks.
- Container removal waits for the log follower to finish, so the tail of a stopped container's output (the part saying why it stopped) is no longer lost to the removal that deletes the runtime's log store.
- A runtime that cannot be reached is no longer read as a container that is not there. Reconciliation kept deleting the record in that case, which strands a running container with nothing pointing at it. The startup leak sweep also asks every runtime the records name, rather than whichever this process defaults to: on a Podman-only host it was running `docker ps` and finding nothing.
- The daemon removes every container it owns on shutdown when a drain overruns `shutdown_timeout`, rather than exiting and leaving the workload holding its reservation.
- The service detail's serving settings resolve `parallel`, `kv_unified`, `flash_attn`, and `ubatch_size` through `extra_args` the way the launch command does, rather than reading the config keys alone, so they no longer disagree with the running server.
- When both `cache_ram_mb` and `-cram` in `extra_args` are set, the estimate uses the `extra_args` value, which is the one llama-server receives.
- A command service without an `openai_proxy` block is no longer offered for chat in the web UI. It is not on the OpenAI-compatible API, so the chat could only fail.
- Picking another model mid-conversation in the chat now switches to it. Before, only the URL changed and the conversation stayed on the old model. The conversation carries over, and each reply is labelled with the model that wrote it.
- Starting a new chat while a message was still being sent, or while its model was starting, no longer leaves that message in the new conversation.
- The chat's reasoning trace follows the model's output as it streams, until the reader scrolls up inside it.
- The chat waits for a model that is slow to start instead of reporting an error. A start request cut by a proxy or the browser while the model is still loading is reissued, a start the daemon declines (start queue full, not enough memory) is reported rather than ignored, and there is no client-side time limit.
- A chat reply that ends without the stream's `[DONE]` marker, or with an error frame, is reported as incomplete rather than shown as finished.
- `pid 0` is no longer registered for memory attribution. It parents init, so attributing its descendants summed the whole machine into one service.

### Notes

- Images are not pulled or built. ananke starts and stops the image named by the service; the image must already be in the runtime's local store.
- A container cannot inherit `PR_SET_PDEATHSIG`, so one survives a `SIGKILL`ed daemon until startup reconciliation removes it. The allocator reads real free VRAM, so a survivor makes ananke conservative rather than wrong. See [What happens when ananke stops](docs/configuration.md#container-workloads).

## 0.2.0

Released before this file existed. See the git history for the changes it carried.
