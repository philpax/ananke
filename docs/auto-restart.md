# Auto-restart

ananke supervises long-lived inference servers. A crashed child is handled by the crash path: the process exits, the supervisor notices, and the service is respawned. The auto-restart triggers handle the remaining cases, where the process stays alive but stops serving correctly.

This document describes each trigger, the failure class it detects, and the guardrails that bound restart frequency. The field-by-field configuration reference is in [configuration.md](configuration.md#auto-restart). The incidents that led to each trigger are listed in the [appendix](#appendix-incident-history).

## Failure classes

A degraded inference server fails in one of three ways:

1. **Requests error.** Every request returns HTTP 5xx. Detectable from status codes.
2. **Requests hang.** The server accepts work but never produces tokens. No status codes are recorded, because no request completes.
3. **Requests return garbage.** Tokens flow, HTTP returns 200, and the output is incorrect. Status codes, completion, and throughput all look normal.

Each trigger covers part of this space.

## Triggers

When any trigger fires, the service drains (SIGTERM with a grace period, then SIGKILL) and returns to `Idle`. The normal ensure path respawns it: on the next request for an on-demand service, or within a few seconds for a persistent one.

### error_rate

Polls the per-request metrics recorded by the proxy. Fires when the error ratio in the recent window reaches `max_error_rate` and the window contains at least `min_requests` requests.

Detects failure class 1. The `min_requests` floor exists because a ratio over few requests is not meaningful; two failures out of two requests should not restart a service. By default only 5xx statuses count, since a 4xx normally indicates a client error. `error_statuses = "4xx+5xx"` counts both.

Limitation: only requests that complete with an error status are visible to it.

### ttft_stall

Watches in-flight streaming requests at the proxy. Fires when no streaming request service-wide has produced a response frame for the timeout (5 minutes by default). A request queued behind a healthy generation does not trip it, because the healthy generation's frames count as service-wide progress.

Detects failure class 2 for streaming traffic. The timeout is deliberately long: healthy prefill produces a first token within seconds, so minutes without any frame indicates a hang rather than load.

Limitation: non-streaming requests produce no observable frames until the full response body arrives, so it cannot watch them.

### generation_stall

Polls the child's Prometheus `/metrics` progress counters (prompt and predicted tokens). Fires when the counters stay flat for the timeout while at least one request is in flight. On llama-cpp services the daemon passes `--metrics` automatically while this trigger is enabled.

Detects failure class 2 for non-streaming traffic. Healthy prefill and decode both advance the counters every batch, so flat counters with requests in flight indicate a hang. An idle service never trips it.

Limitation: it measures progress, not correctness. A process generating incorrect output at normal speed advances the counters normally.

### spec_collapse

Reads the speculative-decoding fields that llama.cpp reports in each response's `timings` object: `draft_n` (tokens proposed by the draft) and `draft_n_accepted` (tokens accepted by the target). The proxy records both per request. Fires when the run has accepted draft tokens at some point and the window then contains drafting requests whose `draft_n` sums to at least `min_draft_tokens` with `draft_n_accepted` summing to zero. Only services with `spec_type` configured run this trigger; other services produce no draft counts, and an explicit per-service enable without `spec_type` is rejected at validation.

Detects failure class 3. Draft acceptance is usable as a correctness signal because the draft and target score the same context: a working pairing accepts a substantial fraction of drafted tokens on nearly every request, while a target with corrupted inference state (degenerate or NaN logits) rejects every draft token on every request.

The trip condition is a zero sum rather than a low-acceptance threshold. Individual short generations can accept nothing, so a "low acceptance" threshold would need tuning and could fire on difficult prompts. A zero sum across `min_draft_tokens` drafted tokens does not occur during normal operation of a workload that accepts at all. The floor counts drafted tokens rather than requests: the failure produces long generations that arrive slowly — a request-count floor can never fill its window on that traffic — while each such generation drafts thousands of tokens, so a token floor is typically reached by a single one.

The prior-acceptance requirement exists because some workloads never accept: grammar-constrained speculative decoding can reject every draft token on a healthy target, since drafts that violate the grammar fail verification. For such a workload, zero is the baseline rather than a failure, and the trigger stays quiet. The requirement also bounds the cost of any residual false positive to a single restart: a fresh run must demonstrate acceptance before the trigger can arm again, so this trigger alone cannot drive a healthy service to the flap cap.

The window is keyed on request completion time (`timestamp_ms + duration_ms`), not start time. Rows are recorded at completion, and the failure produces long garbage generations; a start-keyed window would age those requests out of the window before they were ever recorded.

Aborted requests need one more provision. llama.cpp normally attaches `timings` only to the final chunk of a stream, and a client that gives up on a garbage generation never receives that chunk — so the daemon injects `timings_per_token` into streamed requests to spec-decoding services, making every chunk carry the cumulative draft counts. The proxy records the last values seen when the stream ends, normally or not. Without this, the exact traffic a wedge produces — long generations cut off by client timeouts — would leave no evidence for the trigger to read.

This injection is visible to clients: every chunk of such a stream carries a `timings` object rather than only the last one, which adds a few hundred bytes per token to the response. A request that sets `timings_per_token` itself keeps its own value, so a client that wants the smaller frames can opt out per request — at the cost of that request being invisible to the trigger if it is aborted. Services without `spec_type`, and non-streaming requests, are untouched.

The trigger is passive: it reads metrics from real traffic and sends no probe requests. A synthetic probe (checking a response for degenerate logprobs) would extend coverage to non-speculative services, but would also reset the idle timeout and consume GPU time. Since this failure class only causes damage while traffic is flowing, and the passive signal is present whenever traffic flows, no probe is implemented.

### periodic

A time-based restart, disabled by default. `interval` sets the maximum run age. Three modes control timing once the interval elapses: `immediate` drains at once, `on-idle` waits until no requests are in flight, and `on-request` (the default) marks the run stale and restarts on the next request, which then blocks on the fresh process.

This trigger detects nothing; it bounds run age for services where slow state degradation is expected. Periodic restarts do not count toward the flap cap.

## Engine coverage

ananke proxies more than llama-server, and the triggers differ in what they assume about the upstream. Three observe only HTTP traffic at the proxy and work with any server; two read llama.cpp-specific surfaces and are gated to services that provide them.

| Trigger | Requires from the upstream | Applies to |
|---|---|---|
| `error_rate` | HTTP status codes | any proxied service |
| `ttft_stall` | a streaming (SSE) response | any proxied service, streaming requests only |
| `generation_stall` | llama.cpp-compatible Prometheus `/metrics` counters | `llama-cpp` services automatically; `command` services by explicit opt-in |
| `spec_collapse` | llama.cpp `timings.draft_n` / `draft_n_accepted` response fields | `llama-cpp` services with `spec_type` only |
| `periodic` | nothing | any service |

The boundaries in detail:

- A `command` service wrapping a non-llama.cpp server (vLLM, TabbyAPI, anything OpenAI-compatible) gets `error_rate`, `ttft_stall`, and `periodic`. That covers failure classes 1 and 2-for-streaming; class 2 for non-streaming traffic and class 3 have no detector on such a service.
- `generation_stall` on a `command` service is an explicit opt-in for wrapped servers that expose llama.cpp-compatible counters. If the endpoint is missing or unrecognisable, the trigger logs one warning and never fires.
- `spec_collapse` cannot be enabled outside a `llama-cpp` service with `spec_type`; validation rejects the attempt, since no other configuration produces draft counts. This includes a `command` service wrapping llama-server with speculative decoding — the daemon doesn't manage that server's argv, so it can't establish the preconditions the trigger relies on.
- The statistical triggers (`error_rate`, `spec_collapse`) are fed by per-request metrics, which the proxy records for `/v1/chat/completions` and `/v1/completions` only. Traffic on other endpoints (embeddings, for example) is forwarded without recording and never feeds them.
- `ttft_stall` is armed on the OpenAI multiplexer path. Requests sent directly to a service's own proxy port are metered for `error_rate` but not stall-watched.

## Guardrails

Three limits apply to the watchdog triggers:

- **min_uptime** (default 5 minutes). A fresh run must reach this age before the error-rate, generation-stall, or spec-collapse watchdogs may fire. In addition, every watchdog query is scoped to the current run's `run_id`, so a respawned process starts with empty metrics.
- **max_restarts within flap_window** (default 3 within 30 minutes). Repeated watchdog restarts indicate a fault that restarting does not fix. At the cap, the service is disabled with reason `auto_restart_loop` instead of restarted, and stays disabled until an operator re-enables it. Re-enabling resets the restart budget.
- **A volume floor on the statistical triggers.** Error-rate has `min_requests`; spec-collapse has `min_draft_tokens`, for the reason given above. Either way, a small amount of traffic is not read as a trend.

The stall triggers use a shortened drain wait. Their premise is that in-flight requests will never complete, so waiting the full `max_request_duration` before SIGTERM would only delay the respawn.

## Observability

Each firing is:

- logged by the daemon (`auto-restart: watchdog firing`, with trigger and detail),
- published as an `auto_restarted` event, shown in the web UI's events view, and
- persisted to the daemon's store, capped at the newest 50 entries per service.

Persistence exists because the daemon's event history is in memory, bounded, and lost on restart. The stored history is served in the service detail response (`GET /api/services/{name}`, field `recent_restarts`) and shown on the web UI's service page and in `anankectl show <service>`.

The firing that trips the flap cap is recorded and logged like any other, but broadcasts a state change to `disabled` rather than an `auto_restarted` event, since nothing is restarted.

The Prometheus counter `ananke_auto_restarts_total{service,trigger}` is kept in a separate monotonic tally rather than counted from the stored history, which is capped per service. Counting evictable rows would let a per-trigger count fall, and a falling counter reads to Prometheus as a process restart — turning an eviction into a phantom restart spike on the dashboard.

## Design constraints

Two requirements apply to any new trigger:

1. **No false positives on recorded traffic.** Thresholds are calibrated against captured traffic containing both the target failure and healthy operation. A trigger that can restart a healthy service under load is not accepted.
2. **Run-scoped evidence.** A trigger evaluates the current process only. Metrics from earlier runs, including the run whose failure caused the current respawn, must not count against a fresh process.

## Appendix: incident history

Each watchdog was added in response to a production incident that the existing triggers did not detect.

- **error_rate**: a gemma-4-31b-it-qat service ran for ~8.5 hours, then wedged its grammar stack and returned HTTP 500 on every request while the process stayed alive. The default thresholds were validated against the captured traffic: the watchdog fires about one minute into the wedge and does not fire during the healthy hours.
- **ttft_stall**: a recurring zero-token wedge in the same service. Streaming requests were accepted but no frame was ever emitted, so no request completed and no status code was recorded.
- **generation_stall** (2026-07-11): the same zero-token wedge class reached through non-streaming requests (llama.cpp SWA bug #22450 era), which the TTFT watchdog cannot observe.
- **spec_collapse** (2026-07-24): one hour into healthy MTP traffic on gemma-4-31b-it-qat, client cancellations racing llama-server's `n_past` rollback path corrupted the shared inference state. Every subsequent decode produced NaN logits; the model emitted the token `<unused49>` for every position of every response at a normal ~36 tok/s with HTTP 200 throughout. Draft acceptance was 60–100% per request before the corruption and exactly zero on every request after it, for 45+ minutes. In the healthy phases of the captured traffic, no window of ten drafting requests ever summed to zero accepted tokens, which is the basis for the zero-sum trip condition. The same incident motivated persisting firings: the generation-stall watchdog had restarted the earlier wedged run at 14:37, and this went unnoticed for two hours because the event was only broadcast on the WebSocket.
