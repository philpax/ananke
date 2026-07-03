The OpenAI-compatible API (`/v1/*`) is the primary inference surface. Ananke acts as a smart proxy: it resolves the `model` field in each request to a configured service, ensures the service is running (starting it on-demand if needed), then forwards the request body to the upstream service's private port.

### Proxy behaviour

- The request body is parsed only enough to extract `model`; all other
  fields are passed through verbatim to the upstream.
- Filters may rewrite JSON fields before forwarding (configured per-service
  via `[[service.filters]]`).
- For `openai_proxy` command services, the `model` field is rewritten to
  the upstream's expected name (`openai_proxy.upstream_model`).
- Responses — including SSE streams — are proxied back without buffering.
- Hop-by-hop headers (`connection`, `transfer-encoding`, `keep-alive`) are
  stripped from upstream responses so the browser doesn't misinterpret
  them.

### Streaming

Streaming responses (SSE) are supported on all three POST endpoints. Set `"stream": true` in the request body. The upstream's SSE chunks are proxied to the client as they arrive — there is no buffering.

### 501 stubs

The following OpenAI endpoints return `501 Not Implemented`:

- `/v1/audio/*`
- `/v1/images/*`
- `/v1/files/*`
- `/v1/fine_tuning/*`
- `/v1/batches`
