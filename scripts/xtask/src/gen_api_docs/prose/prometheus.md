The `/metrics` endpoint exposes Prometheus text-format metrics for external scrapers. Prometheus stores history; the JSON `/api/metrics` endpoint serves ananke's own historical data for the dashboard charts.

### Metric families

| Family | Type | Description |
| --- | --- | --- |
| `ananke_requests_total` | counter | Total number of requests proxied. |
| `ananke_tokens_total` | counter | Total tokens processed (labelled by `type`: `prompt` or `completion`). |
| `ananke_inflight_requests` | gauge | Current number of in-flight requests. |
| `ananke_memory_bytes` | gauge | Total memory capacity in bytes (labelled by `device`). |
| `ananke_memory_free_bytes` | gauge | Free memory in bytes (labelled by `device`). |
| `ananke_memory_used_bytes` | gauge | Used memory in bytes (labelled by `device`). |
| `ananke_output_tokens_per_second` | gauge | Average output tokens/sec during decode over the last 5 minutes. |
| `ananke_input_tokens_per_second` | gauge | Average input tokens/sec during prompt processing over the last 5 minutes. |
| `ananke_service_state` | gauge | Numeric service state code (labelled by `service`). |

### State codes

The `ananke_service_state` gauge uses numeric codes:

| Code | State |
| --- | --- |
| 0 | idle |
| 1 | starting |
| 2 | running |
| 3 | draining |
| 4 | stopped |
| 5 | evicted |
| 6 | failed |
| 7 | disabled |
| 8 | unknown |

Code 8 (`unknown`) is used when no supervisor handle exists for a configured service — the service is in the config but hasn't been registered yet.
