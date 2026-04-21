# Ananke stress-test scenarios

Scripts that drive a running daemon through specific scheduling patterns to
validate the allocation algorithm, supervisor state machine, eviction, and
WebSocket event surfaces.

## Shape

- `lib.py` — HTTP + WebSocket helpers, matrix loader, scenario runner.
- `scenarios/<nn>_<name>.py` — individual scenarios. Each prints a narrative
  as it runs and a summary at the end.
- `example-matrix.toml` — committed illustration of how to map scenario
  roles to your own service names.
- `matrix.toml` — your local file, **not committed** (see `.gitignore`).
  Copy `example-matrix.toml` and edit role → service-name mappings to match
  your own daemon's config.

Scenarios reference services by abstract role (e.g. `small_gpu`,
`large_single_gpu`, `hybrid_cpu_offload`, `dynamic_elastic`) rather than by
concrete service name, so the suite runs against any daemon that can populate
the required roles.

## Prerequisites

- A running ananke daemon, reachable at `http://127.0.0.1:7071` by default.
  Override via `ANANKE_ENDPOINT` env.
- [`uv`](https://docs.astral.sh/uv/) for Python environment management.
- A `matrix.toml` next to `lib.py` (see below).

Python dependencies are declared in `pyproject.toml`. On first use, `uv` will
create a `.venv/` automatically; subsequent `uv run` invocations reuse it.

## Setting up the matrix

```bash
cd scripts/stress
cp example-matrix.toml matrix.toml
# Edit matrix.toml so each role points at one of your real service names.
```

Each scenario validates the roles it needs at start-up; unmapped roles cause
the scenario to skip with a clear message.

## Running

```bash
cd scripts/stress

# One scenario:
uv run scenarios/01_estimator_sanity.py

# All scenarios in order:
for s in scenarios/*.py; do uv run "$s" || break; done
```

Scenarios leave the daemon in a drained state at exit (every started service
gets stopped). Pass `--keep-running` to skip the cleanup stop.

## What each scenario exercises

| # | Name | What it exercises |
|---|---|---|
| 01 | `estimator_sanity` | GGUF reading, layer-walker, single-GPU fit, rolling correction. Compares estimator output to NVML observed usage per service. |
| 02 | `eviction_cascade` | Priority-driven eviction: start low-prio services until saturated, then bring in high-prio services and verify the right victims get drained. |
| 03 | `hybrid_cohabitation` | Hybrid CPU-offload model + dynamic-allocation service coexisting on shared devices. Verifies allocation accounting is stable across overlapping reservations. |
| 04 | `concurrent_coalescing` | N simultaneous first-requests to one idle service. Start-queue coalescing: only one spawn actually fires; all N get the same broadcast outcome. |
| 05 | `oneshot_lifecycle` | Submit a oneshot via the command template, observe TTL-driven drain, verify DB row + port-pool cleanup. |
| 06 | `config_reload_live` | PUT `/api/config` while a service is actively serving requests. Verify the reload event fires and the live service isn't disturbed. |
| 07 | `log_flood` | Drive a chatty service for a minute, measure batcher throughput + the retention trim's per-service cap. |

Add your own scenarios under `scenarios/` — the role indirection means they
can be published alongside the suite without leaking specific service names.
