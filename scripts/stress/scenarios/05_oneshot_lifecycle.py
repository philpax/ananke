#!/usr/bin/env python3
"""Scenario 05 — oneshot submit + TTL drain.

Submit a oneshot via `POST /api/oneshot` (command template) with a short TTL.
Observe the supervisor spin it up, run until TTL fires, then drain cleanly.
Verify:
- The oneshot appears under `GET /api/oneshot/{id}` throughout.
- On TTL expiry, the supervisor transitions through draining → stopped.
- The port returns to the pool (next submit reuses or advances).

Matrix config:
- `scenarios.05_oneshot_lifecycle.ttl_seconds`: how long to let the oneshot live.

What this exercises:
- Oneshot spawn path (distinct from persistent services).
- TTL watcher (`oneshot::ttl::spawn_watcher`).
- Port pool allocate/release.
- The oneshots DB table (row update on end).
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import Api, Matrix, Recorder, parse_args, run_scenario  # noqa: E402

SCENARIO = "05_oneshot_lifecycle"


async def body(matrix: Matrix, api: Api, rec: Recorder) -> None:
    cfg = matrix.scenario_config(SCENARIO)
    ttl_s = cfg.get("ttl_seconds", 60)

    # Build a command-template oneshot that just sleeps. The allocation can
    # be tiny because the command doesn't actually use GPUs.
    body = {
        "name": f"stress-oneshot-{int(time.time())}",
        "template": "command",
        "command": ["/bin/sh", "-c", f"sleep {ttl_s + 30}"],
        "allocation": {"mode": "static", "vram_gb": 0.1},
        "devices": {"placement": "cpu-only"},
        "priority": 40,
        "ttl": f"{ttl_s}s",
    }
    print(f"\n>>> submitting oneshot (ttl={ttl_s}s): {body['name']}")
    resp = api.submit_oneshot(body)
    print(f"    response: {resp}")
    oneshot_id = resp["id"]

    # Poll status periodically.
    deadline = time.monotonic() + ttl_s + 30
    last_state = None
    while time.monotonic() < deadline:
        try:
            status = api._get(f"/api/oneshot/{oneshot_id}")  # type: ignore[attr-defined]
            if status.get("state") != last_state:
                print(
                    f"    t+{time.monotonic() - rec.start:.1f}s "
                    f"status: {status.get('state')} "
                    f"(started_at_ms={status.get('started_at_ms')}, "
                    f"ended_at_ms={status.get('ended_at_ms')})"
                )
                last_state = status.get("state")
            if status.get("ended_at_ms") is not None:
                print("    oneshot ended; confirming cleanup")
                break
        except Exception as e:
            print(f"    status poll error: {e}")
        await asyncio.sleep(3)

    # Final check: the oneshot should be absent from the active registry
    # (it's in the DB history but no longer held in the in-memory map for
    # the routing table).
    print("\n  final detail:")
    try:
        print(f"    {api._get(f'/api/oneshot/{oneshot_id}')}")  # type: ignore[attr-defined]
    except Exception as e:
        print(f"    lookup failed: {e}")


def summary(rec: Recorder) -> None:
    drain_events = [e for e in rec.by_type("state_changed") if e["to"] == "draining"]
    print(f"\n{len(drain_events)} drain transitions observed")
    for e in drain_events:
        print(f"  t+{e['at_s']:.1f}s: {e['service']} drained")


def main() -> None:
    args = parse_args()
    asyncio.run(run_scenario("oneshot lifecycle", body, summary=summary))


if __name__ == "__main__":
    main()
