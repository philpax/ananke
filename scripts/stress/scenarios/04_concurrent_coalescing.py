#!/usr/bin/env python3
"""Scenario 04 — start-coalescing under concurrent first-requests.

Fire N simultaneous `POST /start` requests (or `POST /v1/chat/completions`
— change in the scenario config) at an idle service. Verify:
- Only one child process actually spawns (exactly one `StateChanged to
  starting` event).
- All N callers receive a consistent outcome (`StartResponse::Started` or
  `AlreadyRunning`).
- No race artefacts: no extra AllocationChanged events beyond the single
  reserve.

Matrix roles used:
- `scenarios.04_concurrent_coalescing.service`: the role to target.
- `scenarios.04_concurrent_coalescing.concurrent_requests`: how many to fire.

What this exercises:
- The start-queue coalescing path (`SupervisorCommand::Ensure` → broadcast
  subscriber dispatch).
- `start_queue_depth` — if N exceeds depth, some responses become QueueFull.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import Api, Matrix, Recorder, cleanup_all, parse_args, run_scenario  # noqa: E402

SCENARIO = "04_concurrent_coalescing"


async def fire_one(api: Api, svc: str) -> dict:
    return await asyncio.to_thread(api.start, svc)


async def body(matrix: Matrix, api: Api, rec: Recorder) -> None:
    cfg = matrix.scenario_config(SCENARIO)
    (svc,) = matrix.require_roles(cfg.get("service", ""))
    n = cfg.get("concurrent_requests", 12)

    # Make sure the service is idle first.
    api.stop(svc)
    await asyncio.sleep(2)

    print(f"\n>>> firing {n} concurrent start requests at {svc}")
    tasks = [fire_one(api, svc) for _ in range(n)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    statuses: dict[str, int] = {}
    errors = 0
    for r in results:
        if isinstance(r, Exception):
            errors += 1
            continue
        status = r.get("status", "?") if isinstance(r, dict) else "?"
        statuses[status] = statuses.get(status, 0) + 1
    print(f"    status distribution: {statuses} (errors: {errors})")

    # Wait for Running.
    await rec.wait_running(svc, timeout_s=300)

    # How many transitions into Starting were published? Expect exactly 1.
    starting_events = [
        e
        for e in rec.by_type("state_changed")
        if e.get("service") == svc and e.get("to") == "starting"
    ]
    print(f"    state_changed→starting events: {len(starting_events)} (expect 1)")

    alloc_events = [
        e
        for e in rec.by_type("allocation_changed")
        if e.get("service") == svc
    ]
    print(f"    allocation_changed events for {svc}: {len(alloc_events)} (expect 1)")


def summary(rec: Recorder) -> None:
    print("\nstate transitions by service:")
    by_svc: dict[str, list[str]] = {}
    for e in rec.by_type("state_changed"):
        by_svc.setdefault(e["service"], []).append(f"{e['from']}→{e['to']}")
    for name, ts in by_svc.items():
        print(f"  {name}: {' | '.join(ts)}")


def main() -> None:
    args = parse_args()
    matrix = Matrix.load()
    (svc,) = matrix.require_roles(matrix.scenario_config(SCENARIO).get("service", ""))
    try:
        asyncio.run(run_scenario("concurrent coalescing", body, summary=summary))
    finally:
        if not args.keep_running:
            cleanup_all(Api(matrix), [svc])


if __name__ == "__main__":
    main()
