#!/usr/bin/env python3
"""Scenario 02 — priority-driven eviction cascade.

Start services in increasing-priority order. Each successive service raises
pressure on the allocator; the expectation is that lower-priority services
get drained to make room for higher-priority ones.

Matrix roles used (via `scenarios.02_eviction_cascade.sequence`):
Each entry in the sequence is a role name. The scenario starts them in
order, observes the event stream, and prints which services actually stay
Running vs. get drained.

What this exercises:
- `eviction::select_for_slot` victim selection.
- The drain pipeline on evictees (SIGTERM + grace + SIGKILL).
- AllocationChanged event ordering: victim's reservation clears before
  evictor's reservation appears.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import Api, Matrix, Recorder, cleanup_all, parse_args, run_scenario  # noqa: E402

SCENARIO = "02_eviction_cascade"


def print_allocations(api: Api) -> None:
    for g in api.devices():
        used = sum(r["bytes"] for r in g.get("reservations", []))
        print(
            f"  {g['id']}: reserved {used // (1024 * 1024)} MiB, "
            f"free {g['free_bytes'] // (1024 * 1024)} MiB"
        )
        for r in g.get("reservations", []):
            print(f"    {r['service']}: {r['bytes'] // (1024 * 1024)} MiB")


async def body(matrix: Matrix, api: Api, rec: Recorder) -> None:
    cfg = matrix.scenario_config(SCENARIO)
    sequence_roles: list[str] = cfg.get("sequence", [])
    services = matrix.require_roles(*sequence_roles)

    for svc_name in services:
        detail = api.detail(svc_name)
        print(
            f"\n>>> starting {svc_name} (priority {detail.get('priority')})"
        )
        resp = api.start(svc_name)
        print(f"    start response: {resp}")

        running_at = await rec.wait_running(svc_name, timeout_s=300)
        if running_at is None:
            print(f"    !! {svc_name} never reached Running")
        else:
            print(f"    running at t+{running_at:.1f}s")

        print("    allocations:")
        print_allocations(api)
        await asyncio.sleep(2)

    # Summary: which services are currently Running?
    print("\nfinal states:")
    for s in api.services():
        print(f"  {s['name']}: {s['state']}")


def summary(rec: Recorder) -> None:
    drains = [e for e in rec.by_type("state_changed") if e["to"] == "draining"]
    print(f"\n{len(drains)} draining events:")
    for e in drains:
        print(f"  t+{e['at_s']:.1f}s: {e['service']}: {e['from']} → draining")

    allocs = rec.by_type("allocation_changed")
    print(f"\n{len(allocs)} allocation_changed events (last 8):")
    for e in allocs[-8:]:
        reservations = {k: v for k, v in e["reservations"].items() if v > 0}
        print(f"  t+{e['at_s']:.1f}s: {e['service']} = {reservations}")


def main() -> None:
    args = parse_args()
    matrix = Matrix.load()
    services = matrix.require_roles(*matrix.scenario_config(SCENARIO).get("sequence", []))
    try:
        asyncio.run(run_scenario("eviction cascade", body, summary=summary))
    finally:
        if not args.keep_running:
            cleanup_all(Api(matrix), services)


if __name__ == "__main__":
    main()
