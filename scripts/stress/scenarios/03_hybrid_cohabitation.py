#!/usr/bin/env python3
"""Scenario 03 — hybrid / MMProj / dynamic coexistence.

Start services that claim VRAM in three very different ways and verify they
coexist without the allocator over-reserving or thrashing.

Matrix roles used (via `scenarios.03_hybrid_cohabitation.services`):
- vision_with_mmproj: an mmproj-augmented llama-cpp service (tests mmproj
  size accounting).
- hybrid_cpu_offload: override_tensor CPU offload (tests the hybrid path).
- dynamic_elastic: a command-template service with AllocationMode::Dynamic
  (exercises the balloon resolver).

What this exercises:
- Estimator mmproj accounting (weights + mmproj → combined reservation).
- Hybrid placement_override (the one case we still accept a manual split).
- Balloon resolver: first live run. Watch `rolling_mean` converge.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import Api, Matrix, Recorder, cleanup_all, parse_args, run_scenario  # noqa: E402

SCENARIO = "03_hybrid_cohabitation"


async def body(matrix: Matrix, api: Api, rec: Recorder) -> None:
    cfg = matrix.scenario_config(SCENARIO)
    role_names: list[str] = cfg.get("services", [])
    services = matrix.require_roles(*role_names)

    for svc in services:
        print(f"\n>>> starting {svc}")
        resp = api.start(svc)
        print(f"    start → {resp}")
        running_at = await rec.wait_running(svc, timeout_s=600)
        if running_at is None:
            print(f"    !! {svc} never reached Running")
            continue
        print(f"    running at t+{running_at:.1f}s")

        # Report reservations + observed peak (if any).
        detail = api.detail(svc)
        print(f"    observed_peak_bytes: {detail.get('observed_peak_bytes')}")
        print(f"    rolling_mean: {detail.get('rolling_mean')}")

        # Stability check: the previously-running services shouldn't have
        # their reservations change when this new service spins up.
        for previously_running in services[: services.index(svc)]:
            shifts = [
                e
                for e in rec.by_type("allocation_changed")
                if e.get("service") == previously_running
            ]
            print(
                f"    {previously_running}: {len(shifts)} "
                "allocation_changed events so far (1 is normal, N>1 means it moved)"
            )

    # Let the balloon resolver sample.
    print("\nwaiting 20 s for balloon sampler to establish baseline...")
    await asyncio.sleep(20)
    for svc in services:
        d = api.detail(svc)
        print(
            f"  {svc}: rolling_mean={d.get('rolling_mean')} "
            f"observed_peak_mib={(d.get('observed_peak_bytes') or 0) // (1024 * 1024)}"
        )


def summary(rec: Recorder) -> None:
    print("\nstate transitions:")
    for e in rec.by_type("state_changed"):
        print(f"  t+{e['at_s']:.1f}s: {e['service']}: {e['from']} → {e['to']}")
    drift = rec.by_type("estimator_drift")
    if drift:
        print(f"\n{len(drift)} estimator_drift events:")
        for e in drift[:5]:
            print(f"  t+{e['at_s']:.1f}s: {e['service']} mean={e['rolling_mean']:.3f}")


def main() -> None:
    args = parse_args()
    matrix = Matrix.load()
    services = matrix.require_roles(*matrix.scenario_config(SCENARIO).get("services", []))
    try:
        asyncio.run(run_scenario("hybrid cohabitation", body, summary=summary))
    finally:
        if not args.keep_running:
            cleanup_all(Api(matrix), services)


if __name__ == "__main__":
    main()
