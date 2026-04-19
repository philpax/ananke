#!/usr/bin/env python3
"""Scenario 01 — estimator sanity check.

For a list of single-GPU services (small to large), start each, wait for
Running, issue a trivial chat request, and record the daemon's reservation
alongside NVML's observed usage. The comparison exposes drift between what
the estimator thinks it needs and what the child actually consumes.

Matrix roles used (from `scenarios.01_estimator_sanity.services`):
- each entry is a role name that must be populated.

What this exercises:
- GGUF reading + layer-aware estimator.
- Placement engine on single-GPU first-fits.
- Rolling-correction table as services cycle through spawn/drain.
- `/api/events` StateChanged + AllocationChanged publishers.
"""

import asyncio
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import (  # noqa: E402
    Api,
    Matrix,
    Recorder,
    chat,
    cleanup_all,
    parse_args,
    run_scenario,
)

SCENARIO = "01_estimator_sanity"


def nvml_used_mib(gpu: int) -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu),
            ],
            text=True,
            timeout=5,
        )
        return int(out.strip())
    except Exception:
        return None


async def body(matrix: Matrix, api: Api, rec: Recorder) -> None:
    cfg = matrix.scenario_config(SCENARIO)
    role_names: list[str] = cfg.get("services", [])
    timeout_s: float = cfg.get("cold_start_timeout_s", 300)
    services = matrix.require_roles(*role_names)

    for svc_name in services:
        print(f"\n--- {svc_name} ---")
        before = {gi: nvml_used_mib(gi) for gi in (0, 1)}
        print(f"    nvml before: {before}")

        resp = api.start(svc_name)
        print(f"    start → {resp}")

        running_at = await rec.wait_running(svc_name, timeout_s=timeout_s)
        if running_at is None:
            print(f"    !! never reached Running within {timeout_s}s")
            continue
        print(f"    running at t+{running_at:.1f}s")

        detail = api.detail(svc_name)
        print(f"    pid={detail.get('pid')} run_id={detail.get('run_id')}")

        for g in api.devices():
            for r in g.get("reservations", []):
                if r["service"] == svc_name:
                    print(f"    reserved {r['bytes'] // (1024 * 1024)} MiB on {g['id']}")

        await asyncio.sleep(4)
        after = {gi: nvml_used_mib(gi) for gi in (0, 1)}
        delta = {gi: (after[gi] or 0) - (before[gi] or 0) for gi in (0, 1)}
        print(f"    nvml delta: {delta} MiB")

        try:
            resp = chat(matrix, svc_name, "Say hi in three words.")
            print(f"    /v1/chat: HTTP {resp.status_code}, {len(resp.content)} bytes")
        except Exception as e:
            print(f"    chat failed: {e}")

        api.stop(svc_name)
        await asyncio.sleep(4)


def summary(rec: Recorder) -> None:
    running = [e for e in rec.by_type("state_changed") if e["to"] == "running"]
    print("\nstartup timings (service → time to Running):")
    first_by_svc: dict[str, float] = {}
    for e in running:
        first_by_svc.setdefault(e["service"], e["at_s"])
    for name, t in first_by_svc.items():
        print(f"  {name}: {t:.1f}s")


def main() -> None:
    args = parse_args()
    try:
        matrix = Matrix.load()
        services = matrix.require_roles(*matrix.scenario_config(SCENARIO).get("services", []))
        asyncio.run(run_scenario("estimator sanity", body, summary=summary))
    finally:
        if not args.keep_running:
            cleanup_all(Api(Matrix.load()), services)


if __name__ == "__main__":
    main()
