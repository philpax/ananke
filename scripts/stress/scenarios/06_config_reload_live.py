#!/usr/bin/env python3
"""Scenario 06 — config PUT during active traffic.

Start a service, fire a background chat request, then PUT the current config
back to the daemon (unchanged, to force a noop reload). Verify:
- The `config_reloaded` event fires.
- The live service doesn't transition states (`diff_services` returns empty).
- The active request isn't disturbed.

Matrix config:
- `scenarios.06_config_reload_live.service`: the role to keep warm.

What this exercises:
- `ConfigManager::apply` → validate → persist → in-memory swap → publish.
- The notify-watcher dedup we added in the follow-up fix pass.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import Api, Matrix, Recorder, cleanup_all, parse_args, run_scenario  # noqa: E402

SCENARIO = "06_config_reload_live"


async def background_chat(api: Api, svc: str, results: list) -> None:
    try:
        resp = await api.chat(svc, "Count to ten, slowly.")
        results.append(("ok", resp.status_code, len(resp.content)))
    except Exception as e:
        results.append(("err", str(e)))


async def body(matrix: Matrix, api: Api, rec: Recorder) -> None:
    cfg = matrix.scenario_config(SCENARIO)
    (svc,) = matrix.require_roles(cfg.get("service", ""))

    # Warm the service.
    print(f"\n>>> warming {svc}")
    await api.start(svc)
    t = await rec.wait_running(svc, timeout_s=300)
    print(f"    running at t+{t}")

    # Fire background traffic.
    results: list = []
    print("\n>>> firing background chat request")
    traffic = asyncio.create_task(background_chat(api, svc, results))
    await asyncio.sleep(1)  # let the request start

    # GET + PUT the config (idempotent).
    print("\n>>> GET /api/config")
    current = await api.config()
    print(f"    hash: {current['hash'][:16]}..., content length: {len(current['content'])}")

    print("\n>>> PUT /api/config (noop — same content, same hash)")
    code = await api.put_config(current["content"], current["hash"])
    print(f"    status: {code}")

    # The apply() no-op path short-circuits before persisting, so we expect
    # zero config_reloaded events from this PUT. But a content-changing PUT
    # would publish exactly one.
    await asyncio.sleep(3)
    reloaded = rec.by_type("config_reloaded")
    print(f"    config_reloaded events: {len(reloaded)} (expect 0 for noop PUT)")

    # Now do a changing PUT: tweak a comment at the end of the TOML so the
    # hash changes but semantics don't.
    mutated = current["content"].rstrip() + "\n# ananke stress reload marker\n"
    print("\n>>> PUT /api/config (content change: append comment)")
    code = await api.put_config(mutated, current["hash"])
    print(f"    status: {code}")

    # Expect config_reloaded event shortly.
    ev = await rec.wait_for(
        lambda e: e.get("type") == "config_reloaded",
        timeout_s=10,
    )
    if ev:
        print(f"    got config_reloaded at t+{ev['at_s']:.1f}s")
    else:
        print("    !! never observed config_reloaded")

    # Verify the service didn't transition away from Running.
    svc_transitions = [
        e for e in rec.by_type("state_changed")
        if e.get("service") == svc and e.get("to") in ("draining", "stopped")
    ]
    print(
        f"    {svc} transitions to draining/stopped during reload: "
        f"{len(svc_transitions)} (expect 0)"
    )

    # Wait for the background request to finish.
    await traffic
    print(f"\n>>> background chat result: {results}")


def summary(rec: Recorder) -> None:
    cfg_events = rec.by_type("config_reloaded")
    print(f"\n{len(cfg_events)} config_reloaded events observed:")
    for e in cfg_events:
        print(f"  t+{e['at_s']:.1f}s: changed_services={e.get('changed_services', [])}")


async def main_async() -> None:
    args = parse_args()
    matrix = Matrix.load()
    (svc,) = matrix.require_roles(matrix.scenario_config(SCENARIO).get("service", ""))
    try:
        await run_scenario("config reload live", body, summary=summary)
    finally:
        if not args.keep_running:
            async with Api(matrix) as api:
                await cleanup_all(api, [svc])


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
