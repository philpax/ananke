#!/usr/bin/env python3
"""Scenario 07 — log-stream throughput + retention cap.

Start a service, subscribe to its log-stream WebSocket, and fire a stream of
chat requests that provoke verbose token-by-token output. Watch for:
- Overflow frames (subscriber lag: batcher's broadcast channel capacity is
  256, so if we can't keep up, we'll see `{"type": "overflow", "dropped": N}`).
- GET /logs response shape under load (cursor pagination works).
- Retention doesn't kick in during this short run, but we verify the DB
  doesn't crash at volume.

Matrix config:
- `scenarios.07_log_flood.service`: role to flood.
- `scenarios.07_log_flood.duration_s`: total drive time.

What this exercises:
- `db::logs::spawn` batcher under sustained write load.
- `BatcherHandle::subscribe()` broadcast channel lag semantics.
- GET /api/services/{name}/logs pagination with real data.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import websockets  # type: ignore[import]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import Api, Matrix, Recorder, cleanup_all, parse_args, run_scenario  # noqa: E402

SCENARIO = "07_log_flood"


async def ws_consumer(matrix: Matrix, svc: str, stats: dict) -> None:
    url = matrix.log_stream_ws(svc)
    stats["lines"] = 0
    stats["overflow_frames"] = 0
    stats["overflow_dropped"] = 0
    try:
        async with websockets.connect(url) as ws:
            async for frame in ws:
                msg = json.loads(frame)
                t = msg.get("type")
                if t == "line":
                    stats["lines"] += 1
                elif t == "overflow":
                    stats["overflow_frames"] += 1
                    stats["overflow_dropped"] += msg.get("dropped", 0)
    except Exception as e:
        stats["ws_error"] = str(e)


async def fire_request(api: Api, svc: str, results: list) -> None:
    try:
        resp = await api.chat(svc, "Write a haiku.", timeout=90)
        results.append(resp.status_code)
    except Exception as e:
        results.append(f"err:{e}")


async def body(matrix: Matrix, api: Api, rec: Recorder) -> None:
    cfg = matrix.scenario_config(SCENARIO)
    (svc,) = matrix.require_roles(cfg.get("service", ""))
    duration_s = cfg.get("duration_s", 60)

    await api.start(svc)
    await rec.wait_running(svc, timeout_s=300)

    print(f"\n>>> subscribing to log stream of {svc}")
    stats: dict = {}
    ws_task = asyncio.create_task(ws_consumer(matrix, svc, stats))
    await asyncio.sleep(1)

    print(f"\n>>> firing chat requests for {duration_s}s")
    start = time.monotonic()
    request_results: list = []
    request_count = 0
    while time.monotonic() - start < duration_s:
        request_count += 1
        asyncio.create_task(fire_request(api, svc, request_results))
        await asyncio.sleep(0.5)
    await asyncio.sleep(5)  # let in-flight settle

    ws_task.cancel()
    try:
        await ws_task
    except asyncio.CancelledError:
        pass

    # Historical GET /logs.
    print("\n>>> fetching historical /logs")
    logs_resp = await api.logs(svc, limit=500)
    print(
        f"    got {len(logs_resp['logs'])} lines, "
        f"next_cursor {'present' if logs_resp['next_cursor'] else 'none'}"
    )

    # Follow-up page if cursor exists.
    cursor = logs_resp.get("next_cursor")
    if cursor:
        page2 = await api.logs(svc, limit=500, before=cursor)
        print(f"    page 2: {len(page2['logs'])} lines")

    print(f"\n    WS stats: {stats}")
    print(f"    request count: {request_count}")
    print(f"    request results (last 10): {request_results[-10:]}")


def summary(rec: Recorder) -> None:
    alloc = rec.by_type("allocation_changed")
    print(f"\n{len(alloc)} allocation_changed events during the run")
    drift = rec.by_type("estimator_drift")
    if drift:
        print(f"{len(drift)} estimator_drift events:")
        for e in drift[:3]:
            print(f"  t+{e['at_s']:.1f}s: {e['service']} mean={e['rolling_mean']:.3f}")


async def main_async() -> None:
    args = parse_args()
    matrix = Matrix.load()
    (svc,) = matrix.require_roles(matrix.scenario_config(SCENARIO).get("service", ""))
    try:
        await run_scenario("log flood", body, summary=summary)
    finally:
        if not args.keep_running:
            async with Api(matrix) as api:
                await cleanup_all(api, [svc])


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
