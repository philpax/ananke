#!/usr/bin/env python3
"""Scenario 08 — idle-eviction driven by OpenAI API requests.

Drives the new eviction rule (idle candidates are evictable regardless
of priority) purely through `/v1/chat/completions` — no direct
management-API starts. Regression target: before the rule change, two
services at the same priority couldn't displace each other even when
both were sitting at zero in-flight.

Matrix roles used (configurable via
`scenarios.08_idle_eviction_via_openai`):
- `victim` — a service large enough that it contends with `displacer`
  for the same VRAM budget. Hit first.
- `displacer` — a second service that won't cohabit with `victim`.
  Request for it should force an eviction of the now-idle `victim`.

Flow:

  1. Chat → victim. Wait for its state to reach `running` (the model
     is loaded, response came back, in-flight drops to zero).
  2. Chat → displacer. Assert the response is 200 OK. In the event
     log, `victim` should transition `running → draining → (idle |
     evicted)` during or shortly before `displacer`'s own
     `starting → running` climb.
  3. Chat → victim again. Symmetric: `displacer` now gets displaced.

Each of the three chat calls is retried with short backoff if ananke
returns 503 (start_queue_full / insufficient_vram) — the transitions
we care about happen inside ananke's scheduler and are racy against
the inbound request, so the client-side retry is part of exercising
the path, not a bug-hiding shim.

What this exercises:
- `allocator::eviction::select_for_slot` with the relaxed predicate.
- `supervise::collect_eviction_candidates` reading in-flight == 0 as
  idle, not just `ServiceState::Idle`.
- `supervise::persistent_watcher` (implicitly): if the victim is
  declared persistent, the second chat to it would otherwise find it
  in `Evicted` state with nobody re-ensuring.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import (  # noqa: E402
    Api,
    Matrix,
    Recorder,
    parse_args,
    run_scenario,
)

SCENARIO = "08_idle_eviction_via_openai"

# Ananke can queue briefly while drain+place runs; keep the chat retry
# window generous enough to cover a cold start plus drain.
CHAT_TIMEOUT_S = 600.0
CHAT_RETRY_DELAY_S = 2.0
CHAT_RETRY_ATTEMPTS = 3


async def chat_with_retry(api: Api, model: str, prompt: str) -> tuple[int, float]:
    """POST /v1/chat/completions with retry on transient 503.

    Returns (final_status_code, wall_seconds).
    """
    last_status = 0
    start = time.monotonic()
    for attempt in range(1, CHAT_RETRY_ATTEMPTS + 1):
        resp = await api.chat(model, prompt, timeout=CHAT_TIMEOUT_S)
        last_status = resp.status_code
        if resp.status_code == 200:
            return last_status, time.monotonic() - start
        body = resp.text[:200]
        print(
            f"    attempt {attempt}/{CHAT_RETRY_ATTEMPTS}: "
            f"{model} returned {resp.status_code} — {body}"
        )
        if resp.status_code != 503 or attempt == CHAT_RETRY_ATTEMPTS:
            break
        await asyncio.sleep(CHAT_RETRY_DELAY_S)
    return last_status, time.monotonic() - start


async def snapshot_state(api: Api, name: str) -> str:
    detail = await api.detail(name)
    return detail.get("state", "?")


async def print_allocations(api: Api) -> None:
    for g in await api.devices():
        used = sum(r["bytes"] for r in g.get("reservations", []))
        print(
            f"  {g['id']}: reserved {used // (1024 * 1024)} MiB, "
            f"free {g['free_bytes'] // (1024 * 1024)} MiB"
        )
        for r in g.get("reservations", []):
            print(f"    {r['service']}: {r['bytes'] // (1024 * 1024)} MiB")


async def body(matrix: Matrix, api: Api, rec: Recorder) -> None:
    victim, displacer = matrix.require_roles("victim", "displacer")

    prompt = "Reply with a single word."

    async def assert_running(name: str) -> None:
        for _ in range(30):
            st = await snapshot_state(api, name)
            if st == "running":
                return
            if st in ("failed", "disabled"):
                raise RuntimeError(f"{name} went to {st} — aborting scenario")
            await asyncio.sleep(1.0)
        raise RuntimeError(f"{name} did not reach running after chat")

    print(f"\n>>> step 1: chat → {victim}")
    status, elapsed = await chat_with_retry(api, victim, prompt)
    print(f"    status={status}, {elapsed:.1f}s")
    assert status == 200, f"{victim} chat failed with {status}"
    await assert_running(victim)
    print(f"    allocations after step 1:")
    await print_allocations(api)

    print(f"\n>>> step 2: chat → {displacer} (should displace idle {victim})")
    status, elapsed = await chat_with_retry(api, displacer, prompt)
    print(f"    status={status}, {elapsed:.1f}s")
    assert status == 200, f"{displacer} chat failed with {status}"
    await assert_running(displacer)

    # Give the event recorder a breath to catch the drain transitions if
    # they landed after the chat response returned.
    await asyncio.sleep(1.0)
    victim_state = await snapshot_state(api, victim)
    print(f"    {victim} state after displace: {victim_state}")
    print(f"    allocations after step 2:")
    await print_allocations(api)

    print(f"\n>>> step 3: chat → {victim} again (should displace idle {displacer})")
    status, elapsed = await chat_with_retry(api, victim, prompt)
    print(f"    status={status}, {elapsed:.1f}s")
    assert status == 200, f"{victim} second chat failed with {status}"
    await assert_running(victim)

    await asyncio.sleep(1.0)
    displacer_state = await snapshot_state(api, displacer)
    print(f"    {displacer} state after re-displace: {displacer_state}")
    print(f"    allocations after step 3:")
    await print_allocations(api)


def summary(rec: Recorder) -> None:
    drains = [e for e in rec.by_type("state_changed") if e["to"] == "draining"]
    print(f"\n{len(drains)} draining transitions:")
    for e in drains:
        print(f"  t+{e['at_s']:.1f}s: {e['service']}: {e['from']} → draining")

    allocs = rec.by_type("allocation_changed")
    print(f"\n{len(allocs)} allocation_changed events (last 12):")
    for e in allocs[-12:]:
        reservations = {k: v for k, v in e["reservations"].items() if v > 0}
        print(f"  t+{e['at_s']:.1f}s: {e['service']} = {reservations}")


async def main_async() -> None:
    parse_args()
    await run_scenario(
        "idle eviction via OpenAI API",
        body,
        summary=summary,
    )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
