#!/usr/bin/env python3
"""Shared helpers for ananke stress scenarios.

Each scenario imports this module. It centralises:
- HTTP + WebSocket endpoints (overridable via matrix.toml or env).
- Role → service-name resolution from matrix.toml.
- Event-bus subscription with a recording buffer.
- Narrative-friendly scenario runner.

Requires: ``pip install websockets httpx`` (or use the supplied uv env).

The HTTP client is async (``httpx.AsyncClient``). Using a synchronous client
here blocks the asyncio event loop for the duration of each request, which
starves the events-WebSocket pump and causes the recorder to drop frames
while long operations (e.g. model cold-starts) are in flight. Every API
method therefore awaits.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

try:
    import httpx  # type: ignore[import]
    import websockets  # type: ignore[import]
except ImportError as e:  # pragma: no cover
    sys.stderr.write(
        f"missing dependency {e.name!r}; "
        "install with `pip install httpx websockets`\n"
    )
    sys.exit(2)

HERE = Path(__file__).resolve().parent
MATRIX_PATH = HERE / "matrix.toml"


# --- matrix loader ---------------------------------------------------------


@dataclass
class Matrix:
    management: str
    openai: str
    roles: dict[str, str]
    scenarios: dict[str, dict[str, Any]]

    @classmethod
    def load(cls) -> "Matrix":
        if not MATRIX_PATH.exists():
            sys.stderr.write(
                f"no matrix.toml at {MATRIX_PATH}\n"
                f"copy example-matrix.toml → matrix.toml and edit role mappings.\n"
            )
            sys.exit(2)
        with MATRIX_PATH.open("rb") as f:
            data = tomllib.load(f)
        endpoint = data.get("endpoint", {})
        mgmt = os.environ.get(
            "ANANKE_ENDPOINT",
            endpoint.get("management", "http://127.0.0.1:7071"),
        )
        openai = os.environ.get(
            "ANANKE_OPENAI_ENDPOINT",
            endpoint.get("openai", "http://127.0.0.1:7070"),
        )
        return cls(
            management=mgmt,
            openai=openai,
            roles=data.get("roles", {}),
            scenarios=data.get("scenarios", {}),
        )

    def role(self, name: str) -> str | None:
        """Return the service name mapped to `name`, or None if unmapped."""
        return self.roles.get(name)

    def require_roles(self, *names: str) -> list[str]:
        """Resolve every role; abort the scenario if any is unmapped."""
        resolved: list[str] = []
        missing: list[str] = []
        for n in names:
            svc = self.role(n)
            if svc is None:
                missing.append(n)
            else:
                resolved.append(svc)
        if missing:
            sys.stderr.write(
                f"skipping scenario: matrix.toml has no mapping for role(s) "
                f"{', '.join(missing)}\n"
            )
            sys.exit(0)
        return resolved

    def scenario_config(self, scenario: str) -> dict[str, Any]:
        return self.scenarios.get(scenario, {})

    def events_ws(self, service: str | None = None) -> str:
        base = self.management.replace("http://", "ws://").replace("https://", "wss://")
        suffix = f"?service={service}" if service else ""
        return f"{base}/api/events{suffix}"

    def log_stream_ws(self, service: str) -> str:
        base = self.management.replace("http://", "ws://").replace("https://", "wss://")
        return f"{base}/api/services/{service}/logs/stream"


# --- event recorder --------------------------------------------------------


@dataclass
class Recorder:
    events: list[dict[str, Any]] = field(default_factory=list)
    start: float = field(default_factory=time.monotonic)

    def record(self, event: dict[str, Any]) -> None:
        self.events.append({"at_s": round(time.monotonic() - self.start, 3), **event})

    def by_type(self, t: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") == t]

    async def wait_for(
        self,
        predicate: Callable[[dict[str, Any]], bool],
        *,
        timeout_s: float,
        poll_s: float = 0.2,
    ) -> dict[str, Any] | None:
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            for ev in self.events:
                if predicate(ev):
                    return ev
            await asyncio.sleep(poll_s)
        return None

    async def wait_running(self, service: str, *, timeout_s: float) -> float | None:
        ev = await self.wait_for(
            lambda e: e.get("type") == "state_changed"
            and e.get("service") == service
            and e.get("to") == "running",
            timeout_s=timeout_s,
        )
        return ev["at_s"] if ev else None

    async def wait_transition(
        self, service: str, to: str, *, timeout_s: float
    ) -> float | None:
        ev = await self.wait_for(
            lambda e: e.get("type") == "state_changed"
            and e.get("service") == service
            and e.get("to") == to,
            timeout_s=timeout_s,
        )
        return ev["at_s"] if ev else None


# --- event subscription ----------------------------------------------------


@contextlib.asynccontextmanager
async def subscribe_events(
    matrix: Matrix,
    recorder: Recorder,
    *,
    service: str | None = None,
) -> AsyncIterator[asyncio.Task]:
    async def pump() -> None:
        async with websockets.connect(matrix.events_ws(service)) as ws:
            async for frame in ws:
                try:
                    event = json.loads(frame)
                except json.JSONDecodeError:
                    continue
                recorder.record(event)

    task = asyncio.create_task(pump())
    try:
        await asyncio.sleep(0.3)  # let the WS open before the body runs
        yield task
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


# --- API wrappers ----------------------------------------------------------


class Api:
    """Async HTTP client bound to the daemon's management endpoint.

    Use as an async context manager so the underlying ``httpx.AsyncClient``
    is closed cleanly. The default timeout is deliberately generous — model
    cold-starts can run for several minutes on 30B+ weights.
    """

    def __init__(self, matrix: Matrix, *, default_timeout: float = 180.0):
        self.matrix = matrix
        self._client = httpx.AsyncClient(timeout=default_timeout)

    async def __aenter__(self) -> "Api":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def start(self, name: str) -> Any:
        return await self._post(f"/api/services/{name}/start")

    async def stop(self, name: str) -> Any:
        return await self._post(f"/api/services/{name}/stop")

    async def restart(self, name: str) -> Any:
        return await self._post(f"/api/services/{name}/restart")

    async def enable(self, name: str) -> Any:
        return await self._post(f"/api/services/{name}/enable")

    async def disable(self, name: str) -> Any:
        return await self._post(f"/api/services/{name}/disable")

    async def detail(self, name: str) -> Any:
        return await self._get(f"/api/services/{name}")

    async def services(self) -> Any:
        return await self._get("/api/services")

    async def devices(self) -> Any:
        return await self._get("/api/devices")

    async def logs(self, name: str, **params: Any) -> Any:
        qs = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
        path = f"/api/services/{name}/logs"
        if qs:
            path += "?" + qs
        return await self._get(path)

    async def config(self) -> dict[str, Any]:
        return await self._get("/api/config")

    async def put_config(self, content: str, if_match: str) -> int:
        resp = await self._client.put(
            f"{self.matrix.management}/api/config",
            content=content,
            headers={"If-Match": f'"{if_match}"'},
            timeout=30.0,
        )
        return resp.status_code

    async def submit_oneshot(self, body: dict[str, Any]) -> Any:
        resp = await self._client.post(
            f"{self.matrix.management}/api/oneshot", json=body, timeout=30.0
        )
        resp.raise_for_status()
        return resp.json()

    async def get_oneshot(self, oneshot_id: str) -> Any:
        return await self._get(f"/api/oneshot/{oneshot_id}")

    async def chat(
        self, model: str, prompt: str, *, timeout: float = 300.0
    ) -> httpx.Response:
        return await self._client.post(
            f"{self.matrix.openai}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 32,
            },
            timeout=timeout,
        )

    async def _get(self, path: str) -> Any:
        resp = await self._client.get(
            f"{self.matrix.management}{path}", timeout=15.0
        )
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "")
        return resp.json() if ctype.startswith("application/json") else resp.text

    async def _post(self, path: str) -> Any:
        resp = await self._client.post(
            f"{self.matrix.management}{path}", timeout=180.0
        )
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "")
        return resp.json() if ctype.startswith("application/json") else None


# --- scenario runner -------------------------------------------------------


async def run_scenario(
    title: str,
    body: Callable[[Matrix, Api, Recorder], Awaitable[Any]],
    *,
    subscribe_service: str | None = None,
    summary: Optional[Callable[[Recorder], None]] = None,
) -> Recorder:
    matrix = Matrix.load()
    recorder = Recorder()
    async with Api(matrix) as api:
        await confirm_daemon_up(api)

        print(f"\n=== {title} ===")
        print(f"  management: {matrix.management}")
        async with subscribe_events(matrix, recorder, service=subscribe_service):
            await body(matrix, api, recorder)
    print(f"\n--- {len(recorder.events)} events captured ---")
    if summary is not None:
        summary(recorder)
    return recorder


async def confirm_daemon_up(api: Api) -> None:
    try:
        await api.services()
    except (httpx.ConnectError, httpx.ReadTimeout):
        sys.stderr.write(
            f"cannot reach ananke management at {api.matrix.management}\n"
            "start the daemon (and make sure your matrix.toml's endpoint "
            "matches) before running scenarios.\n"
        )
        sys.exit(3)


# --- misc helpers ----------------------------------------------------------


def parse_args(
    extra: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keep-running",
        action="store_true",
        help="skip the cleanup stop() at scenario end",
    )
    if extra is not None:
        extra(parser)
    return parser.parse_args()


async def cleanup_all(api: Api, names: list[str]) -> None:
    print("\n[cleanup] draining services...")
    for name in names:
        try:
            await api.stop(name)
            print(f"  stop({name}) ok")
        except Exception as e:
            print(f"  stop({name}) failed: {e}")
