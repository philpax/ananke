#!/usr/bin/env python3
"""Shared helpers for ananke stress scenarios.

Each scenario imports this module. It centralises:
- HTTP + WebSocket endpoints (overridable via matrix.toml or env).
- Role → service-name resolution from matrix.toml.
- Event-bus subscription with a recording buffer.
- Narrative-friendly scenario runner.

Requires: `pip install websockets requests`.
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
from typing import Any, AsyncIterator, Callable, Optional

try:
    import requests  # type: ignore[import]
    import websockets  # type: ignore[import]
except ImportError as e:  # pragma: no cover
    sys.stderr.write(
        f"missing dependency {e.name!r}; "
        "install with `pip install websockets requests`\n"
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
            endpoint.get("management", "http://127.0.0.1:17777"),
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
    def __init__(self, matrix: Matrix):
        self.matrix = matrix

    def start(self, name: str) -> Any:
        return self._post(f"/api/services/{name}/start")

    def stop(self, name: str) -> Any:
        return self._post(f"/api/services/{name}/stop")

    def restart(self, name: str) -> Any:
        return self._post(f"/api/services/{name}/restart")

    def enable(self, name: str) -> Any:
        return self._post(f"/api/services/{name}/enable")

    def disable(self, name: str) -> Any:
        return self._post(f"/api/services/{name}/disable")

    def detail(self, name: str) -> Any:
        return self._get(f"/api/services/{name}")

    def services(self) -> Any:
        return self._get("/api/services")

    def devices(self) -> Any:
        return self._get("/api/devices")

    def logs(self, name: str, **params: Any) -> Any:
        qs = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
        path = f"/api/services/{name}/logs"
        if qs:
            path += "?" + qs
        return self._get(path)

    def config(self) -> dict[str, Any]:
        return self._get("/api/config")

    def put_config(self, content: str, if_match: str) -> int:
        resp = requests.put(
            f"{self.matrix.management}/api/config",
            data=content,
            headers={"If-Match": f'"{if_match}"'},
            timeout=30,
        )
        return resp.status_code

    def submit_oneshot(self, body: dict[str, Any]) -> Any:
        resp = requests.post(
            f"{self.matrix.management}/api/oneshot", json=body, timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str) -> Any:
        resp = requests.get(f"{self.matrix.management}{path}", timeout=15)
        resp.raise_for_status()
        return resp.json() if resp.headers.get("content-type", "").startswith(
            "application/json"
        ) else resp.text

    def _post(self, path: str) -> Any:
        resp = requests.post(f"{self.matrix.management}{path}", timeout=180)
        resp.raise_for_status()
        return resp.json() if resp.headers.get("content-type", "").startswith(
            "application/json"
        ) else None


# --- scenario runner -------------------------------------------------------


async def run_scenario(
    title: str,
    body: Callable[[Matrix, Api, Recorder], Any],
    *,
    subscribe_service: str | None = None,
    summary: Optional[Callable[[Recorder], None]] = None,
) -> Recorder:
    matrix = Matrix.load()
    api = Api(matrix)
    confirm_daemon_up(api)
    recorder = Recorder()

    print(f"\n=== {title} ===")
    print(f"  management: {matrix.management}")
    async with subscribe_events(matrix, recorder, service=subscribe_service):
        await body(matrix, api, recorder)
    print(f"\n--- {len(recorder.events)} events captured ---")
    if summary is not None:
        summary(recorder)
    return recorder


def confirm_daemon_up(api: Api) -> None:
    try:
        api.services()
    except requests.ConnectionError:
        sys.stderr.write(
            f"cannot reach ananke management at {api.matrix.management}\n"
            "start the daemon (and make sure your matrix.toml's endpoint "
            "matches) before running scenarios.\n"
        )
        sys.exit(3)


# --- misc helpers ----------------------------------------------------------


def chat(matrix: Matrix, model: str, prompt: str, *, timeout: int = 300) -> requests.Response:
    return requests.post(
        f"{matrix.openai}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
        },
        timeout=timeout,
    )


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


def cleanup_all(api: Api, names: list[str]) -> None:
    print("\n[cleanup] draining services...")
    for name in names:
        try:
            api.stop(name)
            print(f"  stop({name}) ok")
        except Exception as e:
            print(f"  stop({name}) failed: {e}")
