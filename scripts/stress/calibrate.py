#!/usr/bin/env python3
"""Estimator calibration sweep — one fresh ``ananke`` per (service, context).

For each LLM service in the source config, double the context length until
the daemon refuses the start or the service never reaches Running. Each
iteration is a cold-boot ``ananke`` subprocess against a synthetic
single-service TOML with ephemeral ports and its own ``data_dir``.

Why a fresh daemon per iteration rather than reloading context on one
long-lived daemon: the rolling-correction feedback loop
(``tracking::rolling``) accumulates observed-vs-estimated-weights
multipliers per service across runs. Measuring at context N would
therefore be biased by the correction learned at context N/2, and so
on. A cold boot resets the correction to 1.0, so each (service, context)
row captures the raw estimator output unperturbed by earlier iterations.

Per-iteration flow:

1. Copy the target ``[[service]]`` block from the source config, override
   its ``context``, mark it persistent so the daemon starts it
   implicitly.
2. Write the synthetic TOML under a per-iteration tempdir (unique
   ``data_dir``, ephemeral management + openai ports).
3. Spawn ``ananke --config=<toml>`` with its stdout/stderr captured to a
   per-iteration log file.
4. Poll ``/api/services`` until the management API answers, then poll
   ``/api/services/<name>`` until ``state == "running"``.
5. Collect the daemon's reservation, NVML-observed usage, observed peak,
   and a trivial chat round-trip.
6. SIGTERM the daemon; SIGKILL if it won't exit inside
   ``--stop-timeout`` seconds.
7. Append the row to ``calibration-<ts>.csv``; checkpoint after each
   service so a crash mid-sweep doesn't lose earlier data.

Sweep stops for a given service when an iteration reports anything other
than a healthy chat — unavailable placement, failed cold-start, OOM, or
``--max-context`` reached.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx  # type: ignore[import]
import tomlkit  # type: ignore[import]

sys.path.insert(0, str(Path(__file__).resolve().parent))

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_BIN = REPO_ROOT / "target" / "release" / "ananke"

INITIAL_CONTEXT: int = 2048


@dataclasses.dataclass
class Measurement:
    service: str
    context: int
    reserved_total_mb: int
    reserved_per_device: str
    nvml_mib_per_gpu: str
    nvml_free_mib_per_gpu: str
    observed_peak_bytes: int
    state: str
    chat_status: str
    chat_ms: int | None
    time_to_running_s: float | None
    error: str

    def to_row(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --- nvml probes -----------------------------------------------------------


def _nvml_query(field: str) -> dict[int, int]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu=index,{field}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except Exception:
        return {}
    res: dict[int, int] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            res[int(parts[0])] = int(parts[1])
        except ValueError:
            continue
    return res


def nvml_used_mib() -> dict[int, int]:
    return _nvml_query("memory.used")


def nvml_free_mib() -> dict[int, int]:
    return _nvml_query("memory.free")


def format_mib_dict(d: dict[int, int]) -> str:
    return ",".join(f"gpu:{k}={v}" for k, v in sorted(d.items()))


# --- synthetic TOML assembly ----------------------------------------------


def _unwrap(v: Any) -> Any:
    """Coerce tomlkit containers into plain Python values so they round-trip
    through a fresh document cleanly."""
    if isinstance(v, dict):
        return {k: _unwrap(sub) for k, sub in v.items()}
    if isinstance(v, list):
        return [_unwrap(sub) for sub in v]
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return float(v)
    return str(v) if v is not None else None


def extract_service_block(source_toml: str, service_name: str) -> dict[str, Any]:
    doc = tomlkit.parse(source_toml)
    services = doc.get("service")
    if services is None:
        raise KeyError("no [[service]] arrays in config")
    for svc in services:
        if str(svc.get("name", "")) == service_name:
            return {k: _unwrap(v) for k, v in svc.items()}
    raise KeyError(f"no service named {service_name!r} in config")


def llm_services(source_toml: str) -> list[str]:
    """Return the names of every llama-cpp service in the source config.

    Command-template services (ComfyUI-style) have no ``context`` knob and
    are skipped; the sweep only applies to llama-server workloads.
    """
    doc = tomlkit.parse(source_toml)
    services = doc.get("service") or []
    out: list[str] = []
    for svc in services:
        if str(svc.get("template", "")) == "llama-cpp":
            out.append(str(svc["name"]))
    return out


def free_port() -> int:
    """Ask the kernel for an unused ephemeral port; close the listener
    before returning. There's a narrow TOCTOU window before the daemon
    binds, but the calibration tool is the only user of this port so the
    race is academic."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def build_synthetic_toml(
    service_block: dict[str, Any],
    context: int,
    *,
    management_port: int,
    openai_port: int,
    data_dir: Path,
) -> str:
    svc = dict(service_block)
    svc["context"] = context
    svc["lifecycle"] = "persistent"
    svc.pop("idle_timeout", None)

    out = tomlkit.document()
    daemon = tomlkit.table()
    daemon["management_listen"] = f"127.0.0.1:{management_port}"
    daemon["data_dir"] = str(data_dir)
    daemon["shutdown_timeout"] = "30s"
    out["daemon"] = daemon

    openai = tomlkit.table()
    openai["listen"] = f"127.0.0.1:{openai_port}"
    out["openai_api"] = openai

    out.add(tomlkit.nl())
    svcs = tomlkit.aot()
    svcs.append(svc)
    out["service"] = svcs
    return tomlkit.dumps(out)


# --- daemon orchestration --------------------------------------------------


class Daemon:
    """Owns a single ``ananke`` subprocess for the duration of one
    measurement. Context-manages startup/shutdown + temp dir cleanup."""

    def __init__(
        self,
        binary: Path,
        config_path: Path,
        mgmt_url: str,
        openai_url: str,
        log_path: Path,
        *,
        startup_timeout_s: float,
        stop_timeout_s: float,
    ):
        self.binary = binary
        self.config_path = config_path
        self.mgmt_url = mgmt_url
        self.openai_url = openai_url
        self.log_path = log_path
        self.startup_timeout_s = startup_timeout_s
        self.stop_timeout_s = stop_timeout_s
        self._proc: subprocess.Popen[bytes] | None = None
        self._log_fh: Any = None

    async def __aenter__(self) -> "Daemon":
        self._log_fh = self.log_path.open("wb")
        env = os.environ.copy()
        env["ANANKE_CONFIG"] = str(self.config_path)
        # LD_LIBRARY_PATH for NVML (and any other runtime deps) is expected
        # to come from the caller's environment — on NixOS the repo's
        # `shell.nix` exposes `/run/opengl-driver/lib`. If you're running
        # calibrate.py outside the dev shell and the daemon logs "NVML init
        # failed" the snapshot will drop to CPU-only and the packer will
        # reject every GPU-bound service.
        self._proc = subprocess.Popen(
            [str(self.binary)],
            env=env,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setsid,  # own process group for clean SIGTERM propagation.
        )
        await self._wait_ready()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()

    async def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_s
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.monotonic() < deadline:
                if self._proc is not None and self._proc.poll() is not None:
                    raise RuntimeError(
                        f"daemon exited during startup (rc={self._proc.returncode}); "
                        f"see {self.log_path}"
                    )
                try:
                    resp = await client.get(f"{self.mgmt_url}/api/services")
                    if resp.status_code == 200:
                        return
                except Exception:
                    pass
                await asyncio.sleep(0.2)
        raise RuntimeError(
            f"daemon never answered /api/services within {self.startup_timeout_s}s; "
            f"see {self.log_path}"
        )

    async def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self._proc.wait), timeout=self.stop_timeout_s
                )
            except asyncio.TimeoutError:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await asyncio.to_thread(self._proc.wait)
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
        self._proc = None


# --- measurement client ----------------------------------------------------


class CalibClient:
    def __init__(self, mgmt_url: str, openai_url: str) -> None:
        self.mgmt_url = mgmt_url
        self.openai_url = openai_url
        self._client = httpx.AsyncClient(timeout=180.0)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "CalibClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def service_detail(self, name: str) -> dict[str, Any]:
        resp = await self._client.get(
            f"{self.mgmt_url}/api/services/{name}", timeout=15.0
        )
        resp.raise_for_status()
        return resp.json()

    async def devices(self) -> list[dict[str, Any]]:
        resp = await self._client.get(f"{self.mgmt_url}/api/devices", timeout=15.0)
        resp.raise_for_status()
        return resp.json()

    async def start(self, name: str) -> dict[str, Any]:
        resp = await self._client.post(
            f"{self.mgmt_url}/api/services/{name}/start", timeout=180.0
        )
        resp.raise_for_status()
        return resp.json() if resp.headers.get("content-type", "").startswith(
            "application/json"
        ) else {"status": "ok"}

    async def chat(self, name: str, prompt: str, *, timeout: float = 120.0) -> httpx.Response:
        return await self._client.post(
            f"{self.openai_url}/v1/chat/completions",
            json={
                "model": name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 32,
            },
            timeout=timeout,
        )


async def wait_running(client: CalibClient, name: str, timeout_s: float) -> float | None:
    deadline = time.monotonic() + timeout_s
    started = time.monotonic()
    while time.monotonic() < deadline:
        try:
            detail = await client.service_detail(name)
            state = detail.get("state", "")
            if state == "running":
                return time.monotonic() - started
            if state in ("failed", "disabled"):
                return None
        except Exception:
            pass
        await asyncio.sleep(1.0)
    return None


# --- one iteration ---------------------------------------------------------


async def measure_one(
    service_name: str,
    context: int,
    service_block: dict[str, Any],
    binary: Path,
    workspace: Path,
    *,
    startup_timeout_s: float,
    running_timeout_s: float,
    stop_timeout_s: float,
) -> Measurement:
    iter_dir = workspace / f"{service_name}-{context}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    data_dir = iter_dir / "data"
    data_dir.mkdir(exist_ok=True)
    config_path = iter_dir / "config.toml"
    log_path = iter_dir / "daemon.log"

    mgmt_port = free_port()
    openai_port = free_port()
    mgmt_url = f"http://127.0.0.1:{mgmt_port}"
    openai_url = f"http://127.0.0.1:{openai_port}"

    config_path.write_text(
        build_synthetic_toml(
            service_block,
            context,
            management_port=mgmt_port,
            openai_port=openai_port,
            data_dir=data_dir,
        )
    )

    t0 = time.monotonic()
    try:
        daemon = Daemon(
            binary,
            config_path,
            mgmt_url,
            openai_url,
            log_path,
            startup_timeout_s=startup_timeout_s,
            stop_timeout_s=stop_timeout_s,
        )
        async with daemon, CalibClient(mgmt_url, openai_url) as client:
            try:
                start_resp = await client.start(service_name)
            except Exception as e:
                return _failure(
                    service_name,
                    context,
                    state="start-failed",
                    error=f"start: {e}",
                    t_start=t0,
                )
            status = start_resp.get("status", "?")
            if status == "unavailable":
                return _failure(
                    service_name,
                    context,
                    state="unavailable",
                    error=str(start_resp.get("reason", "unavailable")),
                    t_start=t0,
                )

            running_delta = await wait_running(client, service_name, running_timeout_s)
            if running_delta is None:
                return _failure(
                    service_name,
                    context,
                    state="timeout",
                    error=f"never reached Running within {running_timeout_s}s",
                    t_start=t0,
                )

            # Let llama-server finish allocating KV + compute buffers and the
            # supervisor's observer tick pick up peak usage.
            await asyncio.sleep(8)

            detail = await client.service_detail(service_name)
            devices = await client.devices()
            nvml_used = nvml_used_mib()
            nvml_free = nvml_free_mib()

            reserved: dict[str, int] = {}
            for g in devices:
                for r in g.get("reservations", []):
                    if r["service"] == service_name:
                        reserved[g["id"]] = r["bytes"] // (1024 * 1024)

            chat_status = "skipped"
            chat_ms: int | None = None
            try:
                chat_t0 = time.monotonic()
                resp = await client.chat(
                    service_name, "Reply with the single word: hi.", timeout=120.0
                )
                chat_ms = int((time.monotonic() - chat_t0) * 1000)
                chat_status = "ok" if resp.status_code == 200 else f"err:{resp.status_code}"
            except Exception as e:
                chat_status = f"err:{type(e).__name__}"

            return Measurement(
                service=service_name,
                context=context,
                reserved_total_mb=sum(reserved.values()),
                reserved_per_device=",".join(
                    f"{k}={v}" for k, v in sorted(reserved.items())
                ),
                nvml_mib_per_gpu=format_mib_dict(nvml_used),
                nvml_free_mib_per_gpu=format_mib_dict(nvml_free),
                observed_peak_bytes=int(detail.get("observed_peak_bytes") or 0),
                state=str(detail.get("state", "?")),
                chat_status=chat_status,
                chat_ms=chat_ms,
                time_to_running_s=round(running_delta, 2),
                error="",
            )
    except RuntimeError as e:
        return _failure(
            service_name, context, state="daemon-failed", error=str(e), t_start=t0
        )


def _failure(
    service_name: str,
    context: int,
    *,
    state: str,
    error: str,
    t_start: float,
) -> Measurement:
    return Measurement(
        service=service_name,
        context=context,
        reserved_total_mb=0,
        reserved_per_device="",
        nvml_mib_per_gpu=format_mib_dict(nvml_used_mib()),
        nvml_free_mib_per_gpu=format_mib_dict(nvml_free_mib()),
        observed_peak_bytes=0,
        state=state,
        chat_status="skipped",
        chat_ms=None,
        time_to_running_s=round(time.monotonic() - t_start, 2),
        error=error,
    )


# --- sweep + main ----------------------------------------------------------


async def sweep_service(
    name: str,
    service_block: dict[str, Any],
    binary: Path,
    workspace: Path,
    *,
    max_context: int | None,
    startup_timeout_s: float,
    running_timeout_s: float,
    stop_timeout_s: float,
) -> list[Measurement]:
    rows: list[Measurement] = []
    ctx = INITIAL_CONTEXT
    while True:
        print(f"\n--- {name} @ context={ctx} ---", flush=True)
        m = await measure_one(
            name,
            ctx,
            service_block,
            binary,
            workspace,
            startup_timeout_s=startup_timeout_s,
            running_timeout_s=running_timeout_s,
            stop_timeout_s=stop_timeout_s,
        )
        rows.append(m)
        print(
            f"    state={m.state} reserved={m.reserved_total_mb} MiB "
            f"chat={m.chat_status} chat_ms={m.chat_ms} "
            f"nvml_used={m.nvml_mib_per_gpu} nvml_free={m.nvml_free_mib_per_gpu}"
            + (f" error={m.error}" if m.error else ""),
            flush=True,
        )
        if m.state != "running" or m.chat_status.startswith("err"):
            print(f"    stopping sweep for {name}: {m.state}/{m.chat_status}", flush=True)
            break
        if max_context is not None and ctx >= max_context:
            print(f"    reached --max-context {max_context}", flush=True)
            break
        ctx *= 2
    return rows


def write_csv(path: Path, rows: list[Measurement]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].to_row().keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_row())


async def main_async(args: argparse.Namespace) -> None:
    source_path = Path(args.source_config).expanduser()
    if not source_path.exists():
        sys.stderr.write(f"source config {source_path} does not exist\n")
        sys.exit(2)
    binary = Path(args.daemon_binary).expanduser()
    if not binary.exists() or not os.access(binary, os.X_OK):
        sys.stderr.write(f"daemon binary {binary} is missing or not executable\n")
        sys.exit(2)

    source = source_path.read_text()
    all_services = llm_services(source)
    targets: list[str]
    if args.services:
        targets = list(args.services)
        missing = [s for s in targets if s not in all_services]
        if missing:
            sys.stderr.write(
                f"requested services not found in {source_path}: {', '.join(missing)}\n"
            )
            sys.exit(2)
    else:
        targets = all_services
    if not targets:
        sys.stderr.write("no llama-cpp services to sweep\n")
        return

    print(f"source config: {source_path}", flush=True)
    print(f"daemon binary: {binary}", flush=True)
    print(f"sweep targets: {targets}", flush=True)
    print(f"initial context: {INITIAL_CONTEXT}, max: {args.max_context or 'unbounded'}", flush=True)

    workspace = Path(tempfile.mkdtemp(prefix="ananke-calib-"))
    print(f"workspace: {workspace}", flush=True)

    all_rows: list[Measurement] = []
    try:
        for name in targets:
            block = extract_service_block(source, name)
            rows = await sweep_service(
                name,
                block,
                binary,
                workspace,
                max_context=args.max_context,
                startup_timeout_s=args.startup_timeout,
                running_timeout_s=args.running_timeout,
                stop_timeout_s=args.stop_timeout,
            )
            all_rows.extend(rows)
            checkpoint = HERE / f"calibration-{int(time.time())}-partial.csv"
            write_csv(checkpoint, all_rows)
            print(f"[checkpoint] {len(all_rows)} rows → {checkpoint}", flush=True)
    finally:
        if not args.keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)
        else:
            print(f"[keep-workspace] per-iteration logs left at {workspace}", flush=True)

    final = HERE / f"calibration-{int(time.time())}.csv"
    write_csv(final, all_rows)
    print(f"\ndone. {len(all_rows)} rows → {final}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--daemon-binary",
        default=os.environ.get("ANANKE_BIN", str(DEFAULT_BIN)),
        help=f"path to the ananke binary (default: $ANANKE_BIN or {DEFAULT_BIN})",
    )
    p.add_argument(
        "--source-config",
        default=os.environ.get("ANANKE_CONFIG", "/tmp/ananke-redline/config.toml"),
        help="config file to copy service blocks from (default: $ANANKE_CONFIG or /tmp/ananke-redline/config.toml)",
    )
    p.add_argument(
        "--services",
        nargs="+",
        help="sweep only the named services (default: every llama-cpp service in the source config)",
    )
    p.add_argument(
        "--max-context",
        type=int,
        default=None,
        help="stop doubling once this context has been measured (default: keep doubling until a step fails)",
    )
    p.add_argument(
        "--startup-timeout",
        type=float,
        default=30.0,
        help="seconds to wait for the daemon's management API to answer (default: 30)",
    )
    p.add_argument(
        "--running-timeout",
        type=float,
        default=600.0,
        help="seconds to wait for the service to reach Running after Start (default: 600)",
    )
    p.add_argument(
        "--stop-timeout",
        type=float,
        default=60.0,
        help="seconds to wait for graceful daemon SIGTERM before SIGKILL (default: 60)",
    )
    p.add_argument(
        "--keep-workspace",
        action="store_true",
        help="don't delete the per-iteration tempdir (keeps daemon.log files for post-mortem)",
    )
    return p.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
