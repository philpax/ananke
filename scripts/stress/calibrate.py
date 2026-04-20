#!/usr/bin/env python3
"""Estimator calibration sweep — direct llama-server, no ananke daemon.

For each target llama-cpp service in the source config and each context
in the doubling sequence (2k, 4k, 8k, 16k, …), this script:

1. Invokes ``cargo run --example estimate`` against the
   service's GGUF to get ananke's *predicted* VRAM reservation per
   device — same estimator + packer the daemon would use, but without
   DB / supervisor / NVML coupling.
2. Records a baseline NVML reading.
3. Spawns ``llama-server`` directly with matching flags (``-m``, ``-c``,
   ``-ngl 999``, ``-fa on``, ``--cache-type-k/v``, ``--tensor-split``,
   ``-ot``, ``--mmproj``) on an ephemeral port.
4. Waits for ``/health`` to return 200 + a stabilisation pause, fires a
   trivial chat request, captures peak + post-chat NVML, SIGTERMs
   llama-server, waits for exit.
5. Appends a row to the output CSV with predicted + actual per-device
   MiB side-by-side, plus deltas.

Bypassing ananke is deliberate: the daemon's rolling-correction table
and startup sequencing add noise to pure "what does the estimator
predict vs what does llama.cpp really allocate" measurements. The
estimator example runs the same code path the daemon uses for pack,
so the predicted numbers are authoritative.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx  # type: ignore[import]
import tomlkit  # type: ignore[import]

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_ESTIMATE_BIN = REPO_ROOT / "target" / "debug" / "examples" / "estimate"

INITIAL_CONTEXT: int = 2048

# Stabilisation pause after /health first returns 200. llama.cpp allocates
# the attention scratch + KV cache lazily on the first forward pass, so
# pure post-load readings under-report unless we at least attempt a chat.
POST_HEALTH_PAUSE_S: float = 5.0
POST_CHAT_PAUSE_S: float = 3.0


@dataclasses.dataclass
class Row:
    service: str
    context: int
    architecture: str
    # Predicted breakdown from the estimator, all in MiB.
    predicted_weights_mib: int
    predicted_kv_total_mib: int
    predicted_compute_buffer_mib: int
    predicted_total_mib: int
    # Actual VRAM delta measured via nvidia-smi, in MiB.
    actual_peak_total_mib: int
    actual_peak_per_gpu: str
    actual_post_chat_total_mib: int
    actual_post_chat_per_gpu: str
    # post-chat minus predicted; positive means the estimator under-reserves.
    delta_post_chat_total_mib: int
    chat_status: str
    chat_ms: int | None
    time_to_healthy_s: float | None
    error: str


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


def format_per_gpu(d: dict[int, int]) -> str:
    return ",".join(f"gpu:{k}={v}" for k, v in sorted(d.items()))


def subtract_per_gpu(current: dict[int, int], base: dict[int, int]) -> dict[int, int]:
    return {k: max(0, v - base.get(k, 0)) for k, v in current.items()}


# --- source-config parsing ------------------------------------------------


def _unwrap(v: Any) -> Any:
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


def llm_service_blocks(source_toml: str) -> dict[str, dict[str, Any]]:
    """Merge `[[service]]` and `[[persistent_service]]` arrays into one
    name → service-block dict. The daemon's config parser accepts both
    top-level array names (the latter is a shortcut for the former with
    `lifecycle = "persistent"`), and calibrate.py needs to pull from
    both without caring about the distinction.
    """
    doc = tomlkit.parse(source_toml)
    out: dict[str, dict[str, Any]] = {}
    for key in ("service", "persistent_service"):
        for s in doc.get(key) or []:
            if str(s.get("template", "")) != "llama-cpp":
                continue
            out[str(s["name"])] = {k: _unwrap(v) for k, v in s.items()}
    return out


# --- estimate example invocation ------------------------------------------


def run_estimator(
    binary: Path, svc: dict[str, Any], context: int, active_devices: int | None = None
) -> dict[str, Any]:
    """Invoke the `estimate` example against `svc` at the given context
    and return its parsed JSON output. When `active_devices` is set, the
    estimate's `total_accounted_*` fields reflect that placement
    (relevant for small single-GPU models vs large dual-GPU + CPU fits).
    """
    args: list[str] = [
        str(binary),
        "--model",
        str(svc["model"]),
        "--context",
        str(context),
    ]
    if svc.get("mmproj"):
        args += ["--mmproj", str(svc["mmproj"])]
    if svc.get("cache_type_k"):
        args += ["--cache-type-k", str(svc["cache_type_k"])]
    if svc.get("cache_type_v"):
        args += ["--cache-type-v", str(svc["cache_type_v"])]
    for rule in svc.get("override_tensor", []) or []:
        args += ["--override-tensor", str(rule)]
    if svc.get("n_cpu_moe") is not None:
        args += ["--n-cpu-moe", str(svc["n_cpu_moe"])]
    if active_devices is not None:
        args += ["--active-devices", str(active_devices)]
    estimation = svc.get("estimation") or {}
    if estimation.get("compute_buffer_mb") is not None:
        args += ["--compute-buffer-mb", str(estimation["compute_buffer_mb"])]
    if estimation.get("allow_fallback"):
        args.append("--allow-fallback")

    result = subprocess.run(args, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        return {"estimator_error": f"exit {result.returncode}: {result.stderr.strip()}"}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        return {"estimator_error": f"bad json: {e}; stdout={result.stdout[:200]}"}


# --- llama-server orchestration -------------------------------------------


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def build_llama_server_argv(
    svc: dict[str, Any], context: int, port: int, ngl: int
) -> tuple[list[str], dict[str, str]]:
    """Render the llama-server argv + env for `svc` at `context`.

    The argv intentionally mirrors what `supervise::spawn::render_argv`
    produces in the daemon so the calibration result is comparable to
    what an operator's production daemon would spawn.
    """
    args: list[str] = [
        "llama-server",
        "-m",
        str(svc["model"]),
        "-c",
        str(context),
        "-ngl",
        str(ngl),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if svc.get("mmproj"):
        args += ["--mmproj", str(svc["mmproj"])]
    if svc.get("flash_attn"):
        args += ["-fa", "on"]
    if svc.get("cache_type_k"):
        args += ["--cache-type-k", str(svc["cache_type_k"])]
    if svc.get("cache_type_v"):
        args += ["--cache-type-v", str(svc["cache_type_v"])]
    for rule in svc.get("override_tensor", []) or []:
        args += ["-ot", str(rule)]

    env: dict[str, str] = os.environ.copy()
    devices = svc.get("devices", {}) or {}
    gpu_allow = devices.get("gpu_allow")
    if gpu_allow:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_allow)
    return args, env


async def wait_healthy(
    base_url: str, timeout_s: float, proc: subprocess.Popen[bytes] | None = None
) -> float | None:
    """Poll `{base_url}/health` until it reports 200 or `timeout_s` elapses.
    If `proc` is passed, also bail out the moment the child exits — OOMs
    near the VRAM ceiling otherwise eat the full startup budget waiting
    on a dead socket.
    """
    start = time.monotonic()
    deadline = start + timeout_s
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.monotonic() < deadline:
            if proc is not None and proc.poll() is not None:
                return None
            try:
                resp = await client.get(f"{base_url}/health")
                if resp.status_code == 200:
                    return time.monotonic() - start
            except Exception:
                pass
            await asyncio.sleep(0.5)
    return None


async def fire_chat(base_url: str, timeout_s: float = 120.0) -> tuple[str, int | None]:
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            t0 = time.monotonic()
            resp = await client.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Reply with: hi."}],
                    "max_tokens": 16,
                    "stream": False,
                },
            )
            dt = int((time.monotonic() - t0) * 1000)
            return ("ok" if resp.status_code == 200 else f"err:{resp.status_code}", dt)
    except Exception as e:
        return (f"err:{type(e).__name__}", None)


# --- one iteration --------------------------------------------------------


async def measure_one(
    name: str,
    svc: dict[str, Any],
    context: int,
    estimate_bin: Path,
    *,
    startup_timeout_s: float,
    log_path: Path,
) -> Row:
    # active_devices=2 tells the estimate example to report GPU VRAM for
    # a dual-GPU redline box (the `gpu_vram_mib` output excludes CPU-
    # resident embeddings and caps cb device count at 2). That matches
    # what nvidia-smi sums to, so deltas track the real calibration signal.
    estimate = run_estimator(estimate_bin, svc, context, active_devices=2)
    architecture = str(estimate.get("architecture", ""))
    predicted_weights_mib = int(estimate.get("weights_bytes", 0)) // (1024 * 1024)
    predicted_kv_total_mib = int(estimate.get("kv_total_mib", 0))
    predicted_compute_mib = int(estimate.get("compute_buffer_mb", 0))
    # Prefer the GPU-only field when present; fall back to total_accounted
    # for older estimate-binary builds that don't emit it.
    predicted_total_mib = int(
        estimate.get("gpu_vram_mib", estimate.get("total_accounted_mib", 0))
    )

    def _err(reason: str) -> Row:
        return Row(
            service=name,
            context=context,
            architecture=architecture,
            predicted_weights_mib=predicted_weights_mib,
            predicted_kv_total_mib=predicted_kv_total_mib,
            predicted_compute_buffer_mib=predicted_compute_mib,
            predicted_total_mib=predicted_total_mib,
            actual_peak_total_mib=0,
            actual_peak_per_gpu="",
            actual_post_chat_total_mib=0,
            actual_post_chat_per_gpu="",
            delta_post_chat_total_mib=0,
            chat_status="skipped",
            chat_ms=None,
            time_to_healthy_s=None,
            error=reason,
        )

    if "estimator_error" in estimate:
        return _err(str(estimate["estimator_error"]))

    # Baseline NVML before spawning.
    baseline = nvml_used_mib()

    # `-ngl 999` offloads every layer to GPU; llama.cpp caps at the actual
    # layer count, so over-shooting is harmless. Using a fixed value keeps
    # the calibration comparable across services without threading the
    # packer's per-device decision into the measurement.
    ngl = 999
    port = free_port()
    argv, env = build_llama_server_argv(svc, context, port, ngl)
    base_url = f"http://127.0.0.1:{port}"

    log_fh = log_path.open("wb")
    proc = subprocess.Popen(
        argv,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )

    try:
        time_to_healthy = await wait_healthy(base_url, startup_timeout_s, proc)
        if time_to_healthy is None:
            if proc.poll() is not None:
                return _err(f"llama-server exited rc={proc.returncode} before healthy")
            return _err(f"never healthy within {startup_timeout_s}s")

        # Let KV + compute buffers stabilise before reading nvml.
        await asyncio.sleep(POST_HEALTH_PAUSE_S)
        peak = subtract_per_gpu(nvml_used_mib(), baseline)

        chat_status, chat_ms = await fire_chat(base_url)
        await asyncio.sleep(POST_CHAT_PAUSE_S)
        post_chat = subtract_per_gpu(nvml_used_mib(), baseline)

        actual_peak_total = sum(peak.values())
        actual_post_total = sum(post_chat.values())
        return Row(
            service=name,
            context=context,
            architecture=architecture,
            predicted_weights_mib=predicted_weights_mib,
            predicted_kv_total_mib=predicted_kv_total_mib,
            predicted_compute_buffer_mib=predicted_compute_mib,
            predicted_total_mib=predicted_total_mib,
            actual_peak_total_mib=actual_peak_total,
            actual_peak_per_gpu=format_per_gpu(peak),
            actual_post_chat_total_mib=actual_post_total,
            actual_post_chat_per_gpu=format_per_gpu(post_chat),
            delta_post_chat_total_mib=actual_post_total - predicted_total_mib,
            chat_status=chat_status,
            chat_ms=chat_ms,
            time_to_healthy_s=round(time_to_healthy, 2),
            error="",
        )
    finally:
        _terminate(proc)
        log_fh.close()
        # Give NVML a moment to reflect the freed VRAM for the next iteration.
        await asyncio.sleep(2.0)


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


# --- sweep ----------------------------------------------------------------


async def sweep_service(
    name: str,
    svc: dict[str, Any],
    estimate_bin: Path,
    workspace: Path,
    *,
    max_context: int | None,
    startup_timeout_s: float,
) -> list[Row]:
    rows: list[Row] = []
    ctx = INITIAL_CONTEXT
    while True:
        print(f"\n--- {name} @ context={ctx} ---", flush=True)
        log_path = workspace / f"{name}-{ctx}-llama-server.log"
        row = await measure_one(
            name,
            svc,
            ctx,
            estimate_bin,
            startup_timeout_s=startup_timeout_s,
            log_path=log_path,
        )
        rows.append(row)
        print(
            f"    predicted={row.predicted_total_mib} MiB "
            f"(w={row.predicted_weights_mib}+kv={row.predicted_kv_total_mib}+cb={row.predicted_compute_buffer_mib}) "
            f"actual_post_chat={row.actual_post_chat_total_mib} MiB ({row.actual_post_chat_per_gpu}) "
            f"delta={row.delta_post_chat_total_mib:+d} MiB "
            f"chat={row.chat_status}"
            + (f" error={row.error}" if row.error else ""),
            flush=True,
        )
        if row.error or row.chat_status.startswith("err"):
            print(f"    stopping sweep for {name}: {row.error or row.chat_status}", flush=True)
            break
        if max_context is not None and ctx >= max_context:
            print(f"    reached --max-context {max_context}", flush=True)
            break
        ctx *= 2
    return rows


def write_csv(path: Path, rows: list[Row]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dataclasses.asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(dataclasses.asdict(r))


# --- main -----------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> None:
    source_path = Path(args.source_config).expanduser()
    if not source_path.exists():
        sys.stderr.write(f"source config {source_path} does not exist\n")
        sys.exit(2)
    estimate_bin = Path(args.estimate_binary).expanduser()
    if not estimate_bin.exists() or not os.access(estimate_bin, os.X_OK):
        sys.stderr.write(
            f"estimate binary {estimate_bin} missing — build with:\n"
            f"  cargo build --example estimate\n"
        )
        sys.exit(2)

    source = source_path.read_text()
    blocks = llm_service_blocks(source)
    if args.services:
        missing = [s for s in args.services if s not in blocks]
        if missing:
            sys.stderr.write(
                f"requested services not found in {source_path}: {', '.join(missing)}\n"
            )
            sys.exit(2)
        targets = list(args.services)
    else:
        targets = list(blocks.keys())
    if not targets:
        sys.stderr.write("no llama-cpp services to sweep\n")
        return

    print(f"source config: {source_path}", flush=True)
    print(f"estimate binary: {estimate_bin}", flush=True)
    print(f"sweep targets ({len(targets)}): {targets}", flush=True)
    print(f"initial context: {INITIAL_CONTEXT}, max: {args.max_context or 'unbounded'}", flush=True)

    workspace = Path(args.workspace or f"/tmp/ananke-calib-{int(time.time())}")
    workspace.mkdir(parents=True, exist_ok=True)
    print(f"workspace: {workspace}", flush=True)

    all_rows: list[Row] = []
    timestamp = int(time.time())
    try:
        for name in targets:
            rows = await sweep_service(
                name,
                blocks[name],
                estimate_bin,
                workspace,
                max_context=args.max_context,
                startup_timeout_s=args.startup_timeout,
            )
            all_rows.extend(rows)
            checkpoint = HERE / f"calibration-{timestamp}-partial.csv"
            write_csv(checkpoint, all_rows)
            print(f"[checkpoint] {len(all_rows)} rows → {checkpoint}", flush=True)
    finally:
        final = HERE / f"calibration-{timestamp}.csv"
        write_csv(final, all_rows)
        print(f"\n{len(all_rows)} rows → {final}", flush=True)
        print(f"llama-server logs left at {workspace}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--estimate-binary",
        default=os.environ.get("ANANKE_ESTIMATE_BIN", str(DEFAULT_ESTIMATE_BIN)),
        help="path to the `estimate` example binary (build with: cargo build --example estimate)",
    )
    p.add_argument(
        "--source-config",
        default=os.environ.get("ANANKE_CONFIG", "/tmp/ananke-redline/config.toml"),
        help="config file to read service blocks from",
    )
    p.add_argument("--services", nargs="+", help="subset of llama-cpp service names to sweep")
    p.add_argument(
        "--max-context",
        type=int,
        default=None,
        help="stop doubling once this context has been measured (default: keep doubling until first failure)",
    )
    p.add_argument("--startup-timeout", type=float, default=600.0)
    p.add_argument("--workspace", type=str, default=None)
    return p.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
