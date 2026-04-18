# Phase 3: GGUF reader + estimator + layer-aware placement — design

Status: accepted 2026-04-18. Parent design: `docs/spec.md` (authoritative). Prior phases: phase-1 (`docs/superpowers/specs/2026-04-18-ananke-phase-1-lean-mvp-daemon.md`), phase-2 (`docs/superpowers/specs/2026-04-18-ananke-phase-2-unified-openai-ondemand-allocator.md`).

## 1. Goal

Replace the phase 1-2 hard requirement that every service declare `placement_override` with an architecture-aware VRAM estimator that reads GGUF headers and decides placement automatically. Once a service runs, rolling correction tunes the estimate from observed usage; OOM triggers a one-shot safety_factor bump. After this phase, the redline lmp config — including the Qwen3-235B hybrid and GLM-4.5-Air — can be imported into ananke without hand-computed memory overrides.

Non-goals for this phase: eviction, dynamic/command template, oneshots, full management-API mutations, WebSocket events, `anankectl`, frontend, Nix module.

## 2. Scope

### 2.1 In scope

**GGUF reader** (`src/gguf/`):

- Parse GGUF v3 header (magic, version, tensor count, kv count).
- Enumerate tensors: name, type, shape, offset, byte size.
- Enumerate metadata key-value pairs as typed `GgufValue`.
- Handle sharded GGUFs per spec §8.3: read shard 0 header, follow `split.count` and `split.tensors.count`; walk remaining shards summing tensor bytes.
- Detect shard-0 from any shard via `split.no` + the conventional `{basename}-{NNNNN}-of-{MMMMM}.gguf` naming.
- **Custom 285-line reader, not the `gguf` crate.** The `gguf` 0.1.2 crate has a stale type table (last update covers only through pre-IQ quants) and hard-fails on BF16, IQ2/IQ3/IQ4/IQ1 variants, and has wrong discriminants for I8/I16/I32. `llama-gguf` pulls a full inference engine; `gguf-utils` / `gguf-rs-lib` have undocumented coverage. Our 285-line reader encodes the current GGML type table exactly and is the safe baseline until one of those crates catches up.

**Architecture-aware estimator** (`src/estimator/`), dispatched on `general.architecture`:

- **Llama-family** (`llama`, `qwen2`, `qwen3`, `mistral`, `gemma`, `gemma2`, `gemma3`, `phi3`, `glm4`): per-layer tensor bytes plus KV cost from `<arch>.attention.head_count_kv`, `key_length`, `value_length`.
- **MoE variants** (`llama4`, `qwen3moe`, `deepseek2`, `mixtral`, `gpt-oss`): identify expert tensors by `_exps` suffix on `blk.N.ffn_{gate,up,down}_exps.weight`. `n_cpu_moe > 0` offloads the top-N expert-bearing layers to CPU.
- **SSM/Mamba** (`mamba`): state cost from `mamba.ssm.state_size`, `conv_kernel`, `inner_size`. `flash_attn` and `cache_type_*` reject at validation for this architecture (already enforced by phase 1 §6.5 surface).
- **Hybrid** (`jamba`): per-layer type read from metadata where available; KV cache only for attention layers.
- **Unknown fallback**: `total_tensor_bytes × 1.15 + 512 MB`, with a `tracing::warn` so the user knows estimates are coarse.
- **`mmproj`**: add projector tensor bytes to GPU 0's weights budget (per paragraph added to main spec in phase 1).
- **Sharded**: sum tensor bytes across all shards; architecture metadata is read from shard 0.
- Final: `weights + kv_per_token × context + compute_buffer_mb × 1MB`, then × `safety_factor` (default 1.1).
- KV byte table per spec §8.3, including q8_0 (~1.06), q5_1 (~0.75), q4_0 (~0.56), etc.

**Layer-aware placement** (`src/placement.rs`, spec §8.2):

- Walk layers in index order; pack onto first allowed device with room, preferring GPUs.
- Non-layer tensors: output head on GPU 0, token embeddings on CPU (always, even in `gpu-only`), other small non-layer tensors on GPU 0 if any GPU used else CPU.
- Compute buffer per active backend (default 400 MB per device that hosts any tensor).
- Per-device one-layer fudge for tensor-split slop (GPUs and CPU when hybrid is in play).
- `override_tensor` rules applied first (before layer walker); matched tensors attributed to declared device; layer walker packs only residual.
- MoE × hybrid composition order: `override_tensor` → `n_cpu_moe` → layer walker.
- Render llama.cpp args: `-ngl N`, `--tensor-split A,B,...` (normalised layer counts in CUDA_VISIBLE_DEVICES-remapped order), `-ot <regex>=<device>` for each user override rule.

**Rolling correction** (`src/rolling.rs`, spec §8.3):

- Per-service `RollingCorrection { rolling_mean: f64, sample_count: u32 }` held in memory; persistence to SQLite is deferred.
- Observed peak captured from the snapshotter's per-PID NVML data (extended to record peak-across-run) plus CPU `VmRSS` from `/proc/<pid>/status` at the same 2s cadence.
- On drain / service exit: update `rolling_mean ← clamp((rolling_mean × (n-1) + observed_peak / base_estimate) / n, 0.8, 1.5)`, then `n += 1`.
- Next start's `adjusted_estimate = base_estimate × rolling_mean`.
- Drift warnings:
  - `rolling_mean > 1.2` sustained → under-estimation warning (logs now; future `/api/events` in phase 5).
  - `rolling_mean < 0.85` sustained → over-reservation warning.
  - `|rolling_mean - 1.0| > 0.3` sustained across ≥5 runs → `estimator_drift` event (logged now; enqueued on the events bus in phase 5).

**OOM retry policy** (spec §8.5):

- If a service OOMs (child SIGKILLs with OOM signal, or exits within 30 s of start) → retry once with `safety_factor` bumped 1.1 → 1.4.
- Second OOM → `Disabled { Oom }` with observed-vs-reserved delta logged.
- Observed-vs-reserved captured from the snapshot at the moment of exit.

**CPU-side observation** (spec §8.3):

- Snapshotter samples `/proc/<pid>/status` `VmRSS` for every known child PID on each 2 s tick.
- Contributes to the observed-peak accounting used by rolling correction whenever a service has a CPU portion (hybrid, cpu-only, or `-ot '...=CPU'` rule in play).

**`placement_override` path** (spec §8.2.6):

- Validator stops requiring `placement_override`; it is now fully optional.
- When set, skip estimator; reserve exactly the declared MB per slot; disable `-ngl`/`--tensor-split` derivation; pass only the user's explicit `extra_args`.
- Rolling correction still runs; warns if observed deviates >30 % from the declared values.

### 2.2 Out of scope (still deferred)

- Eviction / priority-based displacement → phase 4.
- Dynamic / balloon / `command` template / oneshots → phase 4.
- POST/PUT management endpoints, `/api/events` WebSocket → phase 5.
- `anankectl` CLI → phase 5.
- Frontend → phase 6.
- Nix module → phase 7.
- Persistent rolling correction across daemon restarts → phase 5 (small SQLite table).

## 3. Architecture

### 3.1 Per-start flow

```
  config.toml
     |
     v
  parse/merge/validate   ->   EffectiveConfig (placement_override optional)
     |
     v
  At Idle -> Starting:
     1. gguf::read(model_path)          -> GgufSummary (tensors, metadata, mmproj)
     2. estimator::estimate(summary)    -> Estimate { weights_bytes, kv_per_token, compute_buffer }
     3. placement::pack(estimate,       -> Allocation (per-device bytes)
                         capacities,       + CommandArgs (-ngl, --tensor-split, -ot)
                         policy)
     4. allocator::can_fit(alloc)      -> ok or no-fit
     5. spawn with argv rendered from CommandArgs
```

### 3.2 During Running

```
  snapshotter's 2 s tick samples per-PID NVML VRAM + /proc/<pid>/status VmRSS.
  observation::Aggregator tracks per-service peak bytes across run.

  At Running -> Idle / Draining:
     rolling::update(service, peak, base_estimate)
       rolling_mean adjusted and stored in AppState.

  At spawn failure (OOM within 30 s):
     if retry_count == 0: safety_factor 1.1 -> 1.4, re-estimate, re-spawn.
     else: Disabled { Oom } with delta logged.
```

## 4. Module layout

```
src/
├── gguf/
│   ├── mod.rs            // pub: read, GgufSummary
│   ├── reader.rs         // single-file header + tensor table + kv map
│   ├── shards.rs         // multi-shard sum, shard-0 discovery
│   └── types.rs          // GgufValue, GgufTensor, GgufSummary
├── estimator/
│   ├── mod.rs            // dispatch on general.architecture
│   ├── types.rs          // Estimate, PerDeviceReserve, SafetyFactor
│   ├── llama.rs          // llama-family
│   ├── moe.rs            // MoE variants + _exps identification
│   ├── mamba.rs          // SSM/Mamba
│   ├── hybrid.rs         // jamba and future mixed architectures
│   ├── fallback.rs       // unknown architecture
│   └── kv.rs             // kv-bytes-per-element table + helpers
├── placement.rs          // layer walker + tensor-split ratios + -ot rendering
├── rolling.rs            // per-service RollingCorrection map in AppState
└── observation.rs        // /proc/<pid>/status + per-PID NVML aggregator
```

**Modified:**

- `src/config/validate.rs` — remove the `placement_override` required gate; keep it accepted as an escape hatch.
- `src/supervise/mod.rs` — `idle → starting` calls estimator + placement, feeds result into allocator + argv.
- `src/supervise/spawn.rs` — accept `CommandArgs { ngl, tensor_split, override_tensor_rules }` and render the corresponding flags.
- `src/snapshotter.rs` — additionally sample `/proc/<pid>/status` VmRSS for every registered child; surface per-service observed peaks in the shared snapshot.
- `src/app_state.rs` — add `rolling: Arc<RwLock<BTreeMap<SmolStr, RollingCorrection>>>`.
- `src/daemon.rs` — build + pass the rolling table; thread `observation` into the snapshotter.

## 5. Data flow, key design calls

- **Estimator result cached per service per `mtime(model_file)`**: re-reads on GGUF change, not on every start. Cache in-memory; lost on daemon restart — fine for phase 3.
- **`placement_override` bypass is deep**: when set, `-ngl` / `--tensor-split` are not derived; the user owns everything via `extra_args`. Validation enforces consistency (`gpu-only` with CPU bytes = error; already enforced from phase 1).
- **Rolling correction lives in `AppState`**: `Arc<RwLock<BTreeMap<ServiceName, RollingCorrection>>>`. Read at start, written on drain. In-memory per the scope.
- **OOM detection**: child exit within 30 s of spawn AND status code indicating SIGKILL (the kernel OOM-killer uses SIGKILL), OR the phase 1 log pump captures an `out of memory` stderr line. Approximate; false positives trigger a single redundant retry, which is acceptable.
- **GGUF cache locality**: for multi-shard models, only shard 0 is fully re-read on every start; other shards' tensor counts are cached.
- **Compute buffer scales per active backend**: a 2-GPU + CPU hybrid deployment budgets three 400 MB buffers, not one. Per spec §8.2.2.
- **`override_tensor` + `n_cpu_moe` + layer walker order** is enforced in `placement.rs` as three sequential passes; tests cover composition.
- **Tensor-split ratios are in CUDA_VISIBLE_DEVICES-remapped order**: the list passed to llama.cpp matches the renumbered 0, 1, ... set; not the NVML ids.
- **Non-layer tensor attribution is deterministic**: output head → GPU 0 (or first allowed GPU), token embeddings → CPU (always), other small non-layers → GPU 0 if any GPU used else CPU. Matches llama.cpp's actual behaviour.
- **Context default**: `context` becomes effectively required for estimation; if missing, default to `4096` and log a warning. Still overridable per service.

## 6. Testing

### 6.1 Unit

- `gguf` fixtures: hand-crafted tiny GGUF byte streams for parser tests. Round-trip a synthesised header.
- `estimator::llama`: fixture for a tiny Qwen3-shaped model; weights + KV computation within ±5 % of hand calculation.
- `estimator::moe`: fixture with `_exps`-suffixed tensors; n_cpu_moe offload moves the right bytes.
- `estimator::mamba`: fixture with `mamba.ssm.*` metadata.
- `estimator::hybrid`: jamba-shaped fixture.
- `estimator::fallback`: unknown architecture → `total × 1.15 + 512 MB`, warning emitted.
- `placement::pack`: property tests over (layer costs, capacities); invariants: contiguous per device, no overcommit, output head on first GPU, token embeddings on CPU, compute buffer per active backend.
- `placement::tensor_split`: asserts normalised ratios match placed layer counts and match llama.cpp semantics.
- `rolling::update`: clamps to [0.8, 1.5]; drift warnings fire correctly.
- `observation`: `/proc/<pid>/status` parser.

### 6.2 Integration

- `estimator_llama.rs` — real tiny llama-family GGUF fixture, harness builds `Estimate`, asserts within ±10 % of a known-good hand calculation.
- `placement_override_bypass.rs` — service with `placement_override` set; estimator is NOT called, allocator reserves exactly the declared bytes, no `-ngl`/`--tensor-split` emitted.
- `no_placement_override_ok.rs` — service WITHOUT `placement_override`; daemon estimates and places; request succeeds.
- `rolling_correction_warns.rs` — synthesise observed peak 1.5× base; rolling mean crosses 1.2; warning logged.
- `oom_retry.rs` — test-only child that exits nonzero within 30 s; first retry bumps safety_factor to 1.4; second OOM disables.
- `sharded_gguf.rs` — 2-shard fixture; reader enumerates all tensors.
- `multi_gpu_split.rs` — FakeProbe with 2 GPUs, model too large for one; placement produces `--tensor-split` with correct ratios.
- `mmproj_attribution.rs` — vision fixture; GPU 0 weights budget includes projector bytes.
- `override_tensor_moe_hybrid.rs` — `override_tensor=[".ffn_(up|down)_exps.=CPU"]` on a MoE fixture; verifies composition order and final argv.

### 6.3 Manual smoke runbook (`tests/manual/phase-3-smoke.md`)

- Real `qwen3-30b-a3b-instruct` (MoE) on redline without `placement_override`. Ananke estimates, places, spawns. Compare `nvidia-smi` VRAM against the estimate.
- Real `Qwen3-235B-A22B-Instruct` sharded hybrid: daemon sums shards, places across both GPUs + CPU.
- Under-estimated config: watch OOM path trigger retry + bump safety_factor, then disable on second OOM.
- Run the same service 5+ times: rolling correction visibly adjusts `adjusted_estimate`; inspect via `/api/services/{name}`.

## 7. Risks

1. **`gguf` crate may not expose tensor-table.** Mitigation: budget one task for a custom ~200-line reader.
2. **First-run estimates are coarse** (±20 % is realistic). Rolling correction converges within 2-3 runs; OOM retry handles the under-estimation case. No calibration matrix pre-release per spec §18.
3. **Per-pid observation cost is negligible** (reading /proc every 2 s for <20 PIDs).
4. **Multi-shard discovery race**: if the user renames shards while ananke is starting, we fail. Keep the sum-at-start pattern; don't attempt re-discovery.
5. **MoE × hybrid composition order errors**: covered by `override_tensor_moe_hybrid.rs` integration test.
6. **Context default**: `4096` fallback is safe for most models but low for the user's `qwen3-235b` (configured at 16384). Warning is important.

## 8. Success criteria

- Config with no `placement_override` spawns a llama-cpp service correctly (estimator picks placement; allocator admits).
- Sharded 235B config loads without hand-computed overrides.
- Rolling correction visibly tunes the estimate across successive runs; exposed in `/api/services/{name}` as a `rolling_mean: f64` additive field.
- OOM retry path recovers under deliberate under-estimation.
- All prior-phase behaviours (on-demand, coalescing, idle_timeout, management API, openapi.json) continue working.
- `just lint` and `cargo test --workspace --features test-fakes` both pass.
