# Phase 3 manual smoke test

Real-hardware validation for the phase-3 additions: GGUF estimator,
layer-aware placement, rolling correction, OOM retry, and removal of
the `placement_override` requirement.

## Prerequisites

Same as phase 2, plus:

- At least one real GGUF on disk for each architecture family you want
  to validate (llama-family, MoE).

## Steps

1. Build release:
   ```
   cargo build --release
   ```

2. `~/.config/ananke/config.toml` (no `placement_override`):
   ```toml
   [daemon]
   management_listen = "127.0.0.1:17777"
   data_dir = "/tmp/ananke-phase3"

   [openai_api]
   listen = "127.0.0.1:18080"

   [[service]]
   name = "q3-30b-moe"
   template = "llama-cpp"
   model = "/mnt/ssd0/ai/llm/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf"
   port = 17436
   context = 8192
   flash_attn = true
   cache_type_k = "q8_0"
   cache_type_v = "q8_0"
   lifecycle = "on_demand"
   idle_timeout = "60s"
   ```

3. Start: `LD_LIBRARY_PATH=/run/opengl-driver/lib ./target/release/ananke --config ~/.config/ananke/config.toml`

4. First chat request triggers estimator + placement + spawn:
   ```
   curl -s http://127.0.0.1:18080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"q3-30b-moe","messages":[{"role":"user","content":"hi"}]}' | jq
   ```
   Verify it succeeds. Compare `nvidia-smi` output against
   `curl http://127.0.0.1:17777/api/devices | jq` reservation totals.

5. After drain (30 s idle then next request), call
   `curl http://127.0.0.1:17777/api/services/q3-30b-moe | jq`
   and verify `rolling_mean` is populated and `observed_peak_bytes > 0`.

6. Sharded test — swap in the `Qwen3-235B-A22B-Instruct` config pointing
   at the shard-0 file; verify daemon aggregates all shards and places
   across GPU 0, GPU 1, and CPU. The `/api/devices` response should
   show CPU reservation ≥ 80 GB.

7. Manual OOM test — edit the config to demand e.g. `context = 65536` on
   a model that won't fit; send a chat request. Expect 503
   `insufficient_vram` (from allocator) OR, if it squeaks past the
   allocator, observe an actual OOM kill within 30 s of spawn; ananke
   should bump safety_factor and retry once.

8. Repeat the same chat request 5+ times; verify `rolling_mean` moves
   toward 1.0 (converging on reality) in the management API response.

9. Clean shutdown: `kill -TERM $(pidof ananke)`.

Success criteria: every numbered step above produces the expected
result. File a bug for anything that drifts.
