# Phase 2 manual smoke test

Real-hardware validation for the phase-2 additions: unified OpenAI
endpoint, on-demand lifecycle, start coalescing, filters, allocator,
management API.

## Prerequisites

Same as phase 1, plus:

- `jq` for inspecting JSON responses.
- An extra small GGUF to exercise routing by name.

## Steps

1. Build release:
   ```
   cargo build --release
   ```

2. Create `~/.config/ananke/config.toml`:
   ```toml
   [daemon]
   management_listen = "127.0.0.1:17777"
   data_dir = "/tmp/ananke-phase2"

   [openai_api]
   listen = "127.0.0.1:18080"

   [[service]]
   name = "small"
   template = "llama-cpp"
   model = "/mnt/ssd0/ai/llm/Qwen3-4B-Instruct-2507-UD-Q5_K_XL.gguf"
   port = 17435
   context = 4096
   flash_attn = true
   cache_type_k = "q8_0"
   cache_type_v = "q8_0"
   lifecycle = "on_demand"
   idle_timeout = "30s"
   devices.placement = "gpu-only"
   devices.placement_override = { "gpu:0" = 4500 }
   filters.set_params = { max_tokens = 64 }
   ```

3. Start: `LD_LIBRARY_PATH=/run/opengl-driver/lib ./target/release/ananke --config ~/.config/ananke/config.toml`

4. Verify unified OpenAI endpoint:
   - `curl http://127.0.0.1:18080/v1/models | jq` — includes `"id": "small"`.
   - First chat request triggers spawn:
     ```
     curl -s http://127.0.0.1:18080/v1/chat/completions \
       -H 'Content-Type: application/json' \
       -d '{"model":"small","messages":[{"role":"user","content":"say hi"}]}' | jq
     ```
     Response arrives; `max_tokens` is enforced via the filter.

5. Verify on-demand + idle_timeout:
   - After the first request completes, wait 35 seconds.
   - `nvidia-smi` — llama-server is gone.
   - Fire another chat request — starts fresh.

6. Verify start coalescing:
   - Wait 35s so service is idle.
   - Fire 5 concurrent `curl` POSTs in the background (e.g., `for i in 1 2 3 4 5; do curl -s ... & done; wait`). All should return 200; only one llama-server ever exists.

7. Verify management API:
   - `curl http://127.0.0.1:17777/api/services | jq`
   - `curl http://127.0.0.1:17777/api/services/small | jq`
   - `curl http://127.0.0.1:17777/api/devices | jq`
   - `curl http://127.0.0.1:17777/api/openapi.json | jq '.paths | keys'`

8. Verify unimplemented 501:
   - `curl -i http://127.0.0.1:18080/v1/audio/speech -X POST` — returns `501 Not Implemented` with `error.code = "not_implemented"`.

9. Verify insufficient_vram:
   - Edit config: `placement_override = { "gpu:0" = 90000 }` (90 GB).
   - Reload ananke.
   - Fire a chat request — returns 503 with `error.code = "insufficient_vram"`.

10. Clean shutdown: `kill -TERM $(pidof ananke)`.

Success criteria: every numbered step above produces the expected result. File a bug if anything drifts.
