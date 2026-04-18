# Phase 4 manual smoke test

Real-hardware validation for phase 4 additions: command template,
dynamic allocation, balloon resolver, priority-based eviction,
full drain pipeline, oneshots.

## Prerequisites

Same as phase 3, plus:

- ComfyUI installed and a known-good workflow.
- `jq` for inspecting JSON responses.

## Steps

1. Build release:
   ```
   cargo build --release
   ```

2. `~/.config/ananke/config.toml` (command + dynamic):
   ```toml
   [daemon]
   management_listen = "127.0.0.1:17777"
   data_dir = "/tmp/ananke-phase4"

   [openai_api]
   listen = "127.0.0.1:18080"

   [[service]]
   name = "comfy"
   template = "command"
   command = ["/path/to/comfyui-start", "--foreground"]
   port = 8188
   lifecycle = "on_demand"
   idle_timeout = "60s"
   allocation.mode = "dynamic"
   allocation.min_vram_gb = 4
   allocation.max_vram_gb = 20
   devices.placement = "gpu-only"
   devices.gpu_allow = [0]
   health.http = "/system_stats"
   metadata.openai_compat = false

   [[service]]
   name = "q3-4b"
   template = "llama-cpp"
   model = "/mnt/ssd0/ai/llm/Qwen3-4B-Instruct-2507-UD-Q5_K_XL.gguf"
   port = 11434
   context = 4096
   flash_attn = true
   cache_type_k = "q8_0"
   cache_type_v = "q8_0"
   lifecycle = "on_demand"
   priority = 70
   idle_timeout = "60s"
   devices.placement = "gpu-only"
   ```

3. Start: `LD_LIBRARY_PATH=/run/opengl-driver/lib ./target/release/ananke --config ~/.config/ananke/config.toml`

4. Trigger ComfyUI via a workflow POST. Verify `/api/services/comfy`
   shows `state=running`. Observe `nvidia-smi` GPU 0 usage.

5. Priority eviction: while ComfyUI is loaded, send a chat request to
   the llama-cpp service at priority 70. If the GPU is too full,
   ananke should evict ComfyUI (priority 50) and load q3-4b.
   Verify via logs and `/api/services/{name}` detail.

6. Oneshot: POST a short oneshot:
   ```
   curl -s -X POST http://127.0.0.1:17777/api/oneshot \
     -H 'Content-Type: application/json' \
     -d '{
       "template": "command",
       "command": ["/bin/sh", "-c", "echo hi; sleep 30"],
       "allocation": {"mode": "static", "vram_gb": 1},
       "priority": 40,
       "ttl": "10s"
     }' | jq
   ```
   Verify it runs, then expires via TTL. `GET /api/oneshot/{id}`
   reports terminal state.

7. DELETE an oneshot mid-run; verify clean drain via `ps`.

8. Balloon resolver: load a heavy ComfyUI workflow so VRAM climbs
   toward `max_vram_gb`. Submit an elastic borrower (another command
   service) and observe the fast-kill trigger when ComfyUI grows.

9. Clean shutdown: `kill -TERM $(pidof ananke)`. Verify full drain
   completes within the configured bounds.

Success criteria: every numbered step produces the expected result.
