# Phase 1 manual smoke test

Real-hardware validation outside CI. Run once before declaring phase 1 done.

## Prerequisites

- Linux host with at least one NVIDIA GPU and `nvidia-smi` working.
- `llama-server` on `$PATH` (build from llama.cpp `master` or install prebuilt binary).
- A small GGUF (`qwen3-4b-instruct-q5_k_xl.gguf` or similar) under `$HOME/models/`.

## Steps

1. Build release:
   ```
   cargo build --release
   ```

2. Create `~/.config/ananke/config.toml`:
   ```toml
   [daemon]
   management_listen = "127.0.0.1:7777"

   [[service]]
   name = "smoke"
   template = "llama-cpp"
   model = "~/models/qwen3-4b-instruct-q5_k_xl.gguf"
   port = 11435
   context = 4096
   flash_attn = true
   cache_type_k = "q8_0"
   cache_type_v = "q8_0"
   lifecycle = "persistent"
   devices.placement = "gpu-only"
   devices.placement_override = { "gpu:0" = 4500 }
   ```

3. Start: `./target/release/ananke --config ~/.config/ananke/config.toml`

4. Verify:
   - `nvidia-smi` shows one llama-server process with roughly 4.5 GB reserved.
   - `curl http://127.0.0.1:11435/v1/models` returns 200.
   - `curl http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"smoke","messages":[{"role":"user","content":"say hi"}]}'` returns a completion.

5. Orphan recovery:
   - `kill -9 $(pidof ananke)`
   - `ps aux | grep llama-server` — child still alive.
   - Restart ananke. Daemon logs `adopted orphan`.
   - `curl http://127.0.0.1:11435/v1/models` — still 200; no second llama-server started.

6. Clean shutdown:
   - `kill -TERM $(pidof ananke)`
   - Daemon logs drain; llama-server disappears from `ps`.

7. Database:
   - `sqlite3 ~/.local/share/ananke/ananke.sqlite 'select count(*) from service_logs'` — large positive number.
