The daemon embeds the frontend SPA as a fallback handler. Any request that doesn't match an `/api/*`, `/v1/*`, `/metrics`, or WebSocket route is served from the embedded `frontend/dist` directory.

The embedded assets are built at compile time via `rust-embed`. Set the `ANANKE_SKIP_FRONTEND_BUILD=1` environment variable during `cargo build` to skip the frontend build (useful for Rust-only dev loops).
