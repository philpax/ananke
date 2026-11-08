The daemon embeds the frontend SPA as a fallback handler. Any request that doesn't match an `/api/*`, `/v1/*`, `/metrics`, or WebSocket route is served from the embedded `frontend/dist` directory.

The embedded assets are built at compile time via `rust-embed`. Set the `ANANKE_SKIP_FRONTEND_BUILD=1` environment variable during `cargo build` to skip the frontend build (useful for Rust-only dev loops).

The frontend reads live state (services, service detail, devices, the event history, and daemon info) through one `/api/live` WebSocket, loading over HTTP first and polling over HTTP every 5 seconds while the socket is down. Logs load their history over HTTP and then follow new lines over the same socket. Commands such as starting a service or saving the config go over HTTP.
