import { defineConfig } from "vite";
import react, { reactCompilerPreset } from "@vitejs/plugin-react";
import babel from "@rolldown/plugin-babel";
import tailwindcss from "@tailwindcss/vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    babel({ presets: [reactCompilerPreset()] }),
    tailwindcss(),
  ],
  // In dev, proxy `/api/*` and the WebSocket `/api/events` to the
  // daemon's management listener so the frontend can use same-origin
  // fetches without CORS plumbing. The OpenAI API (port 7070) has CORS
  // enabled, so the chat feature talks to it directly. Override the
  // target via `ANANKE_ENDPOINT` (same env var as `anankectl`). In
  // production the daemon serves the built assets from the same origin
  // and the proxy is a no-op.
  server: {
    proxy: {
      "/api": {
        target: process.env.ANANKE_ENDPOINT ?? "http://127.0.0.1:7071",
        changeOrigin: true,
        ws: true,
      },
      "/metrics": {
        target: process.env.ANANKE_ENDPOINT ?? "http://127.0.0.1:7071",
        changeOrigin: true,
      },
    },
  },
});
