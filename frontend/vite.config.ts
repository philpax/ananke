import { defineConfig } from "vite";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [tailwindcss()],
  server: {
    host: "127.0.0.1",
    proxy: {
      "/api": {
        target: process.env.ANANKE_ENDPOINT ?? "http://127.0.0.1:7071",
        ws: true,
      },
      "/v1": {
        target: process.env.ANANKE_ENDPOINT ?? "http://127.0.0.1:7071",
        changeOrigin: true,
      },
    },
  },
});
