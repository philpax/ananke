import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  fullyParallel: true,
  use: {
    baseURL: process.env.ANANKE_UI_URL ?? "http://127.0.0.1:5173",
    viewport: { width: 1420, height: 900 },
    headless: true,
    trace: "retain-on-failure",
  },
  webServer: process.env.ANANKE_UI_URL
    ? undefined
    : {
        command: "npm run dev -- --port 5173",
        url: "http://127.0.0.1:5173",
        reuseExistingServer: true,
      },
});
