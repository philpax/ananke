import { expect, test } from "@playwright/test";

import { hardware, mockBackend, referenceServices } from "./fixtures";

test("one socket carries backfill and updates; the dashboard sends no periodic reads or REST requests", async ({
  page,
}) => {
  const rest: string[] = [];
  page.on("request", (request) => {
    const path = new URL(request.url()).pathname;
    if (path.startsWith("/api/") || path.startsWith("/v1/")) rest.push(path);
  });
  const backend = await mockBackend(page, 2, {
    events: [
      {
        type: "state_changed",
        service: "qwen3.6-27b",
        from: "idle",
        to: "running",
        at_ms: Date.now(),
      },
    ],
  });
  await page.goto("/services/qwen3.6-27b");
  await expect(page.getByRole("log")).toContainText("server is listening");
  expect(backend.control.connectionCount).toBe(1);
  const initialReads = backend.control.messages.filter(
    (message) => message.type === "read",
  ).length;
  await page.clock.install();
  await page.clock.fastForward(60_000);
  expect(
    backend.control.messages.filter((message) => message.type === "read"),
  ).toHaveLength(initialReads);
  backend.control.publish("services", {
    services: referenceServices.map((service) => ({
      ...service,
      state: "idle",
    })),
    openai_api_port: 7070,
  });
  await expect(
    page.getByRole("button", { name: "Start", exact: true }),
  ).toBeEnabled();
  const devices = hardware(2);
  devices[0] = { ...devices[0], free_bytes: devices[0].total_bytes };
  backend.control.publish("devices", devices);
  await expect(
    page.locator("header").getByText("gpu:0", { exact: true }).locator(".."),
  ).toContainText("0%");
  await page
    .getByRole("navigation")
    .getByRole("button", { name: "Events", exact: true })
    .click();
  await expect(page.getByText("Idle → Running", { exact: true })).toBeVisible();
  await page
    .getByRole("navigation")
    .getByRole("button", { name: "Stats", exact: true })
    .click();
  await expect(page.getByRole("main")).toContainText("Fleet activity");
  expect(backend.control.connectionCount).toBe(1);
  expect(rest).toEqual([]);
});

test("reconnecting restores subscriptions and backfill without replaying an action", async ({
  page,
}) => {
  const backend = await mockBackend(page);
  await page.goto("/services/qwen3.6-27b");
  await page.getByRole("button", { name: "Stop", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Start", exact: true }),
  ).toBeEnabled();
  backend.control.disconnect();
  await expect.poll(() => backend.control.connectionCount).toBe(2);
  await expect(
    page.getByRole("button", { name: "Start", exact: true }),
  ).toBeEnabled();
  expect(
    backend.control.messages.filter((message) => message.type === "action"),
  ).toHaveLength(1);
  expect(
    backend.control.messages.filter(
      (message) => message.type === "read" && message.query.type === "services",
    ),
  ).toHaveLength(2);
  await expect(page.getByRole("log")).toContainText("server is listening");
});

test("a lost action acknowledgement reports uncertainty and restores current state without replay", async ({
  page,
}) => {
  const backend = await mockBackend(page);
  await page.goto("/services/qwen3.6-27b");
  await expect(
    page.getByRole("button", { name: "Stop", exact: true }),
  ).toBeEnabled();
  backend.control.dropNextAction();
  await page.getByRole("button", { name: "Stop", exact: true }).click();
  await expect(
    page.getByRole("alert").filter({ hasText: "before acknowledgement" }),
  ).toBeVisible();
  await expect.poll(() => backend.control.connectionCount).toBe(2);
  await expect(
    page.getByRole("button", { name: "Start", exact: true }),
  ).toBeEnabled();
  expect(
    backend.control.messages.filter((message) => message.type === "action"),
  ).toHaveLength(1);
});

test("log backfill fills a gap larger than one page using only socket reads", async ({
  page,
}) => {
  const backend = await mockBackend(page);
  await page.goto("/services/qwen3.6-27b");
  await expect(page.getByRole("log")).toContainText("server is listening");
  const line = (seq: number) => ({
    timestamp_ms: Date.now() - 1000 + seq,
    stream: "stdout",
    line: `captured line ${seq}`,
    run_id: 1,
    seq,
  });
  backend.control.logPage("older", {
    logs: Array.from({ length: 200 }, (_, index) => line(index + 100)),
    next_cursor: "oldest",
  });
  backend.control.logPage("oldest", {
    logs: Array.from({ length: 99 }, (_, index) => line(index + 1)),
    next_cursor: null,
  });
  backend.control.publish("logs", {
    logs: Array.from({ length: 200 }, (_, index) => line(index + 300)),
    next_cursor: "older",
  });
  await expect(page.getByRole("log").locator(":scope > div")).toHaveCount(499);
  await expect(page.getByRole("log")).toContainText("captured line 2");
  await expect(page.getByRole("log")).toContainText("captured line 499");
  expect(
    backend.control.messages.filter(
      (message) =>
        message.type === "read" &&
        message.query.type === "logs" &&
        !message.subscribe,
    ),
  ).toHaveLength(2);
  expect(backend.control.connectionCount).toBe(1);
});

test("editing a long configuration keeps the hardware header at the top of the viewport", async ({
  page,
}) => {
  await mockBackend(page);
  await page.goto("/services/qwen3.6-27b");
  await page
    .getByRole("navigation")
    .getByRole("button", { name: "Config", exact: true })
    .click();
  await page.getByRole("button", { name: "Edit", exact: true }).click();
  const draft =
    "# Long configuration\n" +
    Array.from({ length: 1500 }, (_, index) => `# setting ${index}\n`).join(
      "",
    ) +
    "[daemon]\n";
  await page.getByRole("textbox", { name: "Configuration TOML" }).fill(draft);
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(
    page.getByRole("status").filter({ hasText: "Configuration saved." }),
  ).toBeVisible();
  await page
    .getByRole("navigation")
    .getByRole("button", { name: "Services", exact: true })
    .click();
  await page
    .getByRole("main", { name: "Service qwen3.6-27b" })
    .getByRole("button", { name: "Stop", exact: true })
    .click({ trial: true });
  const layout = await page.evaluate(() => ({
    scroll: window.scrollY,
    top: document.querySelector("header")?.getBoundingClientRect().top,
    bottom: document.querySelector("#root")?.getBoundingClientRect().bottom,
    height: window.innerHeight,
  }));
  expect(layout.scroll).toBe(0);
  expect(layout.top).toBe(0);
  expect(layout.bottom).toBe(layout.height);
});
