import { expect, test } from "@playwright/test";

import { mockBackend } from "./fixtures";

test("selection, favourites, sorting, resizing, hide/show, sticky header, and compact back flow", async ({
  page,
}) => {
  await mockBackend(page);
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Select a service" }),
  ).toBeVisible();
  await expect(
    page.getByRole("separator", { name: "Resize service list" }),
  ).toHaveAttribute("aria-valuenow", "324");
  await page
    .getByRole("button", { name: "Add qwen3.6-27b to favourites", exact: true })
    .click();
  await expect(
    page.getByRole("button", {
      name: "Remove qwen3.6-27b from favourites",
      exact: true,
    }),
  ).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("textbox", { name: "Search services" }).fill("jina");
  await expect(
    page.getByRole("button", {
      name: /^jina-embeddings-v5-text-small-retrieval-vllm Copy/,
    }),
  ).toBeVisible();
  await page.getByRole("textbox", { name: "Search services" }).fill("");
  await page.getByRole("button", { name: "Size", exact: true }).click();
  await page.getByRole("button", { name: /^qwen3.6-27b Copy/ }).click();
  await expect(
    page.getByRole("heading", { name: "qwen3.6-27b", exact: true }),
  ).toBeVisible();
  const separator = page.getByRole("separator", {
    name: "Resize service list",
  });
  await separator.focus();
  await page.keyboard.press("ArrowRight");
  await expect(separator).toHaveAttribute("aria-valuenow", "344");
  const box = await separator.boundingBox();
  if (!box) throw new Error("Missing resize handle");
  await page.mouse.move(box.x + 5, box.y + 100);
  await page.mouse.down();
  await page.mouse.move(box.x + 85, box.y + 100);
  await page.mouse.up();
  await expect(separator).toHaveAttribute("aria-valuenow", "424");
  await page
    .locator("main[data-screen-label=Services]")
    .evaluate((el) => (el.scrollTop = 1600));
  await expect
    .poll(async () =>
      Math.round(
        (await page.locator("[data-service-header]").boundingBox())?.y ?? -1,
      ),
    )
    .toBe(88);
  await page.getByRole("button", { name: "Hide service list" }).click();
  await expect(
    page.getByRole("textbox", { name: "Search services" }),
  ).toBeHidden();
  await page.getByRole("button", { name: /Browse services/ }).click();
  await expect(
    page.getByRole("textbox", { name: "Search services" }),
  ).toBeVisible();
  await page.setViewportSize({ width: 880, height: 900 });
  await expect(
    page.getByRole("button", { name: "All services", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "All services", exact: true }).click();
  await expect(
    page.getByRole("textbox", { name: "Search services" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Hardware resources" }).click();
  await expect(
    page.getByRole("button", { name: "Hardware resources" }),
  ).toHaveAttribute("aria-expanded", "true");
});

test("service lifecycle, ranges, chart resize, copy feedback, and logs", async ({
  page,
  context,
}) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  const calls = await mockBackend(page);
  await page.goto("/services/qwen3.6-27b");
  await page.getByRole("button", { name: "Stop", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Start", exact: true }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "Start", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Stop", exact: true }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "Restart", exact: true }).click();
  await expect
    .poll(() => calls.filter((c) => c.url.endsWith("/restart")).length)
    .toBe(1);
  await page.getByRole("button", { name: "Disable", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Enable", exact: true }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "Enable", exact: true }).click();
  await page.getByRole("button", { name: "24h", exact: true }).click();
  await expect
    .poll(() =>
      calls.some(
        (c) =>
          c.url.includes("/metrics?") &&
          Number(new URL("http://local" + c.url).searchParams.get("until")) -
            Number(
              new URL("http://local" + c.url).searchParams.get("since"),
            ) ===
            86_400_000,
      ),
    )
    .toBe(true);
  const to = await page.getByLabel("To date and time").inputValue();
  await page.getByLabel("From date and time").fill(to);
  await page.getByRole("button", { name: "Apply", exact: true }).click();
  await expect(
    page.getByRole("status").filter({ hasText: "End must be after start." }),
  ).toBeVisible();
  await page.getByRole("button", { name: "6h", exact: true }).click();
  await page
    .locator("[data-service-header]")
    .getByRole("button", { name: "Copy qwen3.6-27b to clipboard", exact: true })
    .click();
  await expect(
    page.locator("[data-service-header]").getByRole("button", {
      name: "Copied qwen3.6-27b to clipboard",
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("log", { name: "Service log viewer" }),
  ).toContainText("server is listening");
  await expect(
    page.locator("svg[aria-label='Requests over the selected time range']"),
  ).toBeVisible();
  await page.setViewportSize({ width: 1920, height: 900 });
  await expect
    .poll(async () => {
      const svg = page.locator(
        "svg[aria-label='Requests over the selected time range']",
      );
      return (await svg.boundingBox())?.width ?? 0;
    })
    .toBeGreaterThan(300);
  await expect(page.locator("input[type=range]")).toHaveCount(0);
  expect(calls.some((c) => c.url.includes("oneshot"))).toBe(false);
});

test("chat roles, system prompt, stream usage, code copying, new chat, events, stats, and config writes", async ({
  page,
}) => {
  const calls = await mockBackend(page);
  await page.goto("/chat?model=qwen3.6-27b");
  await page.getByText("System prompt", { exact: true }).click();
  await page
    .getByRole("textbox", { name: "System prompt" })
    .fill("Be concise.");
  await page
    .getByRole("textbox", { name: "Message", exact: true })
    .fill("Hello");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "You", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "qwen3.6-27b", exact: true }),
  ).toBeVisible();
  await expect(page.getByText("12 in · 5 out · 5 tok/s")).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Copy code to clipboard" }),
  ).toBeVisible();
  expect(calls.find((c) => c.url === "/v1/chat/completions")?.body).toContain(
    "Be concise.",
  );
  await page.getByRole("button", { name: "New chat", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "You", exact: true }),
  ).toHaveCount(0);
  for (const name of ["Events", "Stats", "Config"]) {
    await page
      .getByRole("navigation", { name: "Sections" })
      .getByRole("button", { name, exact: true })
      .click();
    await expect(
      page.getByRole("heading", { name, exact: true }),
    ).toBeVisible();
  }
  await page.getByRole("button", { name: "Edit", exact: true }).click();
  const editor = page.getByRole("textbox", { name: "Configuration TOML" });
  await editor.fill(
    '[daemon]\nmanagement_listen = "127.0.0.1:7071"\n# changed',
  );
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(
    page.getByRole("status").filter({ hasText: "Configuration saved." }),
  ).toBeVisible();
  expect(
    calls.some((c) => c.method === "PUT" && c.body?.includes("changed")),
  ).toBe(true);
});

test("approved section composition, local fonts, dark chart tips, and browser diagnostics", async ({
  page,
}, testInfo) => {
  const errors: string[] = [],
    remote: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error" || message.type() === "warning")
      errors.push(message.text());
  });
  page.on("request", (request) => {
    if (!new URL(request.url()).hostname.match(/^(127\.0\.0\.1|localhost)$/))
      remote.push(request.url());
  });
  await mockBackend(page);
  await page.goto("/");
  await page.evaluate(() => document.fonts.ready);
  await expect(
    page.getByRole("heading", { name: "Select a service" }),
  ).toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("services-empty-1420.png"),
  });
  await page.getByRole("button", { name: /^qwen3.6-27b Copy/ }).click();
  await expect(
    page.getByRole("heading", { name: "Placement", exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("log")).toContainText("server is listening");
  await page.screenshot({
    path: testInfo.outputPath("services-running-1420.png"),
  });
  const plot = page.locator(
    "svg[aria-label='Requests over the selected time range']",
  );
  await plot.scrollIntoViewIfNeeded();
  await plot.locator("circle").filter({ visible: true }).first().hover();
  await expect(plot.locator("g[aria-label=tip]")).toHaveAttribute(
    "fill",
    "#202126",
  );
  await expect(plot.locator("g[aria-label=tip]")).toHaveAttribute(
    "stroke",
    "#777985",
  );
  for (const name of ["Chat", "Events", "Stats", "Config"]) {
    await page
      .getByRole("navigation", { name: "Sections" })
      .getByRole("button", { name, exact: true })
      .click();
    await expect(
      page.getByRole("heading", { name, exact: true }),
    ).toBeVisible();
    if (name === "Stats")
      await expect(
        page.locator("svg[aria-label*='over the selected time range']"),
      ).toHaveCount(6);
    if (name === "Config")
      await expect(
        page.getByRole("textbox", { name: "Configuration TOML" }),
      ).toBeVisible();
    await page.screenshot({
      path: testInfo.outputPath(`${name.toLowerCase()}-1420.png`),
    });
  }
  await page.setViewportSize({ width: 880, height: 900 });
  await page
    .getByRole("navigation", { name: "Sections" })
    .getByRole("button", { name: "Services", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "All services", exact: true }),
  ).toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("services-compact-880.png"),
  });
  expect(errors).toEqual([]);
  expect(remote).toEqual([]);
});

test("configuration validation, navigation persistence, and concurrent-edit protection", async ({
  page,
}) => {
  const calls = await mockBackend(page);
  await page.goto("/config");
  await page.getByRole("button", { name: "Edit", exact: true }).click();
  const editor = page.getByRole("textbox", { name: "Configuration TOML" });
  await editor.fill("invalid");
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(
    page
      .getByRole("alert")
      .filter({ hasText: "Line 1, column 1: Invalid TOML." }),
  ).toBeVisible();
  await expect(editor).toBeEditable();
  await editor.press("ControlOrMeta+A");
  await editor.press("Backspace");
  await expect(editor).toHaveText("");
  await editor.fill("# pending draft\n[daemon]");
  await expect(editor).toHaveText("# pending draft[daemon]");
  await page
    .getByRole("navigation")
    .getByRole("button", { name: "Events", exact: true })
    .click();
  await page
    .getByRole("navigation")
    .getByRole("button", { name: "Config", exact: true })
    .click();
  await expect(editor).toHaveText("# pending draft[daemon]");
  calls.control.conflict();
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(
    page.getByRole("alert").filter({ hasText: "file changed on the server" }),
  ).toBeVisible();
  await expect(editor).toHaveText("# pending draft[daemon]");
});

test("captured events filter correctly and service links select a detail", async ({
  page,
}) => {
  await mockBackend(page, 2, {
    events: [
      {
        type: "state_changed",
        service: "qwen3.6-27b",
        from: "idle",
        to: "running",
        at_ms: Date.now(),
      },
      { type: "config_reloaded", changed_services: [], at_ms: Date.now() },
    ],
  });
  await page.goto("/events");
  await expect(page.getByText("Idle → Running", { exact: true })).toBeVisible();
  await page
    .getByRole("button", { name: "Config reloaded", exact: true })
    .click();
  await expect(page.getByText("Idle → Running", { exact: true })).toHaveCount(
    0,
  );
  await page.getByRole("button", { name: "All", exact: true }).click();
  await page.getByRole("button", { name: "qwen3.6-27b", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "qwen3.6-27b", exact: true }),
  ).toBeVisible();
});
