import { expect, test } from "@playwright/test";

import type { Schemas } from "../src/api/contract";
import { mockBackend, referenceServices } from "./fixtures";

for (const serviceCase of [
  {
    label: "embedding",
    name: "jina-embeddings-v5-text-small-retrieval-vllm",
    openai_compat: true,
    modality: "embedding",
  },
  {
    label: "non-OpenAI",
    name: "comfyui",
    openai_compat: false,
    modality: "chat",
  },
] as const) {
  test(`Chat rejects a selected ${serviceCase.label} service and an unavailable model link`, async ({
    page,
  }) => {
    const { name } = serviceCase;
    const inventory: Schemas["ServicesResponse"] = {
      services: referenceServices.map((service) =>
        service.name === name
          ? {
              ...service,
              openai_compat: serviceCase.openai_compat,
              modality: serviceCase.modality,
            }
          : service,
      ),
      openai_api_port: 7070,
    };
    const calls = await mockBackend(page, 2, { inventory: inventory.services });
    await page.goto("/");
    await page
      .getByRole("button", { name: new RegExp(`^${name} Copy`) })
      .click();
    await expect(
      page.getByRole("main", { name: `Service ${name}` }).getByRole("button", {
        name: "Chat",
        exact: true,
      }),
    ).toBeDisabled();
    await page
      .getByRole("navigation", { name: "Sections" })
      .getByRole("button", { name: "Chat", exact: true })
      .click();
    await expect(
      page.getByRole("combobox", { name: "Chat model" }),
    ).toHaveValue("");
    await page.goto(`/chat?model=${encodeURIComponent(name)}`);
    await expect(
      page.getByRole("combobox", { name: "Chat model" }),
    ).toHaveValue(name);
    await page
      .getByRole("textbox", { name: "Message", exact: true })
      .fill("Hello");
    await expect(
      page.getByRole("button", { name: "Send", exact: true }),
    ).toBeDisabled();
    await page
      .getByRole("textbox", { name: "Message", exact: true })
      .press("Enter");
    await expect(page.locator("article")).toHaveCount(0);
    expect(
      calls.filter((call) => call.url === "/v1/chat/completions"),
    ).toHaveLength(0);
    await page
      .getByRole("combobox", { name: "Chat model" })
      .selectOption("qwen3.6-27b");
    await page.getByRole("button", { name: "Send", exact: true }).click();
    await expect(page.locator("article")).toHaveCount(2);
    await expect(page.locator("article").last()).toContainText("Hello.");
  });
}
