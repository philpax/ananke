import { act, fireEvent, screen } from "@testing-library/react";
import type { QueryClient } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import type { ConfigResponse } from "../../api/client.ts";
import { installLiveFakes, renderLive } from "../../test/renderLive.tsx";
import { ConfigEditorView } from "./ConfigEditorView.tsx";
import { edit } from "./configDraft.ts";
import { resetDraft, updateDraft } from "./configDraftStore.ts";

beforeEach(installLiveFakes);
afterEach(() => act(resetDraft));

function config(content: string, writable = true): ConfigResponse {
  return {
    content,
    hash: `h:${content}`,
    path: "/etc/ananke/config.toml",
    writable,
  };
}

async function serve(client: QueryClient, response: ConfigResponse) {
  await act(async () => {
    client.setQueryData(["config"], response);
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

async function renderEditor(initial: ConfigResponse) {
  const view = renderLive(<ConfigEditorView />, { route: "/config" });
  await serve(view.client, initial);
  return view;
}

function editorText(): string {
  return document.querySelector(".cm-content")?.textContent ?? "";
}

describe("ConfigEditorView", () => {
  it("names the file it is editing", async () => {
    await renderEditor(config("a = 1"));
    expect(screen.getByText(/\/etc\/ananke\/config\.toml/)).toBeVisible();
    expect(editorText()).toBe("a = 1");
  });

  it("keeps unsaved edits when the file changes on disk", async () => {
    const { client } = await renderEditor(config("a = 1"));
    act(() => updateDraft((s) => edit(s, "a = 2")));
    expect(screen.getByText("unsaved")).toBeInTheDocument();

    await serve(client, config("a = 3"));
    expect(editorText()).toBe("a = 2");
    expect(screen.getByRole("alert")).toHaveTextContent(
      "The config changed on disk",
    );

    fireEvent.click(screen.getByRole("button", { name: "Reload from server" }));
    expect(editorText()).toBe("a = 3");
    expect(screen.queryByText("unsaved")).not.toBeInTheDocument();
  });

  it("keeps the draft across leaving the view", async () => {
    const first = await renderEditor(config("a = 1"));
    act(() => updateDraft((s) => edit(s, "a = 2")));
    first.unmount();
    await renderEditor(config("a = 1"));
    expect(editorText()).toBe("a = 2");
    expect(screen.getByText("unsaved")).toBeInTheDocument();
  });

  it("explains a read-only config and offers no save", async () => {
    await renderEditor(config("a = 1", false));
    expect(screen.getByText(/This config is read-only/)).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Save" }),
    ).not.toBeInTheDocument();
  });
});
