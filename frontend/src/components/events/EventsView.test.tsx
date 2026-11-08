import { act, fireEvent, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import type { EventHistory } from "../../api/client.ts";
import { installLiveFakes, renderLive } from "../../test/renderLive.tsx";
import { EventsView } from "./EventsView.tsx";

beforeEach(installLiveFakes);

async function renderWith(history: EventHistory, route = "/events") {
  const view = renderLive(<EventsView />, { route });
  await act(async () => {
    view.client.setQueryData(["events"], history);
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
  return view;
}

const now = Date.now();
const history: EventHistory = {
  since_ms: now - 60_000,
  events: [
    {
      type: "state_changed",
      service: "qwen",
      from: "starting",
      to: "running",
      at_ms: now - 30_000,
    },
    { type: "config_reloaded", changed_services: [], at_ms: now - 20_000 },
  ],
};

describe("EventsView", () => {
  it("lists events newest first and links to their service", async () => {
    await renderWith(history, "/events?range=1h");
    const items = screen.getAllByRole("listitem");
    expect(items[0]).toHaveTextContent("Config reloaded");
    expect(items[1]).toHaveTextContent("starting → running");
    expect(screen.getByRole("link", { name: "qwen" })).toHaveAttribute(
      "href",
      "/services/qwen?range=1h",
    );
  });

  it("filters by more than one type at once", async () => {
    await renderWith(history);
    const filter = screen.getByRole("group", { name: "Filter by event type" });
    const button = (name: string) =>
      Array.from(filter.querySelectorAll("button")).find(
        (b) => b.textContent === name,
      );
    fireEvent.click(button("State changed")!);
    expect(screen.getAllByRole("listitem")).toHaveLength(1);
    fireEvent.click(button("Config reloaded")!);
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
    expect(button("All types")).toHaveAttribute("aria-pressed", "false");
  });

  it("says when nothing falls in the range", async () => {
    await renderWith({ since_ms: now, events: [] });
    expect(
      screen.getByText("No events or recorded restarts in this range."),
    ).toBeInTheDocument();
  });
});
