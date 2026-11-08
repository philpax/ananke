import { act, fireEvent, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  installLiveFakes,
  renderLive,
  service,
} from "../../../test/renderLive.tsx";
import { ServiceList } from "./ServiceList.tsx";

beforeEach(() => {
  installLiveFakes();
  localStorage.clear();
});
afterEach(() => vi.unstubAllGlobals());

const services = [
  service("qwen", "running", { description: "Code model" }),
  service("bge", "idle", { modality: "embedding" }),
  service("llama", "idle"),
  service("broken", "disabled", {
    disabled_reason: "oom",
    fit_verdict: { kind: "does_not_fit", shortfalls: [] },
  }),
];

function renderList(route = "/") {
  return renderLive(<ServiceList selected={undefined} onHide={() => {}} />, {
    route,
    services,
  });
}

describe("ServiceList", () => {
  it("groups services by state with counts", () => {
    renderList();
    expect(screen.getByText("4 / 4 shown")).toBeInTheDocument();
    const idle = screen.getByRole("list", { name: "idle" });
    expect(within(idle).getAllByRole("listitem")).toHaveLength(2);
    expect(
      screen
        .getByRole("list", { name: "running" })
        .compareDocumentPosition(idle),
    ).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
  });

  it("shows a disabled service's reason instead of its fit verdict", () => {
    renderList();
    const row = within(screen.getByRole("list", { name: "disabled" }));
    expect(row.getByText("out of memory")).toBeInTheDocument();
    expect(row.queryByText("won't fit")).not.toBeInTheDocument();
  });

  it("filters by search and favourites", () => {
    renderList();
    fireEvent.change(
      screen.getByRole("searchbox", { name: "Search services" }),
      {
        target: { value: "code" },
      },
    );
    expect(screen.getByText("1 / 4 shown")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "qwen" })).toBeInTheDocument();

    fireEvent.change(
      screen.getByRole("searchbox", { name: "Search services" }),
      {
        target: { value: "" },
      },
    );
    act(() => screen.getByRole("button", { name: "Favourite llama" }).click());
    act(() =>
      screen.getByRole("checkbox", { name: "Favourites only" }).click(),
    );
    expect(screen.getByText("1 / 4 shown")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "llama" })).toBeInTheDocument();
  });

  it("has one tab stop that the arrow keys move", () => {
    renderList();
    const links = screen.getAllByRole("link");
    expect(links.filter((l) => l.tabIndex === 0)).toHaveLength(1);
    const first = links[0]!;
    first.focus();
    fireEvent.keyDown(first, { key: "ArrowDown" });
    expect(document.activeElement).toBe(links[1]);
    expect(links[1]).toHaveAttribute("tabindex", "0");
    fireEvent.keyDown(links[1]!, { key: "End" });
    expect(document.activeElement).toBe(links.at(-1));
  });

  it("carries the time range into service links", () => {
    renderList("/?range=6h");
    expect(screen.getByRole("link", { name: "qwen" })).toHaveAttribute(
      "href",
      "/services/qwen?range=6h",
    );
  });
});
