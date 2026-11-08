import { fireEvent, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  installLiveFakes,
  renderLive,
  service,
} from "../../test/renderLive.tsx";
import { ServicesView } from "./ServicesView.tsx";

// The chart draws on a canvas, which jsdom does not have; these tests are
// about the panes, not the charts.
vi.mock("../ui/chart/Chart.tsx", () => ({ Chart: () => null }));

beforeEach(() => {
  installLiveFakes();
  localStorage.clear();
});
afterEach(() => vi.unstubAllGlobals());

describe("ServicesView", () => {
  it("asks for a selection at the root", () => {
    renderLive(<ServicesView />, { services: [service("qwen")] });
    expect(
      screen.getByText(
        "Select a service to see its state, placement, and activity.",
      ),
    ).toBeInTheDocument();
  });

  it("opens the service a path names, using its summary while loading", () => {
    renderLive(<ServicesView />, {
      route: "/services/qwen",
      services: [service("qwen", "disabled", { disabled_reason: "oom" })],
    });
    expect(
      screen.getByRole("heading", { name: "qwen", level: 1 }),
    ).toBeVisible();
    expect(screen.getAllByText("out of memory").length).toBeGreaterThan(0);
    expect(screen.getByRole("link", { name: "qwen" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });

  it("resizes the list from the keyboard and remembers it", () => {
    renderLive(<ServicesView />, { services: [service("qwen")] });
    const separator = screen.getByRole("separator", {
      name: "Resize the service list",
    });
    expect(separator).toHaveAttribute("aria-valuenow", "324");
    fireEvent.keyDown(separator, { key: "ArrowRight" });
    expect(separator).toHaveAttribute("aria-valuenow", "340");
    expect(localStorage.getItem("ananke-services-list-width")).toBe("340");
  });

  it("hides the list with Enter and offers it back", () => {
    renderLive(<ServicesView />, { services: [service("qwen")] });
    fireEvent.keyDown(screen.getByRole("separator"), { key: "Enter" });
    expect(screen.queryByRole("separator")).not.toBeInTheDocument();
    const browse = screen.getByRole("button", { name: "Browse 1 service" });
    expect(browse).toHaveFocus();
    fireEvent.click(browse);
    expect(screen.getByRole("separator")).toBeInTheDocument();
  });
});
