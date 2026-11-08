import { fireEvent, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  installLiveFakes,
  renderLive,
  service,
} from "../../test/renderLive.tsx";
import { MetricsView } from "./MetricsView.tsx";

// uPlot reads `matchMedia` on import, which jsdom lacks.
vi.mock("../ui/chart/Chart.tsx", () => ({ Chart: () => null }));

beforeEach(installLiveFakes);

const services = [
  service("qwen", "running"),
  service("gemma", "idle"),
  service("embed", "running", { modality: "embedding" }),
];

describe("MetricsView", () => {
  it("counts running services and says when a range is empty", () => {
    renderLive(<MetricsView />, { route: "/stats", services });
    expect(screen.getByText("2 / 3")).toBeInTheDocument();
    expect(
      screen.getAllByText("No samples in this range.").length,
    ).toBeGreaterThan(0);
    expect(
      screen.getByText("No service generated tokens in this range."),
    ).toBeInTheDocument();
  });

  it("narrows to one service, leaving out embedding models", () => {
    renderLive(<MetricsView />, { route: "/stats", services });
    const select = screen.getByRole("combobox", { name: "Service" });
    expect(
      Array.from(select.querySelectorAll("option")).map((o) => o.value),
    ).toEqual(["", "qwen", "gemma"]);
    fireEvent.change(select, { target: { value: "qwen" } });
    expect(screen.queryByText("2 / 3")).not.toBeInTheDocument();
    expect(
      screen.queryByText("Output tokens per second, by service"),
    ).not.toBeInTheDocument();
  });
});
