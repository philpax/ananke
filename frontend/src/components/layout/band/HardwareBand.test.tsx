import { act, fireEvent, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { DeviceSummary } from "../../../api/client.ts";
import { installLiveFakes, renderLive } from "../../../test/renderLive.tsx";
import { HardwareBand } from "./HardwareBand.tsx";

const GIB = 1024 ** 3;

function device(
  id: string,
  usedGib: number,
  reservations: DeviceSummary["reservations"] = [],
): DeviceSummary {
  return {
    id,
    name: id === "cpu" ? "CPU" : "NVIDIA GeForce RTX 4090",
    total_bytes: 24 * GIB,
    free_bytes: (24 - usedGib) * GIB,
    reservations,
  };
}

const DEVICES = [
  device("cpu", 6),
  device("gpu:1", 12),
  device("gpu:0", 22.8, [
    { service: "small", bytes: GIB, elastic: true },
    { service: "big", bytes: 20 * GIB, elastic: false },
  ]),
];

beforeEach(installLiveFakes);
afterEach(() => vi.unstubAllGlobals());

function cells() {
  return within(screen.getByRole("group", { name: "Device memory" }));
}

describe("HardwareBand", () => {
  it("draws one labelled bar per device, GPUs in order, then host memory", () => {
    renderLive(<HardwareBand />, { devices: DEVICES });
    const groups = cells().getAllByRole("group");
    expect(groups.map((g) => g.getAttribute("aria-label"))).toEqual([
      "gpu:0 RTX 4090",
      "gpu:1 RTX 4090",
      "cpu Host memory",
    ]);
    const bars = cells().getAllByRole("img");
    expect(bars).toHaveLength(3);
    expect(bars[0]).toHaveAccessibleName(
      "95% in use: 22.80 GiB of 24.00 GiB, 0 B more pledged",
    );
  });

  it("describes each cell with the services holding memory on it", () => {
    renderLive(<HardwareBand />, { devices: DEVICES });
    const gpu0 = cells().getByRole("group", { name: "gpu:0 RTX 4090" });
    expect(gpu0).toHaveAccessibleDescription(/big.*small/);
    const tip = document.getElementById(
      gpu0.getAttribute("aria-describedby") ?? "",
    );
    expect(tip).toHaveRole("tooltip");
    const holders = within(tip ?? document.body).getAllByRole("listitem");
    expect(holders.map((li) => li.textContent)).toEqual([
      "big20.00 GiB",
      "smallelastic1.00 GiB",
    ]);
    const gpu1 = cells().getByRole("group", { name: "gpu:1 RTX 4090" });
    expect(gpu1).toHaveAccessibleDescription(/No services hold memory here\./);
  });

  it("reports a failed device query", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(new Response("down", { status: 503 }))),
    );
    renderLive(<HardwareBand />);
    expect(await screen.findByRole("alert")).toBeInTheDocument();
  });

  it("toggles the narrow summary, and Escape closes it onto the button", () => {
    renderLive(<HardwareBand />, { devices: DEVICES });
    const button = screen.getByRole("button", {
      name: /Hardware resources:.*gpu 73% \(max 95%\).*ram 25%/,
    });
    expect(button).toHaveAttribute("aria-expanded", "false");

    act(() => button.click());
    expect(button).toHaveAttribute("aria-expanded", "true");

    const firstCell = cells().getAllByRole("group")[0];
    act(() => firstCell?.focus());
    fireEvent.keyDown(firstCell ?? document.body, { key: "Escape" });
    expect(button).toHaveAttribute("aria-expanded", "false");
    expect(button).toHaveFocus();
  });
});
