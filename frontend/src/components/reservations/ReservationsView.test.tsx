import { act, fireEvent, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { DeviceSummary, ReservationsResponse } from "../../api/client.ts";
import { installLiveFakes, renderLive } from "../../test/renderLive.tsx";
import { ReservationsView } from "./ReservationsView.tsx";

beforeEach(installLiveFakes);

const GIB = 1024 ** 3;

const devices: DeviceSummary[] = [
  {
    id: "gpu:0",
    name: "RTX 3090",
    total_bytes: 24 * GIB,
    free_bytes: 6 * GIB,
    reservations: [
      { service: "qwen", bytes: 4 * GIB, elastic: false },
      {
        service: "reservation:3",
        reservation: 3,
        bytes: 12 * GIB,
        elastic: false,
      },
    ],
  },
];

const reservations: ReservationsResponse = {
  manual: [
    {
      id: 3,
      label: "train",
      created_at_ms: Date.now() - 60_000,
      devices: { "gpu:0": 12 * GIB },
    },
  ],
  services: [
    {
      service: "qwen",
      state: "running",
      inflight_count: 0,
      devices: { "gpu:0": 4 * GIB },
    },
  ],
};

async function renderView(data: ReservationsResponse = reservations) {
  const view = renderLive(<ReservationsView />, {
    route: "/reservations",
    devices,
  });
  await act(async () => {
    view.client.setQueryData(["reservations"], data);
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
  return view;
}

function panel(heading: string): HTMLElement {
  const section = screen
    .getByRole("heading", { name: heading })
    .closest("section");
  if (!section) throw new Error(`no panel headed ${heading}`);
  return section;
}

describe("ReservationsView", () => {
  it("shows each device's split and both kinds of reservation", async () => {
    await renderView();
    expect(
      screen.getByText(
        "12.00 GiB reserved · 4.00 GiB services · 8.00 GiB free of 24.00 GiB",
      ),
    ).toBeInTheDocument();
    const manual = panel("Manual reservations");
    expect(within(manual).getByText("train")).toBeInTheDocument();
    expect(
      within(manual).getByRole("button", { name: "Remove reservation train" }),
    ).toBeInTheDocument();
    const services = panel("Services");
    expect(
      within(services).getByRole("link", { name: "qwen" }),
    ).toHaveAttribute("href", "/services/qwen");
    expect(
      within(services).getByRole("button", { name: "Stop qwen" }),
    ).toBeInTheDocument();
  });

  it("says when there are no manual reservations", async () => {
    await renderView({ manual: [], services: [] });
    expect(screen.getByText("No manual reservations.")).toBeInTheDocument();
  });

  it("checks the form before sending it", async () => {
    await renderView();
    fireEvent.click(screen.getByRole("button", { name: "New reservation" }));
    fireEvent.click(screen.getByRole("button", { name: "Reserve" }));
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Enter an amount for at least one device.",
    );
    expect(vi.mocked(fetch)).not.toHaveBeenCalledWith(
      "/api/reservations",
      expect.objectContaining({ method: "POST" }),
    );
  });
});
