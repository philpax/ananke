import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { serviceDetail } from "../../../test/fixtures.ts";
import { PlacementPanel } from "./PlacementPanel.tsx";

describe("PlacementPanel", () => {
  it("shows a tensor split's constraints and the preview verdict", () => {
    render(
      <PlacementPanel
        detail={serviceDetail({
          placement: {
            policy: "gpu-only",
            gpu_allow: [0, 1],
            gpu_headroom_bytes: 0,
            split: { mode: "tensor", tensor_split_weights: [1, 2] },
            allocation: { kind: "estimated" },
          },
          placement_preview: {
            devices: [
              {
                device: "gpu:0",
                bytes: 1024 ** 3,
                max_bytes: 1024 ** 3,
                used_by_others_bytes: 0,
                total_bytes: 8 * 1024 ** 3,
              },
            ],
            verdict: { kind: "needs_eviction" },
            expert_offload_bytes: 0,
            expert_offload_layers: 0,
          },
        })}
      />,
    );
    expect(screen.getByText("by tensor")).toBeInTheDocument();
    expect(screen.getByText("0: 1 · 1: 2")).toBeInTheDocument();
    expect(screen.getByText("0, 1")).toBeInTheDocument();
    expect(screen.getByText("needs eviction")).toBeInTheDocument();
    expect(screen.getByText("gpu:0")).toBeInTheDocument();
  });

  it("names each device that falls short", () => {
    render(
      <PlacementPanel
        detail={serviceDetail({
          placement_preview: {
            devices: [],
            verdict: {
              kind: "does_not_fit",
              shortfalls: [
                {
                  device: "cpu",
                  requested_bytes: 2 * 1024 ** 3,
                  available_bytes: 1024 ** 3,
                },
              ],
            },
            expert_offload_bytes: 0,
            expert_offload_layers: 0,
          },
        })}
      />,
    );
    expect(
      screen.getByText("cpu: needs 2.00 GiB, has 1.00 GiB"),
    ).toBeInTheDocument();
  });
});
