import i18n from "i18next";
import { describe, expect, it } from "vitest";

import type { PlacementConfig } from "../../../api/client.ts";
import { constraintRows } from "./placementConstraints.ts";

const t = i18n.t.bind(i18n);

function rows(placement: PlacementConfig): Record<string, string> {
  return Object.fromEntries(
    constraintRows(placement, t).map((r) => [r.label, r.value]),
  );
}

describe("constraintRows", () => {
  it("describes a weighted tensor split on two GPUs", () => {
    expect(
      rows({
        policy: "gpu-only",
        gpu_allow: [2, 0],
        gpu_headroom_bytes: 512 * 1024 ** 2,
        split: { mode: "tensor", tensor_split_weights: [1, 2] },
        allocation: { kind: "estimated" },
      }),
    ).toEqual({
      Policy: "GPU only",
      GPUs: "2, 0",
      Split: "by tensor",
      "Split weights": "0: 1 · 2: 2",
      "GPU headroom": "512.0 MiB",
      Allocation: "estimated",
    });
  });

  it("omits zero headroom and a missing split, and says all GPUs", () => {
    expect(
      rows({
        policy: "hybrid",
        gpu_headroom_bytes: 0,
        allocation: { kind: "static", reserve_bytes: 2 * 1024 ** 3 },
      }),
    ).toEqual({
      Policy: "GPU, spilling to host",
      GPUs: "all GPUs",
      Allocation: "fixed 2.00 GiB",
    });
  });

  it("shows a layer split without weights, and a dynamic range", () => {
    expect(
      rows({
        policy: "gpu-only",
        gpu_headroom_bytes: 0,
        split: { mode: "layer" },
        allocation: {
          kind: "dynamic",
          min_bytes: 1024 ** 3,
          max_bytes: 4 * 1024 ** 3,
          min_borrower_runtime_ms: 30_000,
        },
      }),
    ).toEqual({
      Policy: "GPU only",
      GPUs: "all GPUs",
      Split: "by layer",
      Allocation: "1.00 GiB to 4.00 GiB, borrowers run at least 30.0s",
    });
  });
});
