import { describe, expect, it } from "vitest";

import type { DeviceSummary } from "../../../api/client.ts";
import {
  bandSummary,
  deviceHolders,
  deviceUsage,
  formatUptime,
  gpuGrid,
  gpuShortId,
  shortDeviceName,
  splitDevices,
  usageTone,
} from "./hardware.ts";

const GIB = 1024 ** 3;

function device(
  id: string,
  usedGib: number,
  reservations: DeviceSummary["reservations"] = [],
  totalGib = 24,
): DeviceSummary {
  return {
    id,
    name: "NVIDIA GeForce RTX 3090",
    total_bytes: totalGib * GIB,
    free_bytes: (totalGib - usedGib) * GIB,
    reservations,
  };
}

describe("deviceUsage", () => {
  it("clamps used to zero when free exceeds total", () => {
    const usage = deviceUsage({ ...device("gpu:0", 0), free_bytes: 30 * GIB });
    expect(usage.used).toBe(0);
    expect(usage.percent).toBe(0);
  });

  it("reports 0% for a device with no total", () => {
    const usage = deviceUsage({ ...device("gpu:0", 0), total_bytes: 0 });
    expect(usage.percent).toBe(0);
    expect(usage.total).toBe(0);
  });

  it("keeps reservations past the total as pledged beyond used", () => {
    const usage = deviceUsage(
      device("gpu:0", 12, [
        { service: "a", bytes: 20 * GIB, elastic: false },
        { service: "b", bytes: 10 * GIB, elastic: true },
      ]),
    );
    expect(usage.percent).toBe(50);
    expect(usage.pledged).toBe(30 * GIB);
    expect(usage.pledgedBeyondUsed).toBe(18 * GIB);
  });
});

describe("usageTone", () => {
  it.each([
    [79.9, "neutral"],
    [80, "warn"],
    [89.9, "warn"],
    [90, "danger"],
  ] as const)("%d%% is %s", (percent, tone) => {
    expect(usageTone(percent)).toBe(tone);
  });
});

describe("gpuGrid", () => {
  it.each([
    [0, 0, 0, false],
    [1, 1, 1, false],
    [3, 1, 3, false],
    [4, 2, 2, false],
    [8, 2, 4, false],
    [9, 3, 3, false],
    [12, 3, 4, false],
    [13, 4, 4, false],
    [16, 4, 4, false],
    [17, 1, 17, true],
    [32, 1, 32, true],
  ] as const)("%d cards: %d × %d", (count, rows, columns, dense) => {
    expect(gpuGrid(count)).toEqual({ rows, columns, dense });
  });
});

describe("splitDevices", () => {
  it("orders GPUs by index, not by string", () => {
    const { gpus, cpu } = splitDevices([
      device("gpu:10", 0),
      device("cpu", 0),
      device("gpu:2", 0),
    ]);
    expect(gpus.map((d) => d.id)).toEqual(["gpu:2", "gpu:10"]);
    expect(cpu?.id).toBe("cpu");
  });
});

describe("gpuShortId", () => {
  it("keeps only the index of a GPU id", () => {
    expect(gpuShortId("gpu:3")).toBe("3");
    expect(gpuShortId("cpu")).toBe("cpu");
  });
});

describe("bandSummary", () => {
  it("leaves the GPU summary out when there are no GPUs", () => {
    const summary = bandSummary(splitDevices([device("cpu", 6)]));
    expect(summary.gpu).toBeUndefined();
    expect(summary.cpu).toEqual({ percent: 25, tone: "neutral" });
  });

  it("leaves the CPU summary out when host memory is not reported", () => {
    const summary = bandSummary(splitDevices([device("gpu:0", 12)]));
    expect(summary.cpu).toBeUndefined();
    expect(summary.gpu?.mean).toBe(50);
  });

  it("tones by the busiest card and shows it when far above the mean", () => {
    const summary = bandSummary(
      splitDevices([
        device("gpu:0", 2.4),
        device("gpu:1", 22.8),
        device("gpu:2", 2.4),
      ]),
    );
    expect(summary.gpu).toEqual({
      mean: 38,
      max: 95,
      tone: "danger",
      showMax: true,
    });
  });

  it("omits the max when the cards are close to the mean", () => {
    const summary = bandSummary(
      splitDevices([device("gpu:0", 12), device("gpu:1", 14.4)]),
    );
    expect(summary.gpu?.showMax).toBe(false);
  });
});

describe("deviceHolders", () => {
  it("lists the largest holder first", () => {
    const holders = deviceHolders(
      device("gpu:0", 0, [
        { service: "small", bytes: GIB, elastic: false },
        { service: "big", bytes: 8 * GIB, elastic: false },
      ]),
    );
    expect(holders.map((r) => r.service)).toEqual(["big", "small"]);
  });
});

describe("formatUptime", () => {
  it.each([
    [59_000, "59s"],
    [3_600_000, "1h 0m"],
    [26 * 3_600_000, "1d 2h"],
    [40 * 86_400_000, "40d 0h"],
  ])("%d ms is %s", (ms, text) => {
    expect(formatUptime(ms)).toBe(text);
  });
});

describe("shortDeviceName", () => {
  it("drops the vendor and consumer line", () => {
    expect(shortDeviceName("NVIDIA GeForce RTX 4090")).toBe("RTX 4090");
  });

  it("keeps a name with no known prefix", () => {
    expect(shortDeviceName("CPU")).toBe("CPU");
  });
});
