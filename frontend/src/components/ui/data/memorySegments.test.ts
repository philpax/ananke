import { describe, expect, it } from "vitest";

import { memorySegments } from "./memorySegments.ts";

describe("memorySegments", () => {
  it("sizes device segments as shares of the total", () => {
    expect(
      memorySegments({ variant: "device", total: 100, used: 40, pledged: 10 }),
    ).toEqual([
      { kind: "used", bytes: 40, percent: 40 },
      { kind: "pledged", bytes: 10, percent: 10 },
    ]);
  });

  it("orders placement segments others, self, growth", () => {
    const kinds = memorySegments({
      variant: "placement",
      total: 100,
      others: 20,
      self: 30,
      growth: 10,
    }).map((s) => s.kind);
    expect(kinds).toEqual(["others", "self", "growth"]);
  });

  it("never runs past the track", () => {
    const segments = memorySegments({
      variant: "placement",
      total: 100,
      others: 70,
      self: 50,
      growth: 20,
    });
    expect(segments.map((s) => s.percent)).toEqual([70, 30]);
  });

  it("drops empty, negative, and non-finite parts", () => {
    expect(
      memorySegments({
        variant: "placement",
        total: 100,
        others: -5,
        self: Number.NaN,
        growth: 0,
      }),
    ).toEqual([]);
  });

  it("draws nothing without a total", () => {
    expect(
      memorySegments({ variant: "device", total: 0, used: 10, pledged: 0 }),
    ).toEqual([]);
  });
});
