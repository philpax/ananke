import { describe, expect, it } from "vitest";

import { compactTick } from "./ticks.ts";

describe("compactTick", () => {
  it.each([
    [0, "0"],
    [0.5, "0.5"],
    [950, "950"],
    [1500, "1.5K"],
    [25_000, "25K"],
    [2_200_000, "2.2M"],
  ])("formats %d as %s", (value, expected) => {
    expect(compactTick(value, "en")).toBe(expected);
  });
});
