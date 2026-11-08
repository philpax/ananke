import { describe, expect, it } from "vitest";

import { categorical, deviceSeries, fillToken } from "./series.ts";

describe("deviceSeries", () => {
  it("gives GPU 0 the accent series", () => {
    expect(deviceSeries("gpu:0")).toBe("series-1");
  });

  it("cycles GPUs through the first five series", () => {
    expect(deviceSeries("gpu:4")).toBe("series-5");
    expect(deviceSeries("gpu:5")).toBe("series-1");
  });

  it("gives anything else the CPU colour", () => {
    expect(deviceSeries("cpu")).toBe("device-cpu");
  });
});

describe("categorical", () => {
  it("wraps after six", () => {
    expect(categorical(0)).toBe("series-1");
    expect(categorical(6)).toBe("series-1");
  });
});

describe("fillToken", () => {
  it("uses the tone background for status colours", () => {
    expect(fillToken("danger")).toBe("danger-bg");
    expect(fillToken("series-2")).toBe("series-2-fill");
  });
});
