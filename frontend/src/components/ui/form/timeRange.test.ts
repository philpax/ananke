import { describe, expect, it } from "vitest";

import {
  DEFAULT_RANGE,
  logWindowFor,
  parseTimeRange,
  rangeSearch,
  resolveTimeRange,
  withTimeRange,
  type TimeRange,
} from "./timeRange.ts";

function parse(query: string): TimeRange {
  return parseTimeRange(new URLSearchParams(query));
}

describe("parseTimeRange", () => {
  it("defaults without parameters", () => {
    expect(parse("")).toEqual(DEFAULT_RANGE);
  });

  it("reads a rolling preset", () => {
    expect(parse("range=6h")).toEqual({ kind: "rolling", preset: "6h" });
  });

  it("reads a fixed range", () => {
    expect(parse("from=1000&to=2000")).toEqual({
      kind: "fixed",
      fromMs: 1000,
      toMs: 2000,
    });
  });

  it("prefers a fixed range over a preset", () => {
    expect(parse("range=6h&from=1000&to=2000").kind).toBe("fixed");
  });

  it.each([
    "range=2d",
    "range=",
    "from=1000",
    "to=2000",
    "from=2000&to=1000",
    "from=1000&to=1000",
    "from=-5&to=2000",
    "from=1e3&to=2000",
    "from=abc&to=2000",
    "from=99999999999999999999&to=99999999999999999999999",
  ])("falls back to the default for %s", (query) => {
    expect(parse(query)).toEqual(DEFAULT_RANGE);
  });
});

describe("withTimeRange", () => {
  it("writes nothing for the default and keeps other parameters", () => {
    const params = new URLSearchParams("model=qwen&range=6h");
    expect(withTimeRange(params, DEFAULT_RANGE).toString()).toBe("model=qwen");
  });

  it("replaces a preset with a fixed range", () => {
    const params = new URLSearchParams("range=6h");
    expect(
      withTimeRange(params, { kind: "fixed", fromMs: 1, toMs: 2 }).toString(),
    ).toBe("from=1&to=2");
  });

  it("round-trips through parseTimeRange", () => {
    const ranges: TimeRange[] = [
      { kind: "rolling", preset: "24h" },
      { kind: "fixed", fromMs: 10, toMs: 20 },
      DEFAULT_RANGE,
    ];
    for (const range of ranges) {
      expect(
        parseTimeRange(withTimeRange(new URLSearchParams(), range)),
      ).toEqual(range);
    }
  });
});

describe("rangeSearch", () => {
  it("keeps only the range, normalised", () => {
    expect(rangeSearch("?range=1h&model=x")).toBe("?range=1h");
    expect(rangeSearch("?from=1&to=2")).toBe("?from=1&to=2");
  });

  it("is empty for the default or an invalid range", () => {
    expect(rangeSearch("")).toBe("");
    expect(rangeSearch("?range=5m")).toBe("");
    expect(rangeSearch("?from=2&to=1")).toBe("");
  });
});

describe("resolveTimeRange", () => {
  it("runs a rolling range to now", () => {
    expect(
      resolveTimeRange({ kind: "rolling", preset: "1h" }, 10_000_000),
    ).toEqual({ sinceMs: 10_000_000 - 3_600_000, untilMs: null });
  });

  it("keeps a fixed range's bounds", () => {
    expect(resolveTimeRange({ kind: "fixed", fromMs: 1, toMs: 2 }, 9)).toEqual({
      sinceMs: 1,
      untilMs: 2,
    });
  });
});

describe("logWindowFor", () => {
  it("follows a rolling range and pins a fixed one", () => {
    expect(logWindowFor({ kind: "rolling", preset: "5m" })).toEqual({
      kind: "relative",
      durationMs: 300_000,
    });
    expect(logWindowFor({ kind: "fixed", fromMs: 1, toMs: 2 })).toEqual({
      kind: "absolute",
      sinceMs: 1,
      untilMs: 2,
    });
  });
});
