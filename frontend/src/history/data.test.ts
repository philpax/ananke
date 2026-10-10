import { describe, expect, test } from "vitest";

import type { Bucket } from "../api/contract";
import { activityCharts, totals } from "./data";
import { parseRange } from "./range";

const bucket: Bucket = {
  service: "model",
  bucket_start: 0,
  request_count: 2,
  prompt_tokens: 20,
  completion_tokens: 10,
  avg_duration_ms: 1000,
  error_count: 1,
  avg_ttft_ms: null,
  output_tps: 2,
  input_tps: null,
};

describe("real history mapping", () => {
  test("weights nullable timings and separates request rates from token speeds", () => {
    const rows = [
      bucket,
      { ...bucket, request_count: 8, output_tps: 4, avg_duration_ms: 2000 },
    ];
    expect(totals(rows)).toMatchObject({
      requests: 10,
      output: 3.6,
      input: null,
      latency: 1800,
    });
    const charts = activityCharts(
      rows,
      [],
      [],
      { since: 0, until: 60_000, preset: 1 },
      false,
    );
    expect(charts[0].series[0].points[0].value).toBe(10);
    expect(charts[1].series[0].points[0].value).toBe(3.6);
    expect(charts[1].series[1].points[0].value).toBeNull();
  });
  test("does not invent samples for an empty range", () =>
    expect(
      activityCharts(
        [],
        [],
        [],
        { since: 0, until: 60_000, preset: null },
        true,
      ).every((c) => c.series.every((s) => s.points.length === 0)),
    ).toBe(true));
  test("rejects invalid and reversed dates", () => {
    expect(typeof parseRange("", "bad")).toBe("string");
    expect(typeof parseRange("2026-10-01T12:00", "2026-10-01T11:00")).toBe(
      "string",
    );
    expect(parseRange("2026-10-01T11:00", "2026-10-01T12:00")).toMatchObject({
      preset: null,
    });
  });
});
