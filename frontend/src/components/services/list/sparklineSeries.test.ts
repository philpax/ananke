import { describe, expect, it } from "vitest";

import { buildServiceSparkline } from "./sparklineSeries.ts";

function bucket(service: string, startMs: number, count: number) {
  return {
    service,
    bucket_start: startMs,
    request_count: count,
    prompt_tokens: 0,
    completion_tokens: 0,
    error_count: 0,
  };
}

describe("buildServiceSparkline", () => {
  it("splits by service, sums shared buckets, and sorts by time", () => {
    const spark = buildServiceSparkline([
      bucket("a", 120_000, 1),
      bucket("a", 60_000, 2),
      bucket("a", 60_000, 3),
      bucket("b", 60_000, 4),
    ]);
    expect(spark.get("a")).toEqual({ ts: [60, 120], counts: [5, 1] });
    expect(spark.get("b")).toEqual({ ts: [60], counts: [4] });
  });
});
