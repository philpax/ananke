import { describe, expect, it } from "vitest";

import type { AggregatedBucket } from "../../../../api/metrics-aggregate.ts";
import { summariseBuckets } from "./historySummary.ts";

function bucket(over: Partial<AggregatedBucket>): AggregatedBucket {
  return {
    ts: 0,
    requestCount: 0,
    promptTokens: 0,
    completionTokens: 0,
    errorCount: 0,
    totalDurationMs: 0,
    timedRequests: 0,
    totalWeightedOutputTps: 0,
    outputTpsRequests: 0,
    totalWeightedInputTps: 0,
    inputTpsRequests: 0,
    totalWeightedEffectiveTps: 0,
    effectiveTpsRequests: 0,
    ...over,
  };
}

describe("summariseBuckets", () => {
  it("has no averages for an empty range", () => {
    const s = summariseBuckets([]);
    expect(s.requests).toBe(0);
    expect(s.avgLatencyMs).toBeNull();
    expect(s.avgEffectiveTps).toBeNull();
    expect(s.hasSplitTps).toBe(false);
  });

  it("totals counts and weights averages by request", () => {
    const s = summariseBuckets([
      bucket({
        requestCount: 1,
        promptTokens: 10,
        totalDurationMs: 100,
        timedRequests: 1,
        totalWeightedOutputTps: 10,
        outputTpsRequests: 1,
      }),
      bucket({
        requestCount: 3,
        errorCount: 1,
        completionTokens: 5,
        totalDurationMs: 900,
        timedRequests: 3,
      }),
    ]);
    expect(s).toMatchObject({
      requests: 4,
      errors: 1,
      inputTokens: 10,
      outputTokens: 5,
      avgLatencyMs: 250,
      avgOutputTps: 10,
      avgInputTps: null,
      hasSplitTps: true,
    });
  });
});
