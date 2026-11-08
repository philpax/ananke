// The totals over a service's history range: requests, errors, tokens, and
// request-weighted averages. An average with no requests behind it is
// `null`, never NaN or a misleading zero.

import type { AggregatedBucket } from "../../../../api/metrics-aggregate.ts";

export type HistorySummary = {
  readonly requests: number;
  readonly errors: number;
  readonly inputTokens: number;
  readonly outputTokens: number;
  readonly avgLatencyMs: number | null;
  readonly avgInputTps: number | null;
  readonly avgOutputTps: number | null;
  readonly avgEffectiveTps: number | null;
  /// Some request carried an input/output split; without one, only the
  /// effective rate means anything.
  readonly hasSplitTps: boolean;
};

export function summariseBuckets(
  buckets: readonly AggregatedBucket[],
): HistorySummary {
  const sum = (f: (b: AggregatedBucket) => number) =>
    buckets.reduce((s, b) => s + f(b), 0);
  const avg = (
    weighted: (b: AggregatedBucket) => number,
    count: (b: AggregatedBucket) => number,
  ) => {
    const n = sum(count);
    return n > 0 ? sum(weighted) / n : null;
  };
  return {
    requests: sum((b) => b.requestCount),
    errors: sum((b) => b.errorCount),
    inputTokens: sum((b) => b.promptTokens),
    outputTokens: sum((b) => b.completionTokens),
    avgLatencyMs: avg(
      (b) => b.totalDurationMs,
      (b) => b.timedRequests,
    ),
    avgInputTps: avg(
      (b) => b.totalWeightedInputTps,
      (b) => b.inputTpsRequests,
    ),
    avgOutputTps: avg(
      (b) => b.totalWeightedOutputTps,
      (b) => b.outputTpsRequests,
    ),
    avgEffectiveTps: avg(
      (b) => b.totalWeightedEffectiveTps,
      (b) => b.effectiveTpsRequests,
    ),
    hasSplitTps: buckets.some(
      (b) => b.outputTpsRequests > 0 || b.inputTpsRequests > 0,
    ),
  };
}
