// Shared metric bucket aggregation. Used by both the stats view
// (MetricsView) and the per-service metrics section (ServiceDetailView).

import type { MetricBucketResponse } from "./client.ts";

export type AggregatedBucket = {
  ts: number;
  requestCount: number;
  promptTokens: number;
  completionTokens: number;
  errorCount: number;
  totalDurationMs: number;
  timedRequests: number;
  totalWeightedOutputTps: number;
  outputTpsRequests: number;
  totalWeightedInputTps: number;
  inputTpsRequests: number;
  totalWeightedAggregateTps: number;
  aggregateTpsRequests: number;
};

function accumulate(ex: AggregatedBucket, b: MetricBucketResponse): void {
  ex.requestCount += b.request_count;
  ex.promptTokens += b.prompt_tokens;
  ex.completionTokens += b.completion_tokens;
  ex.errorCount += b.error_count;
  if (b.avg_duration_ms != null) {
    ex.totalDurationMs += b.avg_duration_ms * b.request_count;
    ex.timedRequests += b.request_count;
  }
  if (b.output_tps != null) {
    ex.totalWeightedOutputTps += b.output_tps * b.request_count;
    ex.outputTpsRequests += b.request_count;
  }
  if (b.input_tps != null) {
    ex.totalWeightedInputTps += b.input_tps * b.request_count;
    ex.inputTpsRequests += b.request_count;
  }
  if (b.aggregate_tps != null) {
    ex.totalWeightedAggregateTps += b.aggregate_tps * b.request_count;
    ex.aggregateTpsRequests += b.request_count;
  }
}

function newBucket(b: MetricBucketResponse): AggregatedBucket {
  const ts = Math.floor(b.bucket_start / 1000);
  return {
    ts,
    requestCount: b.request_count,
    promptTokens: b.prompt_tokens,
    completionTokens: b.completion_tokens,
    errorCount: b.error_count,
    totalDurationMs:
      b.avg_duration_ms != null ? b.avg_duration_ms * b.request_count : 0,
    timedRequests: b.avg_duration_ms != null ? b.request_count : 0,
    totalWeightedOutputTps:
      b.output_tps != null ? b.output_tps * b.request_count : 0,
    outputTpsRequests: b.output_tps != null ? b.request_count : 0,
    totalWeightedInputTps:
      b.input_tps != null ? b.input_tps * b.request_count : 0,
    inputTpsRequests: b.input_tps != null ? b.request_count : 0,
    totalWeightedAggregateTps:
      b.aggregate_tps != null ? b.aggregate_tps * b.request_count : 0,
    aggregateTpsRequests: b.aggregate_tps != null ? b.request_count : 0,
  };
}

export function aggregateBuckets(
  buckets: MetricBucketResponse[],
): AggregatedBucket[] {
  const map = new Map<number, AggregatedBucket>();
  for (const b of buckets) {
    const ts = Math.floor(b.bucket_start / 1000);
    const ex = map.get(ts);
    if (ex) {
      accumulate(ex, b);
    } else {
      map.set(ts, newBucket(b));
    }
  }
  return Array.from(map.values()).sort((a, b) => a.ts - b.ts);
}

export type PerServiceSeries = {
  serviceName: string;
  buckets: AggregatedBucket[];
};

export function groupByService(
  buckets: MetricBucketResponse[],
): PerServiceSeries[] {
  const byService = new Map<string, AggregatedBucket[]>();
  for (const b of buckets) {
    const name = b.service ?? "unknown";
    const arr = byService.get(name) ?? [];
    const ts = Math.floor(b.bucket_start / 1000);
    const ex = arr.find((a) => a.ts === ts);
    if (ex) {
      accumulate(ex, b);
    } else {
      arr.push(newBucket(b));
    }
    byService.set(name, arr);
  }
  return Array.from(byService.entries())
    .map(([serviceName, buckets]) => ({
      serviceName,
      buckets: buckets.sort((a, b) => a.ts - b.ts),
    }))
    .sort(
      (a, b) =>
        b.buckets.reduce((s, x) => s + x.requestCount, 0) -
        a.buckets.reduce((s, x) => s + x.requestCount, 0),
    );
}
