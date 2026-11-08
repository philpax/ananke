// Pure data shaping for the Stats charts: uPlot's column layout from the
// aggregated buckets, per-device memory, and the per-service rankings.
// Timestamps are unix seconds, as uPlot wants them.

import type { DeviceSampleResponse } from "../../api/client.ts";
import type {
  AggregatedBucket,
  PerServiceSeries,
} from "../../api/metrics-aggregate.ts";

export type ChartData = (number | null)[][];

export type MemorySeries = {
  readonly device: string;
  readonly data: readonly { readonly ts: number; readonly usedBytes: number }[];
};

/// Samples grouped per device, each in time order, devices sorted by id.
export function groupDeviceSamples(
  samples: readonly DeviceSampleResponse[],
): MemorySeries[] {
  const byDevice = new Map<string, { ts: number; usedBytes: number }[]>();
  for (const s of samples) {
    const points = byDevice.get(s.device) ?? [];
    points.push({
      ts: Math.floor(s.timestamp_ms / 1000),
      usedBytes: s.used_bytes,
    });
    byDevice.set(s.device, points);
  }
  return Array.from(byDevice.entries())
    .sort(([a], [b]) => a.localeCompare(b, undefined, { numeric: true }))
    .map(([device, data]) => ({
      device,
      data: data.sort((a, b) => a.ts - b.ts),
    }));
}

/// GPUs and host memory apart: host RAM is usually several times a GPU's
/// VRAM, so on one axis it would flatten every GPU line.
export function splitMemorySeries(series: readonly MemorySeries[]): {
  readonly gpus: readonly MemorySeries[];
  readonly host: readonly MemorySeries[];
} {
  return {
    gpus: series.filter((s) => s.device !== "cpu"),
    host: series.filter((s) => s.device === "cpu"),
  };
}

export type ChartField =
  | "requestCount"
  | "promptTokens"
  | "completionTokens"
  | "errorCount"
  | "totalDurationMs"
  | "outputTps"
  | "inputTps"
  | "effectiveTps";

/// A bucket's value for `field`. Latency and rates are request-weighted
/// means, `null` where no request carried them, never a misleading zero.
export function resolveField(
  b: AggregatedBucket,
  field: ChartField,
): number | null {
  switch (field) {
    case "totalDurationMs":
      return mean(b.totalDurationMs, b.timedRequests);
    case "outputTps":
      return mean(b.totalWeightedOutputTps, b.outputTpsRequests);
    case "inputTps":
      return mean(b.totalWeightedInputTps, b.inputTpsRequests);
    case "effectiveTps":
      return mean(b.totalWeightedEffectiveTps, b.effectiveTpsRequests);
    case "requestCount":
    case "promptTokens":
    case "completionTokens":
    case "errorCount":
      return b[field];
    default: {
      const unreachable: never = field;
      return unreachable;
    }
  }
}

/// Whether any bucket in the window carries an input/output split. When
/// none do (all non-streaming with no engine timings), the token-rate
/// chart falls back to the effective line instead of showing two zeros.
export function hasSplitTps(buckets: readonly AggregatedBucket[]): boolean {
  return buckets.some((b) => b.outputTpsRequests > 0 || b.inputTpsRequests > 0);
}

/// One column of timestamps and one per field.
export function toFieldsData(
  buckets: readonly AggregatedBucket[],
  fields: readonly ChartField[],
): ChartData {
  return [
    buckets.map((b) => b.ts),
    ...fields.map((field) => buckets.map((b) => resolveField(b, field))),
  ];
}

/// One column per service over the union of their timestamps; a service
/// with no bucket at a time is `null` there.
export function toMultiSeriesData(
  seriesList: readonly PerServiceSeries[],
  field: ChartField,
): ChartData {
  const ts = unionTimestamps(seriesList.map((s) => s.buckets));
  return [
    ts,
    ...seriesList.map((s) => {
      const byTs = new Map(s.buckets.map((b) => [b.ts, b]));
      return ts.map((t) => {
        const b = byTs.get(t);
        return b === undefined ? null : resolveField(b, field);
      });
    }),
  ];
}

/// Used memory per device in GB over the union of their timestamps.
export function toMemoryData(series: readonly MemorySeries[]): ChartData {
  const ts = unionTimestamps(series.map((s) => s.data));
  return [
    ts,
    ...series.map((s) => {
      const byTs = new Map(s.data.map((d) => [d.ts, d.usedBytes / 1e9]));
      return ts.map((t) => byTs.get(t) ?? null);
    }),
  ];
}

/// Whether a chart has anything to draw: some value column holds a
/// value. The timestamp column alone is not a sample.
export function hasSamples(data: ChartData): boolean {
  return data.slice(1).some((column) => column.some((v) => v !== null));
}

export type ServiceSpeed = {
  readonly service: string;
  readonly tokensPerSecond: number;
  /// The rate is the end-to-end effective one: the service reported no
  /// decode rate in the range.
  readonly effective: boolean;
};

/// Each service's mean output rate over the range, fastest first. A
/// service with no decode rate falls back to its effective rate; one with
/// neither is left out, since it generated nothing to rank.
export function serviceSpeeds(
  seriesList: readonly PerServiceSeries[],
): ServiceSpeed[] {
  const speeds: ServiceSpeed[] = [];
  for (const { serviceName, buckets } of seriesList) {
    const sum = (f: (b: AggregatedBucket) => number) =>
      buckets.reduce((s, b) => s + f(b), 0);
    const output = mean(
      sum((b) => b.totalWeightedOutputTps),
      sum((b) => b.outputTpsRequests),
    );
    const effective = mean(
      sum((b) => b.totalWeightedEffectiveTps),
      sum((b) => b.effectiveTpsRequests),
    );
    if (output !== null) {
      speeds.push({
        service: serviceName,
        tokensPerSecond: output,
        effective: false,
      });
    } else if (effective !== null) {
      speeds.push({
        service: serviceName,
        tokensPerSecond: effective,
        effective: true,
      });
    }
  }
  return speeds.sort((a, b) => b.tokensPerSecond - a.tokensPerSecond);
}

function mean(total: number, count: number): number | null {
  return count > 0 ? total / count : null;
}

function unionTimestamps(
  columns: readonly (readonly { readonly ts: number }[])[],
): number[] {
  const all = new Set<number>();
  for (const column of columns) for (const p of column) all.add(p.ts);
  return Array.from(all).sort((a, b) => a - b);
}
