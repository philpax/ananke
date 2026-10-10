import type { Range } from "../api/client";
import { bucketSize } from "../api/client";
import type { Bucket, Restart, Sample } from "../api/contract";

export type Point = { readonly at: number; readonly value: number | null };
export type Series = {
  readonly label: string;
  readonly color: string;
  readonly points: readonly Point[];
};
export type ChartData = {
  readonly title: string;
  readonly unit: string;
  readonly series: readonly Series[];
};

export function totals(buckets: readonly Bucket[]) {
  const sum = (
    key:
      | "request_count"
      | "error_count"
      | "prompt_tokens"
      | "completion_tokens",
  ) => buckets.reduce((s, b) => s + b[key], 0);
  return {
    requests: sum("request_count"),
    errors: sum("error_count"),
    tokens: sum("prompt_tokens") + sum("completion_tokens"),
    latency: weighted(buckets, "avg_duration_ms"),
    input: weighted(buckets, "input_tps"),
    output: weighted(buckets, "output_tps"),
  };
}

export function activityCharts(
  buckets: readonly Bucket[],
  restarts: readonly Restart[],
  samples: readonly Sample[],
  range: Range,
  fleet: boolean,
): readonly ChartData[] {
  const step = bucketSize(range).ms;
  const grouped = new Map<number, Bucket[]>();
  for (const b of buckets) {
    const rows = grouped.get(b.bucket_start) ?? [];
    rows.push(b);
    grouped.set(b.bucket_start, rows);
  }
  const restartCounts = new Map<number, number>();
  for (const r of restarts) {
    const at = Math.floor(r.at_ms / step) * step;
    restartCounts.set(at, (restartCounts.get(at) ?? 0) + 1);
  }
  const times = new Set<number>();
  if (buckets.length || restarts.length)
    for (
      let at = Math.floor(range.since / step) * step;
      at < range.until;
      at += step
    )
      times.add(at);
  const ordered = [...times].sort((a, b) => a - b);
  const rate = (
    key:
      | "request_count"
      | "prompt_tokens"
      | "completion_tokens"
      | "error_count",
  ) =>
    ordered.map((at) => ({
      at,
      value:
        (grouped.get(at) ?? []).reduce((s, b) => s + b[key], 0) /
        minutes(at, step, range),
    }));
  const average = (
    key: "output_tps" | "input_tps" | "avg_duration_ms",
    divisor = 1,
  ) =>
    ordered.map((at) => ({
      at,
      value: divide(weighted(grouped.get(at) ?? [], key), divisor),
    }));
  const series = (
    label: string,
    color: string,
    points: readonly Point[],
  ): Series => ({ label, color, points });
  const requests: ChartData = {
    title: fleet ? "Request rate" : "Requests",
    unit: "req / min",
    series: [
      series(
        fleet ? "Request rate" : "Requests",
        "var(--accent-data)",
        rate("request_count"),
      ),
    ],
  };
  const speed: ChartData = {
    title: "Token speed",
    unit: "tok / s",
    series: [
      series(
        "Output",
        fleet ? "var(--role-model-accent)" : "#84c9f4",
        average("output_tps"),
      ),
      series("Input", "var(--accent-data)", average("input_tps")),
    ],
  };
  const throughput: ChartData = {
    title: "Throughput",
    unit: "tok / min",
    series: [
      series(
        fleet ? "Output" : "Input",
        fleet ? "var(--accent)" : "var(--accent-data)",
        rate(fleet ? "completion_tokens" : "prompt_tokens"),
      ),
      series(
        fleet ? "Input" : "Output",
        "#84c9f4",
        rate(fleet ? "prompt_tokens" : "completion_tokens"),
      ),
    ],
  };
  const errors: ChartData = {
    title: "Errors & auto-restarts",
    unit: "events / min",
    series: [
      series("Errors", "var(--status-danger)", rate("error_count")),
      series(
        "Auto-restarts",
        "var(--status-warn)",
        ordered.map((at) => ({
          at,
          value: (restartCounts.get(at) ?? 0) / minutes(at, step, range),
        })),
      ),
    ],
  };
  const latency: ChartData = {
    title: fleet ? "Average latency" : "Latency",
    unit: "seconds",
    series: [
      series(
        fleet ? "Average latency" : "Latency",
        "var(--status-warn)",
        average("avg_duration_ms", 1000),
      ),
    ],
  };
  const memory: ChartData = {
    title: "Memory utilisation",
    unit: "% of VRAM",
    series: [
      series(
        "Memory utilisation",
        "var(--text-soft)",
        memoryPoints(samples, step),
      ),
    ],
  };
  return fleet
    ? [requests, throughput, errors, latency, speed, memory]
    : [requests, speed, throughput, errors, latency];
}

export function serviceSpeeds(buckets: readonly Bucket[]) {
  const names = [
    ...new Set(
      buckets
        .map((b) => b.service)
        .filter((name): name is string => name !== null),
    ),
  ];
  return names
    .map((name) => ({
      name,
      value: weighted(
        buckets.filter((b) => b.service === name),
        "output_tps",
      ),
    }))
    .sort((a, b) => (b.value ?? -1) - (a.value ?? -1));
}

function weighted(
  rows: readonly Bucket[],
  key: "output_tps" | "input_tps" | "avg_duration_ms",
) {
  let sum = 0,
    count = 0;
  for (const row of rows) {
    const value = row[key];
    if (value != null && Number.isFinite(value)) {
      sum += value * row.request_count;
      count += row.request_count;
    }
  }
  return count ? sum / count : null;
}

function divide(value: number | null, divisor: number) {
  return value === null ? null : value / divisor;
}
function minutes(at: number, step: number, range: Range) {
  return (
    Math.max(1, Math.min(at + step, range.until) - Math.max(at, range.since)) /
    60_000
  );
}

function memoryPoints(
  samples: readonly Sample[],
  step: number,
): readonly Point[] {
  const bins = new Map<
    number,
    Map<string, { used: number; total: number; count: number }>
  >();
  for (const s of samples) {
    if (!s.device.startsWith("gpu:") || s.total_bytes <= 0) continue;
    const at = Math.floor(s.timestamp_ms / step) * step;
    const devices = bins.get(at) ?? new Map();
    const d = devices.get(s.device) ?? { used: 0, total: 0, count: 0 };
    d.used += s.used_bytes;
    d.total += s.total_bytes;
    d.count++;
    devices.set(s.device, d);
    bins.set(at, devices);
  }
  return [...bins.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([at, devices]) => {
      let used = 0,
        total = 0;
      for (const d of devices.values()) {
        used += d.used / d.count;
        total += d.total / d.count;
      }
      return { at, value: total ? (used / total) * 100 : null };
    });
}
