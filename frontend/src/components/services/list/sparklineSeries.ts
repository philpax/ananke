// Per-service request-rate sparklines, split out of one metrics query over
// every service.

import type { MetricBucketResponse } from "../../../api/client.ts";

export type ServiceSparkData = Map<string, { ts: number[]; counts: number[] }>;

export function buildServiceSparkline(
  buckets: readonly MetricBucketResponse[],
): ServiceSparkData {
  const byService = new Map<string, Map<number, number>>();
  for (const b of buckets) {
    const name = b.service ?? "unknown";
    let m = byService.get(name);
    if (!m) {
      m = new Map();
      byService.set(name, m);
    }
    const ts = Math.floor(b.bucket_start / 1000);
    m.set(ts, (m.get(ts) ?? 0) + b.request_count);
  }
  const result = new Map<string, { ts: number[]; counts: number[] }>();
  for (const [name, m] of byService) {
    const sorted = [...m.entries()].sort((a, b) => a[0] - b[0]);
    result.set(name, {
      ts: sorted.map((e) => e[0]),
      counts: sorted.map((e) => e[1]),
    });
  }
  return result;
}
