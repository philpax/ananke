// Stats / observability view (`/stats`). Charts request rate, token
// throughput, error rate, avg latency, and per-device memory utilisation
// from the daemon's `/api/metrics` and `/api/devices/samples` endpoints.

import { useMemo, useState } from "react";

import { useMetrics, useDeviceSamples, useServices } from "../../api/hooks.ts";
import type {
  MetricBucketResponse,
  DeviceSampleResponse,
} from "../../api/client.ts";
import { Card } from "../ui/Card.tsx";
import { Chart } from "../ui/Chart.tsx";
import { CHART_PALETTE } from "../ui/chart-palette.ts";
import { Spinner } from "../ui/Spinner.tsx";

type TimeRange = { label: string; ms: number; bucket: string };
const RANGES: TimeRange[] = [
  { label: "1h", ms: 3_600_000, bucket: "1m" },
  { label: "6h", ms: 6 * 3_600_000, bucket: "5m" },
  { label: "24h", ms: 24 * 3_600_000, bucket: "1h" },
];

type AggregatedBucket = {
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
};

function aggregateBuckets(buckets: MetricBucketResponse[]): AggregatedBucket[] {
  const map = new Map<number, AggregatedBucket>();
  for (const b of buckets) {
    const ts = Math.floor(b.bucket_start / 1000);
    const ex = map.get(ts);
    if (ex) {
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
    } else {
      map.set(ts, {
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
      });
    }
  }
  return Array.from(map.values()).sort((a, b) => a.ts - b.ts);
}

type PerServiceSeries = {
  serviceName: string;
  buckets: AggregatedBucket[];
};

function groupByService(buckets: MetricBucketResponse[]): PerServiceSeries[] {
  const byService = new Map<string, AggregatedBucket[]>();
  for (const b of buckets) {
    const name = b.service ?? "unknown";
    const arr = byService.get(name) ?? [];
    const ts = Math.floor(b.bucket_start / 1000);
    const ex = arr.find((a) => a.ts === ts);
    if (ex) {
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
    } else {
      arr.push({
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
      });
    }
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

type MemorySeries = {
  device: string;
  data: { ts: number; usedBytes: number }[];
};

function groupDeviceSamples(samples: DeviceSampleResponse[]): MemorySeries[] {
  const byDevice = new Map<string, { ts: number; usedBytes: number }[]>();
  for (const s of samples) {
    const arr = byDevice.get(s.device) ?? [];
    arr.push({
      ts: Math.floor(s.timestamp_ms / 1000),
      usedBytes: s.used_bytes,
    });
    byDevice.set(s.device, arr);
  }
  return Array.from(byDevice.entries()).map(([device, data]) => ({
    device,
    data: data.sort((a, b) => a.ts - b.ts),
  }));
}

type ChartField =
  | "requestCount"
  | "promptTokens"
  | "completionTokens"
  | "errorCount"
  | "totalDurationMs"
  | "outputTps"
  | "inputTps";

function resolveField(b: AggregatedBucket, field: ChartField): number | null {
  switch (field) {
    case "totalDurationMs":
      return b.timedRequests > 0 ? b.totalDurationMs / b.timedRequests : null;
    case "outputTps":
      return b.outputTpsRequests > 0
        ? b.totalWeightedOutputTps / b.outputTpsRequests
        : null;
    case "inputTps":
      return b.inputTpsRequests > 0
        ? b.totalWeightedInputTps / b.inputTpsRequests
        : null;
    default:
      return b[field] as number;
  }
}

function toLineData(
  buckets: AggregatedBucket[],
  field: ChartField,
): (number | null)[][] {
  return [buckets.map((b) => b.ts), buckets.map((b) => resolveField(b, field))];
}

function toMultiSeriesData(
  seriesList: PerServiceSeries[],
  field: ChartField,
): (number | null)[][] {
  const allTs = new Set<number>();
  for (const s of seriesList) {
    for (const b of s.buckets) allTs.add(b.ts);
  }
  const ts = Array.from(allTs).sort((a, b) => a - b);
  const result: (number | null)[][] = [ts];
  for (const s of seriesList) {
    const map = new Map(s.buckets.map((b) => [b.ts, b]));
    const values = ts.map((t) => {
      const b = map.get(t);
      if (!b) return null;
      return resolveField(b, field);
    });
    result.push(values);
  }
  return result;
}

function toMemoryData(series: MemorySeries[]): (number | null)[][] {
  if (series.length === 0) return [[]];
  const allTs = new Set<number>();
  for (const s of series) {
    for (const d of s.data) allTs.add(d.ts);
  }
  const ts = Array.from(allTs).sort((a, b) => a - b);
  const result: (number | null)[][] = [ts];
  for (const s of series) {
    const map = new Map(s.data.map((d) => [d.ts, d.usedBytes / 1e9]));
    result.push(ts.map((t) => map.get(t) ?? null));
  }
  return result;
}

export function MetricsView() {
  const [rangeIdx, setRangeIdx] = useState(0);
  const [since, setSince] = useState(() => Date.now() - RANGES[0].ms);
  const [serviceFilter, setServiceFilter] = useState<string>("");

  const range = RANGES[rangeIdx];
  const xMin = since / 1000;
  const xMax = (since + range.ms) / 1000;

  const services = useServices();
  const metrics = useMetrics({
    service: serviceFilter || undefined,
    since,
    bucket: range.bucket,
  });
  const deviceSamples = useDeviceSamples(undefined, since);

  const loading = metrics.isPending;

  const aggregated = useMemo(
    () => aggregateBuckets(metrics.data?.buckets ?? []),
    [metrics.data],
  );

  const perService = useMemo(
    () => groupByService(metrics.data?.buckets ?? []),
    [metrics.data],
  );

  const memorySeries = useMemo(
    () => groupDeviceSamples(deviceSamples.data ?? []),
    [deviceSamples.data],
  );

  return (
    <div className="flex h-full flex-col">
      <header className="flex h-14 shrink-0 items-center gap-3 border-b border-border-default px-4">
        <h1 className="eyebrow !text-primary">Stats</h1>
        <div className="flex items-center gap-1">
          {RANGES.map((r, i) => (
            <button
              key={r.label}
              onClick={() => {
                setRangeIdx(i);
                setSince(Date.now() - RANGES[i].ms);
              }}
              className={`rounded-sm px-2 py-0.5 text-xs transition-colors ${
                i === rangeIdx
                  ? "bg-elevated text-primary"
                  : "text-tertiary hover:text-secondary"
              }`}
            >
              {r.label}
            </button>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-2">
          <select
            value={serviceFilter}
            onChange={(e) => setServiceFilter(e.target.value)}
            className="h-7 rounded-sm border border-border-default bg-base px-2 text-xs text-primary focus:border-accent focus:outline-none"
          >
            <option value="">All services</option>
            {(services.data ?? [])
              .filter((s) => s.modality !== "embedding")
              .map((s) => (
                <option key={s.name} value={s.name}>
                  {s.name}
                </option>
              ))}
          </select>
        </div>
      </header>

      <div className="flex-1 overflow-auto p-4">
        {loading && !metrics.data ? (
          <div className="flex h-full items-center justify-center">
            <Spinner />
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {/* Request rate */}
            <Card header="Request rate">
              <Chart
                xMin={xMin}
                xMax={xMax}
                data={toLineData(aggregated, "requestCount")}
                series={[
                  {
                    label: "Requests",
                    stroke: CHART_PALETTE[0],
                    fill: "rgba(139,124,248,0.08)",
                  },
                ]}
              />
            </Card>

            {/* Token throughput */}
            <Card header="Token throughput">
              <Chart
                xMin={xMin}
                xMax={xMax}
                data={[
                  aggregated.map((b) => b.ts),
                  aggregated.map((b) => b.promptTokens),
                  aggregated.map((b) => b.completionTokens),
                ]}
                series={[
                  {
                    label: "Prompt",
                    stroke: CHART_PALETTE[0],
                    fill: "rgba(139,124,248,0.08)",
                  },
                  {
                    label: "Completion",
                    stroke: CHART_PALETTE[1],
                    fill: "rgba(69,201,138,0.08)",
                  },
                ]}
              />
            </Card>

            {/* Error rate */}
            <Card header="Error rate">
              <Chart
                xMin={xMin}
                xMax={xMax}
                data={toLineData(aggregated, "errorCount")}
                series={[
                  {
                    label: "Errors",
                    stroke: CHART_PALETTE[5],
                    fill: "rgba(239,90,90,0.08)",
                  },
                ]}
              />
            </Card>

            {/* Avg latency */}
            <Card header="Avg latency">
              <Chart
                xMin={xMin}
                xMax={xMax}
                data={toLineData(aggregated, "totalDurationMs")}
                series={[
                  {
                    label: "Duration (ms)",
                    stroke: CHART_PALETTE[2],
                    fill: "rgba(224,168,60,0.08)",
                  },
                ]}
              />
            </Card>

            {/* Tokens per second */}
            <Card header="Tokens per second">
              <Chart
                xMin={xMin}
                xMax={xMax}
                data={[
                  aggregated.map((b) => b.ts),
                  aggregated.map((b) => resolveField(b, "inputTps")),
                  aggregated.map((b) => resolveField(b, "outputTps")),
                ]}
                series={[
                  {
                    label: "Input",
                    stroke: CHART_PALETTE[0],
                    fill: "rgba(139,124,248,0.08)",
                    unit: "tok/s",
                  },
                  {
                    label: "Output",
                    stroke: CHART_PALETTE[1],
                    fill: "rgba(69,201,138,0.08)",
                    unit: "tok/s",
                  },
                ]}
              />
            </Card>

            {/* Memory utilisation */}
            <Card header="Memory utilisation">
              <Chart
                xMin={xMin}
                xMax={xMax}
                data={toMemoryData(memorySeries)}
                series={memorySeries.map((s, i) => ({
                  label: s.device,
                  stroke: CHART_PALETTE[i % CHART_PALETTE.length],
                  fill: `rgba(139,124,248,${0.04 + 0.06 * i})`,
                  unit: "GB",
                }))}
              />
            </Card>

            {/* Per-service output TPS */}
            {perService.length > 1 && !serviceFilter && (
              <Card header="Per-service output TPS" className="lg:col-span-2">
                <Chart
                  xMin={xMin}
                  xMax={xMax}
                  height={200}
                  data={toMultiSeriesData(perService, "outputTps")}
                  series={perService.map((s, i) => ({
                    label: s.serviceName,
                    stroke: CHART_PALETTE[i % CHART_PALETTE.length],
                    unit: "tok/s",
                  }))}
                />
              </Card>
            )}

            {/* Per-service request breakdown */}
            {perService.length > 1 && !serviceFilter && (
              <Card header="Per-service requests" className="lg:col-span-2">
                <Chart
                  xMin={xMin}
                  xMax={xMax}
                  height={200}
                  data={toMultiSeriesData(perService, "requestCount")}
                  series={perService.map((s, i) => ({
                    label: s.serviceName,
                    stroke: CHART_PALETTE[i % CHART_PALETTE.length],
                  }))}
                />
              </Card>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
