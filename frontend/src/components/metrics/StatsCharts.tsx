// The Stats charts in a grid that goes one, two, or three across by its
// own width: the fleet's (or one service's) activity, then memory per
// device, GPUs and host apart.

import { useTranslation } from "react-i18next";

import type { ServiceRestartEntry } from "../../api/client.ts";
import {
  toErrorsRestartsData,
  type AggregatedBucket,
} from "../../api/metrics-aggregate.ts";
import type { ChartSeries } from "../ui/chart/Chart.tsx";
import { deviceSeries } from "../ui/chart/series.ts";
import { ChartPanel } from "./ChartPanel.tsx";
import {
  hasSamples,
  hasSplitTps,
  toFieldsData,
  toMemoryData,
  type MemorySeries,
} from "./metricsViewData.ts";

export type StatsChartsProps = {
  readonly buckets: readonly AggregatedBucket[];
  readonly restarts: readonly ServiceRestartEntry[];
  readonly gpuMemory: readonly MemorySeries[];
  readonly hostMemory: readonly MemorySeries[];
  readonly sinceMs: number;
  readonly endMs: number;
  readonly bucketMs: number;
};

export function StatsCharts({
  buckets,
  restarts,
  gpuMemory,
  hostMemory,
  sinceMs,
  endMs,
  bucketMs,
}: StatsChartsProps) {
  const { t } = useTranslation();
  const bounds = { xMin: sinceMs / 1000, xMax: endMs / 1000 };
  const split = hasSplitTps(buckets);
  const requests = toFieldsData(buckets, ["requestCount"]);
  const tokens = toFieldsData(buckets, ["promptTokens", "completionTokens"]);
  const latency = toFieldsData(buckets, ["totalDurationMs"]);
  const tps = toFieldsData(
    buckets,
    split ? ["inputTps", "outputTps", "effectiveTps"] : ["effectiveTps"],
  );
  const gpuData = toMemoryData(gpuMemory);
  const hostData = toMemoryData(hostMemory);
  const memorySeries = (series: readonly MemorySeries[]): ChartSeries[] =>
    series.map((s) => ({
      label: s.device,
      color: deviceSeries(s.device),
      fill: true,
      unit: "GB",
    }));
  // Decode and prefill rates are undefined while nothing generates; break
  // the line rather than interpolate a stale value across a stall.
  const rate = { fill: true, unit: "tok/s", spanGaps: false } as const;

  return (
    <div className="@container/charts">
      <div className="grid grid-cols-1 gap-4 @min-chart-2/charts:grid-cols-2 @min-chart-3/charts:grid-cols-3">
        <ChartPanel
          heading={t("stats.requestRate")}
          empty={!hasSamples(requests)}
          data={requests}
          series={[
            { label: t("stats.requests"), color: "series-1", fill: true },
          ]}
          {...bounds}
        />
        <ChartPanel
          heading={t("stats.tokenThroughput")}
          empty={!hasSamples(tokens)}
          data={tokens}
          series={[
            { label: t("stats.tokensIn"), color: "series-1", fill: true },
            { label: t("stats.tokensOut"), color: "series-2", fill: true },
          ]}
          {...bounds}
        />
        {/* The error storm and the watchdog firing it provokes belong on
            the same axis. */}
        <ChartPanel
          heading={t("stats.errorsRestarts")}
          empty={buckets.length === 0 && restarts.length === 0}
          data={toErrorsRestartsData(
            buckets,
            restarts,
            sinceMs,
            endMs,
            bucketMs,
          )}
          series={[
            { label: t("stats.errors"), color: "danger", fill: true },
            { label: t("stats.restarts"), color: "warn", fill: true },
          ]}
          {...bounds}
        />
        <ChartPanel
          heading={t("stats.avgLatency")}
          empty={!hasSamples(latency)}
          data={latency}
          series={[
            {
              label: t("stats.latency"),
              color: "series-3",
              fill: true,
              unit: "ms",
            },
          ]}
          {...bounds}
        />
        {/* The effective line is always there; the input and output rates
            join it when some request in the range reported them. */}
        <ChartPanel
          heading={t("stats.tokensPerSecond")}
          empty={!hasSamples(tps)}
          data={tps}
          series={[
            ...(split
              ? ([
                  { label: t("stats.tpsIn"), color: "series-1", ...rate },
                  { label: t("stats.tpsOut"), color: "series-2", ...rate },
                ] satisfies ChartSeries[])
              : []),
            { label: t("stats.tpsEffective"), color: "series-3", ...rate },
          ]}
          {...bounds}
        />
        <ChartPanel
          heading={t("stats.gpuMemory")}
          empty={!hasSamples(gpuData)}
          data={gpuData}
          series={memorySeries(gpuMemory)}
          {...bounds}
        />
        <ChartPanel
          heading={t("stats.hostMemory")}
          empty={!hasSamples(hostData)}
          data={hostData}
          series={memorySeries(hostMemory)}
          {...bounds}
        />
      </div>
    </div>
  );
}
