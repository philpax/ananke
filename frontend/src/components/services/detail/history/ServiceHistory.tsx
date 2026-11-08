// A service's activity over the shared time range: totals, then request,
// token, throughput, and error/restart charts.

import { useTranslation } from "react-i18next";

import { useMetrics, useRestarts } from "../../../../api/hooks.ts";
import {
  aggregateBuckets,
  toErrorsRestartsData,
} from "../../../../api/metrics-aggregate.ts";
import { bucketMsOf } from "../../../../util.ts";
import { Chart, type ChartSeries } from "../../../ui/chart/Chart.tsx";
import { Stat } from "../../../ui/data/Stat.tsx";
import { rangeLabel, type TimeRange } from "../../../ui/form/timeRange.ts";
import type { ResolvedRange } from "../../../ui/form/useTimeRange.ts";
import { Panel } from "../../../ui/layout/Panel.tsx";
import { summariseBuckets } from "./historySummary.ts";

export function ServiceHistory({
  name,
  range,
  resolved,
}: {
  readonly name: string;
  readonly range: TimeRange;
  readonly resolved: ResolvedRange;
}) {
  const { t } = useTranslation();
  const { since, until, end, bucket } = resolved;
  const metrics = useMetrics({ service: name, since, until, bucket });
  const restarts = useRestarts(name, since, until ?? end);
  const buckets = aggregateBuckets(metrics.data?.buckets ?? []);
  const summary = summariseBuckets(buckets);
  const xMin = since / 1000;
  const xMax = end / 1000;
  const fixed = (v: number | null, digits: number, unit = "") =>
    v === null ? "—" : `${v.toFixed(digits)}${unit}`;
  const ts = buckets.map((b) => b.ts);
  const rate = (sum: number, n: number) => (n > 0 ? sum / n : null);

  const tpsSeries: ChartSeries[] = [
    ...(summary.hasSplitTps
      ? ([
          {
            label: t("stats.tpsIn"),
            color: "series-1",
            fill: true,
            unit: "tok/s",
          },
          {
            label: t("stats.tpsOut"),
            color: "series-2",
            fill: true,
            unit: "tok/s",
          },
        ] satisfies ChartSeries[])
      : []),
    {
      label: t("stats.tpsEffective"),
      color: "series-3",
      fill: true,
      unit: "tok/s",
    },
  ];

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat
          label={t("serviceDetail.requestsInPeriod", {
            range: rangeLabel(range, t),
          })}
          value={summary.requests}
        />
        <Stat label={t("serviceDetail.errors")} value={summary.errors} />
        <Stat
          label={t("serviceDetail.inputTokens")}
          value={summary.inputTokens.toLocaleString()}
        />
        <Stat
          label={t("serviceDetail.outputTokens")}
          value={summary.outputTokens.toLocaleString()}
        />
        <Stat
          label={t("serviceDetail.avgLatency")}
          value={fixed(summary.avgLatencyMs, 0, "ms")}
        />
        {summary.hasSplitTps && (
          <>
            <Stat
              label={t("serviceDetail.avgTpsIn")}
              value={fixed(summary.avgInputTps, 1)}
            />
            <Stat
              label={t("serviceDetail.avgTpsOut")}
              value={fixed(summary.avgOutputTps, 1)}
            />
          </>
        )}
        <Stat
          label={t("serviceDetail.avgTpsEffective")}
          value={fixed(summary.avgEffectiveTps, 1)}
        />
      </div>
      <div className="grid grid-cols-1 gap-4 @min-detail-2col/detail:grid-cols-2">
        <Panel heading={t("stats.requestRate")}>
          <Chart
            xMin={xMin}
            xMax={xMax}
            data={[ts, buckets.map((b) => b.requestCount)]}
            series={[
              { label: t("stats.requests"), color: "series-1", fill: true },
            ]}
          />
        </Panel>
        <Panel heading={t("stats.tokenThroughput")}>
          <Chart
            xMin={xMin}
            xMax={xMax}
            data={[
              ts,
              buckets.map((b) => b.promptTokens),
              buckets.map((b) => b.completionTokens),
            ]}
            series={[
              { label: t("stats.tokensIn"), color: "series-1", fill: true },
              { label: t("stats.tokensOut"), color: "series-2", fill: true },
            ]}
          />
        </Panel>
        <Panel heading={t("stats.tokensPerSecond")}>
          <Chart
            xMin={xMin}
            xMax={xMax}
            data={[
              ts,
              ...(summary.hasSplitTps
                ? [
                    buckets.map((b) =>
                      rate(b.totalWeightedInputTps, b.inputTpsRequests),
                    ),
                    buckets.map((b) =>
                      rate(b.totalWeightedOutputTps, b.outputTpsRequests),
                    ),
                  ]
                : []),
              buckets.map((b) =>
                rate(b.totalWeightedEffectiveTps, b.effectiveTpsRequests),
              ),
            ]}
            series={tpsSeries}
          />
        </Panel>
        {/* The error storm and the watchdog firing it provokes belong on
            the same axis. */}
        <Panel heading={t("stats.errorsRestarts")}>
          <Chart
            xMin={xMin}
            xMax={xMax}
            data={toErrorsRestartsData(
              buckets,
              restarts.data ?? [],
              since,
              end,
              bucketMsOf(bucket),
            )}
            series={[
              { label: t("stats.errors"), color: "danger", fill: true },
              { label: t("stats.restarts"), color: "warn", fill: true },
            ]}
          />
        </Panel>
      </div>
    </div>
  );
}
