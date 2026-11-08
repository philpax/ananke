// Every service's output rate and requests over the range, one line per
// service, full width so the legend has room. Shown only for the whole
// fleet, and only when more than one service was active.

import { useTranslation } from "react-i18next";

import type { PerServiceSeries } from "../../api/metrics-aggregate.ts";
import { categorical } from "../ui/chart/series.ts";
import { ChartPanel } from "./ChartPanel.tsx";
import { hasSamples, toMultiSeriesData } from "./metricsViewData.ts";

export function PerServiceCharts({
  perService,
  sinceMs,
  endMs,
}: {
  readonly perService: readonly PerServiceSeries[];
  readonly sinceMs: number;
  readonly endMs: number;
}) {
  const { t } = useTranslation();
  if (perService.length < 2) return null;
  const bounds = { xMin: sinceMs / 1000, xMax: endMs / 1000 };
  const tps = toMultiSeriesData(perService, "outputTps");
  const requests = toMultiSeriesData(perService, "requestCount");
  return (
    <>
      <ChartPanel
        heading={t("stats.perServiceTpsOut")}
        empty={!hasSamples(tps)}
        height={200}
        data={tps}
        series={perService.map((s, i) => ({
          label: s.serviceName,
          color: categorical(i),
          unit: "tok/s",
          // Break the line where a service is idle rather than
          // interpolate across it.
          spanGaps: false,
        }))}
        {...bounds}
      />
      <ChartPanel
        heading={t("stats.perServiceRequests")}
        empty={!hasSamples(requests)}
        height={200}
        data={requests}
        series={perService.map((s, i) => ({
          label: s.serviceName,
          color: categorical(i),
        }))}
        {...bounds}
      />
    </>
  );
}
