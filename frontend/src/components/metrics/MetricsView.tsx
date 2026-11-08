// Stats (`/stats`): the fleet's activity over the shared time range, or one
// service's. Totals as tiles, then request, token, error and restart,
// latency, token-rate, and memory charts from `/api/metrics`,
// `/api/restarts`, and `/api/devices/samples`, then, across the fleet, each
// service's output rate ranked and each service's activity charted.

import { useState } from "react";
import { useTranslation } from "react-i18next";

import {
  useDeviceSamples,
  useMetrics,
  useRestarts,
  useServices,
} from "../../api/hooks.ts";
import {
  aggregateBuckets,
  groupByService,
} from "../../api/metrics-aggregate.ts";
import { bucketMsOf } from "../../util.ts";
import { summariseBuckets } from "../services/detail/history/historySummary.ts";
import { Message } from "../ui/feedback/Message.tsx";
import { Select } from "../ui/form/Select.tsx";
import { TimeRangeSelect } from "../ui/form/TimeRangeSelect.tsx";
import { useResolvedRange, useTimeRange } from "../ui/form/useTimeRange.ts";
import { PageHeader } from "../ui/layout/PageHeader.tsx";
import { FleetSummary } from "./FleetSummary.tsx";
import { PerServiceCharts } from "./PerServiceCharts.tsx";
import { ServiceSpeeds } from "./ServiceSpeeds.tsx";
import { StatsCharts } from "./StatsCharts.tsx";
import {
  groupDeviceSamples,
  serviceSpeeds,
  splitMemorySeries,
} from "./metricsViewData.ts";

export function MetricsView() {
  const { t } = useTranslation();
  const [range, setRange] = useTimeRange();
  const { since, until, end, bucket } = useResolvedRange(range);
  const [serviceFilter, setServiceFilter] = useState("");
  const service = serviceFilter === "" ? undefined : serviceFilter;

  const services = useServices();
  const metrics = useMetrics({ service, since, until, bucket });
  const deviceSamples = useDeviceSamples(undefined, since, until);
  const restarts = useRestarts(service, since, until ?? end);

  const rawBuckets = metrics.data?.buckets ?? [];
  const buckets = aggregateBuckets(rawBuckets);
  const perService = groupByService(rawBuckets);
  const memory = splitMemorySeries(
    groupDeviceSamples(deviceSamples.data ?? []),
  );
  const fleet = services.data ?? [];
  const errors = [metrics.error, restarts.error, deviceSamples.error].filter(
    (e) => e !== null,
  );

  return (
    <div className="flex h-full min-h-0 flex-col">
      <PageHeader
        title={t("stats.title")}
        meta={service ?? t("stats.allServices")}
        actions={
          <label className="flex items-center gap-2 text-xs text-muted">
            {t("stats.service")}
            <Select
              size="sm"
              className="max-w-48 text-xs"
              value={serviceFilter}
              onChange={(e) => setServiceFilter(e.target.value)}
            >
              <option value="">{t("stats.allServices")}</option>
              {fleet
                .filter((s) => s.modality !== "embedding")
                .map((s) => (
                  <option key={s.name} value={s.name}>
                    {s.name}
                  </option>
                ))}
            </Select>
          </label>
        }
      />
      <div className="shrink-0 border-b border-line bg-surface px-4 py-2">
        <TimeRangeSelect value={range} onChange={setRange} />
      </div>
      <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-auto bg-canvas p-4">
        {errors.map((e) => (
          <Message key={e.message} kind="error" text={e.message} />
        ))}
        <FleetSummary
          summary={summariseBuckets(buckets)}
          services={
            service === undefined
              ? {
                  running: fleet.filter((s) => s.state === "running").length,
                  total: fleet.length,
                }
              : undefined
          }
          loading={metrics.isPending}
        />
        <StatsCharts
          buckets={buckets}
          restarts={restarts.data ?? []}
          gpuMemory={memory.gpus}
          hostMemory={memory.host}
          sinceMs={since}
          endMs={end}
          bucketMs={bucketMsOf(bucket)}
        />
        {/* Each service apart only makes sense across the fleet; with one
            picked, these would repeat the charts above. */}
        {service === undefined && (
          <>
            <ServiceSpeeds speeds={serviceSpeeds(perService)} />
            <PerServiceCharts
              perService={perService}
              sinceMs={since}
              endMs={end}
            />
          </>
        )}
      </div>
    </div>
  );
}
