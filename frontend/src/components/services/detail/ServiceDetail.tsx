// One service's detail pane: the header, its configuration and placement
// panels, and its activity over the shared time range (history, recent
// events, and logs). The pane is its own scroller and container, so the
// panels go two-up by the pane's width rather than the window's.

import { useId } from "react";
import { useTranslation } from "react-i18next";

import type { ServiceSummary } from "../../../api/client.ts";
import { useServiceDetail } from "../../../api/hooks.ts";
import { LogsViewer } from "../../logs/LogsViewer.tsx";
import { Message } from "../../ui/feedback/Message.tsx";
import { TimeRangeSelect } from "../../ui/form/TimeRangeSelect.tsx";
import { useResolvedRange, useTimeRange } from "../../ui/form/useTimeRange.ts";
import { Panel } from "../../ui/layout/Panel.tsx";
import { SectionHeading } from "../../ui/layout/SectionHeading.tsx";
import { ConfigurationPanel } from "./ConfigurationPanel.tsx";
import { DetailHeader, type HeaderService } from "./DetailHeader.tsx";
import { EstimatePanel } from "./EstimatePanel.tsx";
import { LaunchCommandPanel } from "./LaunchCommandPanel.tsx";
import { ModelPanel } from "./ModelPanel.tsx";
import { PlacementPanel } from "./PlacementPanel.tsx";
import { ServingPanel } from "./ServingPanel.tsx";
import { ServiceEvents } from "./history/ServiceEvents.tsx";
import { ServiceHistory } from "./history/ServiceHistory.tsx";

export function ServiceDetail({
  name,
  summary,
  listHidden,
  serviceCount,
  onBrowse,
}: {
  readonly name: string;
  readonly summary: ServiceSummary | undefined;
  readonly listHidden: boolean;
  readonly serviceCount: number;
  readonly onBrowse: () => void;
}) {
  const { t } = useTranslation();
  const titleId = useId();
  const detail = useServiceDetail(name);
  const [range, setRange] = useTimeRange();
  const resolved = useResolvedRange(range);
  const d = detail.data;
  const header: HeaderService | undefined = d
    ? { ...d, has_mmproj: d.model_info?.has_mmproj }
    : summary;

  return (
    <section
      aria-labelledby={titleId}
      className="@container/detail h-full min-h-0 overflow-y-auto bg-canvas"
    >
      {header ? (
        <DetailHeader
          service={header}
          titleId={titleId}
          listHidden={listHidden}
          serviceCount={serviceCount}
          onBrowse={onBrowse}
        />
      ) : (
        <h1 id={titleId} className="sr-only">
          {name}
        </h1>
      )}
      <div className="flex flex-col gap-4 p-4">
        {detail.isPending && <Message kind="loading" />}
        {detail.error && (
          <Message
            kind="error"
            text={
              summary === undefined
                ? t("serviceDetail.notFound")
                : detail.error.message
            }
          />
        )}
        {d && (
          <div className="grid grid-cols-1 gap-4 @min-detail-2col/detail:grid-cols-2">
            <PlacementPanel detail={d} />
            <ConfigurationPanel detail={d} />
            {d.model_info && <ModelPanel model={d.model_info} />}
            {d.serving && (
              <ServingPanel serving={d.serving} runtime={d.runtime} />
            )}
            {d.estimate && <EstimatePanel estimate={d.estimate} detail={d} />}
            <div className="@min-detail-2col/detail:col-span-2">
              <LaunchCommandPanel name={name} />
            </div>
          </div>
        )}
        <SectionHeading
          className="flex flex-wrap items-center gap-2 pt-2"
          actions={<TimeRangeSelect value={range} onChange={setRange} />}
        >
          {t("services.detail.activity")}
        </SectionHeading>
        {d?.modality !== "embedding" && (
          <ServiceHistory name={name} range={range} resolved={resolved} />
        )}
        <ServiceEvents name={name} resolved={resolved} />
        <Panel heading={t("serviceDetail.logs")} padding="none">
          <LogsViewer name={name} range={range} />
        </Panel>
      </div>
    </section>
  );
}
